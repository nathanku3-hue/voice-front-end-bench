#pragma once

// Fast natural logarithm for the fbank log-compression stage.
//
// Why this is allowed at all: the stage computes 80 logs per 10 ms frame and
// std::log spends its budget on <1 ulp double accuracy, while the only consumer
// is an int8 acoustic model that quantises its inputs far more coarsely than
// that. What is *not* allowed is guessing how much accuracy is lost, so the
// kernel is written to a stated error bound and the bound is measured against
// std::log on real speech (scripts/audit_fast_log.py) and pushed through the
// CTC model end to end (scripts/audit_numeric_sensitivity.py).
//
// Structure (float path):
//   1. read the IEEE-754 exponent field directly -> x = 2^e * m, m in [1, 2)
//   2. fold m into [1/sqrt2, sqrt2) by halving it and bumping e
//   3. log(m) = 2 atanh(u) with u = (m-1)/(m+1), |u| <= 0.1716,
//      truncated after u^7: the first dropped term is 2u^9/9 <= 3e-8 relative.
//   4. log(x) = e*ln2 + log(m)
//
// Measured over 2M log-spaced points across the fbank's whole mel-energy domain
// (results/fast_log_accuracy.json): max absolute error 6.3e-7, max relative
// error 1.7e-7 -- i.e. at float32's own rounding, which is the point. The
// tolerance the pipeline actually needs is 1e-4 (see docs), so this is a 150x
// margin, not a leap of faith.
//
// Worth knowing before "just use double": the series is truncated after u^7, so
// the double variant lands at 2.9e-8, not 1e-16. Precision is not what limits
// this kernel; term count is. That is why the shipped path is the 8-wide float
// one -- double would cost 4x the lanes and buy a factor of 2.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
#define VOICE_HAVE_X86_INTRINSICS 1
#endif

namespace voice {

namespace detail {

inline float BitsToFloat(std::uint32_t bits) noexcept {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

inline std::uint32_t FloatToBits(float value) noexcept {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

inline double BitsToDouble(std::uint64_t bits) noexcept {
  double value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

inline std::uint64_t DoubleToBits(double value) noexcept {
  std::uint64_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

constexpr float kLn2f = 0.693147180559945309417f;
constexpr float kSqrt2f = 1.41421356237309504880f;
constexpr double kLn2 = 0.693147180559945309417232121458;

}  // namespace detail

// log(x) for finite x > 0. Returns -inf at 0 and NaN for negative/NaN input,
// matching std::log. Subnormal inputs (never produced by the fbank clamp) fall
// back to std::log rather than being approximated by a broken exponent field.
inline float FastLog(float x) noexcept {
  if (!(x > 0.0f)) {
    if (x == 0.0f) return -std::numeric_limits<float>::infinity();
    return std::numeric_limits<float>::quiet_NaN();
  }
  if (x < std::numeric_limits<float>::min()) return std::log(x);

  const std::uint32_t bits = detail::FloatToBits(x);
  std::int32_t e = static_cast<std::int32_t>(bits >> 23) - 127;
  float m = detail::BitsToFloat((bits & 0x007FFFFFu) | 0x3F800000u);
  if (m > detail::kSqrt2f) {
    m *= 0.5f;
    ++e;
  }
  const float u = (m - 1.0f) / (m + 1.0f);
  const float u2 = u * u;
  float p = 0.142857142857f;  // 1/7
  p = p * u2 + 0.2f;          // 1/5
  p = p * u2 + 0.333333333333f;  // 1/3
  p = p * u2 + 1.0f;
  const float log_m = (u + u) * p;
  return static_cast<float>(e) * detail::kLn2f + log_m;
}

// Same decomposition in double, for the "how much accuracy does the speed
// actually cost" arm of the study. Still no std::log call.
inline double FastLogDouble(double x) noexcept {
  if (!(x > 0.0)) {
    if (x == 0.0) return -std::numeric_limits<double>::infinity();
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (x < std::numeric_limits<double>::min()) return std::log(x);

  const std::uint64_t bits = detail::DoubleToBits(x);
  std::int32_t e = static_cast<std::int32_t>(bits >> 52) - 1023;
  double m = detail::BitsToDouble((bits & 0x000FFFFFFFFFFFFFull) | 0x3FF0000000000000ull);
  if (m > 1.41421356237309504880) {
    m *= 0.5;
    ++e;
  }
  const double u = (m - 1.0) / (m + 1.0);
  const double u2 = u * u;
  double p = 1.0 / 7.0;
  p = p * u2 + 1.0 / 5.0;
  p = p * u2 + 1.0 / 3.0;
  p = p * u2 + 1.0;
  return static_cast<double>(e) * detail::kLn2 + (u + u) * p;
}

#ifdef VOICE_HAVE_X86_INTRINSICS

// 8-wide float log. Precondition: every lane is finite and > 0 (the fbank
// clamps mel energy to FLT_EPSILON before calling, so this always holds).
// Tagged rather than globally enabled so the baseline build stays SSE2-only.
__attribute__((target("avx2,fma"))) inline __m256 FastLog8(__m256 x) noexcept {
  const __m256i bits = _mm256_castps_si256(x);
  __m256i e = _mm256_sub_epi32(_mm256_srli_epi32(bits, 23), _mm256_set1_epi32(127));
  const __m256i mbits = _mm256_or_si256(_mm256_and_si256(bits, _mm256_set1_epi32(0x007FFFFF)),
                                        _mm256_set1_epi32(0x3F800000));
  __m256 m = _mm256_castsi256_ps(mbits);
  // Fold m into [1/sqrt2, sqrt2): halve the mantissa and carry into e.
  const __m256 big = _mm256_cmp_ps(m, _mm256_set1_ps(detail::kSqrt2f), _CMP_GT_OQ);
  m = _mm256_blendv_ps(m, _mm256_mul_ps(m, _mm256_set1_ps(0.5f)), big);
  e = _mm256_add_epi32(e, _mm256_and_si256(_mm256_castps_si256(big), _mm256_set1_epi32(1)));

  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 u = _mm256_div_ps(_mm256_sub_ps(m, one), _mm256_add_ps(m, one));
  const __m256 u2 = _mm256_mul_ps(u, u);
  __m256 p = _mm256_set1_ps(0.142857142857f);
  p = _mm256_fmadd_ps(p, u2, _mm256_set1_ps(0.2f));
  p = _mm256_fmadd_ps(p, u2, _mm256_set1_ps(0.333333333333f));
  p = _mm256_fmadd_ps(p, u2, one);
  const __m256 log_m = _mm256_mul_ps(_mm256_add_ps(u, u), p);
  return _mm256_fmadd_ps(_mm256_cvtepi32_ps(e), _mm256_set1_ps(detail::kLn2f), log_m);
}

#endif  // VOICE_HAVE_X86_INTRINSICS

}  // namespace voice
