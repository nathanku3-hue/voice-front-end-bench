#pragma once

// The two sample-domain kernels that the stage budget blamed for 43% of the
// frame, in both ISA variants.
//
// They live in a header rather than in streaming_fbank.cpp for one reason: the
// disassembly audit (tools/kernel_asm_probe.cpp) has to compile exactly these
// functions and nothing else, so the instruction mix it counts belongs to the
// kernel and not to whatever the inliner did around it.
//
// Naming: "Scalar" is the pre-SIMD reference (modulo ring indexing, one
// dependent accumulator). "Contiguous" is the same arithmetic on the aligned
// staging buffer. "Avx2" is the vector form. All three compute the same
// function; only the rounding order differs, and the artifact that publishes
// the speed-up also publishes the resulting feature difference.

#include <cstddef>
#include <cstdint>

#include "cpu_isa.h"
#include "fast_log.h"

namespace voice {
namespace kernels {

constexpr double kPreemphasis = 0.97;
constexpr std::size_t kFrameSize = 400;

// ---------------------------------------------------------------- scalar ----
// Reads the ring directly with a modulo index. This is what the compiler sees
// when it gives up on vectorising: a data-dependent index it cannot prove
// contiguous, plus a single FP accumulator with a 4-cycle dependent add.

inline double MeanScalarKernel(const float* ring, std::size_t head, std::size_t n) noexcept {
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) sum += static_cast<double>(ring[(head + i) % kFrameSize]);
  return sum;
}

inline void PreemphWindowScalarKernel(const float* ring, std::size_t head, const double* window,
                                      double* out, double mean, std::size_t n) noexcept {
  double previous = static_cast<double>(ring[head]) - mean;
  for (std::size_t i = 0; i < n; ++i) {
    const double centered = static_cast<double>(ring[(head + i) % kFrameSize]) - mean;
    const double preemphasized = centered - kPreemphasis * previous;
    out[i] = preemphasized * window[i];
    previous = centered;
  }
}

// ----------------------------------------------------------- contiguous -----
// Same arithmetic, no modulo: the frame has already been copied into a 400-float
// buffer that starts on a 64-byte boundary.

inline double MeanContiguousKernel(const float* __restrict x, std::size_t n) noexcept {
  const float* p = static_cast<const float*>(__builtin_assume_aligned(x, 64));
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) sum += static_cast<double>(p[i]);
  return sum;
}

inline void PreemphWindowContiguousKernel(const float* __restrict x, const double* __restrict window,
                                          double* __restrict out, double mean,
                                          std::size_t n) noexcept {
  const float* p = static_cast<const float*>(__builtin_assume_aligned(x, 64));
  double previous = static_cast<double>(p[0]) - mean;
  for (std::size_t i = 0; i < n; ++i) {
    const double centered = static_cast<double>(p[i]) - mean;
    const double preemphasized = centered - kPreemphasis * previous;
    out[i] = preemphasized * window[i];
    previous = centered;
  }
}

#ifdef VOICE_HAVE_X86_INTRINSICS

// ----------------------------------------------------------------- avx2 -----
// 400 samples = 50 aligned 8-float loads, split into two 4-wide double
// accumulators so neither the load nor the add chain is the bottleneck.

__attribute__((target("avx2,fma"))) inline double MeanAvx2Kernel(const float* __restrict x,
                                                                 std::size_t n) noexcept {
  const float* p = static_cast<const float*>(__builtin_assume_aligned(x, 64));
  __m256d lo = _mm256_setzero_pd();
  __m256d hi = _mm256_setzero_pd();
  std::size_t i = 0;
  for (; i + 7 < n; i += 8) {
    const __m256 v = _mm256_load_ps(p + i);  // vmovaps: 32-byte aligned, no split
    lo = _mm256_add_pd(lo, _mm256_cvtps_pd(_mm256_castps256_ps128(v)));
    hi = _mm256_add_pd(hi, _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1)));
  }
  const __m256d sum = _mm256_add_pd(lo, hi);
  const __m128d halves = _mm_add_pd(_mm256_castpd256_pd128(sum), _mm256_extractf128_pd(sum, 1));
  double total = _mm_cvtsd_f64(halves) + _mm_cvtsd_f64(_mm_unpackhi_pd(halves, halves));
  for (; i < n; ++i) total += static_cast<double>(p[i]);
  return total;
}

// The frame is 400 samples, so the first four are peeled off to land the vector
// loop on a 32-byte boundary. After that every window load and output store is
// vmovapd; only the deliberately one-sample-shifted pre-emphasis load is
// unaligned, and it has to be -- that is the stencil.
__attribute__((target("avx2,fma"))) inline void PreemphWindowAvx2Kernel(
    const float* __restrict x, const double* __restrict window, double* __restrict out,
    double mean, std::size_t n) noexcept {
  const float* p = static_cast<const float*>(__builtin_assume_aligned(x, 64));
  const double* w = static_cast<const double*>(__builtin_assume_aligned(window, 64));
  double* o = static_cast<double*>(__builtin_assume_aligned(out, 64));
  const __m256d mean_v = _mm256_set1_pd(mean);
  const __m256d pre = _mm256_set1_pd(kPreemphasis);
  std::size_t i = 0;
  for (; i < 4 && i < n; ++i) {
    const double centered = static_cast<double>(p[i]) - mean;
    const double previous = i == 0 ? centered : static_cast<double>(p[i - 1]) - mean;
    o[i] = (centered - kPreemphasis * previous) * w[i];
  }
  for (; i + 3 < n; i += 4) {
    const __m256d centered = _mm256_sub_pd(_mm256_cvtps_pd(_mm_load_ps(p + i)), mean_v);
    const __m256d previous = _mm256_sub_pd(_mm256_cvtps_pd(_mm_loadu_ps(p + i - 1)), mean_v);
    const __m256d y = _mm256_fnmadd_pd(pre, previous, centered);  // centered - 0.97*previous
    _mm256_store_pd(o + i, _mm256_mul_pd(y, _mm256_load_pd(w + i)));
  }
  for (; i < n; ++i) {
    const double centered = static_cast<double>(p[i]) - mean;
    const double previous = static_cast<double>(p[i - 1]) - mean;
    o[i] = (centered - kPreemphasis * previous) * w[i];
  }
}

// 80 mel bins = ten 8-wide logs, widened back into the double Feature array so
// nothing downstream changes type.
__attribute__((target("avx2,fma"))) inline void FastLogAvx2Kernel(const float* __restrict in,
                                                                 double* __restrict out,
                                                                 std::size_t n) noexcept {
  // The input buffer is ours (alignas(32) scratch), so the load is a vmovaps.
  // The output is the caller's Feature array, which is only 8-byte aligned, so
  // the two widening stores stay unaligned on purpose: that alignment is not
  // ours to declare.
  const float* p = static_cast<const float*>(__builtin_assume_aligned(in, 32));
  std::size_t i = 0;
  for (; i + 7 < n; i += 8) {
    const __m256 v = FastLog8(_mm256_load_ps(p + i));
    _mm256_storeu_pd(out + i, _mm256_cvtps_pd(_mm256_castps256_ps128(v)));
    _mm256_storeu_pd(out + i + 4, _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1)));
  }
  for (; i < n; ++i) out[i] = static_cast<double>(FastLog(p[i]));
}

#endif  // VOICE_HAVE_X86_INTRINSICS

}  // namespace kernels
}  // namespace voice
