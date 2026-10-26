// Accuracy audit for the fast-log kernel.
//
// The kernel is only allowed to ship if its error is *bounded and measured*,
// not "small enough in practice". This test sweeps the domain the fbank
// actually produces (mel energy is clamped to FLT_EPSILON at the bottom and
// tops out around 1e6 for a full-scale frame), compares against std::log, and
// fails if the error exceeds the published bound.
//
// It also measures a degree-2 Taylor variant on the same sweep. That variant is
// the thing the "2nd-order is enough" instinct would ship; the numbers here are
// why the shipped kernel carries four terms instead.

#include "fast_log.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Error is not uniform over the domain: the mantissa reduction makes it depend
// on where x falls inside its binade, and the e*ln2 product makes it grow with
// |log x|. A single max number hides that, so the sweep also keeps a per-decade
// profile -- that is what the figure plots.
constexpr int kDecadeLow = -8;
constexpr int kDecadeHigh = 7;
constexpr int kDecadeCount = kDecadeHigh - kDecadeLow + 1;

int DecadeIndex(double x) {
  const int d = static_cast<int>(std::floor(std::log10(x)));
  const int index = d - kDecadeLow;
  return std::max(0, std::min(kDecadeCount - 1, index));
}

struct Sweep {
  double max_abs = 0.0;
  double max_rel = 0.0;
  double rms_abs = 0.0;
  double worst_x = 0.0;
  std::size_t samples = 0;
  std::vector<double> decade_max_abs = std::vector<double>(kDecadeCount, 0.0);
  std::vector<double> decade_max_rel = std::vector<double>(kDecadeCount, 0.0);
  std::vector<std::size_t> decade_count = std::vector<std::size_t>(kDecadeCount, 0);
};

void Accumulate(Sweep& sweep, double x, double abs_error, double rel_error) {
  const int index = DecadeIndex(x);
  sweep.decade_max_abs[static_cast<std::size_t>(index)] =
      std::max(sweep.decade_max_abs[static_cast<std::size_t>(index)], abs_error);
  sweep.decade_max_rel[static_cast<std::size_t>(index)] =
      std::max(sweep.decade_max_rel[static_cast<std::size_t>(index)], rel_error);
  ++sweep.decade_count[static_cast<std::size_t>(index)];
}

// Same decomposition as FastLog but truncated after u^3: the "second order is
// probably fine" strawman.
float FastLogDeg2(float x) {
  if (!(x > 0.0f)) return std::log(x);
  std::uint32_t bits;
  std::memcpy(&bits, &x, sizeof(bits));
  std::int32_t e = static_cast<std::int32_t>(bits >> 23) - 127;
  std::uint32_t mbits = (bits & 0x007FFFFFu) | 0x3F800000u;
  float m;
  std::memcpy(&m, &mbits, sizeof(m));
  if (m > 1.41421356237f) {
    m *= 0.5f;
    ++e;
  }
  const float u = (m - 1.0f) / (m + 1.0f);
  const float log_m = (u + u) * (1.0f + u * u * 0.3333333f);
  return static_cast<float>(e) * 0.69314718f + log_m;
}

template <typename Fn>
Sweep Run(Fn fn, const std::vector<double>& xs) {
  Sweep sweep;
  double sum_sq = 0.0;
  for (double x : xs) {
    const double reference = std::log(x);
    const double approx = static_cast<double>(fn(static_cast<float>(x)));
    const double abs_error = std::abs(approx - reference);
    // Relative error is only meaningful away from the zero crossing: log(1)=0
    // makes the ratio unbounded no matter how good the kernel is. Report it on
    // |log x| >= 1, which is where the fbank lives anyway (a full-scale frame
    // sits near +12, silence near -16).
    const double rel_error = std::abs(reference) >= 1.0 ? abs_error / std::abs(reference) : 0.0;
    if (abs_error > sweep.max_abs) {
      sweep.max_abs = abs_error;
      sweep.worst_x = x;
    }
    sweep.max_rel = std::max(sweep.max_rel, rel_error);
    Accumulate(sweep, x, abs_error, rel_error);
    sum_sq += abs_error * abs_error;
    ++sweep.samples;
  }
  sweep.rms_abs = std::sqrt(sum_sq / static_cast<double>(sweep.samples));
  return sweep;
}

Sweep RunDouble(const std::vector<double>& xs) {
  Sweep sweep;
  double sum_sq = 0.0;
  for (double x : xs) {
    const double reference = std::log(x);
    const double approx = voice::FastLogDouble(x);
    const double abs_error = std::abs(approx - reference);
    if (abs_error > sweep.max_abs) {
      sweep.max_abs = abs_error;
      sweep.worst_x = x;
    }
    const double rel_error = std::abs(reference) >= 1.0 ? abs_error / std::abs(reference) : 0.0;
    sweep.max_rel = std::max(sweep.max_rel, rel_error);
    Accumulate(sweep, x, abs_error, rel_error);
    sum_sq += abs_error * abs_error;
    ++sweep.samples;
  }
  sweep.rms_abs = std::sqrt(sum_sq / static_cast<double>(sweep.samples));
  return sweep;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string json_path;
    std::size_t points = 2000000;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--json" && i + 1 < argc) json_path = argv[++i];
      else if (arg == "--points" && i + 1 < argc) points = std::stoull(argv[++i]);
      else if (arg == "--help") {
        std::cout << "usage: test_fast_log [--points 2000000] [--json out.json]\n";
        return 0;
      } else {
        std::cerr << "unknown argument: " << arg << "\n";
        return 2;
      }
    }
    if (points < 1000) throw std::invalid_argument("need at least 1000 points");

    // The fbank clamps mel energy to FLT_EPSILON, so the domain starts there.
    const double low = static_cast<double>(std::numeric_limits<float>::epsilon());
    const double high = 1.0e6;
    std::vector<double> xs(points);
    const double log_low = std::log(low), log_high = std::log(high);
    for (std::size_t i = 0; i < points; ++i) {
      const double t = static_cast<double>(i) / static_cast<double>(points - 1);
      xs[i] = std::exp(log_low + t * (log_high - log_low));
    }

    const Sweep fast = Run([](float x) { return voice::FastLog(x); }, xs);
    const Sweep deg2 = Run([](float x) { return FastLogDeg2(x); }, xs);
    const Sweep exact = RunDouble(xs);

    std::cout << std::scientific << std::setprecision(3);
    std::cout << "domain [" << low << ", " << high << "], " << points << " log-spaced points\n";
    std::cout << "kernel            max|err|     max rel*     rms|err|      worst x\n";
    std::cout << std::fixed << std::setprecision(3);
    const auto row = [](const char* name, const Sweep& s) {
      std::cout << std::left << std::setw(18) << name << std::scientific << std::setprecision(3)
                << std::setw(12) << s.max_abs << std::setw(12) << s.max_rel << std::setw(12)
                << s.rms_abs << std::setprecision(3) << s.worst_x << "\n";
    };
    row("fast_log(float)", fast);
    row("fast_log(double)", exact);
    row("degree-2 taylor", deg2);

    // Bound the claim. float32 eps is 1.19e-7; the e*ln2 product is where most
    // of the remaining error comes from, so 1e-6 relative is the honest ceiling.
    if (!(fast.max_rel < 1.0e-6)) throw std::runtime_error("fast_log exceeds its 1e-6 relative bound");
    if (!(fast.max_abs < 2.0e-5)) throw std::runtime_error("fast_log exceeds its absolute bound");
    // The double variant is NOT dramatically more accurate, and that is the
    // finding, not a bug: both truncate the same series after u^7, so the
    // truncation floor (~3e-8 relative) is what limits them, not the mantissa
    // width. Buying accuracy here takes another term, not another register.
    if (!(exact.max_rel < 1.0e-7)) throw std::runtime_error("fast_log(double) exceeds 1e-7");
    // The strawman must be visibly worse, otherwise the comparison is theatre.
    if (!(deg2.max_abs > 50.0 * fast.max_abs))
      throw std::runtime_error("degree-2 is not meaningfully worse; the sweep is not discriminating");

    if (!json_path.empty()) {
      std::ofstream out(json_path);
      if (!out) throw std::runtime_error("cannot write " + json_path);
      out.precision(17);
      out << "{\n";
      out << "  \"status\": \"measured_fast_log_accuracy\",\n";
      out << "  \"domain\": [" << low << ", " << high << "],\n";
      out << "  \"points\": " << points << ",\n";
      out << "  \"spacing\": \"log\",\n";
      out << "  \"kernels\": [\n";
      const auto emit = [&](const char* name, const Sweep& s, bool last) {
        out << "    {\"name\": \"" << name << "\", \"max_abs_error\": " << s.max_abs
            << ", \"max_rel_error\": " << s.max_rel << ", \"rms_abs_error\": " << s.rms_abs
            << ", \"worst_x\": " << s.worst_x << "}" << (last ? "\n" : ",\n");
      };
      emit("fast_log_float", fast, false);
      emit("fast_log_double", exact, false);
      emit("degree2_taylor", deg2, true);
      out << "  ],\n";
      out << "  \"decade_low\": " << kDecadeLow << ",\n";
      out << "  \"decade_profiles\": {\n";
      const auto emit_profile = [&](const char* name, const Sweep& s, bool last) {
        out << "    \"" << name << "\": [";
        for (int i = 0; i < kDecadeCount; ++i) {
          const std::size_t index = static_cast<std::size_t>(i);
          out << "{\"decade\": " << (i + kDecadeLow) << ", \"samples\": "
              << s.decade_count[index] << ", \"max_abs\": " << s.decade_max_abs[index]
              << ", \"max_rel\": " << s.decade_max_rel[index] << "}"
              << (i + 1 == kDecadeCount ? "" : ", ");
        }
        out << "]" << (last ? "\n" : ",\n");
      };
      emit_profile("fast_log_float", fast, false);
      emit_profile("fast_log_double", exact, false);
      emit_profile("degree2_taylor", deg2, true);
      out << "  },\n";
      out << "  \"rel_error_definition\": \"|approx-log(x)|/|log(x)| where |log(x)| >= 1\",\n";
      out << "  \"bound\": {\"max_rel\": 1e-6, \"max_abs\": 2e-5, "
             "\"justification\": \"float32 epsilon is 1.19e-7 and the e*ln2 product dominates; "
             "the CTC tolerance is 1e-4, so this is a 100x margin\"}\n";
      out << "}\n";
      std::cout << "wrote " << json_path << "\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "fast log audit failed: " << e.what() << "\n";
    return 1;
  }
}
