// Mel-projection sparsity audit.
//
// The stage budget puts ~88% of frame time in mel_projection, and the reason is
// structural: the dense 80 x 257 filterbank is 164 KB of coefficients re-read
// for every 10 ms frame, while only 501 of its 20,480 entries are non-zero.
//
// This test does two things:
//   1. proves the compressed CSR path is bit-identical to the dense path on
//      real 16 kHz audio (not "close": max |diff| == 0 and zero mismatched
//      doubles), so the optimisation cannot be dismissed as an approximation;
//   2. measures both paths on the same clip so the speed-up is a measurement
//      rather than an extrapolation from the non-zero count.

#include "host_pin.h"
#include "streaming_fbank.h"
#include "wav_reader.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kHop = voice::StreamingFbank::kFrameShift;

struct Run {
  std::vector<voice::StreamingFbank::Feature> frames;
  double nanoseconds = 0.0;
};

Run Extract(const std::vector<float>& pcm, bool sparse, std::size_t capacity) {
  voice::StreamingFbank fbank(capacity);
  fbank.EnableSparseMel(sparse);
  Run run;
  const auto started = std::chrono::steady_clock::now();
  for (std::size_t pos = 0; pos < pcm.size();) {
    const std::size_t n = std::min(kHop, pcm.size() - pos);
    fbank.AcceptWaveform(pcm.data() + pos, n);
    pos += n;
    while (fbank.NumFramesReady()) run.frames.push_back(fbank.GetFrame());
  }
  run.nanoseconds = static_cast<double>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                                           started)
          .count());
  return run;
}

std::string CompilerTag() {
#if defined(__clang__)
  return "clang " __clang_version__;
#elif defined(__GNUC__)
  return "gcc " __VERSION__;
#elif defined(_MSC_VER)
  return "msvc " + std::to_string(_MSC_VER);
#else
  return "unknown";
#endif
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string wav_path, json_path;
    std::size_t repeats = 5;
    double seconds = 20.0;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      const auto value = [&](const char* name) -> std::string {
        if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + name);
        return argv[++i];
      };
      if (arg == "--wav") wav_path = value("--wav");
      else if (arg == "--seconds") seconds = std::stod(value("--seconds"));
      else if (arg == "--repeats") repeats = std::stoull(value("--repeats"));
      else if (arg == "--json") json_path = value("--json");
      else if (arg == "--help") {
        std::cout << "usage: test_mel_sparsity [--wav in.wav] [--seconds 20] [--repeats 5] "
                     "[--json PATH]\n";
        return 0;
      } else {
        std::cerr << "unknown argument: " << arg << "\n";
        return 2;
      }
    }
    if (repeats < 1) throw std::invalid_argument("repeats must be >= 1");
    // Pin, for the same reason the stage profiler does: the dense/sparse ratio
    // is measured in one process, but the absolute us/frame must be comparable
    // with the other audits.
    const bool pinned = voice::tool::PinToOneCore();

    std::vector<float> pcm;
    std::string source;
    if (!wav_path.empty()) {
      const voice::tool::WavClip clip = voice::tool::ReadWav(wav_path);
      pcm = clip.pcm;
      source = wav_path;
    } else {
      const std::size_t n = static_cast<std::size_t>(seconds * voice::StreamingFbank::kSampleRate);
      pcm.resize(n);
      for (std::size_t i = 0; i < n; ++i) {
        pcm[i] = static_cast<float>(.37 * std::sin(.013 * static_cast<double>(i)) +
                                    .11 * std::cos(.037 * static_cast<double>(i)));
      }
      source = "synthetic";
    }

    const std::size_t capacity = 2;
    voice::StreamingFbank probe(capacity);
    const std::size_t taps = probe.MelTapCount();
    const std::size_t dense_entries = probe.MelDenseEntryCount();

    // Correctness first: one pass each, compared value by value.
    const Run sparse_once = Extract(pcm, true, capacity);
    const Run dense_once = Extract(pcm, false, capacity);
    assert(sparse_once.frames.size() == dense_once.frames.size());
    std::size_t compared = 0, mismatched = 0;
    double max_abs_diff = 0.0;
    for (std::size_t i = 0; i < sparse_once.frames.size(); ++i) {
      for (std::size_t b = 0; b < voice::StreamingFbank::kMelBins; ++b) {
        const double a = sparse_once.frames[i][b];
        const double d = dense_once.frames[i][b];
        if (std::memcmp(&a, &d, sizeof(double)) != 0) ++mismatched;
        max_abs_diff = std::max(max_abs_diff, std::abs(a - d));
        ++compared;
      }
    }

    // Timing: interleaved, because this host drifts between blocks.
    std::vector<double> sparse_ns, dense_ns;
    for (std::size_t r = 0; r < repeats; ++r) {
      sparse_ns.push_back(Extract(pcm, true, capacity).nanoseconds);
      dense_ns.push_back(Extract(pcm, false, capacity).nanoseconds);
    }
    const auto median = [](std::vector<double> values) {
      std::sort(values.begin(), values.end());
      const std::size_t mid = values.size() / 2;
      return values.size() % 2 ? values[mid] : 0.5 * (values[mid - 1] + values[mid]);
    };
    const double sparse_med = median(sparse_ns);
    const double dense_med = median(dense_ns);
    const double frames = static_cast<double>(sparse_once.frames.size());
    const double audio_seconds =
        static_cast<double>(pcm.size()) / voice::StreamingFbank::kSampleRate;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "source: " << source << ", " << audio_seconds << " s, " << frames << " frames\n";
    std::cout << "filterbank: " << taps << " non-zero of " << dense_entries << " entries ("
              << std::setprecision(2) << 100.0 * static_cast<double>(taps) /
                     static_cast<double>(dense_entries)
              << "%), dense matrix " << (dense_entries * sizeof(double)) / 1024 << " KiB\n";
    std::cout << std::setprecision(3);
    std::cout << "exactness: " << compared << " values compared, " << mismatched
              << " bit-mismatches, max |diff| " << max_abs_diff << "\n";
    std::cout << "dense : " << dense_med / 1000.0 / frames << " us/frame, "
              << audio_seconds * 1e9 / dense_med << "x realtime\n";
    std::cout << "sparse: " << sparse_med / 1000.0 / frames << " us/frame, "
              << audio_seconds * 1e9 / sparse_med << "x realtime, speed-up "
              << dense_med / sparse_med << "x\n";

    assert(mismatched == 0);
    assert(max_abs_diff == 0.0);
    assert(compared == sparse_once.frames.size() * voice::StreamingFbank::kMelBins);
    assert(taps < dense_entries);

    if (!json_path.empty()) {
      std::ofstream out(json_path);
      if (!out) throw std::runtime_error("cannot write " + json_path);
      out.precision(17);
      out << "{\n";
      out << "  \"status\": \"measured_mel_sparsity\",\n";
      out << "  \"compiler\": \"" << CompilerTag() << "\",\n";
      out << "  \"source\": \"" << source << "\",\n";
      out << "  \"audio_seconds\": " << audio_seconds << ",\n";
      out << "  \"frames\": " << sparse_once.frames.size() << ",\n";
      out << "  \"repeats\": " << repeats << ",\n";
      out << "  \"pinned_to_one_core\": " << (pinned ? "true" : "false") << ",\n";
      out << "  \"nonzeros\": " << taps << ",\n";
      out << "  \"dense_entries\": " << dense_entries << ",\n";
      out << "  \"dense_matrix_bytes\": " << dense_entries * sizeof(double) << ",\n";
      out << "  \"sparse_tap_bytes\": " << taps * sizeof(double) * 2 << ",\n";
      out << "  \"macs_dense\": " << dense_entries << ",\n";
      out << "  \"macs_sparse\": " << taps << ",\n";
      out << "  \"mac_reduction\": " << static_cast<double>(dense_entries) /
                                            static_cast<double>(taps) << ",\n";
      out << "  \"compared_values\": " << compared << ",\n";
      out << "  \"bit_mismatches\": " << mismatched << ",\n";
      out << "  \"max_abs_diff\": " << max_abs_diff << ",\n";
      out << "  \"dense_us_per_frame\": " << dense_med / 1000.0 / frames << ",\n";
      out << "  \"sparse_us_per_frame\": " << sparse_med / 1000.0 / frames << ",\n";
      out << "  \"dense_realtime_factor\": " << audio_seconds * 1e9 / dense_med << ",\n";
      out << "  \"sparse_realtime_factor\": " << audio_seconds * 1e9 / sparse_med << ",\n";
      out << "  \"speedup\": " << dense_med / sparse_med << ",\n";
      out << "  \"sparse_us_per_frame_spread\": ["
          << *std::min_element(sparse_ns.begin(), sparse_ns.end()) / 1000.0 / frames << ", "
          << *std::max_element(sparse_ns.begin(), sparse_ns.end()) / 1000.0 / frames << "],\n";
      out << "  \"dense_us_per_frame_spread\": ["
          << *std::min_element(dense_ns.begin(), dense_ns.end()) / 1000.0 / frames << ", "
          << *std::max_element(dense_ns.begin(), dense_ns.end()) / 1000.0 / frames << "],\n";
      out << "  \"non_claims\": [\"whole-pass timing includes ring copies and frame pushes, not "
             "just the mel stage\", \"host-local, not device latency\"]\n";
      out << "}\n";
      std::cout << "wrote " << json_path << "\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "mel sparsity audit error: " << e.what() << "\n";
    return 1;
  }
}
