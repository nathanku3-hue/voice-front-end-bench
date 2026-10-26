// Stage-wise latency budget for the C++ Fbank frontend.
//
// Answers the question "where does the per-frame time actually go": one RAII
// timer per micro-stage, accumulated over the whole clip, reported as absolute
// time, per-frame cost and share of the total. The split is the one declared in
// StreamingFbank::Stage, so the numbers cannot drift away from the code.
//
// Three axes are switchable at run time, because every claim in the SIMD/log
// document is a paired comparison:
//   --scalar / --vector          sample-domain stages: legacy modulo+scalar vs
//                                aligned staging buffer + AVX2
//   --log std|fast|fastdouble    log-compression kernel
//   --dense-mel / (default)      filterbank layout (from the earlier audit)
//
// --interleave runs the three arms of the SIMD/log study round-robin inside one
// process: baseline (scalar+std log), simd (vector+std log), and
// simd_fastlog. This is the protocol, not a convenience. The host is bimodal
// and FFTW's plan/cache state changes between processes, so measuring one arm
// to completion and then the next attributes drift to the change under test.
// Round-robin sampling gives every arm the same sequence of host conditions and
// the same warm cache, and the median of each arm is then comparable.
//
// Measurement caveat, stated because it bounds the claim: every stage reads
// steady_clock twice, so each row carries roughly one clock-pair of overhead
// (tens of nanoseconds). On a ~2.5 us frame that is low single-digit percent,
// and it is uniform across arms, so the *ratios* are more trustworthy than the
// absolute nanoseconds.

#include "streaming_fbank.h"
#include "host_pin.h"
#include "wav_reader.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using LogMode = voice::StreamingFbank::LogMode;

constexpr std::size_t kHop = voice::StreamingFbank::kFrameShift;
constexpr std::size_t kStages = voice::StreamingFbank::kStageCount;

struct Run {
  std::size_t frames = 0;
  std::size_t samples = 0;
  double wall_ns = 0.0;
  std::array<std::uint64_t, kStages> stage_ns{};
};

struct Config {
  bool sparse = true;
  bool scalar_stages = false;
  LogMode log_mode = LogMode::kFastFloat;
};

struct Arm {
  const char* name;
  Config config;
};

Run TimeOnce(const std::vector<float>& pcm, std::size_t capacity, const Config& config) {
  voice::StreamingFbank fbank(capacity);
  // Must be set before the first frame: the dense path materialises its table
  // on request, and that allocation is deliberately outside the timed loop.
  fbank.EnableSparseMel(config.sparse);
  fbank.EnableScalarStages(config.scalar_stages);
  fbank.SetLogMode(config.log_mode);
  fbank.EnableStageTiming(true);
  Run run;
  const auto started = std::chrono::steady_clock::now();
  for (std::size_t pos = 0; pos < pcm.size();) {
    const std::size_t n = std::min(kHop, pcm.size() - pos);
    fbank.AcceptWaveform(pcm.data() + pos, n);
    pos += n;
    while (fbank.NumFramesReady()) fbank.GetFrame();
  }
  run.wall_ns = static_cast<double>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                                           started)
          .count());
  run.frames = fbank.FramesProduced();
  run.samples = pcm.size();
  for (std::size_t s = 0; s < kStages; ++s) run.stage_ns[s] = fbank.StageNanoseconds(s);
  return run;
}

Run MedianByWall(std::vector<Run> runs) {
  std::sort(runs.begin(), runs.end(), [](const Run& a, const Run& b) {
    return a.wall_ns < b.wall_ns;
  });
  return runs[runs.size() / 2];
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

bool ParseLogMode(const std::string& text, LogMode* mode) {
  if (text == "std") *mode = LogMode::kStd;
  else if (text == "fast") *mode = LogMode::kFastFloat;
  else if (text == "fastdouble") *mode = LogMode::kFastDouble;
  else return false;
  return true;
}

void PrintTable(const Run& run, const Config& config) {
  std::array<double, kStages> stage_ms{};
  double total_ms = 0.0;
  for (std::size_t s = 0; s < kStages; ++s) {
    stage_ms[s] = static_cast<double>(run.stage_ns[s]) / 1e6;
    total_ms += stage_ms[s];
  }
  const double audio_seconds =
      static_cast<double>(run.samples) / voice::StreamingFbank::kSampleRate;
  const double wall_ms = run.wall_ns / 1e6;
  std::cout << "isa: " << voice::IsaName(voice::DetectIsa()) << ", stages: "
            << (config.scalar_stages ? "scalar reference" : "vector") << ", log: "
            << voice::StreamingFbank::LogModeName(config.log_mode) << ", mel: "
            << (config.sparse ? "sparse_csr" : "dense_table") << "\n";
  std::cout << std::left << std::setw(22) << "stage" << std::right << std::setw(12) << "total ms"
            << std::setw(14) << "us/frame" << std::setw(10) << "share" << "\n";
  for (std::size_t s = 0; s < kStages; ++s) {
    const double us_per_frame =
        run.frames ? 1000.0 * stage_ms[s] / static_cast<double>(run.frames) : 0.0;
    const double share = total_ms > 0.0 ? 100.0 * stage_ms[s] / total_ms : 0.0;
    std::cout << std::left << std::setw(22) << voice::StreamingFbank::StageName(s) << std::right
              << std::setw(12) << stage_ms[s] << std::setw(14) << us_per_frame << std::setw(9)
              << share << "%\n";
  }
  std::cout << std::left << std::setw(22) << "instrumented total" << std::right << std::setw(12)
            << total_ms << std::setw(14)
            << (run.frames ? 1000.0 * total_ms / static_cast<double>(run.frames) : 0.0)
            << std::setw(10) << "100%" << "\n";
  std::cout << std::left << std::setw(22) << "whole pass (wall)" << std::right << std::setw(12)
            << wall_ms << std::setw(14)
            << (run.frames ? 1000.0 * wall_ms / static_cast<double>(run.frames) : 0.0) << "\n";
  std::cout << "realtime factor: " << (audio_seconds * 1e3 / wall_ms) << "x\n";
}

void WriteStages(std::ostream& out, const Run& run) {
  double total_ms = 0.0;
  for (std::size_t s = 0; s < kStages; ++s) total_ms += static_cast<double>(run.stage_ns[s]) / 1e6;
  out << "    \"instrumented_total_ms\": " << total_ms << ",\n";
  out << "    \"stages\": [\n";
  for (std::size_t s = 0; s < kStages; ++s) {
    const double stage_ms = static_cast<double>(run.stage_ns[s]) / 1e6;
    const double us_per_frame =
        run.frames ? 1000.0 * stage_ms / static_cast<double>(run.frames) : 0.0;
    const double share = total_ms > 0.0 ? 100.0 * stage_ms / total_ms : 0.0;
    out << "      {\"name\": \"" << voice::StreamingFbank::StageName(s) << "\", "
        << "\"total_ms\": " << stage_ms << ", \"us_per_frame\": " << us_per_frame
        << ", \"share_percent\": " << share << "}" << (s + 1 == kStages ? "\n" : ",\n");
  }
  out << "    ]\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string wav_path, json_path;
    std::size_t repeats = 5;
    double seconds = 20.0;
    bool pin = true;
    bool interleave = false;
    Config config;
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
      else if (arg == "--no-pin") pin = false;
      else if (arg == "--dense-mel") config.sparse = false;
      else if (arg == "--scalar") config.scalar_stages = true;
      else if (arg == "--vector") config.scalar_stages = false;
      else if (arg == "--interleave") interleave = true;
      else if (arg == "--log") {
        const std::string mode = value("--log");
        if (!ParseLogMode(mode, &config.log_mode))
          throw std::invalid_argument("--log must be std|fast|fastdouble");
      } else if (arg == "--help") {
        std::cout << "usage: fbank_stage_profile [--wav in.wav] [--seconds 20] [--repeats 5] "
                     "[--no-pin] [--dense-mel] [--scalar|--vector] "
                     "[--log std|fast|fastdouble] [--interleave] [--json PATH]\n"
                     "  without --wav a deterministic synthetic 16 kHz signal is used\n"
                     "  --interleave runs baseline / simd / simd_fastlog round-robin\n";
        return 0;
      } else {
        std::cerr << "unknown argument: " << arg << "\n";
        return 2;
      }
    }
    if (repeats < 1) throw std::invalid_argument("repeats must be >= 1");
    const bool pinned = pin && voice::tool::PinToOneCore();

    std::vector<float> pcm;
    std::string source;
    if (!wav_path.empty()) {
      const voice::tool::WavClip clip = voice::tool::ReadWav(wav_path);
      pcm = clip.pcm;
      source = wav_path;
    } else {
      if (!(seconds > 0.0)) throw std::invalid_argument("seconds must be > 0");
      const std::size_t n = static_cast<std::size_t>(seconds * voice::StreamingFbank::kSampleRate);
      pcm.resize(n);
      for (std::size_t i = 0; i < n; ++i) {
        pcm[i] = static_cast<float>(.37 * std::sin(.013 * static_cast<double>(i)) +
                                    .11 * std::cos(.037 * static_cast<double>(i)));
      }
      source = "synthetic";
    }
    // One hop per call and an immediate drain, so two slots are always enough;
    // a small ring keeps the measured path identical to the streaming one.
    const std::size_t capacity = 2;

    voice::StreamingFbank probe(capacity);
    const voice::Isa isa = probe.ActiveIsa();
    const auto offset_mod = [](const void* p, std::size_t mod) {
      return static_cast<std::size_t>(reinterpret_cast<std::uintptr_t>(p) % mod);
    };
    const std::size_t frame_offset = offset_mod(probe.FrameBufferAddress(), 64);
    const std::size_t fft_offset = offset_mod(probe.FftInputAddress(), 64);

    std::cout << std::fixed << std::setprecision(3);
    const double audio_seconds =
        static_cast<double>(pcm.size()) / voice::StreamingFbank::kSampleRate;

    if (interleave) {
      const std::array<Arm, 3> arms = {{
          {"baseline", Config{true, true, LogMode::kStd}},
          {"simd", Config{true, false, LogMode::kStd}},
          {"simd_fastlog", Config{true, false, LogMode::kFastFloat}},
      }};
      std::vector<std::vector<Run>> samples(arms.size());
      std::size_t frames = 0;
      // Round-robin: arm k of round r sees the same host state as arm k-1.
      for (std::size_t r = 0; r < repeats; ++r) {
        for (std::size_t a = 0; a < arms.size(); ++a) {
          samples[a].push_back(TimeOnce(pcm, capacity, arms[a].config));
          frames = samples[a].back().frames;
        }
      }
      std::vector<Run> medians;
      for (auto& arm_samples : samples) medians.push_back(MedianByWall(arm_samples));

      std::cout << "source: " << source << ", " << audio_seconds << " s audio, " << frames
                << " frames, " << repeats << " interleaved rounds, "
                << (pinned ? "pinned to one core at high priority" : "unpinned") << "\n";
      std::cout << "isa: " << voice::IsaName(isa) << "\n\n";
      std::cout << std::left << std::setw(16) << "arm" << std::right << std::setw(14)
                << "us/frame" << std::setw(12) << "realtime" << std::setw(12) << "dc+pre" << "\n";
      for (std::size_t a = 0; a < arms.size(); ++a) {
        const Run& run = medians[a];
        const double us = run.frames ? run.wall_ns / 1000.0 / static_cast<double>(run.frames) : 0.0;
        const double sample_domain = static_cast<double>(run.stage_ns[0] + run.stage_ns[1]) / 1000.0 /
                                     static_cast<double>(run.frames);
        std::cout << std::left << std::setw(16) << arms[a].name << std::right << std::setw(14) << us
                  << std::setw(12) << (audio_seconds * 1e6 / (run.wall_ns / 1000.0)) << std::setw(12)
                  << sample_domain << "\n";
      }
      std::cout << "\n";

      if (!json_path.empty()) {
        std::ofstream out(json_path);
        if (!out) throw std::runtime_error("cannot write " + json_path);
        out.precision(17);
        out << "{\n";
        out << "  \"status\": \"measured_interleaved_stage_budget\",\n";
        out << "  \"compiler\": \"" << CompilerTag() << "\",\n";
        out << "  \"isa\": \"" << voice::IsaName(isa) << "\",\n";
        out << "  \"source\": \"" << source << "\",\n";
        out << "  \"audio_seconds\": " << audio_seconds << ",\n";
        out << "  \"frames\": " << frames << ",\n";
        out << "  \"samples\": " << pcm.size() << ",\n";
        out << "  \"repeats\": " << repeats << ",\n";
        out << "  \"pinned_to_one_core\": " << (pinned ? "true" : "false") << ",\n";
        out << "  \"protocol\": \"round-robin arms inside one process, median pass by wall time\",\n";
        out << "  \"frame_buffer_offset_mod64\": " << frame_offset << ",\n";
        out << "  \"fft_input_offset_mod64\": " << fft_offset << ",\n";
        out << "  \"arms\": [\n";
        for (std::size_t a = 0; a < arms.size(); ++a) {
          const Run& run = medians[a];
          const double wall_ms = run.wall_ns / 1e6;
          out << "  {\n";
          out << "    \"name\": \"" << arms[a].name << "\",\n";
          out << "    \"sample_domain_stages\": \""
              << (arms[a].config.scalar_stages ? "scalar_reference" : "vector") << "\",\n";
          out << "    \"log_mode\": \"" << voice::StreamingFbank::LogModeName(arms[a].config.log_mode)
              << "\",\n";
          out << "    \"wall_ms\": " << wall_ms << ",\n";
          out << "    \"wall_us_per_frame\": "
              << (run.frames ? 1000.0 * wall_ms / static_cast<double>(run.frames) : 0.0) << ",\n";
          out << "    \"realtime_factor\": " << (audio_seconds * 1e3 / wall_ms) << ",\n";
          out << "    \"wall_spread_ms\": [" << samples[a].front().wall_ns / 1e6 << ", "
              << samples[a].back().wall_ns / 1e6 << "],\n";
          WriteStages(out, run);
          out << "  }" << (a + 1 == arms.size() ? "\n" : ",\n");
        }
        out << "  ],\n";
        out << "  \"measurement_caveats\": [\n";
        out << "    \"each stage reads steady_clock twice; that overhead is included in the row\",\n";
        out << "    \"host-local decoder-only timing on a shared machine, not device latency\",\n";
        out << "    \"the vector path reorders the floating-point accumulation, so its features are "
               "not bit-identical to the scalar reference; the difference is published separately\"\n";
        out << "  ]\n";
        out << "}\n";
        std::cout << "wrote " << json_path << "\n";
      }
      return 0;
    }

    std::vector<Run> runs;
    runs.reserve(repeats);
    for (std::size_t r = 0; r < repeats; ++r) runs.push_back(TimeOnce(pcm, capacity, config));
    const Run chosen = MedianByWall(runs);

    std::cout << "source: " << source << ", " << audio_seconds << " s audio, " << chosen.frames
              << " frames, median of " << repeats << " passes, "
              << (pinned ? "pinned to one core at high priority" : "unpinned") << "\n";
    PrintTable(chosen, config);
    std::cout << "alignment: frame buffer mod 64 == " << frame_offset
              << ", fft input mod 64 == " << fft_offset << "\n";

    if (!json_path.empty()) {
      std::ofstream out(json_path);
      if (!out) throw std::runtime_error("cannot write " + json_path);
      out.precision(17);
      const double wall_ms = chosen.wall_ns / 1e6;
      out << "{\n";
      out << "  \"status\": \"measured_stage_latency_budget\",\n";
      out << "  \"compiler\": \"" << CompilerTag() << "\",\n";
      out << "  \"isa\": \"" << voice::IsaName(isa) << "\",\n";
      out << "  \"sample_domain_stages\": \"" << (config.scalar_stages ? "scalar_reference" : "vector")
          << "\",\n";
      out << "  \"log_mode\": \"" << voice::StreamingFbank::LogModeName(config.log_mode) << "\",\n";
      out << "  \"source\": \"" << source << "\",\n";
      out << "  \"audio_seconds\": " << audio_seconds << ",\n";
      out << "  \"frames\": " << chosen.frames << ",\n";
      out << "  \"samples\": " << chosen.samples << ",\n";
      out << "  \"repeats\": " << repeats << ",\n";
      out << "  \"pinned_to_one_core\": " << (pinned ? "true" : "false") << ",\n";
      out << "  \"mel_projection\": \"" << (config.sparse ? "sparse_csr" : "dense_table") << "\",\n";
      out << "  \"statistic\": \"median pass by whole-pass wall time\",\n";
      out << "  \"wall_ms\": " << wall_ms << ",\n";
      out << "  \"wall_us_per_frame\": "
          << (chosen.frames ? 1000.0 * wall_ms / static_cast<double>(chosen.frames) : 0.0) << ",\n";
      out << "  \"realtime_factor\": " << (audio_seconds * 1e3 / wall_ms) << ",\n";
      out << "  \"frame_buffer_offset_mod64\": " << frame_offset << ",\n";
      out << "  \"fft_input_offset_mod64\": " << fft_offset << ",\n";
      out << "  \"wall_spread_ms\": [" << runs.front().wall_ns / 1e6 << ", "
          << runs.back().wall_ns / 1e6 << "],\n";
      WriteStages(out, chosen);
      out << "  ,\"measurement_caveats\": [\n";
      out << "    \"each stage reads steady_clock twice; that overhead is included in the row\",\n";
      out << "    \"host-local decoder-only timing on a shared machine, not device latency\",\n";
      out << "    \"FFTW is executed with the pre-created FFTW_ESTIMATE plan; no plan creation "
             "is timed\",\n";
      out << "    \"the vector path reorders the floating-point accumulation, so its features are "
             "not bit-identical to the scalar reference; the difference is published separately\"\n";
      out << "  ]\n";
      out << "}\n";
      std::cout << "wrote " << json_path << "\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "stage profile error: " << e.what() << "\n";
    return 1;
  }
}
