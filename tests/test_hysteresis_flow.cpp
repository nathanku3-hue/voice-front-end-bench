// Hysteresis (dual-watermark) flow-control audit.
//
// The single-watermark bridge pauses the moment the caller's ring is full and
// resumes as soon as one slot frees. The zero-allocation audit shows what that
// costs: at a 50%-rate consumer it produced 9,998 pause and 9,998 resume events
// over 20,002 ticks -- a transition every two ticks. On real silicon each of
// those is an interrupt plus a handshake, and at 100 frames/s that is 100
// handshakes per second for a 10 ms FIFO.
//
// This harness models a firmware-style driver around the same BoundedFbankBridge
// and compares watermark policies:
//
//   single   high = C, low = C-1   (control arm, the current behaviour)
//   dual     high = C, low = k     (k < C-1, the proposed behaviour)
//
// The gate lives in the driver, not in the bridge, because the bridge does not
// own the output ring: whoever owns the FIFO owns the level signal. That is also
// where a DMA controller or an SSP port would put it.
//
// What is claimed and what is not:
//   claimed    transitions drop by the measured ratio, peak occupancy stays
//              bounded by the high watermark, and every delivered frame is
//              bit-identical to the unthrottled reference in every arm
//   not claimed  any absolute interrupt cost; this is a host simulation of a
//              policy, and the tick is audio time, not wall time

#include "bounded_fbank_bridge.h"
#include "flow_hysteresis.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Feature = voice::StreamingFbank::Feature;
using Bridge = voice::BoundedFbankBridge;

constexpr std::size_t kHop = voice::StreamingFbank::kFrameShift;
constexpr std::size_t kDrainPeriod = 2;  // consumer at 50% of the input rate

float WaveSample(std::size_t i) {
  return static_cast<float>(.37 * std::sin(.013 * static_cast<double>(i)) +
                            .11 * std::cos(.037 * static_cast<double>(i)));
}

struct ArmConfig {
  std::size_t capacity = 4;   // ring depth C
  std::size_t low = 3;        // low watermark; C-1 means "single watermark"
  std::size_t frames = 20000;
  bool single_watermark = false;
};

struct ArmReport {
  ArmConfig config;
  std::size_t frames = 0;
  std::size_t ticks = 0;
  std::size_t pauses = 0;
  std::size_t resumes = 0;
  std::size_t transitions = 0;
  std::size_t max_stall_ticks = 0;   // longest continuous producer stall
  std::size_t stall_ticks = 0;       // total ticks the producer was refused
  std::size_t max_occupancy = 0;
  double mean_occupancy = 0.0;
  std::size_t compared_values = 0;
  double max_abs_diff = 0.0;
  std::size_t dropped_frames = 0;    // must stay 0: no sample may be lost
};

std::vector<Feature> BuildReference(std::size_t frames, std::size_t samples) {
  voice::StreamingFbank plain(frames + 2);
  std::vector<Feature> reference;
  reference.reserve(frames);
  std::array<float, kHop> block{};
  for (std::size_t pos = 0; pos < samples;) {
    const std::size_t n = std::min(kHop, samples - pos);
    for (std::size_t i = 0; i < n; ++i) block[i] = WaveSample(pos + i);
    plain.AcceptWaveform(block.data(), n);
    pos += n;
    while (plain.NumFramesReady()) reference.push_back(plain.GetFrame());
  }
  if (reference.size() < frames) throw std::logic_error("reference produced too few frames");
  return reference;
}

ArmReport RunArm(const ArmConfig& cfg, const std::vector<Feature>& reference,
                 const std::vector<float>& pcm) {
  const std::size_t capacity = cfg.capacity;
  const std::size_t low = cfg.single_watermark ? capacity - 1 : cfg.low;
  voice::HysteresisGate gate(capacity, low);

  Bridge bridge;
  std::array<Feature, 64> out{};
  std::array<Feature, 64> ring{};
  std::size_t ring_head = 0, ring_count = 0;
  std::size_t pos = 0, drained = 0, tick = 0;
  std::size_t occupancy_sum = 0;
  std::size_t stall_run = 0;

  ArmReport report;
  report.config = cfg;
  report.config.low = low;

  while (drained < cfg.frames) {
    const bool may_run = gate.ProducerMayRun(ring_count);
    std::size_t produced = 0;
    if (may_run && pos < pcm.size()) {
      const std::size_t n = std::min(kHop, pcm.size() - pos);
      const std::size_t room = capacity - ring_count;
      assert(room > 0);  // high == capacity, so "may run" implies room
      const auto step = bridge.Process(pcm.data() + pos, n, out.data(), std::min(room, out.size()));
      pos += step.samples_consumed;
      produced = step.frames_written;
      for (std::size_t i = 0; i < produced; ++i) {
        ring[(ring_head + ring_count) % capacity] = out[i];
        ++ring_count;
      }
    }

    // A refusal is recorded only while there is still input to offer: the
    // trailing drain (producer out of samples, consumer emptying the ring) is
    // not backpressure and would otherwise inflate the stall statistics.
    const bool refused = pos < pcm.size() && (!may_run || produced == 0);
    if (refused) {
      ++report.stall_ticks;
      ++stall_run;
      report.max_stall_ticks = std::max(report.max_stall_ticks, stall_run);
    } else if (pos < pcm.size()) {
      stall_run = 0;
    }

    report.max_occupancy = std::max(report.max_occupancy, ring_count);
    occupancy_sum += ring_count;

    if (ring_count > 0 && tick % kDrainPeriod == kDrainPeriod - 1) {
      const Feature& got = ring[ring_head];
      const Feature& want = reference[drained];
      for (std::size_t j = 0; j < got.size(); ++j) {
        const double diff = std::abs(got[j] - want[j]);
        report.max_abs_diff = std::max(report.max_abs_diff, diff);
      }
      report.compared_values += got.size();
      ring_head = (ring_head + 1) % capacity;
      --ring_count;
      ++drained;
    }
    ++tick;
  }

  report.frames = drained;
  report.ticks = tick;
  report.pauses = gate.pauses();
  report.resumes = gate.resumes();
  report.transitions = gate.transitions();
  report.mean_occupancy =
      tick == 0 ? 0.0 : static_cast<double>(occupancy_sum) / static_cast<double>(tick);
  // Every offered sample must have been consumed: backpressure delays, never
  // drops. The bridge reports what it actually took.
  report.dropped_frames = bridge.SamplesConsumedTotal() > pcm.size() ? 1 : 0;
  return report;
}

std::string PlatformTag() {
#if defined(_WIN32)
  return "windows";
#elif defined(__linux__)
  return "linux";
#else
  return "unknown";
#endif
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

void WriteArm(std::ostream& os, const ArmReport& r) {
  os << "    {\n";
  os << "      \"mode\": \"" << (r.config.single_watermark ? "single" : "dual") << "\",\n";
  os << "      \"high_watermark\": " << r.config.capacity << ",\n";
  os << "      \"low_watermark\": " << r.config.low << ",\n";
  os << "      \"capacity_frames\": " << r.config.capacity << ",\n";
  os << "      \"frames\": " << r.frames << ",\n";
  os << "      \"ticks\": " << r.ticks << ",\n";
  os << "      \"pauses\": " << r.pauses << ",\n";
  os << "      \"resumes\": " << r.resumes << ",\n";
  os << "      \"transitions\": " << r.transitions << ",\n";
  os << "      \"transitions_per_1000_ticks\": "
     << (r.ticks ? 1000.0 * static_cast<double>(r.transitions) / static_cast<double>(r.ticks) : 0.0)
     << ",\n";
  os << "      \"stall_ticks\": " << r.stall_ticks << ",\n";
  os << "      \"max_stall_ticks\": " << r.max_stall_ticks << ",\n";
  os << "      \"max_occupancy\": " << r.max_occupancy << ",\n";
  os << "      \"mean_occupancy\": " << r.mean_occupancy << ",\n";
  os << "      \"compared_values\": " << r.compared_values << ",\n";
  os << "      \"max_abs_diff\": " << r.max_abs_diff << ",\n";
  os << "      \"dropped_frames\": " << r.dropped_frames << "\n";
  os << "    }";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    ArmConfig base;
    std::string json_path;
    bool sweep = false;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      const auto value = [&](const char* name) -> std::string {
        if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + name);
        return argv[++i];
      };
      if (arg == "--capacity") base.capacity = std::stoull(value("--capacity"));
      else if (arg == "--low") base.low = std::stoull(value("--low"));
      else if (arg == "--frames") base.frames = std::stoull(value("--frames"));
      else if (arg == "--sweep") sweep = true;
      else if (arg == "--json") json_path = value("--json");
      else if (arg == "--help") {
        std::cout << "usage: test_hysteresis_flow [--capacity C] [--low k] [--frames N] "
                     "[--sweep] [--json PATH]\n";
        return 0;
      } else {
        std::cerr << "unknown argument: " << arg << "\n";
        return 2;
      }
    }
    if (base.capacity < 2) throw std::invalid_argument("capacity must be >= 2");
    if (base.frames == 0) throw std::invalid_argument("frames must be >= 1");

    const std::size_t samples = voice::StreamingFbank::kFrameSize + (base.frames - 1) * kHop;
    std::vector<float> pcm(samples);
    for (std::size_t i = 0; i < samples; ++i) pcm[i] = WaveSample(i);
    const std::vector<Feature> reference = BuildReference(base.frames, samples);

    std::vector<ArmReport> arms;
    if (sweep) {
      ArmConfig single = base;
      single.single_watermark = true;
      single.low = base.capacity - 1;
      arms.push_back(RunArm(single, reference, pcm));
      for (std::size_t low = base.capacity - 2; low + 1 >= 1; --low) {
        ArmConfig dual = base;
        dual.single_watermark = false;
        dual.low = low;
        arms.push_back(RunArm(dual, reference, pcm));
        if (low == 0) break;
      }
    } else {
      arms.push_back(RunArm(base, reference, pcm));
    }

    for (const ArmReport& r : arms) {
      std::cout << (r.config.single_watermark ? "single" : "dual  ") << " high " << base.capacity
                << " low " << r.config.low << ": " << r.transitions << " transitions ("
                << r.pauses << " pause / " << r.resumes << " resume), max stall "
                << r.max_stall_ticks << " ticks, max occupancy " << r.max_occupancy
                << ", mean occupancy " << r.mean_occupancy << ", max |diff| " << r.max_abs_diff
                << "\n";
    }

    const ArmReport& control = arms.front();
    for (const ArmReport& r : arms) {
      assert(r.frames == base.frames);
      assert(r.max_abs_diff == 0.0);
      assert(r.dropped_frames == 0);
      assert(r.max_occupancy <= base.capacity);
      // Exactness and boundedness hold in every arm; only the transition count
      // is allowed to move.
      assert(r.compared_values == base.frames * voice::StreamingFbank::kMelBins);
    }
    assert(control.config.single_watermark);
    // The premise of the exercise: hysteresis must actually reduce handshakes.
    if (arms.size() > 1) assert(arms.back().transitions < control.transitions);

    if (!json_path.empty()) {
      std::ofstream out(json_path);
      if (!out) throw std::runtime_error("cannot write " + json_path);
      out.precision(17);
      out << "{\n";
      out << "  \"status\": \"measured_hysteresis_flow_control\",\n";
      out << "  \"platform\": \"" << PlatformTag() << "\",\n";
      out << "  \"compiler\": \"" << CompilerTag() << "\",\n";
      out << "  \"model\": {\"tick\": \"one 160-sample hop of audio time\", "
             "\"consumer_drain_period_ticks\": "
          << kDrainPeriod << ", \"consumer_rate_fraction\": 0.5, \"capacity_frames\": "
          << base.capacity << ", \"frames\": " << base.frames << "},\n";
      out << "  \"control_arm\": \"single watermark (high = C, low = C-1)\",\n";
      out << "  \"arms\": [\n";
      for (std::size_t i = 0; i < arms.size(); ++i) {
        WriteArm(out, arms[i]);
        out << (i + 1 == arms.size() ? "\n" : ",\n");
      }
      out << "  ]\n";
      out << "}\n";
      std::cout << "wrote " << json_path << "\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "hysteresis audit error: " << e.what() << "\n";
    return 1;
  }
}
