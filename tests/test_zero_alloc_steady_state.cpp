// Steady-state zero-allocation audit.
//
// Firmware rule: the hot path must never call the allocator. Fragmentation and
// allocator lock contention are non-deterministic latency, which no amount of
// average-case throughput makes acceptable in an RTOS or an SSD controller.
//
// This test brackets a 10,000-frame streaming window with the global operator
// new probe switched off. Inside the window the pipeline runs under genuine
// backpressure (capacity-2 ring, consumer at 50% of the input rate), so the
// audit covers pause/resume, not just the happy path. It asserts that not one
// byte was allocated -- transient or otherwise -- and that every feature value
// is bit-identical to the unthrottled reference.

#include "alloc_probe.h"
#include "bounded_fbank_bridge.h"

#ifdef NDEBUG
#undef NDEBUG  // The assertions below are the measurement.
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
// One hop per producer tick. The consumer then drains one frame every two
// ticks, i.e. 50% of the 100 frames/s input rate -- the same slow-consumer
// ratio the one-hour soak uses, so the audit covers real backpressure.
constexpr std::size_t kChunk = kHop;
constexpr std::size_t kDrainPeriod = 2;

// Deterministic, allocation-free test signal.
float WaveSample(std::size_t i) {
  return static_cast<float>(.37 * std::sin(.013 * static_cast<double>(i)) +
                            .11 * std::cos(.037 * static_cast<double>(i)));
}

struct AuditConfig {
  std::size_t frames = 10000;
  std::size_t capacity = 2;   // caller-owned output ring, in frames
  std::size_t warmup_frames = 200;
};

struct AuditReport {
  AuditConfig config;
  std::size_t frames = 0;
  std::size_t samples = 0;
  std::size_t ticks = 0;
  std::size_t pause_events = 0;
  std::size_t resume_events = 0;
  std::size_t max_frames_held = 0;
  std::size_t compared_values = 0;
  double max_abs_diff = 0.0;
  double wall_seconds = 0.0;

  // Probe snapshots bracketing the audited window.
  std::size_t warmup_allocations = 0;
  std::size_t warmup_live_bytes = 0;
  std::size_t window_allocations = 0;
  std::size_t window_denied = 0;
  std::size_t window_denied_bytes = 0;
  std::size_t window_live_delta = 0;
  std::size_t total_live_bytes = 0;
  std::size_t peak_live_bytes = 0;
  bool probe_control_passed = false;
};

// Feeds the reference extractor with unlimited output capacity. Runs before the
// audited window, so its allocations are outside the measurement.
std::vector<Feature> BuildReference(std::size_t frames, std::size_t samples) {
  voice::StreamingFbank plain(frames + 2);
  std::vector<Feature> reference;
  reference.reserve(frames);
  std::array<float, kChunk> block{};
  for (std::size_t pos = 0; pos < samples;) {
    const std::size_t n = std::min(kChunk, samples - pos);
    for (std::size_t i = 0; i < n; ++i) block[i] = WaveSample(pos + i);
    plain.AcceptWaveform(block.data(), n);
    pos += n;
    while (plain.NumFramesReady()) reference.push_back(plain.GetFrame());
  }
  if (reference.size() < frames) throw std::logic_error("reference produced too few frames");
  return reference;
}

AuditReport RunAudit(const AuditConfig& cfg) {
  assert(cfg.frames > 0 && cfg.capacity >= 1);
  AuditReport report;
  report.config = cfg;

  const std::size_t samples =
      voice::StreamingFbank::kFrameSize + (cfg.frames - 1) * kHop;

  // ---- phase 0: warm up, with allocations allowed -------------------------
  // Construction allocates the output ring and the FFTW plan. That is init, not
  // steady state; it is counted and reported rather than hidden.
  const voice::probe::Snapshot before_warmup = voice::probe::Capture();
  std::vector<Feature> reference = BuildReference(cfg.frames, samples);
  {
    Bridge bridge;
    std::array<Feature, 1> out{};
    std::array<float, kChunk> block{};
    std::size_t pos = 0, produced = 0;
    while (produced < cfg.warmup_frames && pos < samples) {
      const std::size_t n = std::min(kChunk, samples - pos);
      for (std::size_t i = 0; i < n; ++i) block[i] = WaveSample(pos + i);
      pos += n;
      std::size_t offset = 0;
      while (offset < n) {
        const auto step = bridge.Process(block.data() + offset, n - offset, out.data(), 1);
        offset += step.samples_consumed;
        produced += step.frames_written;
        if (step.samples_consumed == 0 && step.frames_written == 0) break;
      }
    }
  }
  const voice::probe::Snapshot after_warmup = voice::probe::Capture();
  report.warmup_allocations = after_warmup.total_allocations - before_warmup.total_allocations;
  report.warmup_live_bytes = after_warmup.live_bytes;

  // ---- phase 1: the audited window ---------------------------------------
  Bridge bridge;
  // Process() is allowed to write `room` frames in one call, so the output
  // scratch must be at least as large as the largest room this harness offers.
  // Sizing it to 1 here would be an out-of-bounds write whenever the ring has
  // two free slots.
  std::array<Feature, 8> out{};
  std::array<float, kChunk> block{};
  std::array<Feature, 8> ring{};  // fixed storage: the harness must not allocate
  std::size_t ring_head = 0, ring_count = 0;
  std::size_t pos = 0, buf_len = 0, buf_off = 0;
  std::size_t drained = 0;
  bool stalled = false;
  double sum_abs = 0.0;
  const std::size_t capacity = std::min<std::size_t>(cfg.capacity, ring.size());

  voice::probe::ResetDenied();
  const voice::probe::Snapshot window_start = voice::probe::Capture();
  voice::probe::SetAllocationAllowed(false);
  const auto started = std::chrono::steady_clock::now();

  while (drained < cfg.frames) {
    if (buf_off == buf_len && pos < samples) {
      buf_len = std::min(kChunk, samples - pos);
      for (std::size_t i = 0; i < buf_len; ++i) block[i] = WaveSample(pos + i);
      buf_off = 0;
      pos += buf_len;
    }
    const std::size_t room = capacity - ring_count;
    const auto step = (buf_off < buf_len && room > 0)
                          ? bridge.Process(block.data() + buf_off, buf_len - buf_off,
                                           out.data(), std::min(room, out.size()))
                          : Bridge::Step{};
    buf_off += step.samples_consumed;
    for (std::size_t i = 0; i < step.frames_written; ++i) {
      ring[(ring_head + ring_count) % capacity] = out[i];
      ++ring_count;
    }
    report.max_frames_held = std::max(report.max_frames_held, ring_count);

    // A pause is the producer being refused: input is still pending after the
    // tick because the caller-owned ring had no room. It is the offer that is
    // throttled, not the amount the bridge happened to consume in one call.
    const bool now_stalled = buf_off < buf_len;
    if (now_stalled) {
      if (!stalled) {
        ++report.pause_events;
        stalled = true;
      }
    } else if (stalled) {
      ++report.resume_events;
      stalled = false;
    }

    // Consumer slot: one frame per kDrainPeriod ticks, driven by its own clock
    // so a stalled producer cannot stall the consumer (that is what makes this
    // backpressure rather than rate matching).
    if (ring_count > 0 && drained < cfg.frames && report.ticks % kDrainPeriod == kDrainPeriod - 1) {
      const Feature& got = ring[ring_head];
      const Feature& want = reference[drained];
      for (std::size_t j = 0; j < got.size(); ++j) {
        const double diff = std::abs(got[j] - want[j]);
        if (diff > report.max_abs_diff) report.max_abs_diff = diff;
        sum_abs += diff;
      }
      report.compared_values += got.size();
      ring_head = (ring_head + 1) % capacity;
      --ring_count;
      ++drained;
    }
    ++report.ticks;
  }

  const auto finished = std::chrono::steady_clock::now();
  voice::probe::SetAllocationAllowed(true);
  const voice::probe::Snapshot window_end = voice::probe::Capture();

  report.wall_seconds = std::chrono::duration<double>(finished - started).count();
  report.frames = drained;
  report.samples = bridge.SamplesConsumedTotal();
  report.window_allocations = window_end.total_allocations - window_start.total_allocations;
  report.window_denied = voice::probe::DeniedAllocations();
  report.window_denied_bytes = voice::probe::DeniedBytes();
  report.window_live_delta =
      window_end.live_bytes > window_start.live_bytes
          ? window_end.live_bytes - window_start.live_bytes
          : window_start.live_bytes - window_end.live_bytes;
  report.total_live_bytes = window_end.live_bytes;
  report.peak_live_bytes = voice::probe::PeakBytes();

  // ---- phase 2: prove the probe is actually wired -------------------------
  // A zero is only meaningful if a non-zero is possible. Allocate deliberately
  // with the probe live and require it to be seen.
  {
    const std::size_t baseline = voice::probe::TotalAllocations();
    voice::probe::SetAllocationAllowed(false);
    {
      void* volatile p = ::operator new(64);
      ::operator delete(p);
    }
    const std::size_t denied = voice::probe::DeniedAllocations();
    voice::probe::SetAllocationAllowed(true);
    report.probe_control_passed = denied >= report.window_denied + 1 &&
                                  voice::probe::TotalAllocations() > baseline;
    voice::probe::ResetDenied();
  }

  report.max_abs_diff = report.max_abs_diff;
  (void)sum_abs;
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

std::string ToJson(const AuditReport& r) {
  std::ostringstream os;
  os.precision(17);
  os << "{\n";
  os << "  \"status\": \"measured_zero_alloc_steady_state\",\n";
  os << "  \"platform\": \"" << PlatformTag() << "\",\n";
  os << "  \"compiler\": \"" << CompilerTag() << "\",\n";
  os << "  \"config\": {\"frames\": " << r.config.frames << ", \"output_capacity\": "
     << r.config.capacity << ", \"chunk_samples\": " << kChunk << ", \"warmup_frames\": "
     << r.config.warmup_frames << "},\n";
  os << "  \"frames\": " << r.frames << ",\n";
  os << "  \"samples\": " << r.samples << ",\n";
  os << "  \"ticks\": " << r.ticks << ",\n";
  os << "  \"pause_events\": " << r.pause_events << ",\n";
  os << "  \"resume_events\": " << r.resume_events << ",\n";
  os << "  \"max_frames_held\": " << r.max_frames_held << ",\n";
  os << "  \"compared_values\": " << r.compared_values << ",\n";
  os << "  \"max_abs_diff\": " << r.max_abs_diff << ",\n";
  os << "  \"wall_seconds\": " << r.wall_seconds << ",\n";
  os << "  \"warmup_allocations\": " << r.warmup_allocations << ",\n";
  os << "  \"warmup_live_bytes\": " << r.warmup_live_bytes << ",\n";
  os << "  \"window_allocations\": " << r.window_allocations << ",\n";
  os << "  \"window_denied_allocations\": " << r.window_denied << ",\n";
  os << "  \"window_denied_bytes\": " << r.window_denied_bytes << ",\n";
  os << "  \"window_live_delta_bytes\": " << r.window_live_delta << ",\n";
  os << "  \"live_bytes_end\": " << r.total_live_bytes << ",\n";
  os << "  \"peak_live_bytes\": " << r.peak_live_bytes << ",\n";
  os << "  \"probe_control_passed\": " << (r.probe_control_passed ? "true" : "false") << ",\n";
  os << "  \"bytes_per_frame_slot\": " << sizeof(Feature) << ",\n";
  os << "  \"bridge_footprint_bytes\": "
     << sizeof(Bridge) + sizeof(Feature) << "\n";
  os << "}\n";
  return os.str();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    AuditConfig cfg;
    std::string json_path;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      const auto value = [&](const char* name) -> std::string {
        if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + name);
        return argv[++i];
      };
      if (arg == "--frames") cfg.frames = std::stoull(value("--frames"));
      else if (arg == "--capacity") cfg.capacity = std::stoull(value("--capacity"));
      else if (arg == "--json") json_path = value("--json");
      else if (arg == "--help") {
        std::cout << "usage: test_zero_alloc_steady_state [--frames N] [--capacity F] "
                     "[--json PATH]\n";
        return 0;
      } else {
        std::cerr << "unknown argument: " << arg << "\n";
        return 2;
      }
    }

    const AuditReport report = RunAudit(cfg);

    // Report first, assert second: a failed assertion should print the numbers
    // that failed, not just the line number.
    std::cout << "steady state: " << report.frames << " frames / " << report.samples
              << " samples, capacity " << cfg.capacity << " frames, " << report.ticks
              << " ticks, " << report.wall_seconds << " s wall\n";
    std::cout << "flow: " << report.pause_events << " pause / " << report.resume_events
              << " resume, max frames held " << report.max_frames_held << "\n";
    std::cout << "allocations: warm-up " << report.warmup_allocations << " (init only), window "
              << report.window_allocations << " total, " << report.window_denied
              << " denied, live delta " << report.window_live_delta << " B\n";
    std::cout << "exactness: " << report.compared_values << " values, max |diff| "
              << report.max_abs_diff << std::endl;  // flush: assert may abort next

    // The claims, as assertions rather than prose.
    assert(report.frames == cfg.frames);
    assert(report.compared_values == cfg.frames * voice::StreamingFbank::kMelBins);
    assert(report.max_abs_diff == 0.0);
    assert(report.window_allocations == 0);
    assert(report.window_denied == 0);
    assert(report.window_live_delta == 0);
    assert(report.probe_control_passed);
    // The window must actually contain backpressure, or this is not a
    // steady-state audit of the real path.
    assert(report.pause_events > 0 && report.resume_events > 0);
    assert(report.max_frames_held <= cfg.capacity);

    std::cout << "PASS: zero heap allocations across " << report.frames
              << " steady-state frames under backpressure\n";

    if (!json_path.empty()) {
      std::ofstream out(json_path);
      if (!out) throw std::runtime_error("cannot write " + json_path);
      out << ToJson(report);
      std::cout << "wrote " << json_path << "\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "zero-alloc audit error: " << e.what() << "\n";
    return 1;
  }
}
