// AXI4-Stream handshake audit.
//
// The claim under test is narrow and checkable: when the downstream consumer
// drops TREADY, the adapter loses nothing, reorders nothing, and pushes the
// backpressure upstream to S_AXIS_TREADY instead of into a drop counter. That
// is the whole point of the wrapper, and it is exactly what a waveform shows
// and a sentence does not.
//
// Three things are asserted, each against a different oracle:
//
//   1. Protocol legality, checked cycle by cycle against the AXI4-Stream rules
//      themselves: TLAST on every 16th beat, TUSER on the 160th, TVALID held
//      across a stall without the data advancing, TREADY a function of FIFO
//      occupancy alone.
//   2. Data integrity, checked against the frontend itself: the 640 bytes
//      reassembled from the beat stream must equal the feature frames a plain
//      StreamingFbank produces from the same PCM, bit for bit.
//   3. Stall-invariance, checked by running the same input twice under
//      different TREADY scripts and comparing the delivered frame sequence.
//      Backpressure may change *when* a frame arrives; it must not change
//      *what* arrives.
//
// Not claimed: any timing, any synthesisability. The model presents TVALID
// combinationally from a packet cursor; a registered implementation costs one
// cycle and changes no protocol behaviour.
//
// Writes results/axi4s_protocol.json; tests/test_firmware_evidence.py
// recomputes every ratio in it from the same numbers.

#include "alloc_probe.h"
#include "axi4s_adapter.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

using Feature = voice::StreamingFbank::Feature;
using voice::Axi4sFrontend;

constexpr std::size_t kMelBins = voice::StreamingFbank::kMelBins;
constexpr std::size_t kFrameBytes = kMelBins * sizeof(double);  // 640

float WaveSample(std::size_t i) {
  return static_cast<float>(.41 * std::sin(.011 * static_cast<double>(i)) +
                            .13 * std::cos(.029 * static_cast<double>(i)) +
                            .05 * std::sin(.31 * static_cast<double>(i)));
}

std::uint32_t FloatToBits(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

// A TREADY script. Every arm sees the same PCM and the same producer; only the
// consumer's willingness to accept differs.
struct Script {
  const char* name = "always_ready";
  std::size_t warmup = 0;       // cycles of TREADY=1 before anything else
  std::size_t stall_start = 0;  // first cycle of the long stall
  std::size_t stall_cycles = 0;
  std::size_t duty_period = 0;  // 0 = no duty cycling; else ready when phase < duty_high
  std::size_t duty_high = 0;

  bool Ready(std::size_t cycle) const {
    if (cycle < warmup) return true;
    if (stall_cycles != 0 && cycle >= stall_start && cycle < stall_start + stall_cycles)
      return false;
    if (duty_period != 0) return (cycle % duty_period) < duty_high;
    return true;
  }
};

struct ArmResult {
  std::string name;
  std::size_t cycles = 0;
  std::size_t beats_in = 0;          // accepted upstream
  std::size_t beats_out = 0;         // transferred downstream
  std::size_t frames = 0;            // TUSER-terminated packets
  std::size_t stall_cycles = 0;      // cycles with a beat offered and TREADY low
  std::size_t held_beats = 0;        // beats that were held and re-presented
  std::size_t held_mismatch = 0;     // ... with a different TDATA (must be 0)
  std::size_t tlast_errors = 0;
  std::size_t tuser_errors = 0;
  std::size_t bubble_errors = 0;     // TVALID dropped mid-packet (must be 0)
  std::size_t tready_rule_errors = 0;
  std::size_t max_fifo = 0;
  std::size_t s_tready_low_cycles = 0;
  std::size_t first_s_tready_low = 0;
  std::size_t upstream_stall_cycles = 0;
  std::size_t allocations = 0;
  std::size_t denied_allocations = 0;
  std::vector<double> frame_values;  // reassembled, 80 per frame
};

// Reference frames straight from the frontend, no adapter in the way.
std::vector<double> ReferenceFrames(const std::vector<float>& pcm, std::size_t max_frames) {
  voice::StreamingFbank frontend(max_frames + 4);
  std::vector<double> out;
  std::array<float, 160> block{};
  for (std::size_t pos = 0; pos < pcm.size();) {
    const std::size_t n = std::min(block.size(), pcm.size() - pos);
    frontend.AcceptWaveform(pcm.data() + pos, n);
    pos += n;
    while (frontend.NumFramesReady()) {
      const Feature frame = frontend.GetFrame();
      out.insert(out.end(), frame.begin(), frame.end());
    }
  }
  return out;
}

ArmResult RunArm(const Script& script, const std::vector<float>& pcm, std::size_t cycles,
                 std::size_t max_frames) {
  ArmResult result;
  result.name = script.name;

  Axi4sFrontend adapter;
  const std::size_t beats_per_burst = adapter.beats_per_burst();
  const std::size_t beats_per_frame = adapter.beats_per_frame();

  // Every buffer the run needs is sized here, before allocations are forbidden.
  result.frame_values.reserve(2 * max_frames * kMelBins);

  std::size_t producer = 0;
  bool previous_tvalid = false;
  bool previous_tready = false;
  std::uint32_t previous_tdata = 0;
  bool have_previous = false;
  std::size_t level_before = 0;
  std::array<unsigned char, kFrameBytes> packet{};
  std::size_t packet_bytes = 0;

  // TotalAllocations() is cumulative over the process, so the audit has to
  // bracket the window: everything above this line belongs to the reference
  // build and the constructor.
  voice::probe::SetAllocationAllowed(false);
  voice::probe::ResetDenied();
  const std::size_t allocations_before = voice::probe::TotalAllocations();

  for (std::size_t cycle = 0; cycle < cycles; ++cycle) {
    const bool s_tvalid = producer < pcm.size();
    const std::uint32_t s_tdata = s_tvalid ? FloatToBits(pcm[producer]) : 0u;
    const bool m_tready = script.Ready(cycle);

    const Axi4sFrontend::Cycle state = adapter.Tick(s_tvalid, s_tdata, m_tready);
    ++result.cycles;

    // --- TREADY is a function of occupancy alone ---------------------------
    const bool expected_tready = level_before < adapter.FifoCapacity();
    if (state.s_tready != expected_tready) ++result.tready_rule_errors;
    if (!state.s_tready) {
      ++result.s_tready_low_cycles;
      if (result.first_s_tready_low == 0) result.first_s_tready_low = cycle;
    }
    if (state.upstream_stalled) ++result.upstream_stall_cycles;

    // --- a beat presented and refused must be presented again unchanged ----
    if (have_previous && previous_tvalid && !previous_tready && state.m_tvalid) {
      ++result.held_beats;
      if (state.m_tdata != previous_tdata) ++result.held_mismatch;
    }

    // --- downstream transfer ----------------------------------------------
    // The packet-relative beat index is the global transfer counter modulo the
    // packet length: every packet is exactly 160 beats, so idle cycles between
    // packets cannot shift the phase. Using the global counter keeps the check
    // non-circular -- it does not read TUSER to decide where TUSER should be.
    if (state.m_tvalid) {
      if (have_previous && !previous_tvalid && result.beats_out % beats_per_frame != 0) {
        ++result.bubble_errors;  // TVALID dropped in the middle of a packet
      }
      if (state.m_tready) {
        const std::size_t index = result.beats_out;
        ++result.beats_out;
        const bool expect_tlast = (index + 1) % beats_per_burst == 0;
        const bool expect_tuser = (index + 1) % beats_per_frame == 0;
        if (state.m_tlast != expect_tlast) ++result.tlast_errors;
        if (state.m_tuser != expect_tuser) ++result.tuser_errors;

        std::memcpy(packet.data() + packet_bytes, &state.m_tdata, sizeof(state.m_tdata));
        packet_bytes += sizeof(state.m_tdata);
        if (state.m_tuser) {
          assert(packet_bytes == kFrameBytes);
          const double* values = reinterpret_cast<const double*>(packet.data());
          result.frame_values.insert(result.frame_values.end(), values, values + kMelBins);
          packet_bytes = 0;
          ++result.frames;
        }
      } else {
        ++result.stall_cycles;
      }
    }

    if (state.s_tvalid && state.s_tready) {
      ++result.beats_in;
      ++producer;
    }
    result.max_fifo = std::max(result.max_fifo, state.fifo_level);

    level_before = state.fifo_level;
    previous_tvalid = state.m_tvalid;
    previous_tready = state.m_tready;
    previous_tdata = state.m_tdata;
    have_previous = true;
  }

  voice::probe::SetAllocationAllowed(true);
  result.allocations = voice::probe::TotalAllocations() - allocations_before;
  result.denied_allocations = voice::probe::DeniedAllocations();
  return result;
}

void Report(const ArmResult& arm) {
  std::printf("  %-14s cycles %6zu  in %6zu  out %6zu  frames %4zu  "
              "stall %5zu  held %5zu  fifo %3zu\n",
              arm.name.c_str(), arm.cycles, arm.beats_in, arm.beats_out, arm.frames,
              arm.stall_cycles, arm.held_beats, arm.max_fifo);
}

}  // namespace

int main(int argc, char** argv) {
  const std::string out_path = argc > 1 ? argv[1] : "results/axi4s_protocol.json";

  // 4 s of 16 kHz audio: ~397 frames, comfortably more than the 160-beat packet
  // length so the TLAST/TUSER pattern is exercised across many frames.
  constexpr std::size_t kSamples = 64000;
  std::vector<float> pcm(kSamples);
  for (std::size_t i = 0; i < kSamples; ++i) pcm[i] = WaveSample(i);

  const std::size_t max_frames = 420;
  const std::vector<double> reference = ReferenceFrames(pcm, max_frames);
  const std::size_t reference_frames = reference.size() / kMelBins;

  const Axi4sFrontend shape;  // geometry only; constructed outside the audited loop
  std::printf("AXI4-Stream handshake audit: %zu samples, %zu reference frames, "
              "%zu beats/burst, %zu beats/frame\n", kSamples, reference_frames,
              shape.beats_per_burst(), shape.beats_per_frame());

  // Enough cycles for every sample to be accepted even at a 2/3 duty cycle,
  // plus slack for the pipeline to drain.
  constexpr std::size_t kCycles = 140000;

  const std::array<Script, 3> scripts{{
      {"always_ready", 0, 0, 0, 0, 0},
      {"stall_then_duty", 800, 900, 700, 48, 32},
      {"long_stall", 0, 3000, 6000, 0, 0},
  }};

  std::vector<ArmResult> arms;
  arms.reserve(scripts.size());
  for (const Script& script : scripts) {
    arms.push_back(RunArm(script, pcm, kCycles, max_frames));
    Report(arms.back());
  }

  const ArmResult& baseline = arms[0];

  // ---- protocol legality ----------------------------------------------------
  for (const ArmResult& arm : arms) {
    assert(arm.tlast_errors == 0 && "TLAST must land on every 64-byte burst boundary");
    assert(arm.tuser_errors == 0 && "TUSER must land on the beat completing a 640-byte frame");
    assert(arm.bubble_errors == 0 && "TVALID must not drop in the middle of a packet");
    assert(arm.beats_out % shape.beats_per_frame() == 0 && "the beat stream must be whole packets");
    assert(arm.tready_rule_errors == 0 && "S_AXIS_TREADY must depend on occupancy only");
    assert(arm.held_mismatch == 0 && "a stalled beat must be re-presented with the same TDATA");
    assert(arm.max_fifo <= voice::Axi4sFrontend::Config{}.fifo_depth_samples);
    assert(arm.denied_allocations == 0 && "the tick loop must not allocate");
  }

  // ---- data integrity -------------------------------------------------------
  for (const ArmResult& arm : arms) {
    assert(arm.frames == arm.frame_values.size() / kMelBins);
    assert(arm.frames == reference_frames && "the adapter must deliver every frame");
    for (std::size_t i = 0; i < arm.frame_values.size(); ++i) {
      if (arm.frame_values[i] != reference[i]) {
        std::printf("frame mismatch at value %zu: %a vs %a\n", i, arm.frame_values[i], reference[i]);
        assert(false && "reassembled frames must be bit-identical to the frontend");
      }
    }
  }

  // ---- backpressure must actually have engaged ------------------------------
  const ArmResult& stalled = arms[1];
  assert(stalled.stall_cycles > 0 && "the script must actually stall the consumer");
  assert(stalled.held_beats > 0 && "a stall must hold a beat");
  assert(stalled.max_fifo == voice::Axi4sFrontend::Config{}.fifo_depth_samples &&
         "a long enough stall must fill the elastic buffer");
  assert(stalled.s_tready_low_cycles > 0 &&
         "downstream backpressure must reach S_AXIS_TREADY");
  assert(stalled.upstream_stall_cycles > 0 && "the producer must be refused while the FIFO is full");
  assert(arms[2].s_tready_low_cycles > 0);

  // ---- stall invariance -----------------------------------------------------
  // Same input, different TREADY: the delivered frame sequence is identical.
  for (std::size_t a = 1; a < arms.size(); ++a) {
    assert(arms[a].frame_values == baseline.frame_values &&
           "TREADY timing must not change the delivered data");
  }

  // ---- report ---------------------------------------------------------------
  std::printf("\nall assertions passed\n");
  std::printf("  beats per burst        %zu (%zu B)\n", shape.beats_per_burst(), 64u);
  std::printf("  beats per frame        %zu (%zu B)\n", shape.beats_per_frame(), kFrameBytes);
  std::printf("  frames delivered       %zu\n", baseline.frames);
  std::printf("  values compared        %zu\n", baseline.frame_values.size());
  std::printf("  held-beat checks       %zu\n", stalled.held_beats);
  std::printf("  upstream stall cycles  %zu\n", stalled.upstream_stall_cycles);

  std::ofstream out(out_path);
  if (!out) {
    std::fprintf(stderr, "cannot write %s\n", out_path.c_str());
    return 1;
  }
  out.precision(17);
  out << "{\n  \"status\": \"measured_axi4s_handshake\",\n";
  out << "  \"model\": \"BoundedFbankBridge behind an AXI4-Stream shim, one 32-bit beat per cycle\",\n";
  out << "  \"beat_bytes\": 4,\n  \"burst_bytes\": 64,\n";
  out << "  \"beats_per_burst\": " << shape.beats_per_burst()
      << ",\n  \"beats_per_frame\": " << shape.beats_per_frame() << ",\n";
  out << "  \"frame_bytes\": " << kFrameBytes << ",\n";
  out << "  \"fifo_depth_samples\": " << voice::Axi4sFrontend::Config{}.fifo_depth_samples << ",\n";
  out << "  \"samples\": " << kSamples << ",\n";
  out << "  \"cycles\": " << kCycles << ",\n";
  out << "  \"reference_frames\": " << reference_frames << ",\n";
  out << "  \"arms\": [\n";
  for (std::size_t i = 0; i < arms.size(); ++i) {
    const ArmResult& arm = arms[i];
    out << "    {\"name\": \"" << arm.name << "\", \"cycles\": " << arm.cycles
        << ", \"beats_in\": " << arm.beats_in << ", \"beats_out\": " << arm.beats_out
        << ", \"frames\": " << arm.frames
        << ", \"stall_cycles\": " << arm.stall_cycles
        << ", \"held_beats\": " << arm.held_beats
        << ", \"held_mismatch\": " << arm.held_mismatch
        << ", \"tlast_errors\": " << arm.tlast_errors
        << ", \"tuser_errors\": " << arm.tuser_errors
        << ", \"bubble_errors\": " << arm.bubble_errors
        << ", \"tready_rule_errors\": " << arm.tready_rule_errors
        << ", \"max_fifo_level\": " << arm.max_fifo
        << ", \"s_tready_low_cycles\": " << arm.s_tready_low_cycles
        << ", \"first_s_tready_low\": " << arm.first_s_tready_low
        << ", \"upstream_stall_cycles\": " << arm.upstream_stall_cycles
        << ", \"allocations_in_tick_loop\": " << arm.allocations
        << ", \"denied_allocations\": " << arm.denied_allocations
        << "}" << (i + 1 == arms.size() ? "\n" : ",\n");
  }
  out << "  ],\n";
  out << "  \"derived\": {\n";
  out << "    \"all_frames_delivered\": "
      << (std::all_of(arms.begin(), arms.end(),
                      [&](const ArmResult& a) { return a.frames == reference_frames; })
              ? "true" : "false") << ",\n";
  out << "    \"stall_invariant_frames\": "
      << (std::all_of(arms.begin() + 1, arms.end(), [&](const ArmResult& a) {
            return a.frame_values == baseline.frame_values;
          }) ? "true" : "false") << ",\n";
  out << "    \"values_compared\": " << baseline.frame_values.size() << ",\n";
  out << "    \"protocol_violations\": "
      << [&] {
           std::size_t total = 0;
           for (const ArmResult& a : arms) {
             total += a.tlast_errors + a.tuser_errors + a.bubble_errors +
                      a.tready_rule_errors + a.held_mismatch;
           }
           return total;
         }() << ",\n";
  out << "    \"backpressure_reached_upstream\": "
      << (stalled.s_tready_low_cycles > 0 && stalled.upstream_stall_cycles > 0 ? "true" : "false")
      << "\n  },\n";
  out << "  \"non_claims\": [\n";
  out << "    \"cycle counts are model cycles, not nanoseconds on any device\",\n";
  out << "    \"TVALID is combinational from a packet cursor; a registered version costs one cycle\",\n";
  out << "    \"no AXI interconnect, DMA descriptor engine, or clock-domain crossing is modelled\",\n";
  out << "    \"the FIFO is a host vector, not a BRAM with a real read latency\"\n";
  out << "  ]\n}\n";
  std::printf("wrote %s\n", out_path.c_str());
  return 0;
}
