#pragma once

#include "bounded_fbank_bridge.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace voice {

// AXI4-Stream shim around BoundedFbankBridge.
//
// What it models
//   Upstream (adapter is slave):   S_AXIS_TVALID / S_AXIS_TREADY / S_AXIS_TDATA
//   Downstream (adapter is master): M_AXIS_TVALID / M_AXIS_TREADY / M_AXIS_TDATA
//                                   M_AXIS_TLAST / M_AXIS_TUSER
//   One 32-bit beat per clock, one float32 PCM sample on the way in, one
//   32-bit slice of the 80-double feature frame on the way out.
//
// Why TLAST is not the frame boundary
//   A feature frame is 80 doubles = 640 bytes = exactly ten 64-byte bursts.
//   The downstream DMA retires work per 64-byte burst, so TLAST marks the last
//   beat of a burst and TUSER marks the burst that completes a frame. That is
//   the AXI4-Stream-legal split: TLAST delimits a packet, TUSER carries the
//   sideband the consumer needs. A design that put the frame boundary on TLAST
//   would force the DMA to hold a 160-beat packet open across ten lines.
//
// Handshake rules this model keeps, because they are the ones that break real
// designs:
//   * TVALID never depends on TREADY. The master asserts it from state it
//     already owns (the packet cursor); the slave's READY is sampled, not
//     waited on.
//   * TREADY never depends on TVALID. The adapter's S_AXIS_TREADY is a
//     function of FIFO occupancy alone, so a silent producer cannot deadlock it.
//   * A beat is transferred if and only if TVALID && TREADY in the same cycle.
//     Nothing is dropped when TREADY is low: the beat stays in the register and
//     the backpressure walks upstream through the FIFO to S_AXIS_TREADY.
//
// Simplification, stated: TVALID is combinationally derived from the packet
// cursor instead of being registered. A synthesisable version registers it,
// which costs one cycle of latency and changes no protocol behaviour. This is a
// protocol/timing model for the handshake, not an RTL implementation.

class Axi4sFrontend {
 public:
  struct Config {
    std::size_t beat_bytes = 4;            // 32-bit TDATA
    std::size_t burst_bytes = 64;          // one cache line
    std::size_t fifo_depth_samples = 512;  // upstream elastic buffer
  };

  struct Cycle {
    bool s_tvalid = false;
    bool s_tready = false;
    std::uint32_t s_tdata = 0;
    bool m_tvalid = false;
    bool m_tready = false;
    std::uint32_t m_tdata = 0;
    bool m_tlast = false;
    bool m_tuser = false;
    std::size_t fifo_level = 0;
    std::size_t beat_in_burst = 0;
    bool downstream_stalled = false;  // had a beat, TREADY was low
    bool upstream_stalled = false;    // had a sample, TREADY was low
  };

  Axi4sFrontend();
  explicit Axi4sFrontend(Config config);

  // One clock cycle. `s_tvalid`/`s_tdata` are what the producer drives this
  // cycle; `m_tready` is what the consumer drives. Returns the signals as they
  // appear after the edge, which is what a VCD dump wants.
  Cycle Tick(bool s_tvalid, std::uint32_t s_tdata, bool m_tready);

  std::size_t beats_per_burst() const noexcept { return beats_per_burst_; }
  std::size_t beats_per_frame() const noexcept { return beats_per_frame_; }
  std::size_t BeatsTransferred() const noexcept { return beats_transferred_; }
  std::size_t BeatsAccepted() const noexcept { return beats_accepted_; }
  std::size_t FramesTransferred() const noexcept { return frames_transferred_; }
  std::size_t FifoLevel() const noexcept { return fifo_count_; }
  std::size_t FifoCapacity() const noexcept { return fifo_.size(); }
  bool FrameInFlight() const noexcept { return packet_active_; }

 private:
  void TryProduceFrame();

  Config config_;
  std::size_t beats_per_burst_ = 0;
  std::size_t beats_per_frame_ = 0;

  BoundedFbankBridge bridge_;
  std::vector<float> fifo_;
  std::vector<float> scratch_;  // contiguous view handed to the bridge
  std::size_t fifo_head_ = 0;
  std::size_t fifo_count_ = 0;

  StreamingFbank::Feature pending_frame_{};
  bool packet_active_ = false;
  std::size_t packet_bytes_sent_ = 0;
  std::size_t beat_in_burst_ = 0;

  std::size_t beats_transferred_ = 0;
  std::size_t beats_accepted_ = 0;
  std::size_t frames_transferred_ = 0;
};

}  // namespace voice
