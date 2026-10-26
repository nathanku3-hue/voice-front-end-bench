#include "axi4s_adapter.h"

#include <cstring>
#include <stdexcept>

namespace voice {
namespace {

std::uint32_t FloatToBits(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float BitsToFloat(std::uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

}  // namespace

Axi4sFrontend::Axi4sFrontend() : Axi4sFrontend(Config{}) {}

Axi4sFrontend::Axi4sFrontend(Config config) : config_(config) {
  if (config_.beat_bytes != 4) throw std::invalid_argument("only 32-bit TDATA is modelled");
  if (config_.burst_bytes == 0 || config_.burst_bytes % config_.beat_bytes != 0)
    throw std::invalid_argument("burst must be a whole number of beats");
  if (config_.fifo_depth_samples == 0) throw std::invalid_argument("FIFO depth must be >= 1");
  beats_per_burst_ = config_.burst_bytes / config_.beat_bytes;
  // One feature frame is 80 doubles; it must divide into whole bursts or the
  // TLAST/TUSER split below would be a lie.
  const std::size_t frame_bytes = StreamingFbank::kMelBins * sizeof(double);
  if (frame_bytes % config_.burst_bytes != 0)
    throw std::invalid_argument("frame size is not a whole number of bursts");
  beats_per_frame_ = frame_bytes / config_.beat_bytes;
  fifo_.assign(config_.fifo_depth_samples, 0.0f);
  scratch_.assign(config_.fifo_depth_samples, 0.0f);
}

void Axi4sFrontend::TryProduceFrame() {
  if (fifo_count_ == 0) return;
  // The bridge wants contiguous PCM. The elastic buffer is a ring, so copy out
  // the live window before handing it over; this is a model cost, not a
  // hardware cost (real hardware streams from a dual-port RAM).
  for (std::size_t i = 0; i < fifo_count_; ++i) {
    scratch_[i] = fifo_[(fifo_head_ + i) % fifo_.size()];
  }
  StreamingFbank::Feature frame{};
  const BoundedFbankBridge::Step step = bridge_.Process(scratch_.data(), fifo_count_, &frame, 1);
  fifo_head_ = (fifo_head_ + step.samples_consumed) % fifo_.size();
  fifo_count_ -= step.samples_consumed;
  if (step.frames_written == 1) {
    pending_frame_ = frame;
    packet_active_ = true;
    packet_bytes_sent_ = 0;
  }
}

Axi4sFrontend::Cycle Axi4sFrontend::Tick(bool s_tvalid, std::uint32_t s_tdata, bool m_tready) {
  Cycle cycle;
  cycle.s_tvalid = s_tvalid;
  cycle.s_tdata = s_tdata;
  cycle.m_tready = m_tready;

  const unsigned char* payload = reinterpret_cast<const unsigned char*>(pending_frame_.data());
  const bool burst_last = (beat_in_burst_ + 1) % beats_per_burst_ == 0;

  // ---- downstream transfer -------------------------------------------------
  // A beat moves if and only if TVALID && TREADY in the same cycle. TVALID is
  // derived from the packet cursor, never from TREADY.
  if (packet_active_) {
    cycle.m_tvalid = true;
    std::memcpy(&cycle.m_tdata, payload + packet_bytes_sent_, sizeof(cycle.m_tdata));
    cycle.m_tlast = burst_last;
    if (m_tready) {
      const std::size_t frame_bytes = StreamingFbank::kMelBins * sizeof(double);
      packet_bytes_sent_ += config_.beat_bytes;
      ++beats_transferred_;
      ++beat_in_burst_;
      if (beat_in_burst_ == beats_per_burst_) beat_in_burst_ = 0;
      const bool frame_done = packet_bytes_sent_ >= frame_bytes;
      // The frame is 640 bytes = ten bursts, so the frame boundary always
      // coincides with a burst boundary; TUSER rides on that beat.
      cycle.m_tuser = frame_done && burst_last;
      if (frame_done) {
        packet_active_ = false;
        packet_bytes_sent_ = 0;
        ++frames_transferred_;
      }
    } else {
      cycle.downstream_stalled = true;
    }
  }

  // ---- upstream acceptance -------------------------------------------------
  // TREADY is a function of occupancy only, so a producer that never asserts
  // TVALID cannot wedge the adapter.
  cycle.s_tready = fifo_count_ < fifo_.size();
  if (s_tvalid && cycle.s_tready) {
    fifo_[(fifo_head_ + fifo_count_) % fifo_.size()] = BitsToFloat(s_tdata);
    ++fifo_count_;
    ++beats_accepted_;
  } else if (s_tvalid) {
    cycle.upstream_stalled = true;
  }

  // ---- compute -------------------------------------------------------------
  // The frontend only runs when there is nowhere to put a finished frame.
  // That single condition is what turns downstream backpressure into upstream
  // backpressure instead of into dropped data.
  if (!packet_active_) TryProduceFrame();

  cycle.fifo_level = fifo_count_;
  cycle.beat_in_burst = beat_in_burst_;
  return cycle;
}

}  // namespace voice
