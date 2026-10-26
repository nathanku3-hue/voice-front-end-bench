#pragma once

#include "streaming_fbank.h"

#include <cstddef>

namespace voice {

// Caller-owned, bounded-output interface for StreamingFbank. The original
// AcceptWaveform queues all completed frames if its consumer stalls. Here,
// processing pauses when the caller has no room, leaving unconsumed PCM with
// the caller.
//
// Memory: the extractor is constructed with a one-frame output ring, because
// Process() slices input at frame boundaries and pops the frame immediately.
// The whole bridge therefore costs one 648-byte feature slot plus the fixed
// DSP tables, and it performs no heap allocation after construction --
// asserted at the operator new boundary by test_zero_alloc_steady_state.
class BoundedFbankBridge {
 public:
  using Feature = StreamingFbank::Feature;
  struct Step {
    std::size_t samples_consumed = 0;
    std::size_t frames_written = 0;
  };

  BoundedFbankBridge() : extractor_(1) {}

  // Writes at most output_capacity feature frames. With zero output capacity,
  // consumes zero samples. Validates the offered PCM before touching state.
  // The extractor never has a pending output frame when this method returns.
  Step Process(const float* pcm, std::size_t samples,
               Feature* output, std::size_t output_capacity);

  void Reset();
  std::size_t SamplesConsumedTotal() const noexcept { return consumed_total_; }

 private:
  StreamingFbank extractor_;
  std::size_t consumed_total_ = 0;
  std::size_t next_frame_at_ = StreamingFbank::kFrameSize;
};

}  // namespace voice
