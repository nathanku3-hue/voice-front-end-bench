#include "bounded_fbank_bridge.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace voice {

BoundedFbankBridge::Step BoundedFbankBridge::Process(
    const float* pcm, std::size_t samples,
    Feature* output, std::size_t output_capacity) {
  if (samples != 0 && pcm == nullptr)
    throw std::invalid_argument("samples require a non-null PCM buffer");
  if (output_capacity != 0 && output == nullptr)
    throw std::invalid_argument("output capacity requires a feature buffer");
  if (samples > std::numeric_limits<std::size_t>::max() - consumed_total_)
    throw std::overflow_error("sample counter overflow");

  // Reject invalid batches before advancing the stream; the underlying
  // extractor otherwise validates while consuming, allowing partial updates.
  for (std::size_t i = 0; i < samples; ++i)
    if (!std::isfinite(pcm[i])) throw std::invalid_argument("non-finite PCM sample");

  Step step;
  while (step.samples_consumed < samples &&
         step.frames_written < output_capacity) {
    if (next_frame_at_ <= consumed_total_)
      throw std::logic_error("invalid next-frame boundary");
    const auto until_frame = next_frame_at_ - consumed_total_;
    const auto n = std::min(until_frame, samples - step.samples_consumed);
    extractor_.AcceptWaveform(pcm + step.samples_consumed, n);
    consumed_total_ += n;
    step.samples_consumed += n;

    if (consumed_total_ == next_frame_at_) {
      if (extractor_.NumFramesReady() != 1)
        throw std::logic_error("extractor produced unexpected frame count");
      output[step.frames_written++] = extractor_.GetFrame();
      if (next_frame_at_ > std::numeric_limits<std::size_t>::max() -
                               StreamingFbank::kFrameShift)
        throw std::overflow_error("next-frame counter overflow");
      next_frame_at_ += StreamingFbank::kFrameShift;
    } else if (extractor_.NumFramesReady() != 0) {
      throw std::logic_error("unexpected queued feature frame");
    }
  }
  if (extractor_.NumFramesReady() != 0)
    throw std::logic_error("bounded bridge left queued frames");
  return step;
}

void BoundedFbankBridge::Reset() {
  extractor_.Reset();
  consumed_total_ = 0;
  next_frame_at_ = StreamingFbank::kFrameSize;
}

}  // namespace voice
