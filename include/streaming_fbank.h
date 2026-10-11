#pragma once

#include <array>
#include <cstddef>
#include <deque>
#include <memory>

namespace voice {

// Fixed Round-1 configuration: 16 kHz, 25 ms frame, 10 ms hop, 80 mel bands.
// Input: mono float PCM; zero-padding is used only in FFT (400 -> 512).
// snip_edges = true: trailing incomplete frame is discarded.
class StreamingFbank {
 public:
  static constexpr std::size_t kSampleRate = 16000;
  static constexpr std::size_t kFrameSize = 400;
  static constexpr std::size_t kFrameShift = 160;
  static constexpr std::size_t kFftSize = 512;
  static constexpr std::size_t kMelBins = 80;
  using Feature = std::array<double, kMelBins>;

  // Opt-in diagnostic snapshot for a single frame. Not part of the normal
  // feature pipeline or an assertion of float32/official Kaldi parity.
  struct StageTrace {
    std::array<double, kFrameSize> after_preemphasis{};
    std::array<double, kFftSize> windowed_padded{};
    std::array<double, kFftSize / 2 + 1> power_spectrum{};
    std::array<double, kMelBins> mel_energy{};  // before log/floor
    Feature log_fbank{};
  };

  StreamingFbank();
  ~StreamingFbank();
  StreamingFbank(const StreamingFbank&) = delete;
  StreamingFbank& operator=(const StreamingFbank&) = delete;
  StreamingFbank(StreamingFbank&&) = delete;
  StreamingFbank& operator=(StreamingFbank&&) = delete;

  // NULL is valid only when count == 0. Throws on invalid input.
  void AcceptWaveform(const float* samples, std::size_t count);
  std::size_t NumFramesReady() const noexcept;
  Feature GetFrame();  // Pops oldest; throws std::out_of_range if none.
  std::size_t FramesProduced() const noexcept;
  // Allocate diagnostic storage once, before accepting input. Frame indices
  // start at 0. Enabling diagnostics is not intended for real-time callbacks.
  void CaptureStageTraceAt(std::size_t frame_index);
  bool HasStageTrace() const noexcept;
  const StageTrace& GetStageTrace() const;  // throws if not captured
  void Reset();

 private:
  struct Fft;
  std::unique_ptr<Fft> fft_;
  std::array<double, kFrameSize> window_{};
  std::array<std::array<double, kFftSize / 2 + 1>, kMelBins> mel_{};
  std::array<float, kFrameSize> ring_{};
  std::size_t head_ = 0;
  std::size_t filled_ = 0;
  std::size_t produced_ = 0;
  std::deque<Feature> ready_;
  std::unique_ptr<StageTrace> trace_;
  std::size_t trace_target_ = 0;
  bool trace_recorded_ = false;

  void EmitFrame();
};

}  // namespace voice
