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

  void EmitFrame();
};

}  // namespace voice
