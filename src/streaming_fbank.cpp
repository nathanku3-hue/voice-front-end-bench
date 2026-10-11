#include "streaming_fbank.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

// FFTW is the numerical backend, NOT a hand-written FFT. In the test container
// FFTW runtime libraries are present without -dev headers, hence a minimal ABI
// declaration fallback. Installed fftw3.h is used wherever available.
#if __has_include(<fftw3.h>)
#include <fftw3.h>
#else
extern "C" {
typedef struct fftw_plan_s* fftw_plan;
typedef double fftw_complex[2];
void* fftw_malloc(std::size_t n);
void fftw_free(void* p);
fftw_plan fftw_plan_dft_r2c_1d(int n, double* in, fftw_complex* out, unsigned flags);
void fftw_execute(const fftw_plan p);
void fftw_destroy_plan(fftw_plan p);
}
#ifndef FFTW_ESTIMATE
#define FFTW_ESTIMATE (1U << 6)
#endif
#endif

namespace voice {
namespace {
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kPreemphasis = 0.97;
constexpr double kMelLow = 20.0;
constexpr double kMelHigh = 8000.0;

double Mel(double hz) { return 1127.0 * std::log1p(hz / 700.0); }
}  // namespace

struct StreamingFbank::Fft {
  double* input = nullptr;
  fftw_complex* output = nullptr;
  fftw_plan plan = nullptr;

  Fft() {
    input = static_cast<double*>(fftw_malloc(sizeof(double) * kFftSize));
    output = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * (kFftSize / 2 + 1)));
    if (input && output) {
      plan = fftw_plan_dft_r2c_1d(static_cast<int>(kFftSize), input, output, FFTW_ESTIMATE);
    }
    if (!plan) {
      if (output) fftw_free(output);
      if (input) fftw_free(input);
      throw std::runtime_error("FFTW setup failed");
    }
  }
  ~Fft() {
    fftw_destroy_plan(plan);
    fftw_free(output);
    fftw_free(input);
  }
};

StreamingFbank::StreamingFbank() : fft_(std::make_unique<Fft>()) {
  // Kaldi-style Povey (symmetric Hann ^ 0.85, no periodic flag).
  for (std::size_t n = 0; n < kFrameSize; ++n) {
    const double hann = 0.5 - 0.5 * std::cos(2.0 * kPi * n / (kFrameSize - 1));
    window_[n] = std::pow(std::max(0.0, hann), 0.85);
  }
  const double low = Mel(kMelLow);
  const double increment = (Mel(kMelHigh) - low) / (kMelBins + 1);
  for (std::size_t b = 0; b < kMelBins; ++b) {
    const double left = low + b * increment;
    const double center = left + increment;
    const double right = center + increment;
    for (std::size_t k = 0; k < kFftSize / 2; ++k) {
      const double mel_freq = Mel(k * static_cast<double>(kSampleRate) / kFftSize);
      const double up = (mel_freq - left) / (center - left);
      const double down = (right - mel_freq) / (right - center);
      mel_[b][k] = std::max(0.0, std::min(up, down));
    }
    mel_[b][kFftSize / 2] = 0.0;  // torchaudio/Kaldi banks exclude Nyquist.
  }
}

StreamingFbank::~StreamingFbank() = default;

void StreamingFbank::AcceptWaveform(const float* samples, std::size_t count) {
  if (count && !samples) throw std::invalid_argument("non-empty waveform requires samples");
  for (std::size_t i = 0; i < count; ++i) {
    if (!std::isfinite(samples[i])) throw std::invalid_argument("non-finite PCM sample");
    ring_[(head_ + filled_) % kFrameSize] = samples[i];
    ++filled_;
    if (filled_ == kFrameSize) {
      EmitFrame();
      head_ = (head_ + kFrameShift) % kFrameSize;
      filled_ -= kFrameShift;
    }
  }
}

void StreamingFbank::EmitFrame() {
  StageTrace* trace = trace_ && produced_ == trace_target_ ? trace_.get() : nullptr;
  double mean = 0.0;
  for (std::size_t n = 0; n < kFrameSize; ++n) {
    mean += static_cast<double>(ring_[(head_ + n) % kFrameSize]);
  }
  mean /= kFrameSize;

  double previous = static_cast<double>(ring_[head_]) - mean;
  for (std::size_t n = 0; n < kFrameSize; ++n) {
    const double centered = static_cast<double>(ring_[(head_ + n) % kFrameSize]) - mean;
    // Kaldi pre-emphasis is PER FRAME, after optional DC removal. At n=0,
    // the initial previous input is the same first sample (replicate pad).
    const double preemphasized = centered - kPreemphasis * previous;
    if (trace) trace->after_preemphasis[n] = preemphasized;
    fft_->input[n] = preemphasized * window_[n];
    previous = centered;
  }
  std::fill(fft_->input + kFrameSize, fft_->input + kFftSize, 0.0);
  if (trace) std::copy(fft_->input, fft_->input + kFftSize, trace->windowed_padded.begin());
  fftw_execute(fft_->plan);

  std::array<double, kFftSize / 2 + 1> power{};
  for (std::size_t k = 0; k < power.size(); ++k) {
    const double real = fft_->output[k][0], imag = fft_->output[k][1];
    power[k] = real * real + imag * imag;
  }
  if (trace) trace->power_spectrum = power;
  Feature feature{};
  for (std::size_t b = 0; b < kMelBins; ++b) {
    double energy = 0.0;
    for (std::size_t k = 0; k < power.size(); ++k) {
      energy += mel_[b][k] * power[k];
    }
    feature[b] = std::log(std::max(energy, static_cast<double>(std::numeric_limits<float>::epsilon())));
    if (trace) {
      trace->mel_energy[b] = energy;
      trace->log_fbank[b] = feature[b];
    }
  }
  if (trace) trace_recorded_ = true;
  ready_.push_back(feature);
  ++produced_;
}

std::size_t StreamingFbank::NumFramesReady() const noexcept { return ready_.size(); }
std::size_t StreamingFbank::FramesProduced() const noexcept { return produced_; }

void StreamingFbank::CaptureStageTraceAt(std::size_t frame_index) {
  if (filled_ != 0 || produced_ != 0) {
    throw std::logic_error("CaptureStageTraceAt must be called before waveform input");
  }
  trace_target_ = frame_index;
  trace_ = std::make_unique<StageTrace>();
  trace_recorded_ = false;
}

bool StreamingFbank::HasStageTrace() const noexcept { return trace_recorded_; }

const StreamingFbank::StageTrace& StreamingFbank::GetStageTrace() const {
  if (!trace_recorded_ || !trace_) throw std::out_of_range("Stage trace not captured");
  return *trace_;
}

StreamingFbank::Feature StreamingFbank::GetFrame() {
  if (ready_.empty()) throw std::out_of_range("No Fbank frames ready");
  Feature value = ready_.front();
  ready_.pop_front();
  return value;
}

void StreamingFbank::Reset() {
  head_ = filled_ = produced_ = 0;
  ready_.clear();
  trace_.reset();
  trace_recorded_ = false;
}

}  // namespace voice
