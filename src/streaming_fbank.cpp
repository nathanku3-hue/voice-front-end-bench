#include "streaming_fbank.h"

#include "fbank_kernels.h"
#include "fast_log.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
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
constexpr double kMelLow = 20.0;
constexpr double kMelHigh = 8000.0;

double Mel(double hz) { return 1127.0 * std::log1p(hz / 700.0); }

// Single source of truth for the filterbank coefficient. Called once per
// non-zero tap at construction, and per entry if the dense reference is built.
double MelWeightAt(std::size_t band, std::size_t bin) {
  if (bin == StreamingFbank::kFftSize / 2) return 0.0;  // Kaldi banks exclude Nyquist
  const double low = Mel(kMelLow);
  const double increment = (Mel(kMelHigh) - low) / (StreamingFbank::kMelBins + 1);
  const double left = low + static_cast<double>(band) * increment;
  const double center = left + increment;
  const double right = center + increment;
  const double mel_freq =
      Mel(static_cast<double>(bin) * static_cast<double>(StreamingFbank::kSampleRate) /
          StreamingFbank::kFftSize);
  const double up = (mel_freq - left) / (center - left);
  const double down = (right - mel_freq) / (right - center);
  return std::max(0.0, std::min(up, down));
}

// RAII micro-stage timer. When timing is off the object is a null pointer and
// the destructor does nothing, so the only cost is one predictable branch.
class StageTimer {
 public:
  StageTimer(std::uint64_t* slot, bool enabled) noexcept : slot_(enabled ? slot : nullptr) {
    if (slot_) start_ = std::chrono::steady_clock::now();
  }
  ~StageTimer() {
    if (slot_) {
      *slot_ += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                               std::chrono::steady_clock::now() - start_)
                                               .count());
    }
  }
  StageTimer(const StageTimer&) = delete;
  StageTimer& operator=(const StageTimer&) = delete;

 private:
  std::uint64_t* slot_;
  std::chrono::steady_clock::time_point start_{};
};

}  // namespace

struct StreamingFbank::Fft {
  fftw_plan plan = nullptr;

  // Planned on caller-owned memory: the buffers live inside StreamingFbank and
  // are 64-byte aligned, so FFTW picks its aligned SIMD codelets and this
  // object owns no heap block of its own.
  void Create(double* input, double* output) {
    plan = fftw_plan_dft_r2c_1d(static_cast<int>(kFftSize), input,
                                reinterpret_cast<fftw_complex*>(output), FFTW_ESTIMATE);
    if (!plan) throw std::runtime_error("FFTW setup failed");
  }
  ~Fft() {
    if (plan) fftw_destroy_plan(plan);
  }
};

const char* StreamingFbank::StageName(std::size_t stage) noexcept {
  switch (stage) {
    case kStageDcRemove: return "dc_remove";
    case kStagePreemphasisWindow: return "preemphasis_window";
    case kStageFft: return "fftw_execute";
    case kStagePowerSpectrum: return "power_spectrum";
    case kStageMelProjection: return "mel_projection";
    case kStageLogCompress: return "log_compress";
    default: return "unknown";
  }
}

const char* StreamingFbank::LogModeName(LogMode mode) noexcept {
  switch (mode) {
    case LogMode::kFastFloat: return "fast_float";
    case LogMode::kFastDouble: return "fast_double";
    case LogMode::kStd:
    default: return "std_log";
  }
}

const char* StreamingFbank::RangeSignalName(std::size_t signal) noexcept {
  switch (signal) {
    case kRangeInputPcm: return "input_pcm";
    case kRangeDcRemoved: return "dc_removed";
    case kRangePreemphasized: return "preemphasized";
    case kRangeWindowed: return "windowed";
    case kRangeFftComponent: return "fft_component";
    case kRangePowerSpectrum: return "power_spectrum";
    case kRangeMelEnergy: return "mel_energy";
    case kRangeLogFbank: return "log_fbank";
    default: return "unknown";
  }
}

void StreamingFbank::RangeProbe::CloseFrame() noexcept {
  const auto bin_of = [](double magnitude) noexcept {
    if (!(magnitude > 0.0)) return std::size_t{0};
    const int bin = std::ilogb(magnitude) + kPeakHistOffset;
    return static_cast<std::size_t>(std::max(1, std::min(kPeakHistBins - 1, bin)));
  };
  ++peak_hist[bin_of(frame_peak)];
  const double frame_rms =
      frame_count ? std::sqrt(frame_sum_sq / static_cast<double>(frame_count)) : 0.0;
  ++rms_hist[bin_of(frame_rms)];
  if (frame_peak > 0.0 && frame_rms > 0.0) {
    const double crest = 20.0 * std::log10(frame_peak / frame_rms);
    if (crest < frame_crest_min) frame_crest_min = crest;
    if (crest > frame_crest_max) frame_crest_max = crest;
    frame_crest_sum += crest;
    ++frame_crest_count;
  }
  frame_peak = 0.0;
  frame_sum_sq = 0.0;
  frame_count = 0;
}

StreamingFbank::StreamingFbank(std::size_t output_capacity)
    : fft_(std::make_unique<Fft>()), isa_(DetectIsa()) {
  if (output_capacity == 0) throw std::invalid_argument("output capacity must be >= 1");
  // The single allocation this object performs, taken here and never again.
  ready_.resize(output_capacity);
  fft_->Create(fft_input_.data(), fft_output_.data());

  // Kaldi-style Povey (symmetric Hann ^ 0.85, no periodic flag).
  for (std::size_t n = 0; n < kFrameSize; ++n) {
    const double hann = 0.5 - 0.5 * std::cos(2.0 * kPi * n / (kFrameSize - 1));
    window_[n] = std::pow(std::max(0.0, hann), 0.85);
  }

  // Compress the filterbank to its non-zero taps. This is the second and last
  // allocation this object performs; the hot path only reads the result.
  std::size_t nonzeros = 0;
  for (std::size_t b = 0; b < kMelBins; ++b) {
    for (std::size_t k = 0; k < kPowerBins; ++k) {
      if (MelWeightAt(b, k) > 0.0) ++nonzeros;
    }
  }
  taps_.reserve(nonzeros);
  for (std::size_t b = 0; b < kMelBins; ++b) {
    tap_offsets_[b] = static_cast<std::uint32_t>(taps_.size());
    for (std::size_t k = 0; k < kPowerBins; ++k) {
      const double weight = MelWeightAt(b, k);
      if (weight > 0.0) taps_.push_back(MelTap{static_cast<std::uint16_t>(k), weight});
    }
  }
  tap_offsets_[kMelBins] = static_cast<std::uint32_t>(taps_.size());
}

void StreamingFbank::EnableSparseMel(bool enabled) {
  if (!enabled && dense_mel_.empty()) {
    dense_mel_.assign(kMelBins * kPowerBins, 0.0);
    for (std::size_t b = 0; b < kMelBins; ++b) {
      for (std::size_t k = 0; k < kPowerBins; ++k) {
        dense_mel_[b * kPowerBins + k] = MelWeightAt(b, k);
      }
    }
  }
  sparse_mel_ = enabled;
}

void StreamingFbank::EnableRangeStats(bool enabled) {
  if (enabled && !range_) range_ = std::make_unique<std::array<RangeProbe, kRangeSignalCount>>();
  if (!enabled) range_.reset();
}

const StreamingFbank::RangeProbe& StreamingFbank::Range(std::size_t signal) const {
  if (!range_) throw std::logic_error("range stats are not enabled");
  if (signal >= kRangeSignalCount) throw std::out_of_range("range signal out of range");
  return (*range_)[signal];
}

StreamingFbank::~StreamingFbank() = default;

std::size_t StreamingFbank::FramesThisBatchWouldProduce(std::size_t count) const noexcept {
  if (count > std::numeric_limits<std::size_t>::max() - filled_) return kDefaultOutputCapacity + 1;
  const std::size_t total = filled_ + count;
  if (total < kFrameSize) return 0;
  return 1 + (total - kFrameSize) / kFrameShift;
}

void StreamingFbank::AcceptWaveform(const float* samples, std::size_t count) {
  if (count && !samples) throw std::invalid_argument("non-empty waveform requires samples");
  for (std::size_t i = 0; i < count; ++i) {
    if (!std::isfinite(samples[i])) throw std::invalid_argument("non-finite PCM sample");
  }
  // Bounded output is a contract, not a preference: refuse the batch before
  // consuming any sample rather than growing the ring or dropping frames.
  const std::size_t would_produce = FramesThisBatchWouldProduce(count);
  if (would_produce > OutputRoom()) {
    throw std::overflow_error("feature output ring would overflow; drain frames first");
  }
  for (std::size_t i = 0; i < count; ++i) {
    ring_[(head_ + filled_) % kFrameSize] = samples[i];
    ++filled_;
    if (filled_ == kFrameSize) {
      EmitFrame();
      head_ = (head_ + kFrameShift) % kFrameSize;
      filled_ -= kFrameShift;
    }
  }
}

// Copies the 400-sample frame out of the ring into the aligned staging buffer.
// Two memcpys replace 400 modulo-indexed loads; the modulo is what stopped the
// compiler from vectorising the sample-domain stages at all.
void StreamingFbank::GatherFrame() noexcept {
  const std::size_t tail = kFrameSize - head_;
  std::memcpy(frame_pcm_.data(), ring_.data() + head_, tail * sizeof(float));
  if (head_ != 0) std::memcpy(frame_pcm_.data() + tail, ring_.data(), head_ * sizeof(float));
}

double StreamingFbank::MeanScalar() const noexcept {
  return kernels::MeanScalarKernel(ring_.data(), head_, kFrameSize);
}

double StreamingFbank::MeanVector() noexcept {
  GatherFrame();
#ifdef VOICE_HAVE_X86_INTRINSICS
  if (isa_ == Isa::kAvx2Fma) return kernels::MeanAvx2Kernel(frame_pcm_.data(), kFrameSize);
#endif
  return kernels::MeanContiguousKernel(frame_pcm_.data(), kFrameSize);
}

void StreamingFbank::PreemphWindowScalar(double mean, StageTrace* trace) noexcept {
  if (trace) {
    double previous = static_cast<double>(ring_[head_]) - mean;
    for (std::size_t n = 0; n < kFrameSize; ++n) {
      const double centered =
          static_cast<double>(ring_[(head_ + n) % kFrameSize]) - mean;
      const double preemphasized = centered - kernels::kPreemphasis * previous;
      trace->after_preemphasis[n] = preemphasized;
      fft_input_[n] = preemphasized * window_[n];
      previous = centered;
    }
  } else {
    kernels::PreemphWindowScalarKernel(ring_.data(), head_, window_.data(), fft_input_.data(), mean,
                              kFrameSize);
  }
}

void StreamingFbank::PreemphWindowVector(double mean, StageTrace* trace) noexcept {
  if (trace) {
    // Trace capture is a diagnostic build; the vector kernel would have to
    // store every lane separately, so this path stays scalar on purpose.
    double previous = static_cast<double>(frame_pcm_[0]) - mean;
    for (std::size_t n = 0; n < kFrameSize; ++n) {
      const double centered = static_cast<double>(frame_pcm_[n]) - mean;
      const double preemphasized = centered - kernels::kPreemphasis * previous;
      trace->after_preemphasis[n] = preemphasized;
      fft_input_[n] = preemphasized * window_[n];
      previous = centered;
    }
  } else {
#ifdef VOICE_HAVE_X86_INTRINSICS
    if (isa_ == Isa::kAvx2Fma) {
      kernels::PreemphWindowAvx2Kernel(frame_pcm_.data(), window_.data(), fft_input_.data(), mean,
                              kFrameSize);
      return;
    }
#endif
    kernels::PreemphWindowContiguousKernel(frame_pcm_.data(), window_.data(), fft_input_.data(), mean,
                                  kFrameSize);
  }
}

void StreamingFbank::LogCompress(const std::array<double, kMelBins>& mel_energy,
                                 Feature* feature) noexcept {
  constexpr double kFloor = static_cast<double>(std::numeric_limits<float>::epsilon());
  switch (log_mode_) {
    case LogMode::kFastFloat: {
#ifdef VOICE_HAVE_X86_INTRINSICS
      if (isa_ == Isa::kAvx2Fma) {
        alignas(32) float clamped[kMelBins];
        for (std::size_t b = 0; b < kMelBins; ++b) {
          clamped[b] = static_cast<float>(std::max(mel_energy[b], kFloor));
        }
        kernels::FastLogAvx2Kernel(clamped, feature->data(), kMelBins);
        return;
      }
#endif
      for (std::size_t b = 0; b < kMelBins; ++b) {
        (*feature)[b] = static_cast<double>(FastLog(static_cast<float>(
            std::max(mel_energy[b], kFloor))));
      }
      return;
    }
    case LogMode::kFastDouble:
      for (std::size_t b = 0; b < kMelBins; ++b) {
        (*feature)[b] = FastLogDouble(std::max(mel_energy[b], kFloor));
      }
      return;
    case LogMode::kStd:
    default:
      for (std::size_t b = 0; b < kMelBins; ++b) {
        (*feature)[b] = std::log(std::max(mel_energy[b], kFloor));
      }
      return;
  }
}

void StreamingFbank::CloseRangeFrame() noexcept {
  for (std::size_t s = 0; s < kRangeSignalCount; ++s) (*range_)[s].CloseFrame();
}

void StreamingFbank::EmitFrame() {
  const bool timing = stage_timing_;
  StageTrace* trace = trace_ && produced_ == trace_target_ ? trace_.get() : nullptr;
  const bool probe = range_ != nullptr;

  double mean = 0.0;
  {
    StageTimer timer(&stage_ns_[kStageDcRemove], timing);
    mean = scalar_stages_ ? MeanScalar() : MeanVector();
    mean /= static_cast<double>(kFrameSize);
  }

  {
    StageTimer timer(&stage_ns_[kStagePreemphasisWindow], timing);
    if (scalar_stages_) {
      PreemphWindowScalar(mean, trace);
    } else {
      PreemphWindowVector(mean, trace);
    }
    std::fill(fft_input_.begin() + kFrameSize, fft_input_.end(), 0.0);
    if (trace) std::copy(fft_input_.begin(), fft_input_.end(), trace->windowed_padded.begin());
  }

  if (probe) {
    // Dynamic-range audit only. The two intermediates are recomputed rather
    // than stored: same operations, same values, no second pipeline.
    for (std::size_t n = 0; n < kFrameSize; ++n) {
      const double raw = scalar_stages_ ? static_cast<double>(ring_[(head_ + n) % kFrameSize])
                                        : static_cast<double>(frame_pcm_[n]);
      (*range_)[kRangeInputPcm].Add(raw);
      const double centered = raw - mean;
      (*range_)[kRangeDcRemoved].Add(centered);
      const double previous =
          n == 0 ? centered
                 : (scalar_stages_ ? static_cast<double>(ring_[(head_ + n - 1) % kFrameSize]) - mean
                                   : static_cast<double>(frame_pcm_[n - 1]) - mean);
      (*range_)[kRangePreemphasized].Add(centered - kernels::kPreemphasis * previous);
      (*range_)[kRangeWindowed].Add(fft_input_[n]);
    }
  }

  {
    StageTimer timer(&stage_ns_[kStageFft], timing);
    fftw_execute(fft_->plan);
  }

  std::array<double, kPowerBins> power{};
  {
    StageTimer timer(&stage_ns_[kStagePowerSpectrum], timing);
    for (std::size_t k = 0; k < kPowerBins; ++k) {
      const double real = fft_output_[2 * k], imag = fft_output_[2 * k + 1];
      power[k] = real * real + imag * imag;
      if (probe) {
        (*range_)[kRangeFftComponent].Add(real);
        (*range_)[kRangeFftComponent].Add(imag);
      }
    }
    if (probe) {
      for (std::size_t k = 0; k < kPowerBins; ++k) (*range_)[kRangePowerSpectrum].Add(power[k]);
    }
  }
  if (trace) trace->power_spectrum = power;

  std::array<double, kMelBins> mel_energy{};
  {
    StageTimer timer(&stage_ns_[kStageMelProjection], timing);
    if (sparse_mel_) {
      for (std::size_t b = 0; b < kMelBins; ++b) {
        double energy = 0.0;
        for (std::uint32_t i = tap_offsets_[b]; i < tap_offsets_[b + 1]; ++i) {
          energy += taps_[i].weight * power[taps_[i].bin];
        }
        mel_energy[b] = energy;
      }
    } else {
      for (std::size_t b = 0; b < kMelBins; ++b) {
        double energy = 0.0;
        const double* row = dense_mel_.data() + b * kPowerBins;
        for (std::size_t k = 0; k < kPowerBins; ++k) energy += row[k] * power[k];
        mel_energy[b] = energy;
      }
    }
  }

  Feature feature{};
  {
    StageTimer timer(&stage_ns_[kStageLogCompress], timing);
    LogCompress(mel_energy, &feature);
  }

  if (probe) {
    for (std::size_t b = 0; b < kMelBins; ++b) {
      (*range_)[kRangeMelEnergy].Add(mel_energy[b]);
      (*range_)[kRangeLogFbank].Add(feature[b]);
    }
    CloseRangeFrame();
  }

  if (trace) {
    trace->mel_energy = mel_energy;
    trace->log_fbank = feature;
    trace_recorded_ = true;
  }
  ready_[(ready_head_ + ready_count_) % ready_.size()] = feature;
  ++ready_count_;
  ++produced_;
  if (timing) ++timed_frames_;
}

std::size_t StreamingFbank::NumFramesReady() const noexcept { return ready_count_; }
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

void StreamingFbank::ResetStageTimings() noexcept {
  stage_ns_.fill(0);
  timed_frames_ = 0;
}

std::uint64_t StreamingFbank::StageNanoseconds(std::size_t stage) const noexcept {
  return stage < kStageCount ? stage_ns_[stage] : 0;
}

StreamingFbank::Feature StreamingFbank::GetFrame() {
  if (ready_count_ == 0) throw std::out_of_range("No Fbank frames ready");
  Feature value = ready_[ready_head_];
  ready_head_ = (ready_head_ + 1) % ready_.size();
  --ready_count_;
  return value;
}

void StreamingFbank::Reset() {
  head_ = filled_ = produced_ = 0;
  ready_head_ = ready_count_ = 0;
  trace_.reset();
  trace_recorded_ = false;
}

}  // namespace voice
