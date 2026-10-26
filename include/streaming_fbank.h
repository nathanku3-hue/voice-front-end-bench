#pragma once

#include "cpu_isa.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace voice {

// Fixed Round-1 configuration: 16 kHz, 25 ms frame, 10 ms hop, 80 mel bands.
// Input: mono float PCM; zero-padding is used only in FFT (400 -> 512).
// snip_edges = true: trailing incomplete frame is discarded.
//
// Allocation policy: the only heap allocations this object ever performs are
// the caller-sized output ring and the compressed filterbank, both taken in the
// constructor. The FFT scratch lives inside the object, 64-byte aligned, so
// FFTW is planned on memory we own instead of on fftw_malloc blocks. Once
// constructed, the steady-state path (AcceptWaveform -> EmitFrame -> GetFrame)
// performs no allocation at all, which tests/test_zero_alloc_steady_state.cpp
// asserts at the operator new boundary. Output capacity is a declared bound,
// not a suggestion: AcceptWaveform refuses input that would overflow the ring
// instead of growing it.
class StreamingFbank {
 public:
  static constexpr std::size_t kSampleRate = 16000;
  static constexpr std::size_t kFrameSize = 400;
  static constexpr std::size_t kFrameShift = 160;
  static constexpr std::size_t kFftSize = 512;
  static constexpr std::size_t kMelBins = 80;
  static constexpr std::size_t kPowerBins = kFftSize / 2 + 1;
  // 256 frames = 2.56 s of audio. Direct users that hand over a large block in
  // one call must size the ring for the frames that block can produce; the
  // bounded bridge always uses 1.
  static constexpr std::size_t kDefaultOutputCapacity = 256;
  using Feature = std::array<double, kMelBins>;

  // Opt-in diagnostic snapshot for a single frame. Not part of the normal
  // feature pipeline or an assertion of float32/official Kaldi parity.
  struct StageTrace {
    std::array<double, kFrameSize> after_preemphasis{};
    std::array<double, kFftSize> windowed_padded{};
    std::array<double, kPowerBins> power_spectrum{};
    std::array<double, kMelBins> mel_energy{};  // before log/floor
    Feature log_fbank{};
  };

  // Micro-stage split of one frame, used by the latency budget in
  // tools/fbank_stage_profile.cpp. Order matches the pipeline.
  enum Stage : std::size_t {
    kStageDcRemove = 0,
    kStagePreemphasisWindow = 1,
    kStageFft = 2,
    kStagePowerSpectrum = 3,
    kStageMelProjection = 4,
    kStageLogCompress = 5,
    kStageCount = 6,
  };
  static const char* StageName(std::size_t stage) noexcept;

  // How the log-compression stage computes log(mel energy).
  //   kStd        std::log, double, <1 ulp. The reference.
  //   kFastFloat  IEEE-754 exponent split + degree-7 atanh series in float,
  //               vectorised 8-wide. Error budget in include/fast_log.h.
  //   kFastDouble the same decomposition in double, scalar.
  // Only kFastFloat is a shipping candidate; the other two exist so the
  // accuracy/speed trade can be measured instead of asserted.
  enum class LogMode { kStd = 0, kFastFloat = 1, kFastDouble = 2 };
  void SetLogMode(LogMode mode) noexcept { log_mode_ = mode; }
  LogMode GetLogMode() const noexcept { return log_mode_; }
  static const char* LogModeName(LogMode mode) noexcept;

  // Signals whose dynamic range the fixed-point audit needs. One accumulator
  // per signal, each holding global min/max/RMS plus a log2 histogram of the
  // per-frame peak (the quantity that decides the fixed-point headroom).
  enum RangeSignal : std::size_t {
    kRangeInputPcm = 0,
    kRangeDcRemoved = 1,
    kRangePreemphasized = 2,
    kRangeWindowed = 3,
    kRangeFftComponent = 4,
    kRangePowerSpectrum = 5,
    kRangeMelEnergy = 6,
    kRangeLogFbank = 7,
    kRangeSignalCount = 8,
  };
  static constexpr int kPeakHistBins = 64;
  static constexpr int kPeakHistOffset = 48;  // bin = ilogb(peak) + offset
  struct RangeProbe {
    double min = std::numeric_limits<double>::infinity();
    double max = -std::numeric_limits<double>::infinity();
    double abs_max = 0.0;
    double sum_sq = 0.0;
    std::uint64_t count = 0;
    double frame_peak = 0.0;
    double frame_sum_sq = 0.0;
    std::uint64_t frame_count = 0;
    // Within-frame crest factor, peak/RMS in dB, accumulated in CloseFrame().
    // A block-floating-point stage carries one exponent per block, so after
    // normalisation the only level dependence left is the crest factor inside a
    // single block. Measured rather than assumed, because that number is the
    // whole argument for or against BFP.
    double frame_crest_min = std::numeric_limits<double>::infinity();
    double frame_crest_max = 0.0;
    double frame_crest_sum = 0.0;
    std::uint64_t frame_crest_count = 0;
    std::array<std::uint32_t, kPeakHistBins> peak_hist{};
    std::array<std::uint32_t, kPeakHistBins> rms_hist{};

    void Add(double value) noexcept {
      if (value < min) min = value;
      if (value > max) max = value;
      const double magnitude = value < 0.0 ? -value : value;
      if (magnitude > abs_max) abs_max = magnitude;
      if (magnitude > frame_peak) frame_peak = magnitude;
      const double square = value * value;
      sum_sq += square;
      frame_sum_sq += square;
      ++count;
      ++frame_count;
    }
    void CloseFrame() noexcept;
  };
  static const char* RangeSignalName(std::size_t signal) noexcept;
  // Allocates the accumulators on first enable; disabled by default and free
  // (one predictable branch per stage) when off. Enabling it makes the frame
  // path several times slower on purpose -- it is an audit build, not a ship
  // build.
  void EnableRangeStats(bool enabled);
  bool RangeStatsEnabled() const noexcept { return range_ != nullptr; }
  const RangeProbe& Range(std::size_t signal) const;

  // Measurement-only reference path: the pre-SIMD implementation of the two
  // sample-domain stages (modulo indexing, scalar accumulation, no alignment
  // assumption). Same role as EnableSparseMel(false) -- it exists so the
  // before/after table is measured in one process instead of two binaries.
  void EnableScalarStages(bool enabled) noexcept { scalar_stages_ = enabled; }
  bool ScalarStagesEnabled() const noexcept { return scalar_stages_; }

  explicit StreamingFbank(std::size_t output_capacity = kDefaultOutputCapacity);
  ~StreamingFbank();
  StreamingFbank(const StreamingFbank&) = delete;
  StreamingFbank& operator=(const StreamingFbank&) = delete;
  StreamingFbank(StreamingFbank&&) = delete;
  StreamingFbank& operator=(StreamingFbank&&) = delete;

  // NULL is valid only when count == 0. Throws std::invalid_argument on
  // non-finite input and std::overflow_error -- before touching any state --
  // when this batch would produce more frames than the output ring can hold.
  void AcceptWaveform(const float* samples, std::size_t count);
  std::size_t NumFramesReady() const noexcept;
  Feature GetFrame();  // Pops oldest; throws std::out_of_range if none.
  std::size_t FramesProduced() const noexcept;
  std::size_t OutputCapacity() const noexcept { return ready_.size(); }

  // Frames a batch of `count` samples would produce, given the samples already
  // buffered. Exposed so a caller can size its read block instead of guessing.
  std::size_t FramesThisBatchWouldProduce(std::size_t count) const noexcept;
  // Free slots in the output ring.
  std::size_t OutputRoom() const noexcept { return ready_.size() - ready_count_; }

  // Allocate diagnostic storage once, before accepting input. Frame indices
  // start at 0. Enabling diagnostics is not intended for real-time callbacks.
  void CaptureStageTraceAt(std::size_t frame_index);
  bool HasStageTrace() const noexcept;
  const StageTrace& GetStageTrace() const;  // throws if not captured

  // Mel projection layout. The dense table would be 80 x 257 doubles = 164 KB,
  // re-read for every 10 ms frame, and it dominated the stage budget. Only 501
  // of its 20,560 entries are non-zero (2.4%), so the default path walks a
  // compressed CSR of (bin, weight) taps built once in the constructor and the
  // object does NOT carry the dense table at all.
  //
  // The sparse path is bit-exact, not approximate: every skipped term is
  // exactly +0.0 (the filterbank weights and the power spectrum are both
  // non-negative), and adding +0.0 to a partial sum that is never -0.0 leaves
  // it unchanged. tests/test_mel_sparsity.cpp asserts equality against the
  // dense path on real audio.
  //
  // EnableSparseMel(false) materialises the dense table on demand, for A/B
  // measurement only. That allocates ~164 KB and is not a hot-path call.
  void EnableSparseMel(bool enabled);
  bool SparseMelEnabled() const noexcept { return sparse_mel_; }
  std::size_t MelTapCount() const noexcept { return taps_.size(); }
  std::size_t MelDenseEntryCount() const noexcept { return kMelBins * kPowerBins; }

  // Opt-in stage timing. Disabled by default; the disabled path costs one
  // predictable branch per stage and never reads the clock. Counters are
  // cumulative and are never reset by Reset().
  void EnableStageTiming(bool enabled) noexcept { stage_timing_ = enabled; }
  bool StageTimingEnabled() const noexcept { return stage_timing_; }
  void ResetStageTimings() noexcept;
  std::uint64_t StageNanoseconds(std::size_t stage) const noexcept;
  std::uint64_t TimedFrames() const noexcept { return timed_frames_; }

  // ISA the sample-domain stages actually used. Fixed at construction.
  Isa ActiveIsa() const noexcept { return isa_; }
  // Byte offset of the aligned frame staging buffer inside the object, and the
  // offset of the FFT scratch: the two things __builtin_assume_aligned and the
  // AVX2 stores rely on. Reported so the alignment claim is checkable.
  const void* FrameBufferAddress() const noexcept { return frame_pcm_.data(); }
  const void* FftInputAddress() const noexcept { return fft_input_.data(); }

  void Reset();

 private:
  struct Fft;
  // One non-zero filterbank coefficient. `bin` indexes the power spectrum.
  struct MelTap {
    std::uint16_t bin = 0;
    double weight = 0.0;
  };
  std::unique_ptr<Fft> fft_;
  // 64-byte aligned: one cache line per 16 samples, so every vector load in the
  // sample-domain stages is a full-line access at a known offset.
  alignas(64) std::array<float, kFrameSize> frame_pcm_{};
  alignas(64) std::array<double, kFftSize> fft_input_{};
  alignas(64) std::array<double, 2 * kPowerBins> fft_output_{};
  alignas(64) std::array<double, kFrameSize> window_{};
  std::vector<MelTap> taps_;                              // CSR values
  std::array<std::uint32_t, kMelBins + 1> tap_offsets_{};  // CSR row pointers
  // Only populated when the dense reference path is requested; empty otherwise,
  // so the shipped object does not carry 164 KB of mostly-zero coefficients.
  std::vector<double> dense_mel_;
  bool sparse_mel_ = true;
  std::array<float, kFrameSize> ring_{};
  std::size_t head_ = 0;
  std::size_t filled_ = 0;
  std::size_t produced_ = 0;

  std::vector<Feature> ready_;  // fixed size, set once in the constructor
  std::size_t ready_head_ = 0;
  std::size_t ready_count_ = 0;

  std::unique_ptr<StageTrace> trace_;
  std::size_t trace_target_ = 0;
  bool trace_recorded_ = false;

  bool stage_timing_ = false;
  std::array<std::uint64_t, kStageCount> stage_ns_{};
  std::uint64_t timed_frames_ = 0;

  Isa isa_ = Isa::kScalar;
  bool scalar_stages_ = false;
  LogMode log_mode_ = LogMode::kFastFloat;
  std::unique_ptr<std::array<RangeProbe, kRangeSignalCount>> range_;

  // Sample-domain stages, both ISA variants. They write fft_input_ and, when a
  // trace is armed, the trace arrays; everything downstream is shared.
  double MeanScalar() const noexcept;
  double MeanVector() noexcept;
  void PreemphWindowScalar(double mean, StageTrace* trace) noexcept;
  void PreemphWindowVector(double mean, StageTrace* trace) noexcept;
  void GatherFrame() noexcept;
  void LogCompress(const std::array<double, kMelBins>& mel_energy, Feature* feature) noexcept;
  void CloseRangeFrame() noexcept;

  void EmitFrame();
};

}  // namespace voice
