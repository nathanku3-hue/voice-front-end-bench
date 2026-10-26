// Dump the C++ frontend's feature matrix for an external consumer.
//
// Two jobs, one binary, because they must observe exactly the same numbers:
//
//   --out features.f32    raw little-endian float32, row-major [frames x 80].
//                         scripts/audit_numeric_sensitivity.py feeds this to
//                         the int8 CTC model and compares the transcripts that
//                         come back for different --log / --scalar settings.
//                         The frontend is the only thing that varies; the model
//                         file, the decoder and the detokenizer do not.
//
//   --range-json r.json   per-stage dynamic range (min/max/peak/RMS) plus a
//                         log2 histogram of the per-frame peak, for the
//                         fixed-point audit. This is the slow path: the probe
//                         walks every value of every stage.
//
// --repeat loops the clip so a 2.5-minute Dev set can be stretched into the
// hour of audio the range audit asks for, without a 230 MB WAV on disk. The
// loop is continuous (the extractor state carries across the seam), so frames
// straddling the splice are real frames, not padding.

#include "streaming_fbank.h"
#include "wav_reader.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using voice::StreamingFbank;

constexpr std::size_t kHop = StreamingFbank::kFrameShift;

std::string CompilerTag() {
#if defined(__clang__)
  return "clang " __clang_version__;
#elif defined(__GNUC__)
  return "gcc " __VERSION__;
#else
  return "unknown";
#endif
}

bool ParseLogMode(const std::string& text, StreamingFbank::LogMode* mode) {
  if (text == "std") *mode = StreamingFbank::LogMode::kStd;
  else if (text == "fast") *mode = StreamingFbank::LogMode::kFastFloat;
  else if (text == "fastdouble") *mode = StreamingFbank::LogMode::kFastDouble;
  else return false;
  return true;
}

std::string JsonEscape(const std::string& text) {
  std::string out;
  out.reserve(text.size() + 8);
  for (char c : text) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default: out += c;
    }
  }
  return out;
}

struct Totals {
  std::size_t frames = 0;
  std::size_t samples = 0;
  double checksum = 0.0;   // sum of |value|, for a cheap integrity tag
  double min_value = 1e300;
  double max_value = -1e300;
};

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string wav_path, out_path, json_path, range_path;
    std::size_t repeats = 1;
    bool scalar = false;
    StreamingFbank::LogMode log_mode = StreamingFbank::LogMode::kFastFloat;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      const auto value = [&](const char* name) -> std::string {
        if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + name);
        return argv[++i];
      };
      if (arg == "--wav") wav_path = value("--wav");
      else if (arg == "--out") out_path = value("--out");
      else if (arg == "--json") json_path = value("--json");
      else if (arg == "--range-json") range_path = value("--range-json");
      else if (arg == "--repeat") repeats = std::stoull(value("--repeat"));
      else if (arg == "--scalar") scalar = true;
      else if (arg == "--log") {
        const std::string mode = value("--log");
        if (!ParseLogMode(mode, &log_mode)) throw std::invalid_argument("--log must be std|fast|fastdouble");
      } else if (arg == "--help") {
        std::cout << "usage: fbank_feature_dump --wav in.wav [--out feats.f32] [--json meta.json]\n"
                     "                        [--range-json range.json] [--repeat 1]\n"
                     "                        [--scalar] [--log std|fast|fastdouble]\n";
        return 0;
      } else {
        std::cerr << "unknown argument: " << arg << "\n";
        return 2;
      }
    }
    if (wav_path.empty()) throw std::invalid_argument("--wav is required");
    if (repeats == 0) throw std::invalid_argument("--repeat must be >= 1");

    const voice::tool::WavClip clip = voice::tool::ReadWav(wav_path);
    StreamingFbank fbank(4);
    fbank.SetLogMode(log_mode);
    fbank.EnableScalarStages(scalar);
    if (!range_path.empty()) fbank.EnableRangeStats(true);

    std::ofstream sink;
    if (!out_path.empty()) {
      sink.open(out_path, std::ios::binary);
      if (!sink) throw std::runtime_error("cannot write " + out_path);
    }

    Totals totals;
    std::vector<float> row(StreamingFbank::kMelBins);
    for (std::size_t pass = 0; pass < repeats; ++pass) {
      for (std::size_t pos = 0; pos < clip.pcm.size();) {
        const std::size_t n = std::min(kHop, clip.pcm.size() - pos);
        fbank.AcceptWaveform(clip.pcm.data() + pos, n);
        pos += n;
        while (fbank.NumFramesReady()) {
          const StreamingFbank::Feature frame = fbank.GetFrame();
          for (std::size_t b = 0; b < row.size(); ++b) {
            row[b] = static_cast<float>(frame[b]);
            totals.checksum += std::abs(frame[b]);
            totals.min_value = std::min(totals.min_value, frame[b]);
            totals.max_value = std::max(totals.max_value, frame[b]);
          }
          if (sink) sink.write(reinterpret_cast<const char*>(row.data()),
                               static_cast<std::streamsize>(row.size() * sizeof(float)));
          ++totals.frames;
        }
      }
      totals.samples += clip.pcm.size();
    }
    if (sink) sink.close();

    const double audio_seconds = static_cast<double>(totals.samples) / 16000.0;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "frames: " << totals.frames << " x " << StreamingFbank::kMelBins
              << ", audio " << audio_seconds << " s, log " << StreamingFbank::LogModeName(log_mode)
              << ", stages " << (scalar ? "scalar_reference" : "vector") << ", isa "
              << voice::IsaName(fbank.ActiveIsa()) << "\n";
    std::cout << "feature range [" << totals.min_value << ", " << totals.max_value
              << "], checksum " << totals.checksum << "\n";

    if (!json_path.empty()) {
      std::ofstream out(json_path);
      if (!out) throw std::runtime_error("cannot write " + json_path);
      out.precision(17);
      out << "{\n";
      out << "  \"status\": \"dumped_fbank_features\",\n";
      out << "  \"wav\": \"" << JsonEscape(wav_path) << "\",\n";
      out << "  \"compiler\": \"" << CompilerTag() << "\",\n";
      out << "  \"isa\": \"" << voice::IsaName(fbank.ActiveIsa()) << "\",\n";
      out << "  \"sample_domain_stages\": \"" << (scalar ? "scalar_reference" : "vector") << "\",\n";
      out << "  \"log_mode\": \"" << StreamingFbank::LogModeName(log_mode) << "\",\n";
      out << "  \"repeat\": " << repeats << ",\n";
      out << "  \"frames\": " << totals.frames << ",\n";
      out << "  \"bins\": " << StreamingFbank::kMelBins << ",\n";
      out << "  \"dtype\": \"<f4\",\n";
      out << "  \"layout\": \"row-major [frames][bins]\",\n";
      out << "  \"audio_seconds\": " << audio_seconds << ",\n";
      out << "  \"feature_min\": " << totals.min_value << ",\n";
      out << "  \"feature_max\": " << totals.max_value << ",\n";
      out << "  \"abs_sum\": " << totals.checksum << "\n";
      out << "}\n";
    }

    if (!range_path.empty()) {
      std::ofstream out(range_path);
      if (!out) throw std::runtime_error("cannot write " + range_path);
      out.precision(17);
      out << "{\n";
      out << "  \"status\": \"measured_dynamic_range\",\n";
      out << "  \"wav\": \"" << JsonEscape(wav_path) << "\",\n";
      out << "  \"repeat\": " << repeats << ",\n";
      out << "  \"frames\": " << totals.frames << ",\n";
      out << "  \"audio_seconds\": " << audio_seconds << ",\n";
      out << "  \"isa\": \"" << voice::IsaName(fbank.ActiveIsa()) << "\",\n";
      out << "  \"log_mode\": \"" << StreamingFbank::LogModeName(log_mode) << "\",\n";
      out << "  \"peak_hist_bins\": " << StreamingFbank::kPeakHistBins << ",\n";
      out << "  \"peak_hist_offset\": " << StreamingFbank::kPeakHistOffset << ",\n";
      out << "  \"peak_hist_definition\": \"bin b holds frames whose peak |x| has ilogb == b - offset; "
             "bin 0 is an all-zero frame\",\n";
      out << "  \"rms_hist_definition\": \"same binning, applied to the per-frame RMS\",\n";
      out << "  \"crest_definition\": \"per-frame 20*log10(peak/rms) in dB; drives the "
             "block-floating-point SQNR, which is level independent\",\n";
      out << "  \"signals\": [\n";
      for (std::size_t s = 0; s < StreamingFbank::kRangeSignalCount; ++s) {
        const StreamingFbank::RangeProbe& probe = fbank.Range(s);
        const double rms = probe.count ? std::sqrt(probe.sum_sq / static_cast<double>(probe.count)) : 0.0;
        out << "    {\"name\": \"" << StreamingFbank::RangeSignalName(s) << "\", "
            << "\"min\": " << probe.min << ", \"max\": " << probe.max
            << ", \"abs_max\": " << probe.abs_max << ", \"rms\": " << rms
            << ", \"count\": " << probe.count
            << ", \"crest_mean_db\": "
            << (probe.frame_crest_count
                    ? probe.frame_crest_sum / static_cast<double>(probe.frame_crest_count)
                    : 0.0)
            << ", \"crest_min_db\": "
            << (probe.frame_crest_count ? probe.frame_crest_min : 0.0)
            << ", \"crest_max_db\": "
            << (probe.frame_crest_count ? probe.frame_crest_max : 0.0)
            << ", \"crest_frames\": " << probe.frame_crest_count
            << ", \"peak_hist\": [";
        for (int b = 0; b < StreamingFbank::kPeakHistBins; ++b) {
          out << probe.peak_hist[static_cast<std::size_t>(b)];
          if (b + 1 != StreamingFbank::kPeakHistBins) out << ", ";
        }
        out << "], \"rms_hist\": [";
        for (int b = 0; b < StreamingFbank::kPeakHistBins; ++b) {
          out << probe.rms_hist[static_cast<std::size_t>(b)];
          if (b + 1 != StreamingFbank::kPeakHistBins) out << ", ";
        }
        out << "]}" << (s + 1 == StreamingFbank::kRangeSignalCount ? "\n" : ",\n");
      }
      out << "  ]\n";
      out << "}\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "feature dump error: " << e.what() << "\n";
    return 1;
  }
}
