// Pure C++17 closed loop: PCM WAV -> bounded C++ Fbank -> CTC greedy decode ->
// UTF-8 text. No Python, no interpreter, no ONNX runtime in this process.
//
// Scope, stated plainly: the acoustic model is NOT in this binary. The
// posteriors come from a frozen export of the same int8 Zipformer CTC model
// that the Python study uses, so everything downstream of the model is C++.
// The frontend is still load-bearing, not decorative: the tool refuses to
// decode unless the frame count produced by the C++ frontend predicts the
// exported logit length exactly (see ExpectedLogitFrames below), which is
// checked on all 30 Dev utterances by scripts/run_pipeline_parity.py.

#include "bounded_fbank_bridge.h"
#include "ctc_greedy_decoder.h"
#include "wav_reader.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// ------------------------------ NPY input ---------------------------------
struct Matrix {
  std::vector<float> data;
  std::size_t rows = 0;
  std::size_t cols = 0;
};

// Minimal .npy v1/v2/v3 reader for a C-order '<f4'/'<f8' array of 2 or 3 dims.
Matrix ReadNpy(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open posterior dump: " + path);
  std::vector<unsigned char> raw((std::istreambuf_iterator<char>(in)),
                                 std::istreambuf_iterator<char>());
  if (raw.size() < 10) throw std::runtime_error("NPY too short");
  if (std::memcmp(raw.data(), "\x93NUMPY", 6) != 0) throw std::runtime_error("not an NPY file");
  const unsigned major = raw[6];
  std::size_t header_size = 0, header_at = 0;
  if (major == 1) {
    header_size = voice::tool::Le16(raw.data() + 8);
    header_at = 10;
  } else if (major == 2 || major == 3) {
    header_size = voice::tool::Le32(raw.data() + 8);
    header_at = 12;
  } else {
    throw std::runtime_error("unsupported NPY version");
  }
  if (header_at + header_size > raw.size()) throw std::runtime_error("truncated NPY header");
  const std::string header(reinterpret_cast<const char*>(raw.data() + header_at), header_size);

  const auto quoted = [&](const char* key) -> std::string {
    const std::size_t at = header.find(key);
    if (at == std::string::npos) throw std::runtime_error(std::string("NPY header lacks ") + key);
    const std::size_t first = header.find('\'', at + std::strlen(key));
    const std::size_t second = header.find('\'', first + 1);
    if (first == std::string::npos || second == std::string::npos)
      throw std::runtime_error("malformed NPY header");
    return header.substr(first + 1, second - first - 1);
  };
  const std::string descr = quoted("'descr'");
  if (header.find("'fortran_order': True") != std::string::npos)
    throw std::runtime_error("Fortran-ordered NPY is not supported");
  const std::size_t shape_at = header.find("'shape'");
  if (shape_at == std::string::npos) throw std::runtime_error("NPY header lacks 'shape'");
  const std::size_t open = header.find('(', shape_at);
  const std::size_t close = header.find(')', open);
  if (open == std::string::npos || close == std::string::npos)
    throw std::runtime_error("malformed NPY shape");

  std::vector<std::size_t> dims;
  std::size_t cursor = open + 1;
  while (cursor < close) {
    while (cursor < close && !std::isdigit(static_cast<unsigned char>(header[cursor]))) ++cursor;
    if (cursor >= close) break;
    std::size_t value = 0;
    while (cursor < close && std::isdigit(static_cast<unsigned char>(header[cursor]))) {
      value = value * 10 + static_cast<std::size_t>(header[cursor] - '0');
      ++cursor;
    }
    dims.push_back(value);
  }
  if (dims.size() == 3 && dims[0] == 1) dims.erase(dims.begin());
  if (dims.size() != 2) throw std::runtime_error("NPY must hold a 2-D [frames, vocab] matrix");

  const std::size_t element_size = descr == "<f4" ? 4 : descr == "<f8" ? 8 : 0;
  if (element_size == 0) throw std::runtime_error("NPY dtype must be '<f4' or '<f8'");
  const std::size_t payload = raw.size() - header_at - header_size;
  const std::size_t expected = dims[0] * dims[1] * element_size;
  if (payload != expected)
    throw std::runtime_error("NPY payload size does not match its shape");

  Matrix m;
  m.rows = dims[0];
  m.cols = dims[1];
  m.data.resize(m.rows * m.cols);
  const unsigned char* src = raw.data() + header_at + header_size;
  if (element_size == 4) {
    std::memcpy(m.data.data(), src, expected);
  } else {
    const double* d = reinterpret_cast<const double*>(src);
    for (std::size_t i = 0; i < m.data.size(); ++i) m.data[i] = static_cast<float>(d[i]);
  }
  return m;
}

// The exported logit length is the fbank frame count after the model's two
// stride-2 subsampling stages. Verified on 30/30 Dev utterances; this is a
// consistency check against the model, not a derivation of its graph.
std::size_t ExpectedLogitFrames(std::size_t fbank_frames) {
  if (fbank_frames < 4) return 0;
  const std::size_t stage1 = (fbank_frames - 1) / 2;
  return stage1 < 2 ? 0 : (stage1 - 1) / 2;
}

struct FbankRun {
  std::size_t frames = 0;
  std::size_t samples = 0;
  double nanoseconds = 0.0;
};

// Streams the whole clip through the bounded bridge with a one-frame output
// slot, exactly as the soak audit does.
FbankRun RunFbank(const std::vector<float>& pcm, std::size_t chunk, std::size_t capacity,
                  std::vector<double>* sink) {
  voice::BoundedFbankBridge bridge;
  std::vector<voice::StreamingFbank::Feature> out(capacity);
  FbankRun run;
  const auto started = std::chrono::steady_clock::now();
  std::size_t offset = 0;
  while (offset < pcm.size()) {
    const std::size_t offer = std::min(chunk, pcm.size() - offset);
    const auto step = bridge.Process(pcm.data() + offset, offer, out.data(), out.size());
    if (step.samples_consumed == 0 && step.frames_written == 0)
      throw std::logic_error("bridge made no progress");
    offset += step.samples_consumed;
    run.frames += step.frames_written;
    for (std::size_t i = 0; i < step.frames_written; ++i) {
      if (sink) sink->insert(sink->end(), out[i].begin(), out[i].end());
    }
  }
  run.nanoseconds =
      static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::steady_clock::now() - started)
                              .count());
  run.samples = bridge.SamplesConsumedTotal();
  return run;
}

std::string JsonEscape(const std::string& text) {
  std::string out;
  out.reserve(text.size() + 8);
  for (unsigned char c : text) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  return out;
}

double Median(std::vector<double> values) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const std::size_t mid = values.size() / 2;
  return values.size() % 2 ? values[mid] : 0.5 * (values[mid - 1] + values[mid]);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string wav_path, npy_path, tokens_path, json_path;
    std::size_t chunk = 320;
    std::size_t capacity = 1;
    int blank = 0;
    int repeats = 10;
    // Matches the Python reference study (byte_errors="replace"): a greedy CTC
    // tail can end mid-character, and the count is reported rather than hidden.
    voice::Utf8Policy byte_policy = voice::Utf8Policy::kReplace;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      const auto value = [&](const char* name) -> std::string {
        if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + name);
        return argv[++i];
      };
      if (arg == "--wav") wav_path = value("--wav");
      else if (arg == "--posteriors") npy_path = value("--posteriors");
      else if (arg == "--tokens") tokens_path = value("--tokens");
      else if (arg == "--json") json_path = value("--json");
      else if (arg == "--chunk") chunk = std::stoull(value("--chunk"));
      else if (arg == "--capacity") capacity = std::stoull(value("--capacity"));
      else if (arg == "--blank") blank = std::stoi(value("--blank"));
      else if (arg == "--repeats") repeats = std::stoi(value("--repeats"));
      else if (arg == "--byte-errors") {
        const std::string mode = value("--byte-errors");
        if (mode == "replace") byte_policy = voice::Utf8Policy::kReplace;
        else if (mode == "strict") byte_policy = voice::Utf8Policy::kStrict;
        else throw std::invalid_argument("--byte-errors must be 'replace' or 'strict'");
      }
      else if (arg == "--help") {
        std::cout << "usage: fbank_ctc_pipeline --wav in.wav --posteriors logits.npy "
                     "--tokens tokens.txt\n"
                     "       [--blank 0] [--chunk 320] [--capacity 1] [--repeats 10] "
                     "[--json out.json]\n";
        return 0;
      } else {
        std::cerr << "unknown argument: " << arg << "\n";
        return 2;
      }
    }
    if (wav_path.empty() || npy_path.empty() || tokens_path.empty())
      throw std::invalid_argument("--wav, --posteriors and --tokens are required");
    if (chunk == 0 || capacity == 0) throw std::invalid_argument("chunk and capacity must be >= 1");
    if (repeats < 1) throw std::invalid_argument("repeats must be >= 1");

    const voice::tool::WavClip clip = voice::tool::ReadWav(wav_path);
    const Matrix logits = ReadNpy(npy_path);
    const voice::ByteBpeDetokenizer detok = voice::ByteBpeDetokenizer::Load(tokens_path);

    // Correctness pass: keep the feature values so the frontend really ran.
    std::vector<double> features;
    features.reserve(clip.pcm.size() / 160 * 80);
    const FbankRun first = RunFbank(clip.pcm, chunk, capacity, &features);
    if (first.samples != clip.pcm.size())
      throw std::logic_error("frontend did not consume the whole clip");

    const std::size_t predicted = ExpectedLogitFrames(first.frames);
    if (logits.rows != predicted) {
      throw std::runtime_error(
          "frontend/model frame mismatch: " + std::to_string(first.frames) + " fbank frames "
          "predict " + std::to_string(predicted) + " logit frames, dump has " +
          std::to_string(logits.rows));
    }

    std::vector<int> ids(first.frames);
    voice::GreedyDecodeStats stats;
    const std::size_t token_count = voice::GreedyCollapse(
        logits.data.data(), logits.rows, logits.cols, blank, ids.data(), ids.size(), &stats);
    const auto detok_started = std::chrono::steady_clock::now();
    std::size_t replacement_chars = 0;
    const std::string text = detok.Decode(ids.data(), token_count, byte_policy, &replacement_chars);
    const double detok_ns = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                    std::chrono::steady_clock::now() - detok_started)
                                                    .count());

    // Timing pass: median of `repeats` runs, no feature retention, so the
    // numbers are not inflated by the verification buffer.
    std::vector<double> fbank_ns, greedy_ns;
    for (int r = 0; r < repeats; ++r) {
      const FbankRun run = RunFbank(clip.pcm, chunk, capacity, nullptr);
      fbank_ns.push_back(run.nanoseconds);
      const auto t0 = std::chrono::steady_clock::now();
      voice::GreedyCollapse(logits.data.data(), logits.rows, logits.cols, blank, ids.data(),
                            ids.size(), nullptr);
      greedy_ns.push_back(static_cast<double>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0)
              .count()));
    }
    const double fbank_median = Median(fbank_ns);
    const double greedy_median = Median(greedy_ns);
    const double audio_seconds =
        static_cast<double>(clip.pcm.size()) / static_cast<double>(voice::StreamingFbank::kSampleRate);

    std::cout << text << "\n";
    std::cerr << std::fixed << std::setprecision(3);
    std::cerr << "frames: " << first.frames << " fbank -> " << logits.rows << " logit (predicted "
              << predicted << "), tokens " << token_count << " (blank frames "
              << stats.blank_frames << ", collapsed repeats " << stats.collapsed_repeats << ")\n";
    std::cerr << "fbank: " << fbank_median / 1000.0 << " us for " << audio_seconds << " s audio ("
              << fbank_median / 1000.0 / static_cast<double>(first.frames) << " us/frame, "
              << audio_seconds * 1e9 / fbank_median << "x realtime)\n";
    std::cerr << "decode: " << greedy_median / 1000.0 << " us for " << logits.rows << "x"
              << logits.cols << " argmax (" << greedy_median / 1000.0 / static_cast<double>(logits.rows)
              << " us/logit frame)\n";
    std::cerr << "detok: " << detok_ns / 1000.0 << " us; total "
              << (fbank_median + greedy_median + detok_ns) / 1000.0 << " us"
              << "; utf8 replacements " << replacement_chars << "\n";

    if (!json_path.empty()) {
      std::ofstream out(json_path);
      if (!out) throw std::runtime_error("cannot write " + json_path);
      out << std::setprecision(6);
      out << "{\n";
      out << "  \"status\": \"measured_cpp_pipeline\",\n";
      out << "  \"wav\": \"" << JsonEscape(wav_path) << "\",\n";
      out << "  \"posteriors\": \"" << JsonEscape(npy_path) << "\",\n";
      out << "  \"text\": \"" << JsonEscape(text) << "\",\n";
      out << "  \"config\": {\"chunk_samples\": " << chunk << ", \"output_capacity\": " << capacity
          << ", \"blank\": " << blank << ", \"repeats\": " << repeats << "},\n";
      out << "  \"fbank_frames\": " << first.frames << ",\n";
      out << "  \"logit_frames\": " << logits.rows << ",\n";
      out << "  \"logit_vocab\": " << logits.cols << ",\n";
      out << "  \"predicted_logit_frames\": " << predicted << ",\n";
      out << "  \"tokens\": " << token_count << ",\n";
      out << "  \"blank_frames\": " << stats.blank_frames << ",\n";
      out << "  \"collapsed_repeats\": " << stats.collapsed_repeats << ",\n";
      out << "  \"audio_seconds\": " << audio_seconds << ",\n";
      out << "  \"fbank_median_us\": " << fbank_median / 1000.0 << ",\n";
      out << "  \"fbank_us_per_frame\": " << fbank_median / 1000.0 / static_cast<double>(first.frames)
          << ",\n";
      out << "  \"fbank_realtime_factor\": " << audio_seconds * 1e9 / fbank_median << ",\n";
      out << "  \"greedy_median_us\": " << greedy_median / 1000.0 << ",\n";
      out << "  \"greedy_us_per_logit_frame\": "
          << greedy_median / 1000.0 / static_cast<double>(logits.rows) << ",\n";
      out << "  \"detok_us\": " << detok_ns / 1000.0 << ",\n";
      out << "  \"utf8_replacement_chars\": " << replacement_chars << ",\n";
      out << "  \"total_median_us\": " << (fbank_median + greedy_median + detok_ns) / 1000.0 << "\n";
      out << "}\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "fbank_ctc_pipeline error: " << e.what() << "\n";
    return 1;
  }
}
