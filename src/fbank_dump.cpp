#include "streaming_fbank.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Numpy .npy v1.0, little-endian float64 vector. Diagnostics only; no Python
// dependency is introduced into the normal C++ feature pipeline.
void WriteNpy(const std::filesystem::path& path, const double* data, std::size_t count) {
  const std::uint16_t marker = 1;
  if (sizeof(double) != 8 || !std::numeric_limits<double>::is_iec559 ||
      *reinterpret_cast<const unsigned char*>(&marker) != 1) {
    throw std::runtime_error("stage dump requires IEEE754 little-endian float64");
  }
  std::string header = "{'descr': '<f8', 'fortran_order': False, 'shape': (" +
                       std::to_string(count) + ",), }";
  const auto padding = (16 - ((10 + header.size() + 1) % 16)) % 16;
  header.append(padding, ' ');
  header.push_back('\n');
  if (header.size() > std::numeric_limits<std::uint16_t>::max())
    throw std::runtime_error("NPY header too large");
  const auto header_size = static_cast<std::uint16_t>(header.size());
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("Cannot open stage dump: " + path.string());
  const char magic[8] = {static_cast<char>(0x93), 'N', 'U', 'M', 'P', 'Y', 1, 0};
  out.write(magic, sizeof(magic));
  out.put(static_cast<char>(header_size & 0xff));
  out.put(static_cast<char>(header_size >> 8));
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count * sizeof(double)));
  if (!out) throw std::runtime_error("Cannot write stage dump: " + path.string());
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 4 && argc != 6 && argc != 8) {
      std::cerr << "Usage: fbank_dump input_float32_le.raw output.csv chunk_samples "
                   "[--dump-stage output_dir [--frame index]]\n";
      return 2;
    }
    const int chunk = std::stoi(argv[3]);
    if (chunk < 1) throw std::invalid_argument("chunk_samples must be >= 1");
    const bool dump_stage = argc >= 6;
    if (dump_stage && std::string(argv[4]) != "--dump-stage")
      throw std::invalid_argument("Expected --dump-stage output_dir");
    if (argc == 8 && std::string(argv[6]) != "--frame")
      throw std::invalid_argument("Expected --frame index");
    const std::size_t trace_frame = argc == 8 ? std::stoull(argv[7]) : 0;
    std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open input file");
    const auto bytes = input.tellg();
    if (bytes < 0 || bytes % sizeof(float) != 0) throw std::runtime_error("invalid f32 raw byte length");
    input.seekg(0);
    std::vector<float> pcm(static_cast<std::size_t>(bytes) / sizeof(float));
    if (!pcm.empty()) input.read(reinterpret_cast<char*>(pcm.data()), bytes);
    if (!input && !pcm.empty()) throw std::runtime_error("failed reading input");

    std::ofstream output(argv[2]);
    if (!output) throw std::runtime_error("cannot open CSV output");
    output << std::setprecision(17);
    voice::StreamingFbank frontend;
    if (dump_stage) frontend.CaptureStageTraceAt(trace_frame);
    for (std::size_t offset = 0; offset < pcm.size();) {
      const auto n = std::min(static_cast<std::size_t>(chunk), pcm.size() - offset);
      frontend.AcceptWaveform(pcm.data() + offset, n);
      offset += n;
      while (frontend.NumFramesReady()) {
        const auto feat = frontend.GetFrame();
        for (std::size_t j = 0; j < feat.size(); ++j) {
          if (j) output << ',';
          output << feat[j];
        }
        output << '\n';
      }
    }
    if (!output) throw std::runtime_error("failed writing CSV");
    if (dump_stage) {
      if (!frontend.HasStageTrace()) throw std::out_of_range("Requested frame has not been produced");
      const auto& trace = frontend.GetStageTrace();
      const std::filesystem::path directory(argv[5]);
      std::filesystem::create_directories(directory);
      WriteNpy(directory / "after_preemphasis.npy", trace.after_preemphasis.data(), trace.after_preemphasis.size());
      WriteNpy(directory / "windowed_padded.npy", trace.windowed_padded.data(), trace.windowed_padded.size());
      WriteNpy(directory / "power_spectrum.npy", trace.power_spectrum.data(), trace.power_spectrum.size());
      WriteNpy(directory / "mel_energy.npy", trace.mel_energy.data(), trace.mel_energy.size());
      WriteNpy(directory / "log_fbank.npy", trace.log_fbank.data(), trace.log_fbank.size());
      std::cerr << "captured stage trace: frame=" << trace_frame << " at " << directory.string() << '\n';
    }
    std::cerr << "frames=" << frontend.FramesProduced() << '\n';
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "fbank_dump error: " << e.what() << '\n';
    return 1;
  }
}
