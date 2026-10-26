#include "bounded_fbank_bridge.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

// Bounded-memory PCM-to-Fbank CLI for the *existing* Round-1 feature contract:
// mono float32 LE @ 16 kHz, snip_edges=true, mel_high=8000 Hz. A single output
// frame is supplied at a time: the bridge never accumulates a growing backlog.
// The full ONNX CTC model is a separate offline consumer of the resulting CSV.
int main(int argc, char** argv) {
  std::string staged_path;
  try {
    if (argc != 4) {
      std::cerr << "Usage: fbank_bounded_dump input_f32_le.raw output.csv chunk_samples\n";
      return 2;
    }
    const auto chunk = std::stoull(argv[3]);
    if (chunk == 0 || chunk > 1048576)
      throw std::invalid_argument("chunk_samples must be 1..1048576");

    std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open input PCM");
    const auto bytes = input.tellg();
    if (bytes < 0 || bytes % sizeof(float) != 0)
      throw std::runtime_error("invalid input: length is not a multiple of float32");
    input.seekg(0);
    const std::filesystem::path final_path(argv[2]);
    const std::filesystem::path temp_path(final_path.string() + ".part");
    if (std::filesystem::exists(final_path) || std::filesystem::exists(temp_path))
      throw std::runtime_error("refusing to overwrite existing output or partial file");
    staged_path = temp_path.string();
    std::ofstream output(staged_path);
    if (!output) throw std::runtime_error("cannot open temporary CSV");
    output << std::setprecision(17);

    std::vector<float> block(static_cast<std::size_t>(chunk));
    voice::BoundedFbankBridge frontend;
    voice::BoundedFbankBridge::Feature feature[1];
    std::size_t frames = 0;

    while (input) {
      input.read(reinterpret_cast<char*>(block.data()),
                 static_cast<std::streamsize>(block.size() * sizeof(float)));
      const auto got = input.gcount();
      if (got < 0 || got % sizeof(float) != 0)
        throw std::runtime_error("partial float32 sample read");
      const auto count = static_cast<std::size_t>(got) / sizeof(float);
      std::size_t offset = 0;
      while (offset < count) {
        const auto step = frontend.Process(
            block.data() + offset, count - offset, feature, 1);
        if (step.samples_consumed == 0)
          throw std::logic_error("frontend cannot make progress");
        offset += step.samples_consumed;
        if (step.frames_written != 0) {
          for (std::size_t i = 0; i < feature[0].size(); ++i) {
            if (i) output << ',';
            output << feature[0][i];
          }
          output << '\n';
          ++frames;
        }
      }
    }
    if (input.bad() || !output) throw std::runtime_error("PCM/CSV I/O failed");
    output.flush();
    if (!output) throw std::runtime_error("failed flushing CSV");
    output.close();
    if (!output) throw std::runtime_error("failed closing CSV");
    std::filesystem::rename(staged_path, final_path);
    staged_path.clear();
    std::cerr << "frames=" << frames
              << " samples=" << frontend.SamplesConsumedTotal() << '\n';
    return 0;
  } catch (const std::exception& exc) {
    if (!staged_path.empty()) {
      std::error_code ignored;
      std::filesystem::remove(staged_path, ignored);
    }
    std::cerr << "fbank_bounded_dump error: " << exc.what() << '\n';
    return 1;
  }
}
