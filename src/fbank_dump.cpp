#include "streaming_fbank.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  try {
    if (argc != 4) {
      std::cerr << "Usage: fbank_dump input_float32_le.raw output.csv chunk_samples\n";
      return 2;
    }
    const int chunk = std::stoi(argv[3]);
    if (chunk < 1) throw std::invalid_argument("chunk_samples must be >= 1");
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
    std::cerr << "frames=" << frontend.FramesProduced() << '\n';
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "fbank_dump error: " << e.what() << '\n';
    return 1;
  }
}
