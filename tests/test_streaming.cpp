#include "streaming_fbank.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#define CHECK(condition) do { if (!(condition)) throw std::runtime_error("CHECK FAILED: " #condition); } while (0)

using Extractor = voice::StreamingFbank;
using Matrix = std::vector<Extractor::Feature>;

static Matrix Extract(const std::vector<float>& pcm, const std::vector<std::size_t>& chunks) {
  Extractor f;
  Matrix matrix;
  std::size_t i = 0, c = 0;
  while (i < pcm.size()) {
    const auto n = std::min(chunks[c % chunks.size()], pcm.size() - i);
    CHECK(n != 0);
    f.AcceptWaveform(pcm.data() + i, n);
    i += n;
    ++c;
    while (f.NumFramesReady()) matrix.push_back(f.GetFrame());
  }
  CHECK(f.NumFramesReady() == 0);
  CHECK(f.FramesProduced() == matrix.size());
  return matrix;
}

static void Same(const Matrix& expected, const Matrix& actual) {
  CHECK(expected.size() == actual.size());
  for (std::size_t i = 0; i < expected.size(); ++i)
    for (std::size_t b = 0; b < Extractor::kMelBins; ++b) {
      CHECK(std::isfinite(actual[i][b]));
      CHECK(std::abs(expected[i][b] - actual[i][b]) <= 1e-12);
    }
}

int main() {
  try {
    constexpr double pi = 3.141592653589793;
    for (std::size_t length : {0U, 1U, 399U, 400U, 559U, 560U, 561U, 15999U, 16000U, 32017U}) {
      std::vector<float> pcm(length);
      std::mt19937 rng(12345);
      std::normal_distribution<float> noise(0.0f, 0.07f);
      for (std::size_t i = 0; i < length; ++i) {
        pcm[i] = static_cast<float>(0.6 * std::sin(2.0 * pi * 500 * i / 16000)) + noise(rng);
      }
      const Matrix reference = Extract(pcm, {std::max<std::size_t>(1, length)});
      std::vector<std::size_t> random_chunks;
      std::uniform_int_distribution<int> block_size(1, 1024);
      for (int iteration = 0; iteration < 43; ++iteration) random_chunks.push_back(static_cast<std::size_t>(block_size(rng)));
      const std::size_t expected = length < 400 ? 0 : 1 + (length - 400) / 160;
      CHECK(reference.size() == expected);
      for (const auto& chunks : std::vector<std::vector<std::size_t>>{
             {1}, {128}, {160}, {240}, {300}, {1024}, {13, 1, 311, 77, 409, 4, 19}, random_chunks}) {
        Same(reference, Extract(pcm, chunks));
      }
    }
    Extractor f;
    CHECK(f.NumFramesReady() == 0);
    f.AcceptWaveform(nullptr, 0);
    bool threw = false;
    try { f.GetFrame(); } catch (const std::out_of_range&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { f.AcceptWaveform(nullptr, 1); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
    std::vector<float> silence(400, 0.0f);
    f.AcceptWaveform(silence.data(), silence.size());
    CHECK(f.NumFramesReady() == 1);
    const auto feature = f.GetFrame();
    for (auto v : feature) CHECK(std::isfinite(v));
    f.Reset();
    CHECK(f.NumFramesReady() == 0 && f.FramesProduced() == 0);

    // Debug tracing must not change any feature values or frame ordering.
    Extractor traced, plain;
    traced.CaptureStageTraceAt(1);
    CHECK(!traced.HasStageTrace());
    bool no_trace = false;
    try { traced.GetStageTrace(); } catch (const std::out_of_range&) { no_trace = true; }
    CHECK(no_trace);
    std::vector<float> tracing_pcm(1000);
    for (std::size_t n = 0; n < tracing_pcm.size(); ++n)
      tracing_pcm[n] = static_cast<float>(900 * std::sin(2.0 * pi * 1300 * n / 16000));
    traced.AcceptWaveform(tracing_pcm.data(), tracing_pcm.size());
    plain.AcceptWaveform(tracing_pcm.data(), tracing_pcm.size());
    CHECK(traced.HasStageTrace());
    CHECK(traced.NumFramesReady() == plain.NumFramesReady());
    const auto& trace = traced.GetStageTrace();
    for (std::size_t frame = 0; frame < plain.FramesProduced(); ++frame) {
      const auto expected = plain.GetFrame();
      const auto observed = traced.GetFrame();
      for (std::size_t b = 0; b < observed.size(); ++b) {
        CHECK(observed[b] == expected[b]);
        if (frame == 1) CHECK(observed[b] == trace.log_fbank[b]);
      }
    }
    for (double e : trace.mel_energy) CHECK(e >= 0 && std::isfinite(e));
    for (double x : trace.windowed_padded) CHECK(std::isfinite(x));
    traced.Reset();
    CHECK(!traced.HasStageTrace());
    std::cout << "PASS: edge-length framing, 8 chunk patterns, frame count, trace isolation, reset, silence\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
