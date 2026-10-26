// Unit tests for the pure-C++ CTC greedy decoder and Byte-BPE detokeniser.
//
// Scope: the algorithm and its contract, on synthetic data. The end-to-end
// agreement with the Python reference on the 30 Dev utterances is a separate,
// stronger check -- scripts/run_pipeline_parity.py compares full transcripts
// through the real CLI, including the two utterances whose greedy tail ends
// mid-character. Keeping that out of here avoids a unit test that silently
// depends on model assets being present.

#include "ctc_greedy_decoder.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Builds a [frames][vocab] row-major log-probability matrix where row t peaks at
// argmax[t]. Non-peak entries are lower, so argmax is unambiguous unless the
// row is marked as a tie.
std::vector<float> Matrix(const std::vector<int>& argmax, std::size_t vocab,
                          const std::vector<std::size_t>& tie_rows = {}) {
  std::vector<float> data(argmax.size() * vocab, -5.0f);
  for (std::size_t t = 0; t < argmax.size(); ++t) {
    for (std::size_t v = 0; v < vocab; ++v) {
      data[t * vocab + v] = -5.0f - static_cast<float>(v) * 0.01f;
    }
    if (tie_rows.end() != std::find(tie_rows.begin(), tie_rows.end(), t)) {
      data[t * vocab + 0] = 0.0f;
      data[t * vocab + 1] = 0.0f;  // exact tie: argmax must pick index 0
    } else {
      data[t * vocab + static_cast<std::size_t>(argmax[t])] = 0.0f;
    }
  }
  return data;
}

void TestCollapseRules() {
  const std::size_t vocab = 8;
  // blank 0: [0, 5, 5, 0, 5, 6, 0] -> 5 (repeat collapsed), then 5 again after a
  // blank (a real repeat, not a stutter), then 6.
  const std::vector<int> peak = {0, 5, 5, 0, 5, 6, 0};
  const std::vector<float> logits = Matrix(peak, vocab);
  std::vector<int> ids(peak.size());
  voice::GreedyDecodeStats stats;
  const std::size_t count = voice::GreedyCollapse(logits.data(), peak.size(), vocab, 0,
                                                  ids.data(), ids.size(), &stats);
  assert(count == 3);
  assert(ids[0] == 5 && ids[1] == 5 && ids[2] == 6);
  assert(stats.frames == peak.size());
  assert(stats.blank_frames == 3);
  assert(stats.collapsed_repeats == 1);
}

void TestTieBreaksToLowestIndex() {
  const std::size_t vocab = 4;
  const std::vector<int> peak = {0};
  const std::vector<float> logits = Matrix(peak, vocab, {0});
  std::vector<int> ids(1);
  const std::size_t count = voice::GreedyCollapse(logits.data(), 1, vocab, 3, ids.data(),
                                                  ids.size(), nullptr);
  assert(count == 1);
  assert(ids[0] == 0);  // numpy.argmax semantics
}

void TestEmptyAndLimits() {
  std::vector<int> ids(2, -1);
  assert(voice::GreedyCollapse(nullptr, 0, 8, 0, ids.data(), ids.size(), nullptr) == 0);

  const std::vector<int> peak = {3, 3, 4};
  const std::vector<float> logits = Matrix(peak, 8);
  bool threw = false;
  try {
    voice::GreedyCollapse(logits.data(), peak.size(), 8, 0, ids.data(), 1, nullptr);
  } catch (const std::length_error&) {
    threw = true;
  }
  assert(threw);

  threw = false;
  try {
    voice::GreedyCollapse(logits.data(), 1, 8, 8, ids.data(), ids.size(), nullptr);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  assert(threw);  // blank outside the vocabulary
}

void TestDetokenizerRoundTrip() {
  // A miniature byte-BPE table: "▁a" and "▁b" carry the word-boundary marker,
  // "c" does not. SentencePiece turns every marker into a space and strips
  // exactly one leading space, so [0, 1] must render as "a b".
  const std::string path = "tmp_tokens_unit_test.txt";
  {
    std::ofstream out(path, std::ios::binary);
    out << "\xE2\x96\x81" "a 0\n";   // U+2581 LOWER ONE EIGHTH BLOCK
    out << "\xE2\x96\x81" "b 1\n";
    out << "c 2\n";
  }
  const voice::ByteBpeDetokenizer detok = voice::ByteBpeDetokenizer::Load(path);
  std::remove(path.c_str());

  assert(detok.Size() == 3);
  assert(detok.Piece(0) == "\xE2\x96\x81" "a");

  const int words[] = {0, 1};
  assert(detok.Decode(words, 2) == "a b");
  const int no_boundary[] = {2, 2};
  assert(detok.Decode(no_boundary, 2) == "cc");
  // One leading space only: "▁a" then "c" -> "ac", not " ac".
  const int mixed[] = {0, 2};
  assert(detok.Decode(mixed, 2) == "ac");

  std::size_t replacements = 123;
  assert(detok.Decode(words, 2, voice::Utf8Policy::kStrict, &replacements) == "a b");
  assert(replacements == 0);

  bool threw = false;
  try {
    const int bad[] = {7};
    detok.Decode(bad, 1);
  } catch (const std::out_of_range&) {
    threw = true;
  }
  assert(threw);
}

void TestUtf8ValidityHelper() {
  assert(voice::IsValidUtf8("plain ascii"));
  assert(voice::IsValidUtf8("\xE4\xB8\xAD"));          // U+4E2D
  assert(!voice::IsValidUtf8("\xE4\xB8"));             // truncated
  assert(!voice::IsValidUtf8("\xC0\x80"));             // overlong
  assert(!voice::IsValidUtf8("\xED\xA0\x80"));         // surrogate
  assert(!voice::IsValidUtf8("\xF4\x90\x80\x80"));     // > U+10FFFF
}

}  // namespace

int main() {
  TestCollapseRules();
  TestTieBreaksToLowestIndex();
  TestEmptyAndLimits();
  TestDetokenizerRoundTrip();
  TestUtf8ValidityHelper();
  std::cout << "ctc_greedy_decoder: all assertions passed\n";
  return 0;
}
