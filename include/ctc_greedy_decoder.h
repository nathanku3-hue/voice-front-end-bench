#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace voice {

// CTC greedy decode: per-frame argmax, collapse consecutive repeats, drop
// blank. Ties resolve to the lowest index, matching numpy.argmax, so this is
// bit-identical to the Python reference decoder on the same logits.
//
// The caller owns the output storage; `frames` is the hard upper bound on the
// number of emitted tokens, so a frame-indexed buffer is always sufficient.
struct GreedyDecodeStats {
  std::size_t frames = 0;
  std::size_t blank_frames = 0;
  std::size_t collapsed_repeats = 0;
};

std::size_t GreedyCollapse(const float* log_probs, std::size_t frames, std::size_t vocab,
                           int blank, int* out_tokens, std::size_t out_capacity,
                           GreedyDecodeStats* stats = nullptr);

// icefall Byte-BPE detokeniser: SentencePiece pieces -> byte alphabet -> UTF-8.
//
// SentencePiece's decoder replaces U+2581 with a space and strips exactly one
// leading space; icefall then maps each remaining code point back to the byte
// it encodes. Both steps are reproduced here so the C++ text output matches the
// Python detokeniser character for character (asserted in
// tests/test_ctc_greedy_decoder.cpp against the 30 Dev utterances).
//
// Byte-error policy. A greedy CTC decode can end on a partial multi-byte
// character, so the byte stream is not guaranteed to be well-formed UTF-8.
// kReplace follows Python's `bytes.decode("utf-8", "replace")`: one U+FFFD per
// maximal ill-formed subpart, and the count is returned so the caller can
// report it instead of silently shipping corrupted text. kStrict throws.
enum class Utf8Policy { kStrict, kReplace };

class ByteBpeDetokenizer {
 public:
  // Throws std::runtime_error if the table is malformed or ids are not 0..V-1.
  static ByteBpeDetokenizer Load(const std::string& tokens_path);

  std::string Decode(const int* ids, std::size_t count,
                     Utf8Policy policy = Utf8Policy::kReplace,
                     std::size_t* replacement_chars = nullptr) const;

  std::size_t Size() const noexcept { return pieces_.size(); }
  const std::string& Piece(std::size_t id) const;

 private:
  static constexpr std::size_t kCandidateCount = 161;  // bytes 0..31 and 127..255
  static constexpr char32_t kWordBoundary = 0x2581;    // U+2581 LOWER ONE EIGHTH BLOCK

  std::vector<std::string> pieces_;
  // Code point -> byte, or -1 when the code point is outside the byte alphabet.
  std::array<std::int16_t, 0x0800> cp_to_byte_{};
};

// True when `bytes` is well-formed UTF-8 (no overlongs, no surrogates, max
// U+10FFFF). Exposed because the detokeniser's output contract is text, not
// bytes.
bool IsValidUtf8(const std::string& bytes) noexcept;

}  // namespace voice
