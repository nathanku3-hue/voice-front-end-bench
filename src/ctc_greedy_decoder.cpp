#include "ctc_greedy_decoder.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace voice {
namespace {

constexpr std::size_t kByteAlphabetCandidateCount = 161;

#include "byte_alphabet_table.inc"

// Appends the code points of `utf8` to `out`, mapping the SentencePiece word
// boundary marker to a plain space. Throws on malformed input.
void AppendUtf8(const std::string& utf8, std::u32string& out) {
  std::size_t i = 0;
  while (i < utf8.size()) {
    const unsigned char lead = static_cast<unsigned char>(utf8[i]);
    char32_t cp = 0;
    std::size_t extra = 0;
    if (lead < 0x80) {
      cp = lead;
    } else if ((lead & 0xE0) == 0xC0) {
      cp = lead & 0x1F;
      extra = 1;
    } else if ((lead & 0xF0) == 0xE0) {
      cp = lead & 0x0F;
      extra = 2;
    } else if ((lead & 0xF8) == 0xF0) {
      cp = lead & 0x07;
      extra = 3;
    } else {
      throw std::runtime_error("token table is not valid UTF-8");
    }
    if (extra != 0 && i + extra >= utf8.size())
      throw std::runtime_error("truncated UTF-8 sequence in token table");
    for (std::size_t k = 1; k <= extra; ++k) {
      const unsigned char cont = static_cast<unsigned char>(utf8[i + k]);
      if ((cont & 0xC0) != 0x80) throw std::runtime_error("invalid UTF-8 continuation byte");
      cp = (cp << 6) | (cont & 0x3F);
    }
    if (cp == 0x2581) cp = 0x20;  // SentencePiece word boundary -> space
    out.push_back(cp);
    i += extra + 1;
  }
}

// Decodes UTF-8 bytes with Python's `errors="replace"` semantics: one U+FFFD
// per maximal ill-formed subpart, resuming at the first byte that could not
// continue the sequence. This is CPython's utf-8 decoder behaviour, which is
// what the Python reference path used, so the two produce the same text.
std::string DecodeUtf8Bytes(const std::string& bytes, Utf8Policy policy,
                            std::size_t* replacement_chars) {
  std::string out;
  out.reserve(bytes.size());
  std::size_t replacements = 0;
  std::size_t i = 0;
  while (i < bytes.size()) {
    const unsigned char lead = static_cast<unsigned char>(bytes[i]);
    if (lead < 0x80) {
      out.push_back(static_cast<char>(lead));
      ++i;
      continue;
    }
    std::size_t extra = 0;
    unsigned char second_low = 0x80, second_high = 0xBF;
    if (lead >= 0xC2 && lead <= 0xDF) {
      extra = 1;
    } else if (lead >= 0xE0 && lead <= 0xEF) {
      extra = 2;
      if (lead == 0xE0) second_low = 0xA0;        // reject overlong 3-byte forms
      else if (lead == 0xED) second_high = 0x9F;  // reject UTF-16 surrogates
    } else if (lead >= 0xF0 && lead <= 0xF4) {
      extra = 3;
      if (lead == 0xF0) second_low = 0x90;        // reject overlong 4-byte forms
      else if (lead == 0xF4) second_high = 0x8F;  // reject > U+10FFFF
    } else {
      // 0x80..0xC1 and 0xF5..0xFF can never start a sequence.
      if (policy == Utf8Policy::kStrict)
        throw std::runtime_error("detokenised bytes are not valid UTF-8");
      out.append("\xEF\xBF\xBD");
      ++replacements;
      ++i;
      continue;
    }
    std::size_t j = i + 1;
    bool ok = true;
    for (std::size_t k = 0; k < extra; ++k, ++j) {
      if (j >= bytes.size()) {
        ok = false;  // truncated at end of input
        break;
      }
      const unsigned char c = static_cast<unsigned char>(bytes[j]);
      const unsigned char low = k == 0 ? second_low : 0x80;
      const unsigned char high = k == 0 ? second_high : 0xBF;
      if (c < low || c > high) {
        ok = false;
        break;
      }
    }
    if (!ok) {
      if (policy == Utf8Policy::kStrict)
        throw std::runtime_error("detokenised bytes are not valid UTF-8");
      out.append("\xEF\xBF\xBD");
      ++replacements;
      i = j;  // resume at the byte that broke the sequence
      continue;
    }
    out.append(bytes, i, extra + 1);
    i = j;
  }
  if (replacement_chars) *replacement_chars = replacements;
  return out;
}

}  // namespace

bool IsValidUtf8(const std::string& bytes) noexcept {
  std::size_t i = 0;
  while (i < bytes.size()) {
    const unsigned char lead = static_cast<unsigned char>(bytes[i]);
    std::size_t extra = 0;
    char32_t cp = 0;
    if (lead < 0x80) {
      ++i;
      continue;
    }
    if ((lead & 0xE0) == 0xC0) {
      cp = lead & 0x1F;
      extra = 1;
      if (cp < 2) return false;  // overlong
    } else if ((lead & 0xF0) == 0xE0) {
      cp = lead & 0x0F;
      extra = 2;
    } else if ((lead & 0xF8) == 0xF0) {
      cp = lead & 0x07;
      extra = 3;
    } else {
      return false;
    }
    if (extra != 0 && i + extra >= bytes.size()) return false;
    for (std::size_t k = 1; k <= extra; ++k) {
      const unsigned char cont = static_cast<unsigned char>(bytes[i + k]);
      if ((cont & 0xC0) != 0x80) return false;
      cp = (cp << 6) | (cont & 0x3F);
    }
    if (extra == 2 && cp < 0x800) return false;
    if (extra == 3 && cp < 0x10000) return false;
    if (cp > 0x10FFFF) return false;
    if (cp >= 0xD800 && cp <= 0xDFFF) return false;
    i += extra + 1;
  }
  return true;
}

std::size_t GreedyCollapse(const float* log_probs, std::size_t frames, std::size_t vocab,
                           int blank, int* out_tokens, std::size_t out_capacity,
                           GreedyDecodeStats* stats) {
  if (vocab == 0) throw std::invalid_argument("vocabulary size must be >= 1");
  if (frames != 0 && log_probs == nullptr)
    throw std::invalid_argument("frames require a non-null log-probability buffer");
  if (blank < 0 || static_cast<std::size_t>(blank) >= vocab)
    throw std::invalid_argument("blank index outside the vocabulary");
  if (frames > out_capacity && out_tokens == nullptr)
    throw std::invalid_argument("token buffer is required when frames are present");

  std::size_t emitted = 0;
  int previous = -1;  // no previous frame; also never equals a valid id
  GreedyDecodeStats local;
  for (std::size_t t = 0; t < frames; ++t) {
    const float* row = log_probs + t * vocab;
    // Two passes, not one. A single pass has to carry the argmax index through
    // every comparison, which makes it a serial dependency chain the compiler
    // cannot vectorise; the max reduction alone is a clean pattern, and the
    // second pass stops at the first element equal to the max -- which is
    // exactly numpy.argmax's tie rule (lowest index).
    //
    // Four independent accumulators because clang will not turn `if (a > b)`
    // into a vector max without fast-math (NaN semantics), so the next best
    // thing is to give the FP pipeline four independent chains to overlap.
    float m0 = row[0], m1 = row[0], m2 = row[0], m3 = row[0];
    std::size_t v = 1;
    for (; v + 4 <= vocab; v += 4) {
      if (row[v] > m0) m0 = row[v];
      if (row[v + 1] > m1) m1 = row[v + 1];
      if (row[v + 2] > m2) m2 = row[v + 2];
      if (row[v + 3] > m3) m3 = row[v + 3];
    }
    for (; v < vocab; ++v) {
      if (row[v] > m0) m0 = row[v];
    }
    const float best_value = std::max(std::max(m0, m1), std::max(m2, m3));
    std::size_t best = 0;
    while (row[best] != best_value) ++best;  // the max is attained, so this ends
    const int id = static_cast<int>(best);
    ++local.frames;
    if (id == blank) {
      ++local.blank_frames;
      previous = id;
      continue;
    }
    if (id == previous) {
      ++local.collapsed_repeats;
      previous = id;
      continue;
    }
    if (emitted == out_capacity) throw std::length_error("token buffer too small");
    out_tokens[emitted++] = id;
    previous = id;
  }
  if (stats) *stats = local;
  return emitted;
}

ByteBpeDetokenizer ByteBpeDetokenizer::Load(const std::string& tokens_path) {
  std::ifstream in(tokens_path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open token table: " + tokens_path);
  std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (text.size() >= 3 && static_cast<unsigned char>(text[0]) == 0xEF &&
      static_cast<unsigned char>(text[1]) == 0xBB && static_cast<unsigned char>(text[2]) == 0xBF) {
    text.erase(0, 3);  // UTF-8 BOM, matching Python's utf-8-sig
  }

  std::vector<std::pair<std::string, std::size_t>> rows;
  std::size_t start = 0;
  while (start <= text.size()) {
    std::size_t end = text.find('\n', start);
    if (end == std::string::npos) end = text.size();
    std::string line = text.substr(start, end - start);
    start = end + 1;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) {
      if (end == text.size()) break;
      throw std::runtime_error("blank line in token table");
    }
    const std::size_t split = line.find_last_of(" \t");
    if (split == std::string::npos || split + 1 >= line.size())
      throw std::runtime_error("token table line has no id: " + line);
    const std::string digits = line.substr(split + 1);
    if (!std::all_of(digits.begin(), digits.end(),
                     [](unsigned char c) { return std::isdigit(c) != 0; })) {
      throw std::runtime_error("token table line has a non-numeric id: " + line);
    }
    const std::size_t id = std::stoull(digits);
    if (id > 1000000) throw std::runtime_error("implausible token id: " + digits);
    rows.emplace_back(line.substr(0, split), id);
    if (end == text.size()) break;
  }
  if (rows.empty()) throw std::runtime_error("empty token table");

  ByteBpeDetokenizer detok;
  detok.pieces_.assign(rows.size(), std::string());
  std::vector<bool> seen(rows.size(), false);
  for (const auto& row : rows) {
    if (row.second >= rows.size()) throw std::runtime_error("token id out of range");
    if (seen[row.second]) throw std::runtime_error("duplicate token id");
    seen[row.second] = true;
    detok.pieces_[row.second] = row.first;
  }
  if (std::find(seen.begin(), seen.end(), false) != seen.end())
    throw std::runtime_error("token ids must be contiguous 0..V-1");

  detok.cp_to_byte_.fill(-1);
  for (std::size_t i = 0; i < kByteAlphabetCandidateCount; ++i) {
    const std::size_t byte = i < 32 ? i : i + 95;
    detok.cp_to_byte_[kByteAlphabetCandidates[i]] = static_cast<std::int16_t>(byte);
  }
  return detok;
}

const std::string& ByteBpeDetokenizer::Piece(std::size_t id) const {
  if (id >= pieces_.size()) throw std::out_of_range("token id outside the table");
  return pieces_[id];
}

std::string ByteBpeDetokenizer::Decode(const int* ids, std::size_t count,
                                      Utf8Policy policy,
                                      std::size_t* replacement_chars) const {
  if (count != 0 && ids == nullptr) throw std::invalid_argument("ids require a buffer");
  if (replacement_chars) *replacement_chars = 0;
  std::u32string text;
  text.reserve(count * 2 + 1);
  for (std::size_t i = 0; i < count; ++i) {
    const int id = ids[i];
    if (id < 0 || static_cast<std::size_t>(id) >= pieces_.size())
      throw std::out_of_range("token id outside the table");
    AppendUtf8(pieces_[static_cast<std::size_t>(id)], text);
  }
  // SentencePiece strips exactly one leading space after the boundary rewrite.
  if (!text.empty() && text[0] == U' ') text.erase(0, 1);

  std::string bytes;
  bytes.reserve(text.size());
  for (char32_t cp : text) {
    int byte = -1;
    if (cp >= 0x20 && cp <= 0x7E) {
      byte = static_cast<int>(cp);
    } else if (cp < cp_to_byte_.size()) {
      byte = cp_to_byte_[cp];
    }
    if (byte < 0) throw std::runtime_error("token piece contains a character outside the byte alphabet");
    bytes.push_back(static_cast<char>(byte));
  }
  return DecodeUtf8Bytes(bytes, policy, replacement_chars);
}

}  // namespace voice
