#pragma once

// Minimal strict WAV reader shared by the C++ tools. Deliberately narrow:
// 16 kHz mono uncompressed PCM16, the same contract scripts/ctc_export_logits.py
// enforces. Anything else is rejected rather than silently resampled.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace voice {
namespace tool {

struct WavClip {
  std::vector<float> pcm;
  std::uint32_t sample_rate = 0;
  std::uint16_t channels = 0;
  std::uint16_t bits = 0;
};

inline std::uint32_t Le32(const unsigned char* p) {
  return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
         (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}

inline std::uint16_t Le16(const unsigned char* p) {
  return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}

inline WavClip ReadWav(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open WAV: " + path);
  std::vector<unsigned char> raw((std::istreambuf_iterator<char>(in)),
                                 std::istreambuf_iterator<char>());
  if (raw.size() < 44) throw std::runtime_error("WAV too short");
  if (std::memcmp(raw.data(), "RIFF", 4) != 0 || std::memcmp(raw.data() + 8, "WAVE", 4) != 0)
    throw std::runtime_error("not a RIFF/WAVE file");

  WavClip clip;
  bool have_fmt = false, have_data = false;
  std::size_t offset = 12;
  while (offset + 8 <= raw.size()) {
    const char* id = reinterpret_cast<const char*>(raw.data() + offset);
    const std::uint32_t size = Le32(raw.data() + offset + 4);
    const std::size_t body = offset + 8;
    if (body + size > raw.size()) throw std::runtime_error("truncated WAV chunk");
    if (std::memcmp(id, "fmt ", 4) == 0) {
      if (size < 16) throw std::runtime_error("short fmt chunk");
      const std::uint16_t format = Le16(raw.data() + body);
      clip.channels = Le16(raw.data() + body + 2);
      clip.sample_rate = Le32(raw.data() + body + 4);
      clip.bits = Le16(raw.data() + body + 14);
      if (format != 1) throw std::runtime_error("WAV is not uncompressed PCM");
      have_fmt = true;
    } else if (std::memcmp(id, "data", 4) == 0) {
      if (size % 2 != 0) throw std::runtime_error("data chunk is not a whole number of samples");
      const auto* p = reinterpret_cast<const std::int16_t*>(raw.data() + body);
      clip.pcm.resize(size / 2);
      for (std::size_t i = 0; i < clip.pcm.size(); ++i) {
        clip.pcm[i] = static_cast<float>(p[i]) / 32768.0f;
      }
      have_data = true;
    }
    offset = body + size + (size & 1);  // chunks are word-aligned
  }
  if (!have_fmt || !have_data) throw std::runtime_error("WAV is missing fmt or data chunk");
  if (clip.channels != 1 || clip.bits != 16 || clip.sample_rate != 16000)
    throw std::runtime_error("require 16 kHz mono PCM16 WAV");
  return clip;
}

}  // namespace tool
}  // namespace voice
