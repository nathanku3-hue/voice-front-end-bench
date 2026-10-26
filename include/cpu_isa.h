#pragma once

namespace voice {

// Instruction-set level the frontend will actually use on this host.
//
// The hot loops exist twice: a baseline path that any x86-64 can run, and an
// AVX2+FMA path. The choice is made once, at construction, from CPUID, and the
// result is reported in every timing artifact so a number can never be
// attributed to an ISA that did not run.
enum class Isa {
  kScalar = 0,  // SSE2 only: the x86-64 floor
  kAvx2Fma = 1, // 8-wide float / 4-wide double with fused multiply-add
};

// Cached CPUID probe (leaf 1 ECX for AVX/FMA/OSXSAVE, leaf 7 EBX for AVX2).
// Non-x86 builds return kScalar.
Isa DetectIsa() noexcept;

const char* IsaName(Isa isa) noexcept;

}  // namespace voice
