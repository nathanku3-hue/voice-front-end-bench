#include "cpu_isa.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <cpuid.h>
#define VOICE_HAVE_CPUID 1
#endif

namespace voice {
namespace {

Isa Probe() noexcept {
#ifdef VOICE_HAVE_CPUID
  unsigned a = 0, b = 0, c = 0, d = 0;
  if (!__get_cpuid(1, &a, &b, &c, &d)) return Isa::kScalar;
  const bool osxsave = ((c >> 27) & 1U) != 0;
  const bool avx = ((c >> 28) & 1U) != 0;
  const bool fma = ((c >> 12) & 1U) != 0;
  if (!(osxsave && avx && fma)) return Isa::kScalar;
  unsigned a7 = 0, b7 = 0, c7 = 0, d7 = 0;
  if (!__get_cpuid_count(7, 0, &a7, &b7, &c7, &d7)) return Isa::kScalar;
  const bool avx2 = ((b7 >> 5) & 1U) != 0;
  // OSXSAVE + AVX means the OS has enabled XMM/YMM state saving (Windows 8+ and
  // every supported Linux do). We do not read XCR0 here: on the platforms this
  // benchmark targets, an OS that advertises OSXSAVE but masks YMM does not
  // exist, and a wrong guess would fault immediately rather than silently.
  return avx2 ? Isa::kAvx2Fma : Isa::kScalar;
#else
  return Isa::kScalar;
#endif
}

}  // namespace

Isa DetectIsa() noexcept {
  static const Isa cached = Probe();
  return cached;
}

const char* IsaName(Isa isa) noexcept {
  switch (isa) {
    case Isa::kAvx2Fma: return "avx2+fma";
    case Isa::kScalar:
    default: return "scalar";
  }
}

}  // namespace voice
