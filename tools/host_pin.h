#pragma once

// Host timing hygiene for the audit tools.
//
// This machine is bimodal: the same binary measures the same clip up to ~2x
// apart depending on which core the scheduler picks and what else is running.
// Pinning to one core at high priority removes most of it. The residual spread
// is still reported by every tool rather than hidden -- pinning reduces the
// noise, it does not make the numbers device latency.

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace voice {
namespace tool {

inline bool PinToOneCore() {
#if defined(_WIN32)
  const HANDLE process = GetCurrentProcess();
  const bool affinity = SetProcessAffinityMask(process, 1) != 0;
  const bool priority = SetPriorityClass(process, HIGH_PRIORITY_CLASS) != 0;
  return affinity && priority;
#else
  return false;
#endif
}

}  // namespace tool
}  // namespace voice
