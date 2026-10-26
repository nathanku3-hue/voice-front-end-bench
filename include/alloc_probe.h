#pragma once

#include <cstddef>

namespace voice {
namespace probe {

// Replaceable global operator new/delete are defined in src/alloc_probe.cpp.
// Every allocation that goes through the C++ allocation functions is counted
// exactly. Allocations that bypass them (malloc, FFTW's fftw_malloc, mmap,
// OS/Vulkan/driver buffers) are NOT visible here -- see the non-claims in
// docs/ZERO_ALLOC_AND_PIPELINE.md.

std::size_t LiveBytes() noexcept;
std::size_t PeakBytes() noexcept;
std::size_t TotalAllocations() noexcept;

// While allocations are disallowed, a request is still served (so a violation
// cannot corrupt the audit by throwing from inside a hot path) but it is
// recorded. Real-time steady state must show zero denials.
void SetAllocationAllowed(bool allowed) noexcept;
bool AllocationAllowed() noexcept;
std::size_t DeniedAllocations() noexcept;
std::size_t DeniedBytes() noexcept;
void ResetDenied() noexcept;

// Snapshot used by the audit to bracket a window without touching counters
// that the measurement itself would perturb.
struct Snapshot {
  std::size_t live_bytes = 0;
  std::size_t total_allocations = 0;
  std::size_t denied_allocations = 0;
  std::size_t denied_bytes = 0;
};

Snapshot Capture() noexcept;

}  // namespace probe
}  // namespace voice
