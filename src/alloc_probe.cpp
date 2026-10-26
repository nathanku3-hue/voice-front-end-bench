#include "alloc_probe.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>

// ---------------------------------------------------------------------------
// Replaceable global allocation functions.
//
// Counting at the operator new/delete boundary is exact and cheap: resident-set
// size is page-granular and can hide a leak inside already-touched pages, and
// Valgrind does not exist on this host while ASan has no runtime for the mingw
// target used here. The probe therefore answers one question precisely -- "did
// the hot path call the allocator?" -- and deliberately says nothing about
// out-of-bounds access or malloc-only callers.
// ---------------------------------------------------------------------------
namespace {

constexpr std::size_t kAllocHeader = 32;  // base pointer + size + alignment slack

std::atomic<std::size_t> g_live_bytes{0};
std::atomic<std::size_t> g_peak_bytes{0};
std::atomic<std::size_t> g_total_allocations{0};
std::atomic<std::size_t> g_denied_allocations{0};
std::atomic<std::size_t> g_denied_bytes{0};
std::atomic<bool> g_allowed{true};

void* ProbeAllocate(std::size_t bytes, std::size_t alignment) noexcept {
  g_total_allocations.fetch_add(1, std::memory_order_relaxed);
  if (!g_allowed.load(std::memory_order_relaxed)) {
    // Served, not failed: a denied allocation must not turn a measurement bug
    // into a std::bad_alloc thrown out of a real-time callback.
    g_denied_allocations.fetch_add(1, std::memory_order_relaxed);
    g_denied_bytes.fetch_add(bytes, std::memory_order_relaxed);
  }
  if (alignment < alignof(std::max_align_t)) alignment = alignof(std::max_align_t);
  if (bytes > std::numeric_limits<std::size_t>::max() - kAllocHeader - alignment)
    return nullptr;
  char* raw = static_cast<char*>(std::malloc(bytes + kAllocHeader + alignment));
  if (raw == nullptr) return nullptr;
  const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(raw);
  const std::uintptr_t aligned = (base + kAllocHeader + alignment - 1) & ~(alignment - 1);
  char* user = reinterpret_cast<char*>(aligned);
  void* base_ptr = raw;
  const std::size_t size = bytes;
  std::memcpy(user - sizeof(void*), &base_ptr, sizeof(void*));
  std::memcpy(user - 2 * sizeof(void*), &size, sizeof(size));

  const std::size_t live = g_live_bytes.fetch_add(bytes, std::memory_order_relaxed) + bytes;
  std::size_t peak = g_peak_bytes.load(std::memory_order_relaxed);
  while (live > peak &&
         !g_peak_bytes.compare_exchange_weak(peak, live, std::memory_order_relaxed)) {
  }
  return user;
}

void ProbeRelease(void* p) noexcept {
  if (p == nullptr) return;
  char* user = static_cast<char*>(p);
  void* base = nullptr;
  std::size_t size = 0;
  std::memcpy(&base, user - sizeof(void*), sizeof(void*));
  std::memcpy(&size, user - 2 * sizeof(void*), sizeof(size));
  g_live_bytes.fetch_sub(size, std::memory_order_relaxed);
  std::free(base);
}

}  // namespace

void* operator new(std::size_t n) {
  void* p = ProbeAllocate(n, alignof(std::max_align_t));
  if (p == nullptr) throw std::bad_alloc();
  return p;
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
  return ProbeAllocate(n, alignof(std::max_align_t));
}
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept {
  return ProbeAllocate(n, alignof(std::max_align_t));
}
void* operator new(std::size_t n, std::align_val_t a) {
  void* p = ProbeAllocate(n, static_cast<std::size_t>(a));
  if (p == nullptr) throw std::bad_alloc();
  return p;
}
void* operator new[](std::size_t n, std::align_val_t a) { return ::operator new(n, a); }
void* operator new(std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
  return ProbeAllocate(n, static_cast<std::size_t>(a));
}
void* operator new[](std::size_t n, std::align_val_t a, const std::nothrow_t&) noexcept {
  return ProbeAllocate(n, static_cast<std::size_t>(a));
}
void operator delete(void* p) noexcept { ProbeRelease(p); }
void operator delete[](void* p) noexcept { ProbeRelease(p); }
void operator delete(void* p, std::size_t) noexcept { ProbeRelease(p); }
void operator delete[](void* p, std::size_t) noexcept { ProbeRelease(p); }
void operator delete(void* p, std::align_val_t) noexcept { ProbeRelease(p); }
void operator delete[](void* p, std::align_val_t) noexcept { ProbeRelease(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { ProbeRelease(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { ProbeRelease(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { ProbeRelease(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { ProbeRelease(p); }

namespace voice {
namespace probe {

std::size_t LiveBytes() noexcept { return g_live_bytes.load(std::memory_order_relaxed); }
std::size_t PeakBytes() noexcept { return g_peak_bytes.load(std::memory_order_relaxed); }
std::size_t TotalAllocations() noexcept {
  return g_total_allocations.load(std::memory_order_relaxed);
}

void SetAllocationAllowed(bool allowed) noexcept {
  g_allowed.store(allowed, std::memory_order_relaxed);
}
bool AllocationAllowed() noexcept { return g_allowed.load(std::memory_order_relaxed); }
std::size_t DeniedAllocations() noexcept {
  return g_denied_allocations.load(std::memory_order_relaxed);
}
std::size_t DeniedBytes() noexcept { return g_denied_bytes.load(std::memory_order_relaxed); }
void ResetDenied() noexcept {
  g_denied_allocations.store(0, std::memory_order_relaxed);
  g_denied_bytes.store(0, std::memory_order_relaxed);
}

Snapshot Capture() noexcept {
  Snapshot s;
  s.live_bytes = LiveBytes();
  s.total_allocations = TotalAllocations();
  s.denied_allocations = DeniedAllocations();
  s.denied_bytes = DeniedBytes();
  return s;
}

}  // namespace probe
}  // namespace voice
