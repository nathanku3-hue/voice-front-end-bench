#include "bounded_fbank_bridge.h"

#ifdef NDEBUG
#undef NDEBUG  // Keep the regression assertions live in Release builds.
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#else
#include <unistd.h>
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <limits>
#include <new>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Replaceable global allocation functions.
//
// The audit below claims "no unbounded growth". Resident-set size alone is a
// coarse, page-granular proxy: an allocator can leak inside already-touched
// pages and RSS will not move. Counting live bytes at the operator new/delete
// boundary is exact, cheap, and works on every host, including the Windows
// host used here where Valgrind does not exist and ASan is unavailable.
// ---------------------------------------------------------------------------
namespace {
constexpr std::size_t kAllocHeader = 32;  // room for base pointer + size + align
std::size_t g_heap_live = 0;
std::size_t g_heap_peak = 0;

void* ProbeAllocate(std::size_t bytes, std::size_t alignment) {
  if (alignment < alignof(std::max_align_t)) alignment = alignof(std::max_align_t);
  if (bytes > std::numeric_limits<std::size_t>::max() - kAllocHeader - alignment) return nullptr;
  char* raw = static_cast<char*>(std::malloc(bytes + kAllocHeader + alignment));
  if (raw == nullptr) return nullptr;
  const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(raw);
  std::uintptr_t aligned = (base + kAllocHeader + alignment - 1) & ~(alignment - 1);
  char* user = reinterpret_cast<char*>(aligned);
  void* base_ptr = raw;
  const std::size_t size = bytes;
  std::memcpy(user - sizeof(void*), &base_ptr, sizeof(void*));
  std::memcpy(user - 2 * sizeof(void*), &size, sizeof(size));
  g_heap_live += bytes;
  if (g_heap_live > g_heap_peak) g_heap_peak = g_heap_live;
  return user;
}

void ProbeRelease(void* p) noexcept {
  if (p == nullptr) return;
  char* user = static_cast<char*>(p);
  void* base = nullptr;
  std::size_t size = 0;
  std::memcpy(&base, user - sizeof(void*), sizeof(void*));
  std::memcpy(&size, user - 2 * sizeof(void*), sizeof(size));
  g_heap_live -= size;
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

namespace {
using Feature = voice::StreamingFbank::Feature;

// --------------------------- resident set probes ---------------------------
std::size_t ResidentSetBytes() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS_EX pmc{};
  if (!GetProcessMemoryInfo(GetCurrentProcess(),
                            reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc),
                            sizeof(pmc)))
    return 0;
  return static_cast<std::size_t>(pmc.WorkingSetSize);
#else
  std::ifstream statm("/proc/self/statm");
  std::size_t total = 0, resident = 0;
  if (!(statm >> total >> resident)) return 0;
  const long page = ::sysconf(_SC_PAGESIZE);
  return resident * static_cast<std::size_t>(page > 0 ? page : 4096);
#endif
}

std::size_t PeakResidentSetBytes() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS_EX pmc{};
  if (!GetProcessMemoryInfo(GetCurrentProcess(),
                            reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc),
                            sizeof(pmc)))
    return 0;
  return static_cast<std::size_t>(pmc.PeakWorkingSetSize);
#else
  std::ifstream status("/proc/self/status");
  std::string key;
  while (status >> key) {
    if (key == "VmHWM:") {
      std::size_t kib = 0;
      status >> kib;
      return kib * 1024;
    }
    status.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
  return 0;
#endif
}

const char* PlatformTag() {
#if defined(_WIN32)
  return "windows";
#elif defined(__linux__)
  return "linux";
#else
  return "unknown";
#endif
}

// ------------------------------ existing tests ------------------------------
std::vector<float> TestWave(std::size_t n) {
  std::vector<float> out(n);
  for (std::size_t i = 0; i < n; ++i)
    out[i] = static_cast<float>(.37 * std::sin(.013 * static_cast<double>(i)) +
                                .11 * std::cos(.037 * static_cast<double>(i)));
  return out;
}

// Same generator, evaluated lazily so an hour of audio never has to be
// materialised: the soak must not allocate the signal it is auditing.
float WaveSample(std::size_t i) {
  return static_cast<float>(.37 * std::sin(.013 * static_cast<double>(i)) +
                            .11 * std::cos(.037 * static_cast<double>(i)));
}

std::vector<Feature> Reference(const std::vector<float>& samples) {
  // This reference hands over the whole clip in one call, so the output ring
  // must be sized for every frame the call can produce -- that is the documented
  // contract for direct StreamingFbank users, and the bridge is the component
  // that removes the need for it.
  voice::StreamingFbank plain(samples.size() / voice::StreamingFbank::kFrameShift + 2);
  plain.AcceptWaveform(samples.data(), samples.size());
  std::vector<Feature> frames;
  while (plain.NumFramesReady()) frames.push_back(plain.GetFrame());
  return frames;
}

void Compare(const std::vector<Feature>& actual, const std::vector<Feature>& expected) {
  assert(actual.size() == expected.size());
  for (std::size_t i = 0; i < actual.size(); ++i)
    for (std::size_t j = 0; j < actual[i].size(); ++j)
      assert(std::abs(actual[i][j] - expected[i][j]) < 1e-12);
}

std::vector<Feature> WithBackpressure(const std::vector<float>& samples,
                                      unsigned seed, std::size_t max_frames) {
  std::mt19937 rng(seed);
  voice::BoundedFbankBridge bridge;
  std::vector<Feature> all;
  std::size_t pos = 0;
  while (pos < samples.size()) {
    const auto offered = std::min<std::size_t>(1 + rng() % 8001, samples.size() - pos);
    std::size_t inside = 0;
    while (inside < offered) {
      std::vector<Feature> out(max_frames);
      const auto step = bridge.Process(samples.data() + pos + inside,
                                       offered - inside, out.data(), out.size());
      assert(step.frames_written <= max_frames);
      assert(step.samples_consumed > 0 && step.samples_consumed <= offered - inside);
      all.insert(all.end(), out.begin(), out.begin() + step.frames_written);
      inside += step.samples_consumed;
    }
    pos += offered;
  }
  assert(bridge.SamplesConsumedTotal() == samples.size());
  return all;
}

// ------------------------- slow-consumer soak audit -------------------------
//
// The producer offers fixed PCM chunks and refuses to drop or over-write any
// unconsumed sample: when the caller-owned frame ring is full the producer
// simply stops offering input. The consumer drains at exactly half the input
// rate, so the ring saturates and every steady-state iteration is a genuine
// backpressure cycle (stall -> resume -> stall).
struct SoakConfig {
  double audio_seconds = 20.0;
  std::size_t capacity = 4;   // caller-owned output ring, in feature frames
  std::size_t chunk = 320;    // PCM samples offered per producer step
};

struct SoakReport {
  SoakConfig config;
  double audio_seconds = 0.0;
  double wall_seconds = 0.0;
  std::size_t samples = 0;
  std::size_t frames = 0;
  std::size_t ticks = 0;
  std::size_t producer_steps = 0;
  std::size_t stall_steps = 0;
  std::size_t pause_events = 0;
  std::size_t resume_events = 0;
  std::size_t frames_after_resume = 0;
  std::size_t max_pcm_backlog = 0;
  std::size_t max_frames_held = 0;
  std::size_t rss_warm = 0;
  std::size_t rss_end = 0;
  std::size_t rss_min = 0;
  std::size_t rss_max = 0;
  std::size_t rss_peak = 0;
  std::size_t heap_warm = 0;
  std::size_t heap_end = 0;
  std::size_t heap_min = 0;
  std::size_t heap_max = 0;
  std::size_t heap_peak = 0;
  std::size_t compared_values = 0;
  double max_abs_diff = 0.0;
  double mean_abs_diff = 0.0;
  std::vector<std::pair<double, std::size_t>> rss_series;  // (audio s, bytes)
};

SoakReport RunSlowConsumerSoak(const SoakConfig& cfg) {
  constexpr std::size_t kSampleRate = voice::StreamingFbank::kSampleRate;
  constexpr std::size_t kHop = voice::StreamingFbank::kFrameShift;
  assert(cfg.capacity >= 1 && cfg.chunk >= 1 && cfg.chunk <= 65536);

  const std::size_t total_samples =
      static_cast<std::size_t>(cfg.audio_seconds * static_cast<double>(kSampleRate));
  // Time-driven simulation. One tick is one producer chunk of input time, the
  // producer offers at most one chunk per tick, and the consumer drains one
  // frame per 320 samples of input time -- exactly 50% of the 100 frames/s
  // input rate. Because the consumer is driven by its own clock rather than by
  // how much the bridge managed to consume, a stalled producer cannot deadlock
  // the loop: that is what makes this a backpressure test and not a merely
  // rate-matched one.
  const std::size_t warmup_samples = 2 * kSampleRate;
  // At most ~300 RSS points, never finer than one per second of audio.
  const std::size_t series_stride = std::max(kSampleRate, total_samples / 300);
  std::size_t series_bucket = 0;

  SoakReport report;
  report.config = cfg;

  voice::BoundedFbankBridge bridge;   // audited path: throttled, stalls
  voice::StreamingFbank baseline;     // unthrottled reference: always has room
  std::vector<Feature> out(cfg.capacity);
  std::vector<Feature> ring(cfg.capacity);
  std::size_t ring_head = 0, ring_count = 0;

  std::vector<float> chunk_buf(cfg.chunk);
  std::deque<Feature> ref_frames;  // bounded: one entry per uncompared frame

  std::size_t pos = 0, buf_len = 0, buf_off = 0, tick = 0, consumer_clock = 0;
  bool stalled = false;
  bool just_resumed = false;
  double sum_abs = 0.0;

  report.rss_min = report.heap_min = std::numeric_limits<std::size_t>::max();
  // Reserve the whole RSS series up front: the audit must not attribute its
  // own measurement buffer growth to the code under test.
  report.rss_series.reserve(320);

  const auto feed_baseline = [&](const float* samples, std::size_t count) {
    baseline.AcceptWaveform(samples, count);
    while (baseline.NumFramesReady()) ref_frames.push_back(baseline.GetFrame());
  };
  // Draining the caller ring and comparing against the baseline are the same
  // event: a frame that leaves the ring without being compared would be a hole
  // in the audit, so the two are fused here.
  const auto consume_one = [&]() {
    assert(ring_count > 0 && !ref_frames.empty());
    const Feature& got = ring[ring_head];
    const Feature& want = ref_frames.front();
    for (std::size_t j = 0; j < got.size(); ++j) {
      const double diff = std::abs(got[j] - want[j]);
      if (diff > report.max_abs_diff) report.max_abs_diff = diff;
      sum_abs += diff;
    }
    report.compared_values += got.size();
    ring_head = (ring_head + 1) % cfg.capacity;
    --ring_count;
    ref_frames.pop_front();
  };

  const auto started = std::chrono::steady_clock::now();
  while (buf_off < buf_len || pos < total_samples || ring_count > 0) {
    if (buf_off == buf_len && pos < total_samples) {
      // Producer refills only when its previous chunk has been fully accepted;
      // otherwise it keeps the unconsumed PCM and offers nothing new.
      buf_len = std::min(cfg.chunk, total_samples - pos);
      for (std::size_t i = 0; i < buf_len; ++i) chunk_buf[i] = WaveSample(pos + i);
      buf_off = 0;
      pos += buf_len;
      report.max_pcm_backlog = std::max(report.max_pcm_backlog, buf_len);
    }

    const std::size_t room = cfg.capacity - ring_count;
    const auto step = buf_off < buf_len
                          ? bridge.Process(chunk_buf.data() + buf_off, buf_len - buf_off,
                                           out.data(), room)
                          : voice::BoundedFbankBridge::Step{};
    if (buf_off < buf_len) ++report.producer_steps;
    buf_off += step.samples_consumed;

    // The unthrottled baseline sees the identical samples with unlimited output
    // capacity. It is fed once per tick, whereas the bounded bridge slices the
    // same span at frame boundaries, so the two paths do not share a chunking.
    if (step.samples_consumed != 0)
      feed_baseline(chunk_buf.data() + buf_off - step.samples_consumed,
                    step.samples_consumed);

    for (std::size_t i = 0; i < step.frames_written; ++i) {
      ring[(ring_head + ring_count) % cfg.capacity] = out[i];
      ++ring_count;
    }
    report.frames += step.frames_written;
    report.max_frames_held = std::max(report.max_frames_held, ring_count);

    if (just_resumed && step.frames_written > 0) {
      report.frames_after_resume += step.frames_written;
      just_resumed = false;
    }
    // A stall is the producer still holding PCM at the end of a tick while
    // more input remains: the downstream ring was full, so no new chunk could
    // be accepted. This is the backpressure event the audit counts.
    const bool now_stalled = buf_off < buf_len && pos < total_samples;
    if (now_stalled) {
      ++report.stall_steps;
      if (!stalled) {
        ++report.pause_events;
        stalled = true;
      }
    } else if (stalled) {
      ++report.resume_events;
      stalled = false;
      just_resumed = true;
    }
    // Consumer slot: one frame per 320 samples of input time, i.e. exactly 50%
    // of the 100 frames/s input rate, independent of the offered chunk size.
    consumer_clock += cfg.chunk;
    while (ring_count > 0 && consumer_clock >= 2 * kHop) {
      consume_one();
      consumer_clock -= 2 * kHop;
    }
    // An idle consumer must not bank credit and then burst-drain the ring.
    if (ring_count == 0) consumer_clock = std::min(consumer_clock, 2 * kHop);
    if (ring_count == 0) consumer_clock = std::min(consumer_clock, 2 * kHop);  // no idle banking
    ++tick;
    report.ticks = tick;

    report.samples = bridge.SamplesConsumedTotal();
    if (report.samples >= warmup_samples) {
      const std::size_t rss = ResidentSetBytes();
      const std::size_t heap = g_heap_live;
      if (report.rss_warm == 0) {
        report.rss_warm = rss;
        report.heap_warm = heap;
      }
      report.rss_min = std::min(report.rss_min, rss);
      report.rss_max = std::max(report.rss_max, rss);
      report.heap_min = std::min(report.heap_min, heap);
      report.heap_max = std::max(report.heap_max, heap);
      const std::size_t bucket = report.samples / series_stride;
      if (bucket > series_bucket) {
        series_bucket = bucket;
        report.rss_series.emplace_back(static_cast<double>(report.samples) / kSampleRate, rss);
      }
    }
  }

  // The consumer runs at half rate by construction, so the tail of the stream
  // is still in the ring: drain it before the equality check, otherwise the
  // audit would silently skip the last frames.
  while (ring_count > 0) consume_one();
  const auto finished = std::chrono::steady_clock::now();
  report.wall_seconds = std::chrono::duration<double>(finished - started).count();

  assert(ref_frames.empty() && ring_count == 0);
  assert(report.samples == total_samples);
  report.audio_seconds = static_cast<double>(report.samples) / kSampleRate;
  report.rss_end = ResidentSetBytes();
  report.rss_peak = PeakResidentSetBytes();
  report.heap_end = g_heap_live;
  report.heap_peak = g_heap_peak;
  report.mean_abs_diff = report.compared_values
                             ? sum_abs / static_cast<double>(report.compared_values)
                             : 0.0;
  if (report.rss_min == std::numeric_limits<std::size_t>::max()) report.rss_min = report.rss_end;
  if (report.heap_min == std::numeric_limits<std::size_t>::max()) report.heap_min = report.heap_end;
  return report;
}

void PrintSoak(const SoakReport& r, std::ostream& os) {
  os << "soak: " << r.audio_seconds << " s audio, " << r.samples << " samples, "
     << r.frames << " frames, capacity " << r.config.capacity << " frames, chunk "
     << r.config.chunk << " samples, " << r.wall_seconds << " s wall\n";
  os << "flow: " << r.pause_events << " pause / " << r.resume_events << " resume events, "
     << r.stall_steps << " stalled steps of " << r.ticks << " ticks (" << r.producer_steps
     << " PCM offers), max PCM backlog " << r.max_pcm_backlog << " samples, max frames held "
     << r.max_frames_held << "\n";
  os << "exactness: " << r.compared_values << " values compared, max |diff| "
     << r.max_abs_diff << ", MAE " << r.mean_abs_diff << ", frames after resume "
     << r.frames_after_resume << "\n";
  os << "memory: RSS warm " << r.rss_warm << " -> end " << r.rss_end << " (min "
     << r.rss_min << ", max " << r.rss_max << ", peak " << r.rss_peak << "); heap live warm "
     << r.heap_warm << " -> end " << r.heap_end << " (min " << r.heap_min << ", max "
     << r.heap_max << ", peak " << r.heap_peak << ")\n";
}

std::string SoakJson(const SoakReport& r) {
  std::ostringstream os;
  os.precision(17);
  os << "{\n";
  os << "  \"status\": \"measured_slow_consumer_soak\",\n";
  os << "  \"platform\": \"" << PlatformTag() << "\",\n";
  os << "  \"compiler\": \"" <<
#if defined(__clang__)
      "clang " << __clang_version__
#elif defined(__GNUC__)
      "gcc " << __VERSION__
#elif defined(_MSC_VER)
      "msvc " << _MSC_VER
#else
      "unknown"
#endif
     << "\",\n";
  os << "  \"config\": {\"audio_seconds\": " << r.config.audio_seconds
     << ", \"capacity_frames\": " << r.config.capacity
     << ", \"chunk_samples\": " << r.config.chunk
     << ", \"consumer_rate_ratio\": 0.5},\n";
  os << "  \"audio_seconds\": " << r.audio_seconds << ",\n";
  os << "  \"wall_seconds\": " << r.wall_seconds << ",\n";
  os << "  \"samples\": " << r.samples << ",\n";
  os << "  \"frames\": " << r.frames << ",\n";
  os << "  \"ticks\": " << r.ticks << ",\n";
  os << "  \"producer_steps\": " << r.producer_steps << ",\n";
  os << "  \"stall_steps\": " << r.stall_steps << ",\n";
  os << "  \"pause_events\": " << r.pause_events << ",\n";
  os << "  \"resume_events\": " << r.resume_events << ",\n";
  os << "  \"frames_after_resume\": " << r.frames_after_resume << ",\n";
  os << "  \"max_pcm_backlog_samples\": " << r.max_pcm_backlog << ",\n";
  os << "  \"max_frames_held\": " << r.max_frames_held << ",\n";
  os << "  \"compared_values\": " << r.compared_values << ",\n";
  os << "  \"max_abs_diff\": " << r.max_abs_diff << ",\n";
  os << "  \"mean_abs_diff\": " << r.mean_abs_diff << ",\n";
  os << "  \"rss_bytes\": {\"warm\": " << r.rss_warm << ", \"end\": " << r.rss_end
     << ", \"min\": " << r.rss_min << ", \"max\": " << r.rss_max
     << ", \"peak\": " << r.rss_peak << "},\n";
  os << "  \"heap_live_bytes\": {\"warm\": " << r.heap_warm << ", \"end\": " << r.heap_end
     << ", \"min\": " << r.heap_min << ", \"max\": " << r.heap_max
     << ", \"peak\": " << r.heap_peak << "},\n";
  os << "  \"rss_series\": [";
  for (std::size_t i = 0; i < r.rss_series.size(); ++i) {
    os << (i ? ", " : "") << "[" << r.rss_series[i].first << ", " << r.rss_series[i].second << "]";
  }
  os << "]\n}\n";
  return os.str();
}
}  // namespace

int main(int argc, char** argv) {
  SoakConfig soak;
  std::string json_path;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto value = [&](const char* name) -> std::string {
      if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + name);
      return argv[++i];
    };
    if (arg == "--soak-seconds") soak.audio_seconds = std::stod(value("--soak-seconds"));
    else if (arg == "--soak-capacity") soak.capacity = std::stoul(value("--soak-capacity"));
    else if (arg == "--soak-chunk") soak.chunk = std::stoul(value("--soak-chunk"));
    else if (arg == "--json") json_path = value("--json");
    else if (arg == "--help") {
      std::cout << "usage: test_bounded_fbank_bridge [--soak-seconds S] [--soak-capacity F] "
                   "[--soak-chunk N] [--json PATH]\n";
      return 0;
    } else {
      std::cerr << "unknown argument: " << arg << "\n";
      return 2;
    }
  }

  const auto wave = TestWave(100001);
  const auto reference = Reference(wave);
  assert(reference.size() == 623);
  for (unsigned seed : {1U, 17U, 734U})
    for (std::size_t capacity : {1UL, 2UL, 7UL})
      Compare(WithBackpressure(wave, seed, capacity), reference);

  voice::BoundedFbankBridge bridge;
  const auto blocked = bridge.Process(wave.data(), wave.size(), nullptr, 0);
  assert(blocked.samples_consumed == 0 && blocked.frames_written == 0);
  assert(bridge.SamplesConsumedTotal() == 0);
  Feature one[1];
  const auto resumed = bridge.Process(wave.data(), wave.size(), one, 1);
  assert(resumed.samples_consumed == 400 && resumed.frames_written == 1);
  Compare(std::vector<Feature>{one[0]}, std::vector<Feature>{reference[0]});

  auto bad = TestWave(600);
  bad[550] = std::numeric_limits<float>::quiet_NaN();
  const auto prior = bridge.SamplesConsumedTotal();
  bool invalid_rejected = false;
  try { bridge.Process(bad.data(), bad.size(), one, 1); }
  catch (const std::invalid_argument&) { invalid_rejected = true; }
  assert(invalid_rejected && bridge.SamplesConsumedTotal() == prior);

  bridge.Reset();
  assert(bridge.SamplesConsumedTotal() == 0);
  const auto first_again = bridge.Process(wave.data(), 400, one, 1);
  assert(first_again.samples_consumed == 400 && first_again.frames_written == 1);
  Compare(std::vector<Feature>{one[0]}, std::vector<Feature>{reference[0]});

  voice::BoundedFbankBridge short_clip;
  const auto short_wave = TestWave(399);
  const auto no_frame = short_clip.Process(short_wave.data(), short_wave.size(), one, 1);
  assert(no_frame.samples_consumed == 399 && no_frame.frames_written == 0);

  std::cout << "bounded-fbank PASS: randomized chunks/capacities, backpressure, "
               "atomic invalid-input rejection, reset, and short input\n";

  // Slow-consumer audit. The default duration keeps CTest fast; the recorded
  // evidence run uses --soak-seconds 3600.
  const auto soak_report = RunSlowConsumerSoak(soak);

  // Backpressure must never change a single feature value.
  assert(soak_report.max_abs_diff == 0.0);
  assert(soak_report.mean_abs_diff == 0.0);
  assert(soak_report.samples == static_cast<std::size_t>(soak.audio_seconds * 16000));
  // The stall really happened: otherwise this is not a backpressure audit.
  assert(soak_report.pause_events > 0 && soak_report.resume_events > 0);
  assert(soak_report.frames_after_resume > 0);
  // Bounded watermarks, known a priori rather than observed after the fact.
  assert(soak_report.max_frames_held <= soak.capacity);
  assert(soak_report.max_pcm_backlog <= soak.chunk);
  // No unbounded growth: live heap and RSS after warm-up must be flat.
  assert(soak_report.heap_end <= soak_report.heap_warm + 65536);
  assert(soak_report.rss_end <= soak_report.rss_warm + 8 * 1024 * 1024);

  PrintSoak(soak_report, std::cout);
  std::cout << "soak PASS: strictly bounded watermarks, leak-free steady state, "
               "bit-exact output across pause/resume\n";
  if (!json_path.empty()) {
    std::ofstream out(json_path);
    if (!out) throw std::runtime_error("cannot write " + json_path);
    out << SoakJson(soak_report);
    std::cout << "wrote " << json_path << "\n";
  }
  return 0;
}
