#pragma once

#include <cstddef>
#include <stdexcept>

namespace voice {

// Two-watermark (hysteresis) flow gate for a fixed-capacity FIFO.
//
// Single-watermark control pauses the producer the moment the FIFO fills and
// resumes it as soon as one slot frees. When the consumer runs near the
// producer's rate that makes the pause line chatter once per slot, and on real
// silicon every transition is an interrupt plus a handshake. A low watermark
// forces the FIFO to drain to `low` before production resumes: fewer
// transitions, at the cost of a bounded amount of extra queueing delay.
//
// high == low + 1 reproduces single-watermark behaviour exactly, which is what
// the audit uses as its control arm.
//
// This is a policy object, not a queue: it observes occupancy and answers
// "may the producer run". It never allocates and never blocks.
class HysteresisGate {
 public:
  HysteresisGate(std::size_t high_watermark, std::size_t low_watermark)
      : high_(high_watermark), low_(low_watermark) {
    if (high_ == 0) throw std::invalid_argument("high watermark must be >= 1");
    if (low_ >= high_) throw std::invalid_argument("low watermark must be below the high watermark");
  }

  // Call once per scheduling opportunity with the current FIFO occupancy.
  // Returns true when the producer is allowed to offer more input.
  bool ProducerMayRun(std::size_t occupancy) noexcept {
    if (!paused_) {
      if (occupancy >= high_) {
        paused_ = true;
        ++pauses_;
      }
    } else if (occupancy <= low_) {
      paused_ = false;
      ++resumes_;
    }
    return !paused_;
  }

  bool paused() const noexcept { return paused_; }
  std::size_t pauses() const noexcept { return pauses_; }
  std::size_t resumes() const noexcept { return resumes_; }
  // One pause + one resume is one handshake cycle. Counting transitions rather
  // than cycles keeps the two control arms directly comparable.
  std::size_t transitions() const noexcept { return pauses_ + resumes_; }
  std::size_t high_watermark() const noexcept { return high_; }
  std::size_t low_watermark() const noexcept { return low_; }

 private:
  std::size_t high_;
  std::size_t low_;
  bool paused_ = false;
  std::size_t pauses_ = 0;
  std::size_t resumes_ = 0;
};

}  // namespace voice
