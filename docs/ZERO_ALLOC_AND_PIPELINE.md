# Zero-allocation hot path, hysteresis flow control, and the pure-C++ pipeline

**Status:** local engineering evidence for four specific claims about the C++ frontend. Not production
firmware, not real-time audio, not device latency. Every number below is recomputed from a committed
artifact by `tests/test_firmware_evidence.py`; nothing is hand-transcribed from a console.

Companion to [the bounded frontend proof](CPP_CTC_BOUNDARY_PROOF.md) (feature correctness) and
[the flow-control / beam-width audit](FLOW_CONTROL_AND_BEAM_PARETO.md) (sustained load, decoder
pruning). This document covers the four properties a firmware reviewer actually asks about:
allocation discipline, control-loop chatter, the decoder's runtime, and where the frame time goes.

Everything here is reproducible on a machine with no MSVC, no g++ and no CMake:

```bash
bash scripts/build_windows_zig.sh                 # zig cc 21.1.0 + FFTW 3.3.5 import lib
.workbuddy-ai/venv-cpp/Scripts/python.exe scripts/run_firmware_audits.py
.workbuddy-ai/venv-cpp/Scripts/python.exe scripts/run_pipeline_parity.py
```

---

## 1. Steady state performs zero heap allocations

### Why it matters

An SSD controller's data path cannot call `malloc`. A single allocation inside a hot loop buys
non-deterministic latency (allocator lock contention) and, over hours, fragmentation. "Fast on
average" is the wrong property; "never calls the allocator" is the right one.

### What had to change first

The extractor as originally written pushed completed frames into a `std::deque`, which allocates a
map node per frame. That is fine for offline decoding and disqualifying for a hot path. It was
replaced with a caller-sized fixed ring: the constructor allocates it once, and `AcceptWaveform`
**refuses** a batch that would overflow it rather than growing it or dropping frames. A declared
bound that the code enforces is worth more than a comment that says "bounded".

### The audit

`tests/test_zero_alloc_steady_state.cpp`. The global `operator new`/`delete` are replaced
(`src/alloc_probe.cpp`) and every request is counted exactly. A window of 10,000 frames runs with
`g_allocation_allowed = false` — and, deliberately, with genuine backpressure (capacity-2 ring,
consumer at 50% of the input rate), so the window covers pause/resume and not just the happy path.
A control step then allocates on purpose and requires the probe to see it: a zero is only meaningful
if a non-zero is reachable.

| Measurement | Result |
| --- | --- |
| Frames / samples in the window | 10,000 / 1,600,240 |
| Allocations inside the window | **0** |
| Denied allocations (attempted) | **0** |
| Live-byte delta across the window | **0 B** |
| Feature values compared, bit-exact | 800,000, max \|diff\| = 0 |
| Backpressure inside the window | 9,998 pause / 9,998 resume |
| Init-only allocations (constructor) | 7 |
| Bridge footprint, `sizeof` | 5,992 B (640 B feature slot + extractor) |

The 7 allocations counted before the window belong to object construction (the FFT plan and its two
buffers, the output ring, the CSR tap array) and to the harness materialising its reference clip.
They are reported rather than excluded, because "zero allocations" would be false if it meant "zero
ever".

### What this probe cannot see

`malloc`, FFTW's `fftw_malloc`, `mmap` and driver buffers bypass `operator new` and are invisible
here. Valgrind does not exist on this host (Linux only) and AddressSanitizer has no runtime for the
mingw target, so **this is an allocation audit, not a memory-error audit** — it cannot detect an
out-of-bounds read. The functional unit tests cover edge cases and invalid input; none of them
substitutes for a sanitizer, and no such run is claimed here.

---

## 2. Dual watermark cuts pause/resume chatter

### Why it matters

The soak audit shows the cost of the naive policy: with a 50%-rate consumer and a 4-frame FIFO, the
producer is paused and resumed **once per drained frame** — 39,988 transitions over 40,002 ticks,
i.e. **one transition per 10 ms tick** and a full pause/resume cycle every 20 ms. On silicon each
transition is an interrupt plus a handshake, so that is ~100 interrupts/s to move 100 frames/s
through a 4-frame FIFO. Pure overhead.

### The mechanism

`include/flow_hysteresis.h` — a policy object, not a queue. High watermark `C` pauses, low watermark
`k` resumes, and `k = C-1` reproduces single-watermark behaviour exactly, which is what makes the
control arm comparable. The gate lives in the **driver**, not in the bridge, because the bridge does
not own the output ring: whoever owns the FIFO owns the level signal. That is also where a DMA
controller or an SSP port would put it.

### Measured (ring depth 4, 20,000 frames, consumer at 50%)

| Policy | Transitions | vs control | Longest stall | Mean occupancy |
| --- | --- | --- | --- | --- |
| single (high 4, low 3) | 39,988 | — | 2 ticks | 4.00 |
| dual (high 4, low 2) | 13,330 | **−66.7%** | 3 ticks | 3.50 |
| dual (high 4, low 1) | 7,998 | **−80.0%** | 5 ticks | 3.00 |
| dual (high 4, low 0) | 5,714 | **−85.7%** | 7 ticks | 2.50 |

All four arms: max occupancy ≤ 4, zero dropped frames, and every one of the 1.6 M compared feature
values bit-identical to the unthrottled reference. Hysteresis changes *when* the producer runs, never
*what* it produces.

**The saving is not free, and the table says so.** Every step down in the low watermark buys
transitions with a longer stall (2 → 7 ticks) and more FIFO headroom consumed. `low = 0` drains the
FIFO completely between bursts, which on an audio *output* FIFO is an underrun waiting to happen;
`low = 1..2` is the defensible operating point.

The originally proposed expectation was a 50–75% reduction. The measured range is 67–86%, so the
claim was updated to the measurement rather than the other way round. `low = 2` lands inside the
original window and is the recommended setting.

---

## 3. The decode path is C++ too

### What is in the binary, and what is not

`tools/fbank_ctc_pipeline` takes a WAV and a frozen posterior dump and prints text:

```
PCM WAV -> BoundedFbankBridge (C++) -> greedy CTC collapse (C++) -> byte-BPE detokenise (C++) -> UTF-8
```

**The acoustic model is not in this binary.** The posteriors are a frozen export of the same int8
Zipformer CTC model the Python study used. Calling this "C++ inference" would be false; what it is,
is that everything downstream of the model — and the frontend that feeds it — runs without Python.

The frontend is load-bearing, not decorative: the tool refuses to decode unless the frame count it
produced predicts the exported logit length exactly, and that check passes on all 30 Dev utterances.

### Parity (`results/pipeline_parity.json`)

| Check | Result |
| --- | --- |
| Transcripts identical to the Python reference | **30/30** |
| Token counts identical | **30/30** |
| fbank frames predict the logit length | **30/30** |
| Hypothesis characters compared | 856 |
| UTF-8 replacement characters | 2, matching CPython exactly |

The last row is a bug the parity harness found: a greedy CTC tail can end mid-character, and
CPython's `bytes.decode("utf-8", "replace")` emits one U+FFFD per maximal ill-formed subpart. The
first C++ detokeniser threw instead. It now implements the same replacement rule (and counts the
replacements, so corrupted text is reported rather than shipped silently); the two utterances that
end on a partial character now agree character for character.

### Timing, and a claim I had to walk back

Per-utterance frontend cost on this host: 5.3–10.1 µs/frame (shorter clips cost more per frame
because the bridge call overhead is amortised over fewer frames). Greedy decode: **0.34–0.69 µs per
logit frame** (mean 0.46). Detokenise: ~25 µs per utterance. All host-local, unpinned here.

The first version of the C++ argmax was **1.43 µs/frame — about 4× slower than the Python it
replaced**, because `numpy.argmax` is a vectorised reduction while a scalar loop that carries the
argmax index is a serial dependency chain. Two changes fixed that: split the argmax into a max
reduction plus a first-index search (which also *is* numpy's lowest-index tie rule), and give the
reduction four independent accumulators so clang can overlap the FP chains it will not vectorise
without `-ffast-math`. That is 3× faster and lands within ~1.5× of numpy.

Which means the honest claim for this section is **not** "C++ is faster than Python". Numpy's greedy
pass was already 0.045 ms per utterance. What the C++ path buys is that the runtime, the interpreter
and the `numpy`/`sentencepiece` dependency are gone, and that the decode sits in the same process and
address space as the frontend. The frontend is where the order-of-magnitude win was.

---

## 4. Stage-wise budget: the bottleneck was not the FFT

`tools/fbank_stage_profile.cpp` instruments six micro-stages with an RAII timer, pinned to one core
at high priority (this host is bimodal — the same binary measured the same clip 2× apart unpinned).

### Before

| Stage | µs/frame | Share |
| --- | --- | --- |
| `mel_projection` | 21.558 | **88.5%** |
| `preemphasis_window` | 0.843 | 3.5% |
| `fftw_execute` | 0.826 | 3.4% |
| `log_compress` | 0.550 | 2.3% |
| `dc_remove` | 0.444 | 1.8% |
| `power_spectrum` | 0.130 | 0.5% |
| **whole pass** | **25.51** | 392× realtime |

The expected answer — "the FFT is the hot spot" — is wrong by a factor of 26. The mel projection was
a dense 80 × 257 dot product: **164 KB of coefficients re-read for every 10 ms frame**, which is a
memory-traffic problem wearing an arithmetic costume.

### The finding

Of those 20,560 coefficients, **501 are non-zero (2.4%)**; the mean band has 6.3 non-zero bins out of
256. The dense loop was spending 97.6% of its multiplies on exact zeros.

### The change, and why it is exact

The filterbank is now compressed to a CSR of `(bin, weight)` taps at construction (8,016 B) and the
object no longer carries the dense table at all — `sizeof` dropped from 170,440 B to 5,992 B.

Skipping a term that is exactly `+0.0` is not an approximation. Both factors are non-negative, so the
partial sum is never `-0.0`, and `x + 0.0 == x` exactly in IEEE-754 for every other `x`. The audit
proves it rather than asserting it: **159,840 values compared, 0 bit-mismatches, max |diff| = 0**.

### After

| Stage | µs/frame | Share |
| --- | --- | --- |
| `preemphasis_window` | 0.800 | 28.4% |
| `fftw_execute` | 0.597 | 21.2% |
| `log_compress` | 0.520 | 18.5% |
| `dc_remove` | 0.416 | 14.7% |
| `mel_projection` | 0.360 | **12.8%** |
| `power_spectrum` | 0.124 | 4.4% |

| | Before | After |
| --- | --- | --- |
| Whole pass | 25.51 µs/frame, 392× realtime | **4.09 µs/frame, 2,445× realtime** |
| `mel_projection` | 21.56 µs/frame | **0.360 µs/frame (60× less)** |
| One hour of audio through the soak harness | 43.8 s wall | **9.6 s wall** |

The dense-vs-sparse A/B on the same clip measures **6.74× end-to-end** — less than the 41× arithmetic
headroom, because the rest of the frame is now the cost.

### Two honest caveats about these numbers

1. **Timer overhead is now material, and the two runs prove it.** Each stage reads `steady_clock`
   twice. Subtracting the stage sum from the whole pass leaves **1.16 µs/frame before** and
   **1.28 µs/frame after** — the same fixed per-frame cost (six clock reads, the ring copy, the frame
   push/pop), a 9% difference between two runs of the same harness. That is 4.5% of the dense frame
   and **31.2% of the sparse one**. The consistency is the best evidence that the 6× drop is real:
   the unaccounted work did not move, only the instrumented work did. It also means the per-stage
   absolutes are inflated by roughly one clock read each at 4 µs/frame; the *shares* are the
   trustworthy part.
2. **Pinning reduces the noise, it does not remove it.** The five sparse passes span 7.92–10.89 ms
   (1.37×) and the dense ones 41.3–52.0 ms (1.26×). This is a shared desktop, not a device.

### What is left

With the memory-bound matmul gone, the remaining time is spread across per-sample loops
(`preemphasis_window` 28%, `dc_remove` 15%), the FFT (21%) and 80 `log()` calls per frame (18%). The
next wins are vectorising the 400-sample loops and replacing `std::log` with a table-plus-polynomial
approximation — both change results in the last bits, so they would need their own tolerance argument
rather than this one's bit-exactness.

---

## Claim audit

| Claim | Artifact | Test |
| --- | --- | --- |
| 0 allocations over 10,000 steady-state frames under backpressure | `results/zero_alloc_steady_state.json` | `tests/test_firmware_evidence.py::ZeroAllocationTests` |
| Bridge footprint is 5,992 B | same | `::test_footprint_is_a_declared_constant` |
| Dual watermark cuts transitions 67–86% at bounded occupancy | `results/hysteresis_flow.json` | `::HysteresisTests` |
| Hysteresis is bit-exact and lossless | same | `::test_every_arm_is_bounded_and_lossless` |
| C++ text == Python text, 30/30 | `results/pipeline_parity.json` | `::PipelineParityTests` |
| Mel projection is 2.4% dense and 6.74× faster compressed, bit-exactly | `results/mel_sparsity.json` | `::MelSparsityTests` |
| Stage shares and per-frame costs | `results/stage_latency_budget.json` | `::StageBudgetTests` |

## Non-claims

- **Not real-time.** The harness processes as fast as the host allows. Nothing here bounds worst-case
  latency, and no scheduler or interrupt behaviour is modelled.
- **Not device latency.** Host-local, desktop-class, x86-64 numbers from a shared machine.
- **No acoustic model in C++.** Section 3 is a frontend + decoder + detokeniser claim only.
- **No memory-error detection.** No ASan, no Valgrind on this host; see section 1.
- **Feature parity is frame-count consistency, not numerical equality** with the Python frontend.
