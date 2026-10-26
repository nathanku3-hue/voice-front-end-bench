# Flow control and beam-width evidence — two measured audits

**Status:** local engineering evidence for two specific claims. Not production firmware, not real-time
audio, not device latency. Every number below is recomputed from a committed artifact by
`tests/test_flow_control_evidence.py`; nothing here is hand-transcribed from a console.

Companion to [the bounded C++ frontend proof](CPP_CTC_BOUNDARY_PROOF.md). That document covers
*correctness of the features*; this one covers *what happens under sustained load and under decoder
pruning*.

---

## 1. Backpressure: bounded watermarks and a leak-free steady state

### The property under test

`BoundedFbankBridge::Process` writes into a caller-owned buffer of `output_capacity` frames. When the
caller has no room it stops consuming PCM and returns the unconsumed samples to the caller, which
keeps them. The interface therefore has a hard upper bound on everything it can hold:

| Held by | Bound |
| --- | --- |
| The bridge (extractor + pending frame) | 1 frame, by construction |
| The caller's output ring | `output_capacity` frames |
| The caller's unconsumed PCM | one offered chunk |

The audit checks that these bounds are *reached but never exceeded* over a long stream, and that
refusing input never changes a feature value.

### Harness

`tests/test_bounded_fbank_bridge.cpp` (`--soak-seconds`, `--soak-capacity`, `--soak-chunk`):

- **Time-driven simulation.** One tick advances one producer chunk of input time. The producer offers
  a new chunk only when its previous chunk was fully accepted; otherwise it holds the PCM.
- **Consumer at exactly 50%.** One frame drained per 320 samples of input time, i.e. half of the
  100 frames/s input rate. The consumer is driven by its own clock, not by how much the bridge
  managed to consume — otherwise a stalled producer would deadlock the loop and the test would
  measure a rate-matched system instead of a backpressured one.
- **Unthrottled baseline.** A second `StreamingFbank` receives the identical samples with unlimited
  output capacity. The bounded bridge slices each span at frame boundaries; the baseline does not, so
  the two paths do not share a chunking.
- **Exactness.** Draining the ring and comparing against the baseline are fused into one operation: a
  frame that left the ring uncompared would be a hole in the audit. Every element of every frame is
  compared; `compared_values == frames × 80` is asserted.
- **Memory.** Live bytes are counted at the `operator new`/`delete` boundary (exact, allocator
  independent) and RSS is sampled throughout.

**Valgrind and AddressSanitizer are not used, and that is a real limitation, not a preference.**
Valgrind is Linux-only and this host is Windows. ASan was tried and does not link: with zig cc the
instrumentation compiles but the link fails on undefined `__asan_option_detect_stack_use_after_return`
and `__asan_shadow_memory_dynamic_address` — no ASan runtime is shipped for the mingw target.
Byte-exact allocation accounting is a substitute for *leak detection*, not for *memory-error
detection* — it cannot see an out-of-bounds read. The unit tests and the Debug CTest run cover that
separately.

### Result — one hour of audio

Artifact: `results/backpressure_soak.json`, figure: `results/backpressure_watermark.png`.

| Quantity | Value |
| --- | --- |
| Audio streamed | 3,600 s (57,600,000 samples, 359,998 frames) |
| Wall time | 43.8 s (82× faster than real time — **not** a real-time run) |
| Backpressure cycles | 179,995 pause / 179,995 resume events (49.999% of ticks blocked) |
| Max frames held | 4 = capacity |
| Max unconsumed PCM | 320 samples = one chunk |
| Feature values compared | 28,799,840 (all of them) |
| Max \|difference\| vs baseline | **0.0** |
| MAE vs baseline | **0.0** |
| Live heap, warm-up → end | 1,142,752 → 1,142,752 B (**Δ = 0**, min = max) |
| RSS, warm-up → end | 13,021,184 → 12,853,248 B (**−164 KiB**), band 12.24–12.42 MiB |
| RSS start-up peak | 13,819,904 B (FFTW plan + buffers, one-time) |

The RSS *decrease* is ordinary Windows working-set trimming; the claim it supports is "no growth",
not "flat to the byte". The live-heap figure is the stronger one: it is exactly constant, with
`min == max` over the whole hour.

### Result — watermark scales with capacity, not with elapsed time

Same harness, 60 s of audio each:

| Capacity | Max frames held | Pause fraction | Live-heap Δ |
| --- | --- | --- | --- |
| 1 | 1 | 0.333 | 0 B |
| 4 | 4 | 0.499 | 0 B |
| 64 | 64 | 0.489 | −30,720 B |

The high-water mark equals the configured capacity in all three cases: memory is bounded by
*configuration*, and the pause fraction stays near 50% regardless of capacity, which is what a 50%
consumer should produce.

### What this does **not** show

- Not real time. One hour of *audio content* was processed in 44 s; no deadline was ever at risk.
- Not a device. x86-64 Windows, single-threaded harness; no interrupts, DMA, cache contention or
  scheduler preemption.
- Not memory-error free. No ASan/Valgrind (see above).
- Not a claim about the CLI's file staging path (`src/fbank_bounded_dump.cpp`).

---

## 2. Beam width: latency versus CER

### Method

`scripts/sweep_beam_pareto.py` over the frozen 30-utterance Dev logits
(`build/ctc_round1_dev`, hash-pinned by `results/ctc_logit_inventory.json`). Unpruned prefix beam
search only (`strategy="none"`), so the single variable is width. Timing covers
`prefix_beam_search()` alone: no ONNX session, no audio I/O, no detokenization, no scoring.

Two measurement decisions were forced by the host:

- **Widths are interleaved round-robin**, nine passes, median pass published. Measuring one width to
  completion and then the next attributed host drift to the width: an early run reported W=1 at
  20.4 ms and W=4 at 10.2 ms, which is not physically possible.
- **The process is pinned to one CPU at high priority.** Unpinned, the same width varied by ~3×
  between passes, consistent with P/E-core scheduling.

### Result

Artifact: `results/ctc_beam_pareto.json` / `.csv`, figure: `results/ctc_beam_pareto.png`.

| Beam | CER | Mean ms/utt | p95 ms | ms per CTC frame | vs W=32 | Pass spread | Hypotheses ≠ W=32 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2.94% | 9.99 | 16.18 | 0.077 | 4.41× | 8.74–16.56 | 0/30 |
| 2 | 2.94% | 12.95 | 24.82 | 0.100 | 3.40× | 9.96–16.92 | 0/30 |
| 4 | 2.94% | 14.51 | 21.35 | 0.111 | 3.03× | 12.46–22.50 | 0/30 |
| 8 | 2.94% | 19.45 | 35.01 | 0.149 | 2.26× | 16.34–22.30 | 0/30 |
| 16 | 2.94% | 26.15 | 42.99 | 0.201 | 1.68× | 22.80–51.85 | 0/30 |
| 32 | 2.94% | 44.01 | 65.34 | 0.338 | 1.00× | 35.35–51.48 | 0/30 |
| Greedy | 2.94% | 0.045 | — | — | — | — | 0/30 |

### The finding: there is no trade-off to make

CER is *identical* at every width, and the decoded token sequences are identical too — 0/30
utterances differ from W=32 at any width. So the flat CER is not detokenization aliasing; the beam
genuinely converges to the same path. The accuracy/latency plane collapses to a single point, and
every width above the narrowest is strictly dominated.

**Why.** The acoustic posterior is effectively one-hot. Over 3,904 CTC frames:

| Statistic | Value |
| --- | --- |
| Mean top-1 probability | 0.9748 |
| Median top-1 probability | 0.9988 |
| Frames with top-1 > 0.99 | 76.5% |
| Mean top-1 − top-2 margin | 0.9555 |
| Frames with margin > 0.9 | 91.2% |

A beam only changes the output where the frame distribution is ambiguous. On 91% of frames the top
candidate wins by more than 0.9 probability, so width buys nothing. This is a property of *this*
int8 Zipformer-CTC model on *this* Dev set — a model with a language model, a noisier channel, or a
larger vocabulary could behave differently.

**Engineering reading.** Beam width is a cost knob here, not an accuracy knob. If a beam is needed
for a future LM or a different acoustic model, the narrowest width that preserves all outputs is the
right choice; on this data that is W=1 at 4.4× less decoder time than W=32. And relative to plain
framewise greedy (0.045 ms/utterance, same 2.94% CER, same tokens), *any* beam search is a 200×+
cost for zero measured benefit.

### What this does **not** show

- Not device latency. This is a CPython prototype on an x86-64 laptop; the ratios transfer to a
  ported C decoder only as a hypothesis, not as a result.
- Not a general claim about beam search. It is a claim about this model, this 30-utterance Dev split,
  and no language model.
- Dev was already used to choose the Round 2 configuration grid; it is not a fresh test set.
- Absolute milliseconds drifted by up to ~50% between sweep runs before pinning. Treat the ratio as
  "about 4×", not as a constant.

---

## 3. Claim audit — draft wording that does not survive the measurement

The draft bullet proposed for this work was:

> 遍歷 Beam Width ∈ [4,8,16,32] … 明確給出結論：「在 Beam=8 時達成 4.1 倍加速，且 CER 僅微幅劣化 0.1%」

| Draft element | Measured | Verdict |
| --- | --- | --- |
| "Beam=8 → 4.1× speed-up" | W=8 is **2.26×** faster than W=32. The 4.41× figure belongs to **W=1**, not W=8. | **Wrong number, wrong width** |
| "CER degrades by 0.1%" | CER is **2.94% at every width**; 0/30 hypotheses differ. There is no degradation to report. | **Wrong direction of the claim** |
| "Beam=8 is the optimal trade-off point" | W=8 is dominated by W=1, W=2 and W=4 on both axes. | **Not supported** |

Two related confusions are worth flagging because they are easy to repeat:

- The repo's existing **3.68×** figure is *Hard Skip θ=0.90 vs unpruned at the same width 16* — a
  blank-shortcut result, not a beam-width result. It also altered 2/30 transcripts.
- A 4.1× number does exist in the neighbourhood (W=1 vs W=32 measured 4.41×), but attaching it to
  "Beam=8" and pairing it with a fabricated 0.1% CER cost would not survive one question in an
  interview.

## 4. Wording that does survive

**Backpressure (Chinese, for review):**

> 在自寫 C++ 串流 Fbank 上實作「呼叫方持有、容量有界」的輸出佇列與背壓：下游只以 50% 速率消費時，生產端暫停送入而非丟棄或無限緩衝。以 1 小時音訊（5,760 萬取樣、359,998 幀）連續驗證，記憶體水位嚴格等於佇列容量（1/4/64 幀），live heap 成長 0 bytes、RSS 結束時低於暖機基準 164 KiB；18 萬次暫停／恢復後輸出與無背壓基準逐位元一致（2,880 萬個特徵值，max|diff| = 0、MAE = 0）。

**Beam width (Chinese, for review):**

> 在凍結的 30 句 Dev logits 上掃描束寬 1–32：CER 恆為 2.94%、30/30 解碼序列與 W=32 完全相同，解碼時間卻隨寬度成長 4.4 倍（0.077 → 0.338 ms/CTC 幀）。量測顯示該 int8 CTC 模型後驗近乎 one-hot（幀級 top-1 機率中位數 0.9988、91% 幀 top1−top2 差距 > 0.9），因此束寬是成本旋鈕而非精度旋鈕；邊緣端應取最窄束寬或直接貪心，而非 W=8。

**English one-liners:**

- "Added caller-owned bounded output and backpressure to a C++17 streaming Fbank; a 1-hour
  (57.6 M-sample) soak at a 50% consumer rate held the frame watermark at exactly the configured
  capacity, kept live heap growth at 0 bytes, and stayed bit-identical (MAE 0 over 28.8 M values)
  across 180 k pause/resume cycles."
- "Measured the beam-width/CER Pareto on 30 frozen Dev utterances: CER is invariant from greedy to
  W=32 with token-identical hypotheses, while decoder time grows 4.4×; the posterior is near-one-hot
  (median top-1 0.9988), so beam width is a pure cost knob for this model."

**Do not claim:** real-time audio, device latency, ASan/Valgrind coverage, memory-error freedom,
"Beam 8 = 4.1× with 0.1% CER cost", or that these numbers transfer to a C decoder on embedded
hardware.

---

## 5. Reproduce

```bash
# 1. Build the audited binary (Linux/WSL2: normal CMake; see README)
cmake -S . -B build/bridge-release -DCMAKE_BUILD_TYPE=Release
cmake --build build/bridge-release --parallel 2
ctest --test-dir build/bridge-release --output-on-failure

# 2. One hour of audio, plus the 1/4/64 capacity sweep, plus the figure
python scripts/run_backpressure_soak.py \
  --binary build/bridge-release/test_bounded_fbank_bridge \
  --headline-seconds 3600

# 3. Beam-width sweep (needs the CTC model dir for tokens.txt + bbpe.model)
python scripts/sweep_beam_pareto.py --model-dir ~/.cache/voice-front-end/models/sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03

# 4. Re-check that the artifacts still support the claims above
python -m unittest discover -s tests -p 'test_flow_control_evidence.py' -v
```

The soak runner fails closed: it refuses to write `results/backpressure_soak.json` unless every
frame was compared, `max_abs_diff` is exactly 0, the watermark never exceeded capacity, and heap
growth stayed inside tolerance.

On this Windows host the binary was built with zig cc 21.1.0 against FFTW 3.3.5 (win64 import
library generated with `zig dlltool`); `CMakeLists.txt` links `psapi` on Windows for the RSS probe.
The measured binary's SHA-256 is recorded in the artifact.
