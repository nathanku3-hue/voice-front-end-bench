# CTC-Decoding-Bench — Round 2 closure: mathematically checked decoding, Dev-only sweep

**Executed:** 2026-10-12, existing local WSL2 / CPU environment.
**Status: CLOSED for the explicitly scoped 30-utterance Dev algorithm and measurement gate.**
**50 held-out Test utterances remain unscored**; final Test, Pareto figure and full Bad Case report are Round 3 tasks.

## 1. Question and scope

With **the same** frozen `[T,1000]` CTC log probabilities, does full-vocabulary
log-domain Prefix Beam Search change Dev CER and how much decoder-only CPU time
do Forced Blank and Hard Skip approximate shortcuts save? We do **not** train a
model, connect the C++ frontend to inference, or call decoder timing end-to-end
RTF or microphone response delay.

The public fixed offline Zipformer-CTC ONNX model is distinct from Project 1's
online Transducer. Model SHA-256:
`e291b9c468b651e2697caa09bc684326c3addc6a019e78eb537cfd1a8248ca07`.
Dev source: 30 source-verified AISHELL-1 test utterances, 442 normalized
reference characters, 3,904 CTC frames in total. Data are
utterance-disjoint from the 50 held-out Test and the earlier endpoint corpus.
The split is **not speaker-disjoint**, nor was original model-training overlap
independently established.

## 2. Numerically validated prefix recurrence

For each token-ID tuple prefix `l`, maintain log-domain probabilities
`Pb(l)` (paths ending blank) and `Pnb(l)` (paths ending nonblank). Merge
with stable `log_add`, including `-inf`. Blank extends the same prefix;
a token equal to the previous prefix's last token continues the existing
prefix only from `Pnb`, or appends a fresh duplicate only from `Pb`.
Other tokens append from `logadd(Pb,Pnb)`.

**Independent oracle:** `tests/test_ctc_prefix_beam.py` independently
enumerates each of the 64 paths for `T=3,V=4` and 81 paths for `T=4,V=3`,
collapses each path using group-by-then-blank-removal and compares **all**
reachable prefix log probabilities (not just the best hypothesis). Multiple
random seeds, nonzero blank IDs, `-inf` paths, and repeated-symbol toy cases
are tested with absolute tolerance `1e-6`. The vectorized exact full-vocab
top-beam update is also compared with an intentionally literal full-enumeration
prefix recurrence for finite beam widths and tied scores.

**Performance implementation note:** `scripts/ctc_prefix_beam_search.py`
does **not** prune acoustic tokens to a hidden Top-K. It scores all
`V-1=999` nonblank tokens for every expanded active prefix with NumPy,
merges the existing prefix's self/parent contributions, then uses an exact
cutoff (including lexically deterministic boundary ties) for the best new
prefixes. New extension prefixes have a unique parent; only pre-existing
prefixes have other incoming contributions, so this shortlist after merging
is equivalent to ranking the full expanded candidate set. This is an
**exact per-frame beam update** given the incoming beam, not a claim that
a finite-width beam reproduces the exhaustive unlimited-width CTC model.

## 3. Two intentionally different approximate Blank policies

- **Unpruned:** apply every frame's blank and all nonblank token arcs.
- **Forced Blank:** if `p(blank) >= θ`, set
  `Pb_new(l)=logadd(Pb_old(l),Pnb_old(l))+log p(blank)`, and
  `Pnb_new(l)=-inf`. This is mathematically exact for the **restricted
  only-blank-at-that-frame** search, **not** lossless for the original model.
  A frame is still processed; report it as *forced*, **not** a Hard Skip.
- **Hard Skip:** if `p(blank) >= θ`, leave both `Pb` and `Pnb` unchanged.
  This can erase Blank separators between two identical symbols, and is a
  different heuristic than Forced Blank. Toy input `A, blank, A` proves
  Forced Blank can output `AA` while Hard Skip outputs `A`.

Both policies are **approximations**. An identical aggregate CER does not
prove mathematical or transcript equivalence. All comparisons retain
Round 1's explicit SentencePiece Byte-BPE `--byte-errors replace` policy:
unrecoverable UTF-8 becomes `U+FFFD`, counted as an error rather than
silently discarded.

## 4. Experiment contract and timing

**Matrix: 36 distinct Prefix Beam configurations**, namely `W={1,4,8,16}`
× (one unpruned configuration + four thresholds
`θ={0.90,0.95,0.98,0.99}` for **each** of Forced Blank and Hard Skip).
A **separate Framewise Greedy** serves as the independent baseline;
`Prefix Beam W=1` is **not defined** as Framewise Greedy.

All 30 Dev `.npy` tensors are loaded and SHA-256 checked **before** timing.
Only `prefix_beam_search()`, or `greedy_ids()` for the independent Greedy,
is inside a `time.perf_counter_ns()` window. A warmup call precedes
**three timed repetitions per utterance and condition**; the per-utterance
median is used for group mean, group median, P95, and sum. Excluded: WAV
loading, Fbank, ONNX inference, disk I/O, Byte-BPE detokenization, CER/edit
alignment and experiment serialization. Run sequentially with the same
frozen inputs on **CPython 3.12.3 / NumPy 2.3.5 / WSL2 Linux**
(12th Gen Intel Core i5-12500H, 16 logical CPUs exposed to WSL2). Exact
processor frequency, scheduler interference and inter-run order effects
were not controlled; these are **exploratory local Python timings**, not
portable throughput, tail-latency SLAs, or deployment profiling.

To avoid a misleading benchmark, the speedups below compare each policy's
**sum of per-utterance median times** against the **same Beam Width without
Blank shortcut**. They are not the ratio of the displayed group medians.

## 5. Measured Dev results (actual, not anticipated)

| Beam W | Unpruned median ms/utt | Forced Blank θ=.90 median | Forced speedup | Hard Skip θ=.90 median | Hard speedup |
|---|---:|---:|---:|---:|---:|
| 1 | 5.771 | 2.044 | 2.77× | 2.006 | 2.76× |
| 4 | 7.039 | 2.439 | 2.99× | 2.256 | 3.16× |
| 8 | 9.969 | 2.974 | 3.24× | 2.905 | 3.33× |
| 16 | 14.369 | 4.059 | 3.27× | 3.759 | 3.68× |

**Accuracy:** all **36/36 configurations** yielded corpus
**CER = 13/442 = 2.9412%** on **this 30-utterance Dev only**.
The independent Framewise Greedy baseline also had 2.9412% CER
(S/D/I = **12/0/1**) with approximately **0.0241 ms/utterance** median
decode time. Even pruned Prefix Beam remains **much slower than simple
Framewise Greedy** in this local Python implementation, and no accuracy
benefit versus Greedy was demonstrated. Wider beams likewise offered no
CER improvement on this small corpus.

**Frame triggers:** θ=0.90 marked approximately **80.40% of CTC frames**.
At θ=0.95, 0.98 and 0.99, the triggered fractions are approximately
**79.30%, 77.15%, 74.49%**, respectively. With Forced Blank these are
restricted-frame updates, not skipped frames (**actual Hard Skip rate 0**);
with Hard Skip the same fractions are truly bypassed. The complete matrix
with exact counts, CER/S/D/I, total nonblank arcs, elapsed time metrics and
per-utterance outcomes is in
`results/ctc_dev_sweep_results.json` and
`results/ctc_dev_sweep_results.csv`.

**Important audit finding:** In all 16 Forced Blank configurations,
normalised transcripts matched their same-width unpruned counterpart on
these 30 samples. In **all 16 Hard Skip configurations**, **2/30**
normalized transcripts differed despite identical aggregate CER:

1. `BAC009S0768W0427`: Hard Skip changed the correct `平衡` to
   `平�`, introducing a substitution / malformed UTF-8 replacement.
2. `BAC009S0906W0212`: Hard Skip removed the extra `生` in
   `棉花生生产`, correcting an insertion.

The two errors **cancel in aggregate**. In Beam W=16, Hard Skip
increased Byte-BPE replacement characters from **2 to 3** compared
with unpruned. The equality of 2.94% CER is **not proof of losslessness**.
This is a useful negative/neutral engineering result: Blank transition
semantics matter even when a headline score does not move.

## 6. Exit artifacts and independent checking

- `scripts/ctc_prefix_beam_search.py`: log-domain Pb/Pnb algorithm,
  deterministic exact beam selection, independent shortcut strategies.
- `tests/test_ctc_prefix_beam.py`: exhaustive alignment Oracle,
  finite-width literal reference comparison, repeated symbols and edge
  cases, -inf and blank policies.
- `scripts/run_ctc_dev_sweep.py`: frozen Dev SHA guards, separate Greedy
  baseline, 36 conditions, 3× decoder-only times, JSON/CSV.
- `results/ctc_dev_sweep_results.json` and `.csv`: all actual
  Dev measurements and per-utterance S/D/I and timing.
- `scripts/ctc_round2_audit.py` and `results/ctc_round2_audit.json`:
  independently re-score all 36×30 Dev hypotheses and check CSV/JSON
  correspondence, time/skip aggregation, input hashes, strict Dev IDs and
  no scoring artifacts under the held-out Test export directory.
- `docs/CTC_ROUND2_STATUS.md`: this closure and limitations.
- `tests/test_ctc_dev_sweep.py` and
  `tests/test_ctc_tokenization_boundary.py`: grid completeness,
  exact merging and SentencePiece tuple→list decoder boundary.

**Final audit status:** `PASS`; exactly **36** conditions, **30**
Dev utterances each, aggregate results revalidated against reference text.
The full Python suite and original C++ stream invariance test are checked
separately; no held-out Test logits were decoded or scored.

## 7. Re-run on the existing WSL2 environment

From `E:\\Code\\voice` (mounted as `/mnt/e/Code/voice` in WSL2):

```bash
PY="$HOME/.venvs/voice-bench/bin/python"
MODEL="$HOME/.cache/voice-front-end/models/sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03"

"$PY" -m unittest discover -s tests -p 'test_ctc_prefix_beam.py' -v
for W in 1 4 8 16; do
  "$PY" scripts/run_ctc_dev_sweep.py --part-width "$W" --model-dir "$MODEL" --repeats 3
done
"$PY" scripts/run_ctc_dev_sweep.py --merge-only
"$PY" scripts/ctc_round2_audit.py
"$PY" -m unittest discover -s tests -p 'test_*.py' -q
```

Re-runs may produce different **timing** numbers due to host conditions; model
outputs, configuration IDs, Dev CER and source hashes should be reproducible.
The partial `build/ctc_round2_width_*.json` files and `build/` tensors
are deliberately ignored by Git; `results/` contains the human-inspectable
measurement and evidence summary. Re-creating a different source corpus does
not count as an exact reproduction unless SHA-256 identities match.

## 8. Transition to Round 3 (not performed here)

Freeze a Dev-selected policy *before* any Test CER inspection; then decode the
50 untouched Test utterances once. Measure decoder-only time with the same
protocol, construct a properly labeled latency–CER comparison/Pareto plot,
audit the 2 observed Hard Skip Dev changes and additional Test bad cases,
and finalize resume bullets with Test-qualified findings.

**No new model training, unseen-speaker generalization, universal CER gain,
lossless Blank pruning, production speech latency or vehicle deployment
operating point is claimed.** No commit, remote push, or public release
is implied by Round 2 closure.
