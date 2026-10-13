# CTC Round 3 — held-out Test evaluation status

**Status:** Held-out Test one-shot completed and machine-audited; manual audio-level bad-case attribution remains a separate review.

## Protocol and custody

- Pre-registration SHA-256: `686a56f717f1c04935f784bd448afb18f545f7aeda6533d15f093ac6243bd55c`.
- Frozen Test manifest SHA-256: `11651297f9dd5abdd6b985b45a383a095b8d0a1c1ab8fc9b658fb6a0e117d056`.
- Model SHA-256: `e291b9c468b651e2697caa09bc684326c3addc6a019e78eb537cfd1a8248ca07`.
- Exactly four conditions fixed from the 30 Dev samples before evaluating 50 Test samples.
- Independent audit: 4 × 50 utterance hypotheses, corpus edit counts, UTF-8 replacements, timings, four-row CSV, claim file, and Pareto identities: PASS.
- Data are utterance-disjoint, **not** speaker-disjoint; original training overlap unverified.

## Test vs Dev — observed, not forecast

| Locked decoder | Dev corpus CER | Test corpus CER | Test S/D/I | Test mean decoder-only ms/utt | Test median decoder-only ms/utt | Replacement chars |
|---|---:|---:|---:|---:|---:|---:|
| `framewise_greedy` | 2.94% | 1.65% | 11/0/1 | 0.026 | 0.025 | 0 |
| `prefix_w16_unpruned` | 2.94% | 1.65% | 11/0/1 | 15.541 | 15.305 | 0 |
| `prefix_w16_forced_blank_t0.95` | 2.94% | 1.65% | 11/0/1 | 5.122 | 4.816 | 0 |
| `prefix_w04_forced_blank_t0.95` | 2.94% | 1.65% | 11/0/1 | 3.294 | 3.180 | 0 |

## Measured decision evidence

The comparisons below use sums of per-utterance medians (not ratios of group medians).
- `prefix_w16_unpruned`: 588.6× Greedy decoder time; 0/50 normalized hypotheses differ and 0/50 token sequences differ.
- `prefix_w16_forced_blank_t0.95`: 194.0× Greedy decoder time; 0/50 normalized hypotheses differ and 0/50 token sequences differ.
- `prefix_w04_forced_blank_t0.95`: 124.8× Greedy decoder time; 0/50 normalized hypotheses differ and 0/50 token sequences differ.
- `prefix_w16_forced_blank_t0.95` vs Beam 16 unpruned: 3.03× decoder-only speedup (different beam width if W=4); 0/50 normalized hypotheses differ; Forced Blank triggered on 82.25% of frames.
- `prefix_w04_forced_blank_t0.95` vs Beam 16 unpruned: 4.72× decoder-only speedup (different beam width if W=4); 0/50 normalized hypotheses differ; Forced Blank triggered on 82.25% of frames.

## Pareto assessment

Test non-dominated condition(s), using mean of utterance decoder medians vs corpus CER: **`framewise_greedy`**.
All four conditions remain in the reported plot and results. CER equality does not imply identical token sequences or normalized hypotheses.

## Deterministically selected bad-case evidence

Selection rule: highest *Greedy* absolute edit count (S+D+I), then utterance ID. This is an audit queue, not a confirmed causal acoustic diagnosis.

### `BAC009S0765W0361`

- Reference: 二零一四  年  的  冰壶  世锦赛  也  在  北京  举行
- Source audio: `data/ctc_eval/audio/BAC009S0765W0361.wav`; duration: 5.155 s; CTC frames: 127; mean blank posterior: 0.7937; mean maximum frame posterior: 0.9862.
- `framewise_greedy`: 二 零 一 四 年 的 滨 湖 世 锦 赛 也 在 北 京 举 行 (S/D/I=2/0/0, U+FFFD=0).
- `prefix_w16_unpruned`: 二 零 一 四 年 的 滨 湖 世 锦 赛 也 在 北 京 举 行 (S/D/I=2/0/0, U+FFFD=0).
- `prefix_w16_forced_blank_t0.95`: 二 零 一 四 年 的 滨 湖 世 锦 赛 也 在 北 京 举 行 (S/D/I=2/0/0, U+FFFD=0).
- `prefix_w04_forced_blank_t0.95`: 二 零 一 四 年 的 滨 湖 世 锦 赛 也 在 北 京 举 行 (S/D/I=2/0/0, U+FFFD=0).
- **Manual listening status:** not performed by this script; do not claim homophone/deletion/acoustic cause until verified.

### `BAC009S0915W0484`

- Reference: 湖北  二五  岁  女子  从未  来  例假  基因  检查  是男  身
- Source audio: `data/ctc_eval/audio/BAC009S0915W0484.wav`; duration: 7.640 s; CTC frames: 189; mean blank posterior: 0.8616; mean maximum frame posterior: 0.9714.
- `framewise_greedy`: 湖 北 二 十 五 岁 女 子 从 未 来 例 假 基 金 检 查 是 男 身 (S/D/I=1/0/1, U+FFFD=0).
- `prefix_w16_unpruned`: 湖 北 二 十 五 岁 女 子 从 未 来 例 假 基 金 检 查 是 男 身 (S/D/I=1/0/1, U+FFFD=0).
- `prefix_w16_forced_blank_t0.95`: 湖 北 二 十 五 岁 女 子 从 未 来 例 假 基 金 检 查 是 男 身 (S/D/I=1/0/1, U+FFFD=0).
- `prefix_w04_forced_blank_t0.95`: 湖 北 二 十 五 岁 女 子 从 未 来 例 假 基 金 检 查 是 男 身 (S/D/I=1/0/1, U+FFFD=0).
- **Manual listening status:** not performed by this script; do not claim homophone/deletion/acoustic cause until verified.

### `BAC009S0764W0234`

- Reference: 政府  将  制订  境外  投资  总体  战略
- Source audio: `data/ctc_eval/audio/BAC009S0764W0234.wav`; duration: 4.115 s; CTC frames: 101; mean blank posterior: 0.8691; mean maximum frame posterior: 0.9953.
- `framewise_greedy`: 政 府 将 制 定 境 外 投 资 总 体 战 略 (S/D/I=1/0/0, U+FFFD=0).
- `prefix_w16_unpruned`: 政 府 将 制 定 境 外 投 资 总 体 战 略 (S/D/I=1/0/0, U+FFFD=0).
- `prefix_w16_forced_blank_t0.95`: 政 府 将 制 定 境 外 投 资 总 体 战 略 (S/D/I=1/0/0, U+FFFD=0).
- `prefix_w04_forced_blank_t0.95`: 政 府 将 制 定 境 外 投 资 总 体 战 略 (S/D/I=1/0/0, U+FFFD=0).
- **Manual listening status:** not performed by this script; do not claim homophone/deletion/acoustic cause until verified.

## Measurement boundary

Same local sequential Python decoder-only timing as Round 2: one warmup and three repetitions per utterance/config; exclude WAV, Fbank, ONNX inference, disk I/O, BBPE text reconstruction and CER scoring.
No Language Model or hidden token Top-K. Forced Blank is an approximate restriction.
No universal accuracy result, lossless pruning, deployment RTF, speaker-disjoint generalization, or independent training-data overlap audit is claimed.

## Evidence

- `docs/CTC_ROUND3_PROTOCOL.md`
- `results/ctc_round3_unseal_claim.json`
- `results/ctc_test_final_results.json` and `.csv`
- `results/ctc_test_pareto.png`
- `scripts/ctc_round3_audit.py`
