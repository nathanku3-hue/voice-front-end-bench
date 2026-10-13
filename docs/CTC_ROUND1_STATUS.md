# CTC-Decoding-Bench — Round 1 closure record

**Date:** 2026-10-12. **Status: CLOSED for the model/data/logits/DEV Greedy baseline gate.**
**The separate 50-utterance Test split was deliberately unscored at Round 1 closure.** The later log-domain Prefix Beam/Blank Dev study is now documented in [CTC Round 2 Status](CTC_ROUND2_STATUS.md); the Test set, Pareto figure and final bad-case audit remain Round 3 tasks.

## Scope and verified evidence

1. **Genuine CTC model, not the existing Transducer.** Official k2-fsa offline
   `sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03` model. The downloaded release
   archive SHA-256 is `f3ad1814fea34c407eab0cc3df6f6b625419ac9a60d8aebd8efe772a8e85ef67`,
   matching the SHA-256 of k2-fsa's own pinned Hugging Face release file.
   `model.int8.onnx` SHA-256: `e291b9c468b651e2697caa09bc684326c3addc6a019e78eb537cfd1a8248ca07`.
   `tokens.txt`: `6fed8c6c248516f38e7faa19404b57413e8ce259f1cbc1fa4aebc86eac32fdfd`;
   `bbpe.model`: `503204e0690eff065e30d0e01898c9ab06d0e6dc376a741eb6846198f95b2f82`.
   The original model archive and binaries remain in WSL's external model cache,
   **not checked into Git**. A clear standalone redistribution license for these
   **specific model weights** has not been independently established; do not
   claim rights to redistribute them, or conflate their license with the code's.

2. **Inspected ONNX tensor interface, not guessed.** CPU ONNX Runtime 1.23.2
   loaded the real file with `x: float32[1,T,80]`, `x_lens: int64[1]`
   → `log_probs: float32[1,T_out,1000]`, `log_probs_len: int64[1]`.
   Log-probability rows are validated; returned output lengths strip padded
   frames. Blank ID **0** was confirmed against the downloaded token table,
   which has **1,000 indexed SentencePiece Byte-BPE tokens**.
   The frontend currently uses `torchaudio.compliance.kaldi.fbank`, 16kHz
   normalized PCM16, 80 bands, 25/10ms, `snip_edges=false`, high frequency
   offset `-400`, dither zero. Model input parity with sherpa's native frontend,
   C++ Fbank insertion and user-facing latency are **not** claimed.

3. **Single real-utterance pilot.** Original AISHELL-1 ID
   `BAC009S0764W0350` produced 447 Fbank frames and **110 × 1000** log
   probabilities. Greedy was normalized-text equal to native sherpa and matched
   its 15-character reference (pilot CER 0%); this single example is not
   a performance estimate. Real model inspection: `scripts/ctc_model_inspect.py`.

4. **New frozen data, Project 1 untouched.** `data/ctc_eval/` contains
   80 independently source-matched AISHELL-1 **test-split** WAVs with
   official-transcript cross-check and individual WAV SHA-256, from 20 speakers,
   explicitly excluding all 30 former endpoint evaluation IDs. Deterministic
   study-specific **Dev=30** (tuning) / **Test=50** (held-out) split, seed
   `20261012`, at *utterance* level only; the same speaker can occur in
   both splits. This is not a claim of speaker-independent generalization or
   of zero overlap with the checkpoint's unknown original training data.
   Overall manifest SHA-256:
   `978f40d0e2e18d4c4c4c1071703cf86c3ae884a01ac566bd16eda927a9383f8b`.
   Original 30-sample manifest was left unmodified. One candidate WAV with
   an invalid duration/format was rejected, not counted.

5. **80/80 real Logits exported and independently audited.** Dev=30 and
   Test=50 arrays reside under ignored `build/ctc_round1_dev/log_probs` and
   `build/ctc_round1_test/log_probs`. SHA-256 per matrix, WAV identity and
   tensor dimensions are recorded in `results/ctc_logit_inventory.json`.
   `scripts/ctc_round1_audit.py` rechecked source IDs, manifest hashes,
   audio SHA, model/token/BPE SHA, each .npy SHA and log-probability contract;
   **PASS** in `results/ctc_round1_audit.json`. This audit does **not** decode
   or compute CER on held-out Test.

## DEV Greedy baseline (exploratory, not held-out performance)

| Metric | Measured value |
|---|---:|
| Dev utterances | **30** |
| Normalized Chinese reference characters | **442** |
| Corpus substitutions / deletions / insertions | **12 / 0 / 1** |
| Corpus CER | **13 / 442 = 2.9412%** |
| Malformed UTF-8 Byte-BPE output | **2 utterances, 2 replacement characters** |
| Sherpa native normalized transcript agreement | **28 / 30** |
| Native sherpa corpus S/D/I | **10 / 2 / 1** |
| Native sherpa corpus CER | **13 / 442 = 2.9412%** |

The strict Byte-BPE decoder correctly refused malformed UTF-8 on the complete
Dev batch. A **declared `--byte-errors replace` policy** replaces each
malformed sequence with U+FFFD and **counts** it in the CER. Native sherpa
instead drops the corresponding incomplete decoded characters in exactly two
utterances (`BAC009S0912W0412` and `BAC009S0769W0340`).
Thus the **corpus CER is numerically equal** in this small Dev sample, but the
S/D categories differ. Neither the matched CER nor 28/30 equal normalized
hypotheses establish frame/logit parity between torchaudio and sherpa's native
frontends. See `results/ctc_baseline_dev.json` and
`results/ctc_native_audit_dev.json`.

These results are **exploratory and small**. The upstream release publishes its
own AISHELL-1 **WER** from a different full test protocol; do not compare that
figure to this study's **CER** as if they were directly equivalent.

## Inspectable files

- `scripts/fetch_ctc_eval.py`: separate dataset creation (refuses to
  overwrite frozen manifests), reuse existing verified AISHELL source API.
- `data/ctc_eval/{manifest,dev,test}.jsonl`, `provenance.json`:
  frozen transcript/WAV hashes and selection protocol.
- `scripts/ctc_model_inspect.py`: actual ONNX input/output and hash audit.
- `scripts/ctc_export_logits.py`: CTC model input contract, real inference,
  explicit output lengths and .npy artifacts.
- `scripts/ctc_tokenization.py`: indexed token table and Byte-BPE decoding,
  strict vs replacement policy.
- `scripts/ctc_greedy_baseline.py`: deterministic Greedy and S/D/I/CER.
- `scripts/ctc_native_compare.py` (single pilot),
  `scripts/ctc_native_audit.py` (Dev native comparison).
- `scripts/ctc_round1_audit.py` and `results/ctc_round1_audit.json`:
  frozen asset integrity, **Test remains unscored**.
- `results/ctc_logit_inventory.json`: every model-output SHA, including
  sealed Test tensor identities, without Test hypotheses or CER.
- `tests/test_ctc_*.py`: offline contract, data, and CTC unit tests.

## Reproduction on the recorded WSL2 host

Python environment: `/home/enovo/.venvs/voice-bench/bin/python`; recorded
runtime: Python 3.12, NumPy 2.3.5, torch/torchaudio 2.10.0+cpu,
sherpa-onnx 1.13.8, ONNX Runtime 1.23.2, SentencePiece 0.2.1.
See `requirements-ctc.txt` for new CTC-specific dependencies, and install
matching **CPU** PyTorch/torchaudio wheels separately if recreating the venv.

From the repo root in WSL2:

```bash
PY="$HOME/.venvs/voice-bench/bin/python"
MODEL="$HOME/.cache/voice-front-end/models/sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03"
"$PY" scripts/ctc_model_inspect.py --model "$MODEL/model.int8.onnx" --tokens "$MODEL/tokens.txt" --bpe-model "$MODEL/bbpe.model"
"$PY" scripts/ctc_export_logits.py --model-dir "$MODEL" --source-manifest data/ctc_eval/dev.jsonl --limit 30 --output-root build/ctc_round1_dev
"$PY" scripts/ctc_export_logits.py --model-dir "$MODEL" --source-manifest data/ctc_eval/test.jsonl --limit 50 --output-root build/ctc_round1_test
"$PY" scripts/ctc_greedy_baseline.py --manifest build/ctc_round1_dev/manifest.jsonl --tokens "$MODEL/tokens.txt" --blank-id 0 --tokenizer bbpe --bpe-model "$MODEL/bbpe.model" --byte-errors replace --model-info build/ctc_round1_dev/model.json --output build/ctc_round1_dev/greedy_replace.json
"$PY" scripts/ctc_native_audit.py --model-dir "$MODEL" --source-manifest data/ctc_eval/dev.jsonl --exported-report build/ctc_round1_dev/greedy_replace.json --output build/ctc_round1_dev/native_audit.json
"$PY" scripts/ctc_round1_audit.py --model-dir "$MODEL"
"$PY" -m unittest discover -s tests -p 'test_*.py' -q
```

Source audio and external weights are deliberately **not** committed. To
recreate the dataset from upstream rather than reuse frozen WAVs, point
`fetch_ctc_eval.py` at a **fresh output directory**; mirror data can change,
so exact wav SHA identities must be compared rather than assumed. A fresh
regeneration cannot silently overwrite the frozen manifest.

## Future gates

**Round 2:** implement log-domain CTC Prefix Beam Search and exhaustive
alignment oracle; use ONLY Dev to explore Beam Width and approximate Blank
Thresholding. Explicitly define detokenization policy and how invalid UTF-8
is treated across decoders.

**Round 3:** freeze decoder/threshold selection, decode 50 untouched Test
utterances once, measure decoder-only runtime under a declared timing protocol,
draw Pareto frontier, review bad cases, then revise final resume bullets.
No Beam CER benefit, speedup, Pareto optimum, speaker-generalized gain,
real-device latency, or Test CER is claimed in Round 1.
