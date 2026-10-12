# Round 3 / P3 — Closure Record

**Date:** 2026-10-12

**Status:** **CLOSED — locally validated portfolio and controlled follow-up experiments**. This records the final phase acceptance and supersedes the earlier provisional **21/21 tests / release pending** snapshot in [PROJECT_PLAN.md](PROJECT_PLAN.md). A local Git commit is the closure artifact; remote push, release tag, and public open-source licensing are **not** part of this closure.

## Acceptance: delivered and checkable

- **Round 3 portfolio**: [README](../README.md) with measured endpoint figures, explicit DSP/ASR/TN boundaries, reproduction commands, an [interview grilling guide](INTERVIEW_GRILL.md), an [evidence-driven CV](../CV.md) and [resume bullets](../RESUME_BULLETS.md).
- **Mandarin TN mini-demo**: [21 deterministic rule cases](../scripts/tts_tn_demo.py) with [regression tests](../tests/test_tts_tn_demo.py); visual pause hints only, **no synthesized TTS voice**.
- **Round 2 retrospective and reproducibility**: [one-page closure summary](ROUND2_SUMMARY.md), [full original 30-WAV evaluation](ROUND2_EVALUATION_REPORT.md), [speaker-cluster bootstrap](../results/paired_endpoint_ci.json) and [independent manifest/WAV integrity verifier](../scripts/verify_round2_manifest.py).
- **P0 paired endpoint audit**: [same-utterance finalization stream-time deltas](ROUND3_VALIDATION_REPORT.md), independent of an estimated end-of-speech timestamp. Example: 700→1200 ms, mean **+1179.2 ms**, median **+320 ms**, on 30 matched read-speech WAVs. **Not** measured microphone-to-response latency.
- **P1 feature contract audit**: actual sherpa-onnx `OnlineStream.get_frames` vs independent C++ Fbank, **559 × 80** features. With mismatched config, MAE **0.937197**; with sherpa's diagnostic feature settings matched to C++, MAE **4.58×10⁻⁶**. This is **feature-level**, not C++→ONNX inference equivalence.
- **P3-B hypothetical cost envelope**: [unit-consistent cost model and graph](P3_DECISION_REPORT.md). Assumed deletion penalty λ switchpoints approximately **332.53**, **494.34**, **758.87 ms/deletion** select 300→500→900→1200 ms respectively; **700 ms never minimizes this particular assumed linear cost** on the sample. Neither λ nor finalization stream-time is a real vehicle SLA.
- **P3-A native 2×2 recognition sensitivity**: **eight executed native-sherpa runs**, varying `snip_edges` and Mel high cutoff, with endpoint disabled / Rule2=1200. Matched-to-C++ *native configuration* changed **9/30** hypotheses; 1200ms corpus CER **13.65%→14.89%**, a **+1.24 percentage point** observed difference. Paired 11-speaker bootstrap interval **[−0.58, +3.26] pp** includes zero, so no generalized CER-penalty claim. The C++ feature tensor was **never injected** into the Encoder. [Raw run JSON](../results/p3_frontend_runs/) · [summary and bootstrap intervals](../results/p3_frontend_summary.json).

## Final local verification

The following checks passed on 2026-10-12 in the existing Windows/WSL2 repo:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
# 34 / 34 tests PASS

ctest --test-dir build --output-on-failure
# 1 / 1 test PASS (WSL2 C++ baseline)

git diff --check
# PASS (Git may print nonfatal CRLF conversion warnings)
```

The Python suite checks source custody, four-corner comparison constraints, raw P3 evidence roundtrip, speaker-cluster CI sign, Markdown links, stored endpoint results and TN cases. The original Round 1 C++ implementation is unmodified by this closure.

## Final scope and constraints

1. **Round 2 R2-A/B CLOSED as exploratory.** Optional noise-suppression R2-C deferred. No human-verified speech-end annotations for the original 30-WAV study (28/30 uncertain tails).
2. **Round 3 and P3 CLOSED as local portfolio evidence**, not production vehicle speech, model training, custom ONNX execution, AEC/NS, a real-time-safe audio callback, a business-calibrated loss or a finished spoken TTS system.
3. Code and locally reproducible evidence are in the Git commit associated with this record. **No remote push or Git tag is implied.** This checkout has no first-party LICENSE; committing source is not a license grant.
4. Future work is a **new scope**, not an outstanding acceptance condition. Reopen only against an explicit question and metric, such as speech-end annotated car commands, paced real-device latency, or the C++ feature-to-model interface contract.

**Decision:** Close Round 3 / P3. Preserve all measured limitations instead of inflating the claims to production readiness.
