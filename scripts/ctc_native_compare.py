#!/usr/bin/env python3
"""Cross-check offline sherpa-onnx CTC native recognition vs exported greedy output.

The native and independently exported paths can have different frontend defaults,
so disagreement is reported rather than silently treated as a decoder bug.
"""
import argparse
import json
from pathlib import Path

from asr_metrics import evaluate, normalize
from ctc_export_logits import read_wav, sha256


def compare(model_dir, wav, greedy_report, output, *, utterance_id=None):
    import sherpa_onnx
    model_dir = Path(model_dir).resolve()
    wav = Path(wav).resolve()
    greedy_report = json.loads(Path(greedy_report).read_text(encoding="utf-8"))
    items = greedy_report.get("utterances", [])
    if utterance_id is not None:
        items = [item for item in items if item["id"] == utterance_id]
    if len(items) != 1:
        raise ValueError("Native comparison needs exactly one selected Greedy utterance")
    model_path = model_dir / "model.int8.onnx"
    token_path = model_dir / "tokens.txt"
    if sha256(model_path) != greedy_report["model"]["model_sha256"]:
        raise ValueError("Native and exported model fingerprints differ")
    recognizer = sherpa_onnx.OfflineRecognizer.from_zipformer_ctc(
        model=str(model_path), tokens=str(token_path),
        num_threads=2, sample_rate=16000, feature_dim=80, decoding_method="greedy_search")
    stream = recognizer.create_stream()
    stream.accept_waveform(16000, read_wav(wav))
    recognizer.decode_stream(stream)
    native = stream.result.text
    reported = items[0]
    reference = reported["reference"]
    edits = evaluate(reference, native)
    result = dict(model_sha256=sha256(model_path), wav_sha256=sha256(wav),
                  reference=reference, exported_greedy=reported["hypothesis"],
                  native_greedy=native,
                  normalized_hypotheses_equal=normalize(native) == normalize(reported["hypothesis"]),
                  native_edits=dict(s=edits.s, d=edits.d, i=edits.i, n=edits.n, cer=edits.cer),
                  note="Native sherpa and torchaudio frontend inputs not proven identical; matching text is only a smoke check")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--greedy-report", type=Path, required=True)
    parser.add_argument("--id", help="Utterance ID when report contains more than one")
    parser.add_argument("--output", type=Path, default=Path("build/ctc_pilot/native_compare.json"))
    args = parser.parse_args()
    print(json.dumps(compare(args.model_dir, args.wav, args.greedy_report, args.output, utterance_id=args.id),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
