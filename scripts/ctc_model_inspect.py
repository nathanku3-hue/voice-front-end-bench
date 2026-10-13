#!/usr/bin/env python3
"""Inspect a genuine ONNX CTC model before designing its exporter."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect(model_path, tokens_path, bpe_model=None):
    import onnxruntime as ort

    model_path, tokens_path = Path(model_path), Path(tokens_path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
    metadata = session.get_modelmeta()
    rows = tokens_path.read_text(encoding="utf-8-sig").splitlines()
    explicit = all(len(line.rsplit(maxsplit=1)) == 2 and line.rsplit(maxsplit=1)[-1].isdigit() for line in rows)
    if explicit:
        ids = [int(line.rsplit(maxsplit=1)[-1]) for line in rows]
        if sorted(ids) != list(range(len(ids))):
            raise ValueError("Noncontiguous or duplicated token IDs")
        sample_tokens = {int(line.rsplit(maxsplit=1)[-1]): line.rsplit(maxsplit=1)[0]
                         for line in rows[:min(15, len(rows))]}
    else:
        sample_tokens = dict(enumerate(rows[:min(15, len(rows))]))
    output = dict(
        model=str(model_path),
        model_sha256=sha256(model_path),
        model_size=model_path.stat().st_size,
        token_table=str(tokens_path),
        token_table_sha256=sha256(tokens_path),
        vocab_rows=len(rows),
        tokens_have_explicit_ids=explicit,
        sample_tokens=sample_tokens,
        inputs=[dict(name=x.name, type=x.type, shape=x.shape) for x in session.get_inputs()],
        outputs=[dict(name=x.name, type=x.type, shape=x.shape) for x in session.get_outputs()],
        model_metadata=metadata.custom_metadata_map,
        graph_name=metadata.graph_name,
        runtime_version=ort.__version__,
    )
    if bpe_model:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=str(bpe_model))
        output["sentencepiece"] = dict(path=str(bpe_model), sha256=sha256(bpe_model),
                                       piece_size=sp.get_piece_size(),
                                       first_pieces=[sp.id_to_piece(i) for i in range(min(12, sp.get_piece_size()))])
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--bpe-model", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.model, args.tokens, args.bpe_model), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
