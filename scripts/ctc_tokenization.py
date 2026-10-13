"""Token-table and icefall Byte-BPE utilities for the CTC study.

The byte alphabet matches icefall/icefall/byte_utils.py (derived from fairseq):
ASCII 32..126; other byte values map to NFKC-stable code points starting at U+0100.
"""
from pathlib import Path
import unicodedata


def load_tokens(path):
    rows = Path(path).read_text(encoding="utf-8-sig").splitlines()
    if not rows or any(not row for row in rows):
        raise ValueError("Empty token table or blank token line")
    explicit = [row.rsplit(maxsplit=1) for row in rows]
    if all(len(parts) == 2 and parts[1].isdigit() for parts in explicit):
        table = {}
        for piece, index in explicit:
            i = int(index)
            if i in table:
                raise ValueError("Duplicate token ID")
            table[i] = piece
        if sorted(table) != list(range(len(rows))):
            raise ValueError("Token IDs must be contiguous 0..V-1")
        return [table[i] for i in range(len(rows))]
    if any(len(parts) == 2 and parts[1].isdigit() for parts in explicit):
        raise ValueError("Mixed explicit and implicit token-table formats")
    return rows


def _byte_alphabet():
    candidates = [chr(c) for c in range(256, 2048)
                  if unicodedata.normalize("NFKC", chr(c)) == chr(c)]
    if len(candidates) < 161:
        raise RuntimeError("Insufficient NFKC-stable byte characters")
    table = {}
    for byte in range(256):
        if 32 <= byte <= 126:
            table[byte] = chr(byte)
        else:
            table[byte] = candidates[byte if byte < 32 else byte - 95]
    if len(set(table.values())) != 256:
        raise RuntimeError("Byte alphabet must have unique symbols")
    return table


BYTE_ALPHABET = _byte_alphabet()
BCHAR_TO_BYTE = {ch: byte for byte, ch in BYTE_ALPHABET.items()}


def decode_encoded_bytes(encoded, *, errors="strict"):
    """Recover UTF-8 from icefall's NFKC-stable byte alphabet.

    SentencePiece converts ▁ word-boundary markers to literal spaces.
    In strict mode fail on invalid UTF-8. With errors='replace', malformed
    sequences turn into U+FFFD, *not silently deleted*; the caller must
    count these replacement characters as a diagnostic in its CER report.
    """
    if errors not in ("strict", "replace"):
        raise ValueError("Use UTF-8 error policy 'strict' or 'replace'")
    try:
        return bytes(BCHAR_TO_BYTE[ch] for ch in encoded).decode("utf-8", errors=errors)
    except (KeyError, UnicodeDecodeError) as exc:
        raise ValueError("Invalid byte-BPE output; inspect emitted tokens") from exc


def build_detokenizer(tokens, *, kind="char", bpe_model=None, byte_errors="strict"):
    if kind == "char":
        if bpe_model is not None or byte_errors != "strict":
            raise ValueError("Character tokenizer cannot take BPE model or byte error policy")
        def detokenize(ids):
            pieces = [tokens[i] for i in ids]
            if any(piece.startswith("<") and piece.endswith(">") for piece in pieces):
                raise ValueError("Special token emitted; review model-specific policy")
            return "".join(pieces)
        return detokenize
    if kind != "bbpe":
        raise ValueError(f"Unsupported tokenizer: {kind}")
    if bpe_model is None:
        raise ValueError("--bpe-model required for bbpe")
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=str(bpe_model))
    if sp.get_piece_size() != len(tokens) or any(sp.id_to_piece(i) != p for i, p in enumerate(tokens)):
        raise ValueError("SentencePiece model and tokens.txt ID mapping differ")
    def detokenize(ids):
        if any(tokens[i].startswith("<") and tokens[i].endswith(">") for i in ids):
            raise ValueError("Special token emitted; review model-specific policy")
        # SentencePiece Python bindings accept a list of IDs, not a tuple.
        # Beam prefixes remain immutable tuples until this final boundary.
        return decode_encoded_bytes(sp.decode(list(ids)), errors=byte_errors)
    return detokenize
