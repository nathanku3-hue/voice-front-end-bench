"""Round 2 CER utilities; codepoint normalization documented in output."""
import unicodedata
from dataclasses import dataclass

NORMALIZATION = "NFKC + casefold + strip Unicode punctuation and spaces"

def normalize(text: str) -> str:
    src = unicodedata.normalize("NFKC", text).casefold()
    return "".join(ch for ch in src if not (ch.isspace() or unicodedata.category(ch).startswith(("P", "Z"))))

@dataclass
class Counts:
    s: int
    d: int
    i: int
    n: int

    @property
    def cer(self):
        if not self.n:
            raise ValueError("empty reference")
        return (self.s + self.d + self.i) / self.n

def evaluate(reference: str, hypothesis: str) -> Counts:
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        raise ValueError("reference empty after normalization")
    n, m = len(ref), len(hyp)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for a in range(n+1):
        dp[a][0] = a
    for b in range(m+1):
        dp[0][b] = b
    for a in range(1, n+1):
        for b in range(1, m+1):
            dp[a][b] = min(dp[a-1][b]+1, dp[a][b-1]+1, dp[a-1][b-1]+(ref[a-1]!=hyp[b-1]))
    a, b, s, d, i = n, m, 0, 0, 0
    while a or b:
        if a and b and ref[a-1] == hyp[b-1] and dp[a][b] == dp[a-1][b-1]:
            a -= 1; b -= 1
        elif a and b and dp[a][b] == dp[a-1][b-1]+1:
            a -= 1; b -= 1; s += 1
        elif a and dp[a][b] == dp[a-1][b]+1:
            a -= 1; d += 1
        else:
            b -= 1; i += 1
    return Counts(s, d, i, n)

def corpus(items: list[Counts]) -> dict:
    s = sum(x.s for x in items)
    d = sum(x.d for x in items)
    i = sum(x.i for x in items)
    n = sum(x.n for x in items)
    return dict(labeled_utterances=len(items), substitutions=s, deletions=d, insertions=i,
                reference_chars=n, cer=(s+d+i)/n if n else None)
