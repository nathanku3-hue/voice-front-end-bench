#!/usr/bin/env python3
"""Small, deterministic Mandarin vehicle-command text-normalization demo.

No TTS acoustic model, audio synthesis, prosody predictor, or quality evaluation.
Rules intentionally cover only the patterns exercised by DEMO_CASES.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata

_DIGITS = "零一二三四五六七八九"
_POWERS = {3: "千", 2: "百", 1: "十", 0: ""}
_NUMBER = r"\d+(?:\.\d+)?"
_TIME_RE = re.compile(r"(?<![\d:])(?P<h>\d{1,2}):(?P<m>\d{2})(?![\d:])")
_UNIT_RE = re.compile(
    rf"(?<![A-Za-z0-9.])(?P<n>{_NUMBER})\s*(?P<unit>km/h|km|m|°C|%)"
    rf"(?![A-Za-z])", flags=re.IGNORECASE
)
_BARE_NUMBER_RE = re.compile(rf"(?<![A-Za-z0-9.]){_NUMBER}(?![A-Za-z0-9.])")
_TURN_RE = re.compile(r"(前方[零一二三四五六七八九十百千两点]+(?:米|公里))(?=(?:左转|右转|掉头))")
_PUNCT_SHORT_RE = re.compile(r"[，,、；;：:]")
_PUNCT_LONG_RE = re.compile(r"[。！？!?]")


def integer_to_hanzi(number: int) -> str:
    """Read a nonnegative integer in [0, 9999] as a Mandarin cardinal."""
    if not 0 <= number <= 9999:
        raise ValueError("Demo only supports integers from 0 to 9999")
    if number == 0:
        return "零"
    out = []
    pending_zero = False
    for index, char in enumerate(str(number)):
        digit = int(char)
        power = len(str(number)) - index - 1
        if digit == 0:
            pending_zero = bool(out)
            continue
        if pending_zero:
            out.append("零")
            pending_zero = False
        if not (digit == 1 and power == 1 and not out):
            out.append("两" if digit == 2 and power >= 2 else _DIGITS[digit])
        out.append(_POWERS[power])
    return "".join(out)


def number_to_hanzi(text: str) -> str:
    parts = text.split(".")
    if len(parts) > 2 or not parts[0].isdigit() or len(parts[0]) > 4:
        raise ValueError("Only nonnegative numbers up to 9999 are supported")
    result = integer_to_hanzi(int(parts[0]))
    if len(parts) == 2:
        if not parts[1].isdigit() or not 1 <= len(parts[1]) <= 2:
            raise ValueError("Only 1-2 decimal places are supported")
        result += "点" + "".join(_DIGITS[int(c)] for c in parts[1])
    return result


def normalize_text(text: str) -> str:
    """Normalize times, measured quantities, then remaining simple cardinals."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    value = unicodedata.normalize("NFKC", text.strip())
    if re.search(r"[-−]\d", value):
        raise ValueError("Negative numbers are outside this narrow TN demo")

    def replace_time(match: re.Match[str]) -> str:
        h, m = int(match["h"]), int(match["m"])
        if not (0 <= h <= 23 and 0 <= m <= 59):
            raise ValueError(f"Invalid 24-hour clock time: {match.group()}")
        hour = integer_to_hanzi(h)
        if m == 0:
            return f"{hour}点整"
        minute = ("零" + _DIGITS[m] if m < 10 else integer_to_hanzi(m))
        return f"{hour}点{minute}分"

    def replace_unit(match: re.Match[str]) -> str:
        numeral = number_to_hanzi(match["n"])
        unit = match["unit"].lower()
        if match["n"] == "2" and unit in {"km/h", "km", "m"}:
            numeral = "两"  # natural standalone quantity: 两公里 / 两米
        if unit == "km/h":
            return numeral + "公里每小时"
        if unit == "km":
            return numeral + "公里"
        if unit == "m":
            return numeral + "米"
        if unit == "°c":
            return numeral + "摄氏度"
        if unit == "%":
            return "百分之" + numeral
        raise AssertionError("Unreachable unit")

    value = _TIME_RE.sub(replace_time, value)
    value = _UNIT_RE.sub(replace_unit, value)
    return _BARE_NUMBER_RE.sub(lambda m: number_to_hanzi(m.group()), value).replace(",", "，")


def add_pause_tags(normalized: str) -> str:
    """Visual pause hints only: | short, || long; NOT SSML or speech timing."""
    tagged = _TURN_RE.sub(r"\1 | ", normalized)
    tagged = _PUNCT_SHORT_RE.sub(" | ", tagged)
    tagged = _PUNCT_LONG_RE.sub(" || ", tagged)
    return re.sub(r"\s+", " ", tagged).strip()


def transform(text: str) -> dict[str, str]:
    normalized = normalize_text(text)
    return {"input": text, "normalized": normalized, "pause_tags": add_pause_tags(normalized)}


# Curated, deterministic examples; these are rule expectations, NOT a TTS MOS study.
# (raw command, expected normalized text, expected pause-hint display)
DEMO_CASES = (
    ("前方300m右转", "前方三百米右转", "前方三百米 | 右转"),
    ("前方1.5km左转", "前方一点五公里左转", "前方一点五公里 | 左转"),
    ("距离目的地2km", "距离目的地两公里", "距离目的地两公里"),
    ("车速80km/h", "车速八十公里每小时", "车速八十公里每小时"),
    ("限速120km/h", "限速一百二十公里每小时", "限速一百二十公里每小时"),
    ("预计15:30到达", "预计十五点三十分到达", "预计十五点三十分到达"),
    ("请在08:05出发", "请在八点零五分出发", "请在八点零五分出发"),
    ("预计12:00到达", "预计十二点整到达", "预计十二点整到达"),
    ("车外温度26°C", "车外温度二十六摄氏度", "车外温度二十六摄氏度"),
    ("车内温度21℃", "车内温度二十一摄氏度", "车内温度二十一摄氏度"),
    ("电量80%", "电量百分之八十", "电量百分之八十"),
    ("电量降至20%", "电量降至百分之二十", "电量降至百分之二十"),
    ("前方500m有测速", "前方五百米有测速", "前方五百米有测速"),
    ("继续行驶1000m", "继续行驶一千米", "继续行驶一千米"),
    ("前方200m掉头", "前方两百米掉头", "前方两百米 | 掉头"),
    ("距离服务区3.5km", "距离服务区三点五公里", "距离服务区三点五公里"),
    ("空调调到22°C", "空调调到二十二摄氏度", "空调调到二十二摄氏度"),
    ("将音量调到30%", "将音量调到百分之三十", "将音量调到百分之三十"),
    ("前方10m靠右停车", "前方十米靠右停车", "前方十米靠右停车"),
    ("导航到3号停车场", "导航到三号停车场", "导航到三号停车场"),
    ("前方300m右转，注意行人", "前方三百米右转，注意行人", "前方三百米 | 右转 | 注意行人"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", help="Normalize one custom Mandarin command")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable samples")
    args = parser.parse_args()

    if args.text is not None and not args.text.strip():
        parser.error("--text must not be blank")
    try:
        if args.text is not None:
            results = transform(args.text)
        else:
            results = [transform(raw) for raw, _, _ in DEMO_CASES]
    except ValueError as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    elif isinstance(results, dict):
        print(f"Input:      {results['input']}\nNormalized: {results['normalized']}\nPause hints: {results['pause_tags']}")
    else:
        for i, row in enumerate(results, 1):
            print(f"{i:02d} {row['input']}\n   -> {row['normalized']}\n   -> {row['pause_tags']}")
        print(f"\n{len(results)} rule examples; pause hints are visual only (no TTS audio).")


if __name__ == "__main__":
    main()
