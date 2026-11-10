"""Code point -> writing system ("script") table.

This is a small hand-written subset of Unicode ``Scripts.txt``: only the writing
systems langlock can ban.  Everything else (digits, punctuation, symbols,
emoji, whitespace, CJK punctuation ...) is ``Common`` and is never banned.
"""

from __future__ import annotations

from bisect import bisect_right

COMMON = "Common"

_RAW: dict[str, tuple[tuple[int, int], ...]] = {
    "Han": (
        (0x2E80, 0x2FDF),  # CJK radicals, Kangxi radicals
        (0x3400, 0x4DBF),  # Extension A
        (0x4E00, 0x9FFF),  # Unified ideographs
        (0xF900, 0xFAFF),  # Compatibility ideographs
        (0x20000, 0x2A6DF),  # Extension B
        (0x2A700, 0x2EBEF),  # Extensions C-F
        (0x2F800, 0x2FA1F),  # Compatibility supplement
        (0x30000, 0x323AF),  # Extensions G-H
    ),
    "Hangul": (
        (0x1100, 0x11FF),
        (0x3130, 0x318F),
        (0xA960, 0xA97F),
        (0xAC00, 0xD7AF),
        (0xD7B0, 0xD7FF),
        (0xFFA0, 0xFFDC),
    ),
    "Kana": (
        (0x3041, 0x3096),
        (0x309D, 0x309F),
        (0x30A1, 0x30FA),
        (0x30FD, 0x30FF),
        (0x31F0, 0x31FF),
        (0xFF66, 0xFF9F),
        (0x1B000, 0x1B16F),
    ),
    "Latin": (
        (0x0041, 0x005A),
        (0x0061, 0x007A),
        (0x00AA, 0x00AA),
        (0x00BA, 0x00BA),
        (0x00C0, 0x00D6),
        (0x00D8, 0x00F6),
        (0x00F8, 0x02B8),
        (0x1E00, 0x1EFF),
        (0x2C60, 0x2C7F),
        (0xA720, 0xA7FF),
        (0xFF21, 0xFF3A),
        (0xFF41, 0xFF5A),
    ),
    "Cyrillic": (
        (0x0400, 0x052F),
        (0x1C80, 0x1C8F),
        (0x2DE0, 0x2DFF),
        (0xA640, 0xA69F),
    ),
    "Greek": ((0x0370, 0x03FF), (0x1F00, 0x1FFF)),
    "Hebrew": ((0x0590, 0x05FF),),
    "Arabic": ((0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)),
    "Devanagari": ((0x0900, 0x097F),),
    "Thai": ((0x0E01, 0x0E5B),),
}

SCRIPT_NAMES: tuple[str, ...] = tuple(_RAW)


def _build() -> tuple[list[int], list[int], list[str]]:
    rows = sorted((lo, hi, name) for name, ranges in _RAW.items() for lo, hi in ranges)
    for (_, prev_hi, _), (lo, _, _) in zip(rows, rows[1:]):
        assert lo > prev_hi, "script ranges must not overlap"
    return [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]


_STARTS, _ENDS, _NAMES = _build()


def script_of(codepoint: int) -> str:
    """Name of the writing system a code point belongs to (``Common`` if none)."""
    i = bisect_right(_STARTS, codepoint) - 1
    if i >= 0 and codepoint <= _ENDS[i]:
        return _NAMES[i]
    return COMMON


def ranges_for(scripts: "frozenset[str] | set[str] | tuple[str, ...]") -> list[tuple[int, int]]:
    """Sorted, merged code point ranges covering the given scripts."""
    spans = sorted((lo, hi) for name in scripts for lo, hi in _RAW[name])
    merged: list[tuple[int, int]] = []
    for lo, hi in spans:
        if merged and lo <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def banned_fraction(low: int, high: int, scripts: "frozenset[str]") -> float:
    """Fraction of code points in ``[low, high]`` that belong to a banned script."""
    total = high - low + 1
    if total <= 0:
        return 0.0
    hit = 0
    for lo, hi in ranges_for(scripts):
        overlap = min(hi, high) - max(lo, low) + 1
        if overlap > 0:
            hit += overlap
    return hit / total
