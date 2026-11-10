"""Byte-level helpers: the GPT-2 byte<->unicode table and incremental UTF-8 splitting.

Byte-level BPE vocabularies write every token as a string of "stand-in"
characters, one per byte.  Recovering the real bytes lets us reason about
*partial* UTF-8 characters, which is the whole point of this library.
"""

from __future__ import annotations

from functools import lru_cache

# Highest code point whose lead byte announces the given sequence length.
_MAX_CODEPOINT = 0x10FFFF


def bytes_to_unicode() -> dict[int, str]:
    """Return the GPT-2 table mapping each byte value to a printable character."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    chars = printable[:]
    extra = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            chars.append(256 + extra)
            extra += 1
    return {b: chr(c) for b, c in zip(printable, chars)}


BYTE_TO_CHAR: dict[int, str] = bytes_to_unicode()
CHAR_TO_BYTE: dict[str, int] = {c: b for b, c in BYTE_TO_CHAR.items()}


def decode_piece(piece: str) -> bytes | None:
    """Turn a vocabulary string into real bytes, or ``None`` if it is not byte-level."""
    try:
        return bytes(CHAR_TO_BYTE[ch] for ch in piece)
    except KeyError:
        return None


def is_continuation(byte: int) -> bool:
    """True for UTF-8 continuation bytes (10xxxxxx)."""
    return 0x80 <= byte <= 0xBF


def _sequence_length(lead: int) -> int:
    """Length of the UTF-8 sequence a lead byte starts, or 0 if it cannot be a lead."""
    if lead < 0x80:
        return 1
    if 0xC2 <= lead <= 0xDF:
        return 2
    if 0xE0 <= lead <= 0xEF:
        return 3
    if 0xF0 <= lead <= 0xF4:
        return 4
    return 0


def _valid_second_byte(lead: int, second: int) -> bool:
    """Reject overlong forms, surrogates and code points above U+10FFFF."""
    if lead == 0xE0:
        return 0xA0 <= second <= 0xBF
    if lead == 0xED:
        return 0x80 <= second <= 0x9F
    if lead == 0xF0:
        return 0x90 <= second <= 0xBF
    if lead == 0xF4:
        return 0x80 <= second <= 0x8F
    return is_continuation(second)


def split_utf8(data: bytes) -> tuple[list[int], bytes, bool]:
    """Split bytes into complete code points and an unfinished tail.

    Returns ``(code_points, tail, ok)``.  ``tail`` is a valid *prefix* of a
    multi-byte character that needs more bytes.  ``ok`` is False when the bytes
    can never become valid UTF-8 (orphan continuation byte, bad lead byte, ...).
    """
    cps: list[int] = []
    i, n = 0, len(data)
    while i < n:
        lead = data[i]
        need = _sequence_length(lead)
        if need == 0:
            return cps, b"", False
        end = min(i + need, n)
        chunk = data[i:end]
        if len(chunk) >= 2 and not _valid_second_byte(lead, chunk[1]):
            return cps, b"", False
        if any(not is_continuation(b) for b in chunk[2:]):
            return cps, b"", False
        if end - i < need:
            return cps, bytes(chunk), True
        cps.append(ord(chunk.decode("utf-8")))
        i = end
    return cps, b"", True


@lru_cache(maxsize=4096)
def tail_range(tail: bytes) -> tuple[int, int]:
    """Smallest and largest code point an unfinished tail can still complete to."""
    lead = tail[0]
    need = _sequence_length(lead)
    bits = {1: 0x7F, 2: 0x1F, 3: 0x0F, 4: 0x07}[need]
    low = high = lead & bits
    for index in range(need):
        if index == 0:
            continue
        if index < len(tail):
            low = (low << 6) | (tail[index] & 0x3F)
            high = (high << 6) | (tail[index] & 0x3F)
        else:
            low = low << 6  # fill with 0x80 -> payload 000000
            high = (high << 6) | 0x3F  # fill with 0xBF -> payload 111111
    floor = {1: 0, 2: 0x80, 3: 0x800, 4: 0x10000}[need]
    return max(low, floor), min(high, _MAX_CODEPOINT)
