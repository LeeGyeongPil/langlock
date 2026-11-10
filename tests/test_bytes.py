from langlock._bytes import BYTE_TO_CHAR, bytes_to_unicode, decode_piece, split_utf8, tail_range


def test_bytes_to_unicode_is_a_bijection_over_256_bytes():
    table = bytes_to_unicode()
    assert len(table) == 256 and len(set(table.values())) == 256
    assert table[ord("a")] == "a" and table[0x20] == "Ġ"  # the famous "G with dot" for space


def test_decode_piece_roundtrip_and_rejects_foreign_characters():
    piece = "".join(BYTE_TO_CHAR[b] for b in "한".encode())
    assert decode_piece(piece) == "한".encode()
    assert decode_piece("한") is None  # a real Hangul character is not a byte stand-in


def test_split_utf8_complete_tail_and_invalid():
    assert split_utf8("한a".encode()) == ([0xD55C, 0x61], b"", True)
    assert split_utf8(b"a\xeb\xb7") == ([0x61], b"\xeb\xb7", True)
    assert split_utf8(b"\xa2")[2] is False  # orphan continuation byte
    assert split_utf8(b"\xe6a")[2] is False  # lead byte followed by a non-continuation byte
    assert split_utf8(b"\xc0\x80")[2] is False  # overlong
    assert split_utf8(b"\xed\xa0\x80")[2] is False  # surrogate
    assert split_utf8(b"\xf5")[2] is False  # above U+10FFFF


def test_tail_range_covers_reachable_code_points():
    assert tail_range(b"\xe4") == (0x4000, 0x4FFF)
    assert tail_range(b"\xe4\xb7") == (0x4DC0, 0x4DFF)
    assert tail_range(b"\xe0") == (0x800, 0xFFF)  # overlong floor
    assert tail_range(b"\xc3") == (0xC0, 0xFF)
    lo, hi = tail_range(b"\xf4")
    assert lo == 0x100000 and hi == 0x10FFFF
