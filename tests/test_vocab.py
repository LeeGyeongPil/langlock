import json

import pytest

from langlock import UnsupportedTokenizerError, load_vocab
from langlock.errors import LanglockError


def test_fixture_bytes_roundtrip(vocab):
    # every byte exists as a single-byte token and maps back to itself
    for value in range(256):
        token_id = vocab.token_for_bytes(bytes([value]))
        assert token_id is not None, value
        assert vocab.token_bytes(token_id) == bytes([value])
    # added tokens are special and keep their literal text
    assert len(vocab.special_ids) == 3
    assert vocab.token_bytes(1) == b"<think>" and vocab.is_special(1)
    assert vocab.size == 500


def test_roundtrip_matches_the_tokenizers_library(vocab, fixture_path):
    tokenizers = pytest.importorskip("tokenizers")
    tok = tokenizers.Tokenizer.from_file(str(fixture_path))
    for text in ["한국어로 답하세요.", "漢字 and 日本語", "뷁 똠 emoji 😀"]:
        ids = tok.encode(text).ids
        assert b"".join(vocab.token_bytes(i) for i in ids).decode("utf-8") == text


def test_classification_counts(vocab):
    kinds = vocab.kinds
    counts = {name: kinds.count(name) for name in sorted(set(kinds))}
    # snapshot for src/langlock/_data/tiny_bytebpe.json; regenerate with scripts/make_tiny_tokenizer.py
    assert counts == {"complete": 276, "continuation": 109, "invalid": 13, "special": 3, "tail": 99}
    # single-byte tokens: 128 ASCII complete, 64 continuation bytes, 64 lead bytes (tails) -- minus invalid C0/C1/F5..FF
    assert kinds[vocab.token_for_bytes(b"\xa2")] == "continuation"
    assert kinds[vocab.token_for_bytes(b"\xe6")] == "tail"
    assert kinds[vocab.token_for_bytes(b"a")] == "complete"
    assert kinds[vocab.token_for_bytes(b"\xff")] == "invalid"


def test_from_hf_tokenizer_object_matches_json(vocab, fixture_path):
    transformers = pytest.importorskip("transformers")
    hf = transformers.PreTrainedTokenizerFast(tokenizer_file=str(fixture_path))
    from_object = load_vocab(hf)
    from_dict = load_vocab(json.loads(fixture_path.read_text(encoding="utf-8")))
    assert from_object.size == from_dict.size == vocab.size
    assert from_object.special_ids == from_dict.special_ids == vocab.special_ids
    assert all(from_object.token_bytes(i) == vocab.token_bytes(i) for i in range(vocab.size))


def test_rejects_non_bytelevel():
    wordpiece = {"model": {"type": "WordPiece", "vocab": {"[UNK]": 0}}}
    with pytest.raises(UnsupportedTokenizerError, match="byte-level"):
        load_vocab(wordpiece)
    sentencepiece_like = {
        "model": {"type": "BPE", "vocab": {"▁a": 0, "<0x41>": 1}, "byte_fallback": True},
        "pre_tokenizer": None,
        "decoder": {"type": "Sequence", "decoders": [{"type": "ByteFallback"}]},
    }
    with pytest.raises(UnsupportedTokenizerError, match="SentencePiece"):
        load_vocab(sentencepiece_like)
    with pytest.raises(UnsupportedTokenizerError):
        load_vocab(42)


def test_missing_or_broken_file_gives_a_clear_error(tmp_path):
    with pytest.raises(LanglockError, match="not found"):
        load_vocab(tmp_path / "nope.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(LanglockError, match="not a valid tokenizer.json"):
        load_vocab(broken)


def test_stats_as_dict_has_the_documented_keys(vocab):
    stats = vocab.stats("ko").as_dict()
    assert stats["vocab_size"] == 500 and stats["special_tokens"] == 3
    assert stats["dynamic_candidates"] == 109
    assert {"Han", "Hangul", "Common"} <= set(stats["scripts"])
