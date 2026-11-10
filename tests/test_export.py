import re

import pytest

from langlock import Policy
from langlock.errors import LanglockError
from langlock.export import to_gbnf, to_logit_bias, to_regex
from langlock.vocab import CONTINUATION

GBNF_KO = (
    "# langlock: no Han, Kana characters\n"
    "root ::= char*\n"
    "char ::= [^\\u2E80-\\u2FDF\\u3041-\\u3096\\u309D-\\u309F\\u30A1-\\u30FA\\u30FD-\\u30FF\\u31F0-\\u31FF"
    "\\u3400-\\u4DBF\\u4E00-\\u9FFF\\uF900-\\uFAFF\\uFF66-\\uFF9F\\U0001B000-\\U0001B16F"
    "\\U00020000-\\U0002A6DF\\U0002A700-\\U0002EBEF\\U0002F800-\\U0002FA1F\\U00030000-\\U000323AF]\n"
)


def test_llamacpp_logit_bias_equals_static_set(vocab, mask):
    entries = to_logit_bias(vocab, "ko", style="llamacpp")
    assert {token_id for token_id, _ in entries} == set(mask.static_ids())
    assert all(value is False for _, value in entries)
    ids = {token_id for token_id, _ in entries}
    # orphan continuation bytes are NOT in the static list: banning them would trap half-written characters
    assert not any(vocab.kinds[i] == CONTINUATION for i in ids)
    assert vocab.token_for_bytes(b"\xe6") in ids
    soft = to_logit_bias(vocab, "ko", bias=-5.0)
    assert {v for _, v in soft} == {-5.0}


def test_openai_bias_range(vocab):
    table = to_logit_bias(vocab, "ko", style="openai")
    assert set(table.values()) == {-100.0} and all(key.isdigit() for key in table)
    assert set(to_logit_bias(vocab, "ko", style="openai", bias=-30).values()) == {-30.0}
    with pytest.raises(ValueError, match="between -100 and 100"):
        to_logit_bias(vocab, "ko", style="openai", bias=-101)
    with pytest.raises(ValueError, match="unknown style"):
        to_logit_bias(vocab, "ko", style="anthropic")


def test_regex_rejects_han_accepts_korean():
    pattern = re.compile(to_regex("ko"))
    for ok in ["안녕하세요, 반갑습니다.", "Hello, world! 123", "좋아요 😀🎉", "뷁 똠", "한국어 (English)"]:
        assert pattern.fullmatch(ok), ok
    for bad in ["이것은 漢字 입니다", "中文", "ひらがな"]:
        assert not pattern.fullmatch(bad), bad


def test_gbnf_snapshot():
    text = to_gbnf("ko")
    assert text == GBNF_KO
    assert "[^" in text and "\\u4E00-\\u9FFF" in text


def test_grammar_exports_refuse_soft_policy_and_note_exemptions():
    with pytest.raises(LanglockError, match="hard ban"):
        to_gbnf(Policy.preset("ko", mode="soft"))
    with pytest.raises(LanglockError, match="hard ban"):
        to_regex(Policy.preset("ko", mode="soft"))
    assert "cannot be expressed" in to_gbnf(Policy.preset("ko", exempt_think=True))
