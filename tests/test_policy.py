import pytest

from langlock import Policy, PolicyError, UnknownPresetError
from conftest import byte_ids


def test_presets_and_unknown_preset():
    assert Policy.preset("ko").ban_scripts == {"Han", "Kana"}
    assert Policy.preset("en").ban_scripts == {"Han", "Kana", "Hangul", "Cyrillic"}
    assert Policy.preset("ru").ban_scripts == {"Han", "Kana", "Hangul"}
    assert Policy.preset("ko", mode="soft", bias=-3).mode == "soft"
    with pytest.raises(UnknownPresetError, match="available presets: en, ko, ru"):
        Policy.preset("fr")


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"ban_scripts": frozenset({"Klingon"})}, "unknown script"),
        ({"mode": "medium"}, "mode must be"),
        ({"bias": 3.0}, "bias must be"),
        ({"tail_ban_ratio": 0.0}, "tail_ban_ratio"),
    ],
)
def test_invalid_policy_values_explain_the_fix(kwargs, message):
    with pytest.raises(PolicyError, match=message):
        Policy(**kwargs)


def test_code_fence_exempt(vocab):
    policy = Policy.preset("ko", exempt_code_blocks=True)
    mask = vocab.script_mask(policy)
    han_lead = vocab.token_for_bytes(b"\xe6")
    tick = vocab.token_for_bytes(b"`")
    state = mask.new_state()
    assert mask.decide(state).is_banned(han_lead)
    for _ in range(3):  # opening fence
        state.feed(tick)
    assert mask.decide(state).exempt and not mask.decide(state).is_banned(han_lead)
    for _ in range(3):  # closing fence
        state.feed(tick)
    assert not mask.decide(state).exempt and mask.decide(state).is_banned(han_lead)


def test_fence_is_ignored_unless_requested(vocab, mask):
    state = mask.new_state()
    for _ in range(3):
        state.feed(vocab.token_for_bytes(b"`"))
    assert mask.decide(state).is_banned(vocab.token_for_bytes(b"\xe6"))


def test_think_exempt(vocab):
    policy = Policy.preset("ko", exempt_think=True)
    mask = vocab.script_mask(policy)
    han_lead = vocab.token_for_bytes(b"\xe6")
    state = mask.new_state()
    state.feed(1)  # <think>
    assert mask.decide(state).exempt
    state.feed(han_lead)  # the model may reason in Chinese here
    assert state.pending == b"\xe6"
    state.feed(2)  # </think> ends the exemption and drops the half-written character
    decision = mask.decide(state)
    assert not decision.exempt and state.pending == b"" and decision.is_banned(han_lead)


def test_think_tags_written_as_plain_bytes_also_count(vocab):
    policy = Policy.preset("ko", exempt_think=True)
    mask = vocab.script_mask(policy)
    state = mask.new_state()
    for token_id in byte_ids(vocab, b"<think>"):
        state.feed(token_id)
    assert state.exempt
