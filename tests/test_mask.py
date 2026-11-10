import pytest

from langlock import Policy, ScriptMask
from conftest import byte_ids


def feed_all(mask, state, ids):
    for token_id in ids:
        state.feed(token_id)


def test_han_fragments_blocked(vocab, mask):
    han = byte_ids(vocab, "漢".encode())  # E6 BC A2
    state = mask.new_state()
    assert mask.decide(state).is_banned(han[0])  # E6 alone can only become a Han character
    assert mask.decide(state).is_banned(vocab.token_for_bytes("漢".encode()))  # whole-character token
    assert not mask.decide(state).is_banned(vocab.token_for_bytes("한".encode()))
    # even if something forced the lead byte in, the completion is refused
    state.feed(han[0])
    decision = mask.decide(state)
    assert decision.allowed is not None
    assert han[1] not in decision.allowed
    # a Hangul character that shares nothing with it is still fine afterwards
    assert not mask.decide(mask.new_state()).is_banned(vocab.token_for_bytes("국".encode()))


def test_rare_hangul_fragments_kept(vocab, mask):
    pieces = byte_ids(vocab, "뷁".encode())  # EB B7 81
    state = mask.new_state()
    for step, token_id in enumerate(pieces):
        assert not mask.decide(state).is_banned(token_id), f"piece {step} was blocked"
        state.feed(token_id)
    assert state.pending == b""
    assert b"".join(vocab.token_bytes(i) for i in pieces).decode("utf-8") == "뷁"


def test_no_orphan_continuation(vocab, mask):
    state = mask.new_state()
    decision = mask.decide(state)
    from langlock.vocab import CONTINUATION

    continuation_ids = [i for i, kind in enumerate(vocab.kinds) if kind == CONTINUATION]
    assert continuation_ids and all(decision.is_banned(i) for i in continuation_ids)
    # while a character is unfinished, a fresh ASCII/lead token may not interrupt it
    state.feed(byte_ids(vocab, b"\xeb")[0])
    pending = mask.decide(state)
    assert pending.is_banned(vocab.token_for_bytes(b"a"))
    assert pending.is_banned(vocab.token_for_bytes(b"\xec"))
    assert not pending.is_banned(vocab.token_for_bytes(b"\xb7"))


def test_tail_ban_ratio_e4_banned_e3_allowed(vocab, mask):
    base = mask.decide(mask.new_state())
    assert base.is_banned(vocab.token_for_bytes(b"\xe4"))  # 98% of U+4000..4FFF is Han
    assert not base.is_banned(vocab.token_for_bytes(b"\xe3"))  # U+3000..3FFF has Hangul jamo, symbols
    # the threshold is configurable: with ratio 1.0 even E4 stays open
    lenient = ScriptMask(vocab, Policy.preset("ko", tail_ban_ratio=1.0))
    assert not lenient.decide(lenient.new_state()).is_banned(vocab.token_for_bytes(b"\xe4"))


def test_all_banned_allows_special(vocab):
    only_hangul = ScriptMask(vocab, Policy(ban_scripts=frozenset({"Hangul"})))
    state = only_hangul.new_state()
    state.feed(vocab.token_for_bytes(b"\xeb"))  # force an unfinished Hangul character
    decision = only_hangul.decide(state)
    assert decision.fallback is True
    assert decision.allowed == vocab.special_ids
    assert not decision.is_banned(0) and decision.is_banned(vocab.token_for_bytes(b"a"))


def test_special_tokens_are_never_statically_banned(vocab, mask):
    base = mask.decide(mask.new_state())
    assert not any(base.is_banned(i) for i in vocab.special_ids)
    assert not any(mask.static[i] for i in vocab.special_ids)


def test_invalid_utf8_resets_pending_instead_of_crashing(vocab, mask):
    state = mask.new_state()
    feed_all(mask, state, byte_ids(vocab, b"\xeb"))
    state.feed(vocab.token_for_bytes(b"a"))  # a lead byte followed by ASCII is invalid: start over
    assert state.pending == b""


def test_sync_resets_when_history_changes(vocab, mask):
    state = mask.new_state()
    a, b = vocab.token_for_bytes(b"\xeb"), vocab.token_for_bytes(b"a")
    state.sync([a])
    assert state.pending == b"\xeb"
    state.sync([b, b])  # a different history of the same length family
    assert state.pending == b"" and state.fed == 2


def test_soft_mode_adds_exact_bias(vocab):
    np = pytest.importorskip("numpy")
    from langlock.llamacpp import make_logits_processor

    policy = Policy.preset("ko", mode="soft", bias=-6.0)
    mask = ScriptMask(vocab, policy)
    rng = np.random.default_rng(0)
    scores = rng.normal(size=vocab.size).astype(np.single)
    before = scores.copy()
    out = make_logits_processor(vocab, policy, prompt_len=1)(np.array([5], dtype=np.intc), scores.copy())
    decision = mask.decide(mask.new_state())
    banned = np.array([decision.is_banned(i) for i in range(vocab.size)])
    assert banned.any() and (~banned).any()
    assert np.allclose((out - before)[banned], -6.0)
    assert np.array_equal(out[~banned], before[~banned])
    assert np.isfinite(out).all()
