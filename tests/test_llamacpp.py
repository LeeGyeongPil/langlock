import random

import pytest

np = pytest.importorskip("numpy")

from langlock.llamacpp import make_logits_processor  # noqa: E402


def test_numpy_processor_matches_core(vocab, mask):
    prompt = [5, 6, 7]
    processor = make_logits_processor(vocab, "ko", prompt_len=len(prompt))
    state = mask.new_state()
    rng = random.Random(0)
    generated: list[int] = []
    saw_pending = False
    for _ in range(400):
        ids = np.array(prompt + generated, dtype=np.intc)
        scores = processor(ids, np.zeros(vocab.size, dtype=np.single))
        decision = mask.decide(state)
        saw_pending = saw_pending or decision.allowed is not None
        blocked = np.isneginf(scores)
        assert blocked.tolist() == [decision.is_banned(i) for i in range(vocab.size)]
        choice = rng.choice(np.flatnonzero(~blocked).tolist())
        generated.append(choice)
        state.feed(choice)
    assert saw_pending, "the walk never hit an unfinished character; the test would prove little"


def test_processor_restarts_for_a_new_completion(vocab):
    processor = make_logits_processor(vocab, "ko", prompt_len=1)
    lead = vocab.token_for_bytes(b"\xeb")
    first = processor(np.array([1, lead], dtype=np.intc), np.zeros(vocab.size, dtype=np.single))
    assert np.isfinite(first).sum() < 120  # unfinished character: few ids open
    again = processor(np.array([1], dtype=np.intc), np.zeros(vocab.size, dtype=np.single))
    assert np.isfinite(again).sum() > 300  # new completion, nothing pending


def test_wider_scores_are_left_alone(vocab):
    processor = make_logits_processor(vocab, "ko", prompt_len=1)
    out = processor(np.array([1], dtype=np.intc), np.zeros(vocab.size + 8, dtype=np.single))
    assert np.isfinite(out[vocab.size :]).all()
