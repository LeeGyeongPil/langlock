import codecs

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import LogitsProcessor, LogitsProcessorList, Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from langlock import Policy  # noqa: E402
from langlock.hf import LanglockLogitsProcessor  # noqa: E402
from langlock.scripts import script_of  # noqa: E402

EOS = 0  # <|endoftext|> in the bundled tokenizer


def tiny_model(vocab, seed: int):
    """A random-weight Qwen2 with the bundled vocabulary size: no download, no training."""
    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=vocab.size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=512,
        pad_token_id=EOS,
        eos_token_id=EOS,
    )
    return Qwen2ForCausalLM(config).eval()


def decode_complete(vocab, ids):
    """Text of ``ids``; an unfinished character at the very end is left out, not turned into U+FFFD."""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    return decoder.decode(b"".join(vocab.token_bytes(i) for i in ids if not vocab.is_special(i)))


def prompt(vocab):
    ids = [vocab.token_for_bytes("한국어".encode()) or vocab.token_for_bytes("한".encode())]
    return torch.tensor([ids])


def generate(model, vocab, processors, seed):
    torch.manual_seed(seed)
    ids = prompt(vocab)
    out = model.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=200,
        do_sample=True,
        top_k=0,
        temperature=1.0,
        logits_processor=LogitsProcessorList(processors),
        pad_token_id=EOS,
        eos_token_id=EOS,
    )
    return out[0, ids.shape[1] :].tolist()


def test_generate_random_tiny_qwen2(vocab):
    for seed in range(5):
        model = tiny_model(vocab, seed)
        new_ids = generate(model, vocab, [LanglockLogitsProcessor(vocab, "ko")], seed)
        assert len(new_ids) > 10
        text = decode_complete(vocab, new_ids)
        assert not any(script_of(ord(ch)) in ("Han", "Kana") for ch in text), (seed, text)
        assert "�" not in text, (seed, text)


def test_batch_left_padding_rows_independent(vocab, mask):
    processor = LanglockLogitsProcessor(vocab, "ko")
    pad = EOS
    a, eb = vocab.token_for_bytes(b"a"), vocab.token_for_bytes(b"\xeb")
    prompts = torch.tensor([[pad, pad, 5, 6], [7, 8, 9, 10]])  # row 0 is left-padded
    width = vocab.size
    processor(prompts, torch.zeros(2, width))  # first call fixes prompt_len = 4
    ids = torch.cat([prompts, torch.tensor([[eb], [a]])], dim=1)
    scores = processor(ids, torch.zeros(2, width))
    row0_open = set(torch.nonzero(torch.isfinite(scores[0])).flatten().tolist())
    row1_blocked = set(torch.nonzero(torch.isneginf(scores[1])).flatten().tolist())
    # row 0 has an unfinished Hangul byte (EB): only continuation-byte tokens stay open
    assert row0_open and eb not in row0_open and a not in row0_open
    assert all(vocab.token_bytes(i)[0] & 0xC0 == 0x80 for i in row0_open)
    assert vocab.token_for_bytes(b"\xb7") in row0_open
    # row 1 is in the normal state and is unaffected by row 0
    assert row1_blocked == {i for i in range(width) if mask.decide(mask.new_state()).is_banned(i)}


def test_logits_wider_than_vocab_untouched(vocab):
    processor = LanglockLogitsProcessor(vocab, "ko")
    extra = 40
    prompt_ids = torch.tensor([[5, 6]])
    base = torch.randn(1, vocab.size + extra)
    processor(prompt_ids, base.clone())
    # normal state
    out = processor(torch.cat([prompt_ids, torch.tensor([[vocab.token_for_bytes(b"a")]])], 1), base.clone())
    assert torch.equal(out[0, vocab.size :], base[0, vocab.size :])
    # unfinished-character state (everything inside the vocabulary but a few ids is blocked)
    lead = torch.tensor([[vocab.token_for_bytes(b"\xeb")]])
    out = processor(torch.cat([prompt_ids, torch.tensor([[vocab.token_for_bytes(b"a")]]), lead], 1), base.clone())
    assert torch.equal(out[0, vocab.size :], base[0, vocab.size :])
    assert torch.isneginf(out[0, : vocab.size]).any()


class AllowList(LogitsProcessor):
    """A stand-in for another constraint (a JSON or allow-list processor)."""

    def __init__(self, allowed, width):
        self.blocked = torch.ones(width, dtype=torch.bool)
        self.blocked[list(allowed)] = False

    def __call__(self, input_ids, scores):
        return scores.masked_fill(self.blocked, float("-inf"))


def test_composes_with_allowlist_processor(vocab):
    from langlock.vocab import COMPLETE

    allowed = {vocab.token_for_bytes(bytes([b])) for b in range(256)} - {None}
    allowed |= {i for i, kind in enumerate(vocab.kinds) if kind == COMPLETE} | {EOS}
    for seed in range(3):
        model = tiny_model(vocab, 100 + seed)
        new_ids = generate(model, vocab, [AllowList(allowed, vocab.size), LanglockLogitsProcessor(vocab, "ko")], seed)
        assert new_ids and all(i in allowed for i in new_ids)
        text = decode_complete(vocab, new_ids)
        assert not any(script_of(ord(ch)) in ("Han", "Kana") for ch in text)


def test_soft_mode_keeps_scores_finite_and_shifts_by_bias(vocab, mask):
    policy = Policy.preset("ko", mode="soft", bias=-4.0)
    processor = LanglockLogitsProcessor(vocab, policy)
    base = torch.randn(1, vocab.size)
    out = processor(torch.tensor([[5]]), base.clone())
    decision = mask.decide(mask.new_state())
    banned = torch.tensor([decision.is_banned(i) for i in range(vocab.size)])
    assert torch.isfinite(out).all()
    assert torch.allclose((out - base)[0][banned], torch.tensor(-4.0))
    assert torch.equal(out[0][~banned], base[0][~banned])
