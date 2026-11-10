"""Optional checks against a real tokenizer / model.  Run with ``uv run pytest -q -m real``."""

import os

import pytest

from langlock import Policy, ScriptMask, detect, load_vocab

pytestmark = pytest.mark.real
TOKENIZER = os.environ.get("LANGLOCK_TOKENIZER_JSON")
MODEL_DIR = os.environ.get("LANGLOCK_MODEL_DIR")


@pytest.mark.skipif(not TOKENIZER, reason="set LANGLOCK_TOKENIZER_JSON to a byte-level BPE tokenizer.json")
def test_qwen_tokenizer_scan():
    vocab = load_vocab(TOKENIZER)
    assert vocab.size > 100_000
    mask = ScriptMask(vocab, Policy.preset("ko"))
    han = [vocab.token_for_bytes(bytes([b])) for b in "漢".encode()]
    rare = [vocab.token_for_bytes(bytes([b])) for b in "뷁".encode()]
    assert mask.decide(mask.new_state()).is_banned(han[0])
    state = mask.new_state()
    for token_id in rare:
        assert not mask.decide(state).is_banned(token_id)
        state.feed(token_id)


@pytest.mark.skipif(not MODEL_DIR, reason="set LANGLOCK_MODEL_DIR to a local causal LM directory")
def test_real_model_reduces_han():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from langlock.hf import LanglockLogitsProcessor

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_DIR)
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_DIR).eval()
    inputs = tokenizer("한국어로만 자기소개를 해 주세요.", return_tensors="pt")
    processor = LanglockLogitsProcessor(load_vocab(tokenizer), "ko")
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=60, logits_processor=transformers.LogitsProcessorList([processor])
        )
    text = tokenizer.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    assert detect(text, "ko").banned_chars == 0
