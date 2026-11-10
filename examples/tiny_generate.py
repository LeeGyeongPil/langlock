"""Random-weight tiny Qwen2 + the bundled tokenizer: generate() without and with langlock.

No download and no training: the model is random, so the text is noise.  What
matters is that real ``generate()`` sampling is run twice from the same seed,
and only the second run has the processor.

    uv run python examples/tiny_generate.py
"""

from __future__ import annotations

import codecs
import time

import torch
from transformers import LogitsProcessorList, Qwen2Config, Qwen2ForCausalLM

from langlock.demo import bundled_vocab
from langlock.hf import LanglockLogitsProcessor
from langlock.scripts import script_of

SEEDS = 5
NEW_TOKENS = 200


def count(text: str) -> tuple[int, int, int]:
    han = sum(script_of(ord(c)) == "Han" for c in text)
    hangul = sum(script_of(ord(c)) == "Hangul" for c in text)
    return han, hangul, text.count("�")


def main() -> None:
    vocab = bundled_vocab()
    eos = 0  # <|endoftext|>
    print("연습용 목(mock): 무작위 가중치 초소형 모델이라 문장은 의미 없는 잡음입니다.")
    print(f"seeds={SEEDS}, new tokens per run={NEW_TOKENS}, vocab={vocab.size}")
    totals = {"without": [0, 0, 0, 0.0], "with langlock": [0, 0, 0, 0.0]}
    prompt = torch.tensor([[vocab.token_for_bytes("한국어".encode()) or 0]])
    for seed in range(SEEDS):
        torch.manual_seed(seed)
        config = Qwen2Config(
            vocab_size=vocab.size, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=512,
            pad_token_id=eos, eos_token_id=eos,
        )
        model = Qwen2ForCausalLM(config).eval()
        for label in totals:
            processors = LogitsProcessorList([LanglockLogitsProcessor(vocab, "ko")]) if label != "without" else None
            torch.manual_seed(seed)
            started = time.perf_counter()
            out = model.generate(
                input_ids=prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=NEW_TOKENS,
                do_sample=True, top_k=0, temperature=1.0, logits_processor=processors,
                pad_token_id=eos, eos_token_id=eos,
            )
            elapsed = time.perf_counter() - started
            raw = b"".join(vocab.token_bytes(i) for i in out[0, prompt.shape[1]:].tolist() if not vocab.is_special(i))
            # an unfinished character at the very end is not counted as a broken one
            text = codecs.getincrementaldecoder("utf-8")(errors="replace").decode(raw)
            han, hangul, fffd = count(text)
            row = totals[label]
            row[0] += han; row[1] += hangul; row[2] += fffd; row[3] += elapsed
    print(f"{'':15}{'Han chars':>10}{'Hangul':>8}{'U+FFFD':>8}{'seconds':>9}")
    for label, (han, hangul, fffd, seconds) in totals.items():
        print(f"{label:15}{han:>10}{hangul:>8}{fffd:>8}{seconds:>9.2f}")


if __name__ == "__main__":
    main()
