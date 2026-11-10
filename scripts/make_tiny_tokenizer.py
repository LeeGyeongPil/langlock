"""Regenerate src/langlock/_data/tiny_bytebpe.json from scripts/tiny_corpus.txt.

Usage:  uv run python scripts/make_tiny_tokenizer.py

The fixture is a small byte-level BPE tokenizer trained on a hand-written
corpus (MIT, like the rest of the repository).  The corpus deliberately never
contains rare Hangul such as the syllables used in the tests, so those stay
split into single-byte tokens, which is exactly the situation langlock must
handle.  Commit the generated JSON: tests only read the committed file.
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "src" / "langlock" / "_data" / "tiny_bytebpe.json"


def main() -> None:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=700,
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<|endoftext|>", "<think>", "</think>"],
        show_progress=False,
    )
    tokenizer.train([str(ROOT / "tiny_corpus.txt")], trainer)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(OUT))
    print(f"wrote {OUT} ({tokenizer.get_vocab_size()} tokens)")


if __name__ == "__main__":
    main()
