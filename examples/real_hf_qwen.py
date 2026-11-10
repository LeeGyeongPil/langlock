"""Optional: compare a real local model with and without langlock.

    # 1) get a model once (about 1 GB), for example with huggingface-cli:
    #      huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir ./qwen25
    # 2) run:
    uv run python examples/real_hf_qwen.py --model-dir ./qwen25 --limit 3

Runs on CPU/MPS and can be slow when other programs share the machine; slowness is not an error.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

from langlock import detect, load_vocab
from langlock.hf import LanglockLogitsProcessor

PROMPTS = Path(__file__).with_name("prompts_ko.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True, help="local Hugging Face model directory")
    parser.add_argument("--limit", type=int, default=10, help="how many prompts to use (default 10)")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--policy", default="ko")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir).eval()
    vocab = load_vocab(tokenizer)
    prompts = PROMPTS.read_text(encoding="utf-8").splitlines()[: args.limit]
    totals = {"without": [0, 0], "with langlock": [0, 0]}
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        for label in totals:
            processors = None if label == "without" else LogitsProcessorList([LanglockLogitsProcessor(vocab, args.policy)])
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=args.max_new_tokens, do_sample=False, logits_processor=processors)
            text = tokenizer.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
            report = detect(text, args.policy)
            totals[label][0] += report.banned_chars
            totals[label][1] += report.letters
    for label, (banned, letters) in totals.items():
        share = banned / letters if letters else 0.0
        print(f"{label:15} banned chars: {banned:5d} of {letters:6d} letters ({share:.2%})")


if __name__ == "__main__":
    main()
