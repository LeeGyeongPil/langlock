"""Smallest example: ask the mask, byte by byte, what may come next. Needs nothing but langlock.

    uv run python examples/quickstart.py
"""

from langlock import detect
from langlock.demo import bundled_vocab

vocab = bundled_vocab()           # the tiny bundled byte-level BPE tokenizer
mask = vocab.script_mask("ko")    # ban Han and Kana

for char in "한뷁漢":
    state = mask.new_state()      # one state per generated sequence
    for i, byte in enumerate(char.encode(), start=1):
        token = vocab.token_for_bytes(bytes([byte]))
        if mask.decide(state).is_banned(token):
            print(f"{char} byte {i}/3 {byte:#04x}: BLOCKED")
            break
        print(f"{char} byte {i}/3 {byte:#04x}: allowed")
        state.feed(token)

print(detect("한국어 문장에 漢字가 끼었다", "ko").spans)
