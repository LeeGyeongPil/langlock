"""``langlock demo``: a practice (mock) run on the bundled tiny tokenizer.

Nothing here is a real model.  A scripted "model" always prefers to write one
character as *single-byte tokens* (the way a byte-level BPE vocabulary spells a
character it has rarely seen) and gives a Korean word a lower score.  The three
strategies then run the real decoding loop: ask the mask what is blocked, take
the highest-scoring token that is still open, feed it back.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import resources

from .mask import ScriptMask
from .policy import Policy
from .scripts import script_of
from .vocab import VocabIndex, load_vocab

HEADER = (
    "연습용 목(mock): 작은 합성 토크나이저와 가짜 점수로 원리만 보여 줍니다. "
    "실제 모델 품질 측정이 아닙니다."
)

# (label, character the fake model wants to write, spelled as single-byte tokens)
SCENARIOS = (
    ("漢 (한자)", "漢"),
    ("뷁 (드문 한글)", "뷁"),
)
FILLER = ("한", "국", "어")  # whole-character tokens the fake model likes less


def bundled_vocab() -> VocabIndex:
    """The tiny byte-level BPE tokenizer shipped inside the package."""
    path = resources.files("langlock").joinpath("_data/tiny_bytebpe.json")
    with resources.as_file(path) as real_path:
        return load_vocab(real_path)


def naive_banned_ids(vocab: VocabIndex) -> set[int]:
    """The common recipe: block tokens whose text has a Han character or a broken ``\\ufffd``."""
    banned: set[int] = set()
    for token_id in range(vocab.size):
        if vocab.is_special(token_id):
            continue
        text = vocab.token_bytes(token_id).decode("utf-8", errors="replace")
        if "�" in text or any(script_of(ord(ch)) == "Han" for ch in text):
            banned.add(token_id)
    return banned


def _scripted_decode(
    vocab: VocabIndex,
    wanted: str,
    is_banned: Callable[[int], bool],
    feed: Callable[[int], None],
) -> str:
    fillers = [vocab.token_for_bytes(ch.encode()) for ch in FILLER]
    steps = list(wanted.encode("utf-8"))
    out: list[int] = []
    for step, byte in enumerate(steps):
        preferred = vocab.token_for_bytes(bytes([byte]))
        fallback = fillers[step % len(fillers)]
        assert preferred is not None and fallback is not None, "bundled tokenizer changed"
        # fake logits: preferred token 10, filler 5, everything else 0
        choice = next((t for t in (preferred, fallback) if not is_banned(t)), fallback)
        out.append(choice)
        feed(choice)
    return b"".join(vocab.token_bytes(t) for t in out).decode("utf-8", errors="replace")


def run_scenarios(vocab: VocabIndex | None = None) -> dict[str, dict[str, str]]:
    """Return ``{method: {scenario label: decoded text}}`` for none / naive / langlock."""
    vocab = vocab or bundled_vocab()
    naive = naive_banned_ids(vocab)
    mask = ScriptMask(vocab, Policy.preset("ko"))
    results: dict[str, dict[str, str]] = {"none": {}, "naive": {}, "langlock": {}}
    for label, wanted in SCENARIOS:
        results["none"][label] = _scripted_decode(vocab, wanted, lambda t: False, lambda t: None)
        results["naive"][label] = _scripted_decode(vocab, wanted, lambda t: t in naive, lambda t: None)
        state = mask.new_state()
        results["langlock"][label] = _scripted_decode(
            vocab, wanted, lambda t: mask.decide(state).is_banned(t), state.feed
        )
    return results


def render(results: dict[str, dict[str, str]]) -> str:
    labels = [label for label, _ in SCENARIOS]
    rows = [["method", *labels]] + [[name, *(results[name][lb] for lb in labels)] for name in results]
    lines = [HEADER, "", "가짜 모델이 아래 글자를 바이트 조각 토큰으로 쓰려고 할 때 방식별 결과:", ""]
    for row in rows:
        lines.append(" | ".join(cell.ljust(14) for cell in row).rstrip())
    lines += [
        "",
        "none     = 아무것도 막지 않음",
        "naive    = 한자가 든 토큰과 깨진 글자(U+FFFD) 토큰을 모두 막는 흔한 레시피",
        "langlock = 바이트를 추적해 '한자로 완성될 때만' 막음",
    ]
    return "\n".join(lines)


def run_demo() -> str:
    """Run the three strategies and return the printable table."""
    return render(run_scenarios())
