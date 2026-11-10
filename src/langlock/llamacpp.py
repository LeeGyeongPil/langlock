"""llama-cpp-python adapter (needs numpy, which llama-cpp-python already depends on)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .policy import Policy, resolve_policy
from .vocab import VocabIndex, load_vocab


def make_logits_processor(
    vocab: "VocabIndex | Any",
    policy: "Policy | str" = "ko",
    prompt_len: int = 0,
) -> Callable[[Any, Any], Any]:
    """Build a function usable in ``LogitsProcessorList`` for ``Llama.create_completion``.

    llama-cpp-python calls ``f(input_ids, scores)`` with *all* tokens so far
    (prompt included) and a 1-D float array of scores, and expects the scores
    back. Pass the prompt length so only generated tokens are tracked, and
    build a new function for every completion.
    """
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise ImportError('this adapter needs numpy: pip install "langlock[numpy]"') from exc

    index = vocab if isinstance(vocab, VocabIndex) else load_vocab(vocab)
    resolved = resolve_policy(policy)
    mask = index.script_mask(resolved)
    state = mask.new_state()
    static_cache: dict[int, Any] = {}

    def processor(input_ids: Any, scores: Any) -> Any:
        state.sync(np.asarray(input_ids)[prompt_len:].tolist())
        decision = mask.decide(state)
        if decision.exempt:
            return scores
        width = scores.shape[-1]
        known = min(width, index.size)
        blocked = np.zeros(width, dtype=bool)
        if decision.allowed is not None:
            blocked[:known] = True
            open_ids = [i for i in decision.allowed if i < known]
            if open_ids:
                blocked[open_ids] = False
        else:
            flags = static_cache.get(id(decision.banned))
            if flags is None:
                flags = static_cache[id(decision.banned)] = np.frombuffer(
                    bytes(decision.banned), dtype=np.uint8
                ).astype(bool)
            blocked[:known] = flags[:known]
        if resolved.mode == "soft":
            scores[blocked] += resolved.bias
        else:
            scores[blocked] = -np.inf
        return scores

    return processor
