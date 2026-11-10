"""Hugging Face transformers adapter (needs ``pip install "langlock[hf]"``)."""

from __future__ import annotations

from typing import Any

try:
    import torch
    from transformers import LogitsProcessor
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        'langlock.hf needs torch and transformers. Install them with: pip install "langlock[hf]"'
    ) from exc

from .mask import RowState
from .policy import Policy, resolve_policy
from .vocab import VocabIndex, load_vocab


class LanglockLogitsProcessor(LogitsProcessor):
    """A ``LogitsProcessor`` that keeps banned scripts out of ``model.generate()``.

    Create one per ``generate()`` call (it remembers per-row state). Passing it in
    a ``LogitsProcessorList`` next to other processors combines the constraints:
    each one only ever lowers scores.

    Args:
        vocab: a :class:`VocabIndex`, or anything :func:`load_vocab` accepts.
        policy: a :class:`Policy` or preset name.
        prompt_len: number of prompt columns in ``input_ids``. Defaults to the
            width seen on the first call, which is right for ``generate()``
            (with left padding every row has the same prompt width).

    Beam search reorders rows between steps; row state is re-synchronised from
    the ids, but ``num_beams > 1`` is not part of the tested behaviour.
    """

    def __init__(
        self,
        vocab: "VocabIndex | Any",
        policy: "Policy | str" = "ko",
        prompt_len: int | None = None,
    ) -> None:
        self.vocab = vocab if isinstance(vocab, VocabIndex) else load_vocab(vocab)
        self.policy = resolve_policy(policy)
        self.mask = self.vocab.script_mask(self.policy)
        self.prompt_len = prompt_len
        self._states: list[RowState] = []
        self._static: dict[tuple[int, str], Any] = {}

    def _static_tensor(self, banned: "bytes | bytearray", device: Any) -> Any:
        key = (id(banned), str(device))
        tensor = self._static.get(key)
        if tensor is None:
            tensor = torch.frombuffer(bytearray(banned), dtype=torch.uint8).to(device).bool()
            self._static[key] = tensor
        return tensor

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        if self.prompt_len is None:
            self.prompt_len = int(input_ids.shape[1])
        rows, width = scores.shape
        while len(self._states) < rows:
            self._states.append(self.mask.new_state())
        known = min(width, self.vocab.size)  # ids beyond the tokenizer vocabulary are never touched
        blocked = torch.zeros((rows, width), dtype=torch.bool, device=scores.device)
        touched = False
        for row in range(rows):
            state = self._states[row]
            state.sync(input_ids[row, self.prompt_len :].tolist())
            decision = self.mask.decide(state)
            if decision.exempt:
                continue
            touched = True
            if decision.allowed is not None:
                blocked[row, :known] = True
                open_ids = [i for i in decision.allowed if i < known]
                if open_ids:
                    blocked[row, open_ids] = False
            else:
                blocked[row, :known] = self._static_tensor(decision.banned, scores.device)[:known]
        if not touched:
            return scores
        if self.policy.mode == "soft":
            return scores + blocked.to(scores.dtype) * self.policy.bias
        return scores.masked_fill(blocked, float("-inf"))
