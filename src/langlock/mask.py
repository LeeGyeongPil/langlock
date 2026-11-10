"""The engine-independent core: which token ids are blocked right now?

A byte-level BPE token can end in the middle of a UTF-8 character, so the text
of a token alone does not say which writing system it will end up in.  The mask
therefore remembers, per generated sequence, the unfinished bytes ("pending")
and decides for the next token:

* pending is empty  -> block every token that is, or can only become, a banned
  character, plus orphan continuation bytes (they would print U+FFFD);
* pending is not empty -> only continuation-byte tokens that finish the
  character into an allowed script may follow.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ._bytes import split_utf8, tail_range
from .policy import Policy
from .scripts import banned_fraction, script_of
from .vocab import COMPLETE, CONTINUATION, INVALID, TAIL, VocabIndex

_CACHE_LIMIT = 8192  # pending-prefix cache cap: at most a few thousand distinct prefixes exist


@dataclass(frozen=True)
class Decision:
    """Which ids are blocked for the next token of one sequence.

    Exactly one of three shapes:

    * ``exempt`` -- nothing is blocked (inside a code fence or ``<think>``);
    * ``allowed`` is a set -- *only* those ids are open (a character is unfinished);
    * otherwise ``banned`` holds one 0/1 flag per vocabulary id.
    """

    banned: bytes | bytearray | None = None
    allowed: frozenset[int] | None = None
    exempt: bool = False
    fallback: bool = False  # True when nothing else was possible and only special tokens are open
    vocab_size: int = 0

    def is_banned(self, token_id: int) -> bool:
        """Whether one id is blocked. Ids outside the vocabulary are never touched."""
        if self.exempt or not 0 <= token_id < self.vocab_size:
            return False
        if self.allowed is not None:
            return token_id not in self.allowed
        return bool(self.banned[token_id]) if self.banned is not None else False


class RowState:
    """What has been generated so far in one sequence (unfinished bytes, exempt zones)."""

    __slots__ = ("_mask", "pending", "in_fence", "in_think", "_ticks", "_recent", "fed", "_last_ids")

    def __init__(self, mask: "ScriptMask") -> None:
        self._mask = mask
        self.reset()

    def reset(self) -> None:
        self.pending = b""
        self.in_fence = False
        self.in_think = False
        self._ticks = 0  # consecutive backticks seen
        self._recent = b""
        self.fed = 0
        self._last_ids: tuple[int, ...] = ()

    @property
    def exempt(self) -> bool:
        policy = self._mask.policy
        return (policy.exempt_code_blocks and self.in_fence) or (policy.exempt_think and self.in_think)

    def feed(self, token_id: int) -> None:
        """Record one generated token."""
        vocab, policy = self._mask.vocab, self._mask.policy
        data = vocab.token_bytes(token_id)
        if vocab.is_special(token_id):
            self.pending = b""
        else:
            _, tail, ok = split_utf8(self.pending + data)
            self.pending = tail if ok else b""
        if policy.exempt_code_blocks:
            for byte in data:
                self._ticks = self._ticks + 1 if byte == 0x60 else 0
                if self._ticks == 3:  # a ``` fence opens or closes
                    self._ticks = 0
                    self.in_fence = not self.in_fence
                    self.pending = b""
        if policy.exempt_think:
            self._recent = (self._recent + data)[-8:]
            if self._recent.endswith(b"</think>"):
                self.in_think, self.pending = False, b""
            elif self._recent.endswith(b"<think>"):
                self.in_think, self.pending = True, b""
        self.fed += 1
        self._last_ids = (*self._last_ids, token_id)[-8:]

    def sync(self, generated: Sequence[int]) -> None:
        """Catch up with the full list of generated ids; start over if it does not extend what we saw."""
        n = len(generated)
        window = len(self._last_ids)
        if n < self.fed or tuple(generated[self.fed - window : self.fed]) != self._last_ids:
            self.reset()
        for token_id in generated[self.fed :]:
            self.feed(int(token_id))


_EXEMPT = Decision(exempt=True)


class ScriptMask:
    """Per-vocabulary, per-policy analysis plus the rule that turns a state into a :class:`Decision`."""

    def __init__(self, vocab: VocabIndex, policy: Policy) -> None:
        self.vocab = vocab
        self.policy = policy
        kinds = vocab.kinds
        static = bytearray(vocab.size)
        dynamic: list[int] = []
        for token_id, kind in enumerate(kinds):
            if kind == CONTINUATION:
                dynamic.append(token_id)
            elif kind == INVALID:
                static[token_id] = 1
            elif kind in (COMPLETE, TAIL) and self._banned_text(vocab.token_bytes(token_id)):
                static[token_id] = 1
        self.static = static  # blocked regardless of state (this is what export writes)
        self.dynamic_ids: tuple[int, ...] = tuple(dynamic)  # tokens that start with a continuation byte
        base = bytearray(static)
        for token_id in dynamic:
            base[token_id] = 1  # with nothing pending they are orphan bytes
        self.base = base
        self.static_banned_count = sum(static)
        self.special_ids: frozenset[int] = vocab.special_ids
        self._pending_cache: dict[bytes, Decision] = {}
        self._base_decision = Decision(banned=self.base, vocab_size=vocab.size)

    # -- classification -------------------------------------------------
    def _tail_banned(self, tail: bytes) -> bool:
        """Dead-end rule: ban an unfinished character if (almost) all its completions are banned."""
        low, high = tail_range(tail)
        return banned_fraction(low, high, self.policy.ban_scripts) >= self.policy.tail_ban_ratio

    def _banned_text(self, data: bytes) -> bool:
        cps, tail, ok = split_utf8(data)
        if not ok:
            return True
        ban = self.policy.ban_scripts
        if any(script_of(cp) in ban for cp in cps):
            return True
        return bool(tail) and self._tail_banned(tail)

    def static_ids(self) -> list[int]:
        """Ids that are blocked whatever came before (usable as a static ``logit_bias``)."""
        return [i for i, flag in enumerate(self.static) if flag]

    # -- decisions ------------------------------------------------------
    def new_state(self) -> RowState:
        return RowState(self)

    def _pending_decision(self, pending: bytes) -> Decision:
        cached = self._pending_cache.get(pending)
        if cached is not None:
            return cached
        allowed = [
            token_id
            for token_id in self.dynamic_ids
            if not self._banned_text(pending + self.vocab.token_bytes(token_id))
        ]
        if allowed:
            decision = Decision(allowed=frozenset(allowed), vocab_size=self.vocab.size)
        elif self.special_ids:  # dead end: let the sequence end instead of printing a broken character
            decision = Decision(allowed=self.special_ids, fallback=True, vocab_size=self.vocab.size)
        else:  # no special tokens to fall back on: do not block anything
            decision = Decision(exempt=True, fallback=True, vocab_size=self.vocab.size)
        if len(self._pending_cache) >= _CACHE_LIMIT:
            self._pending_cache.clear()
        self._pending_cache[pending] = decision
        return decision

    def decide(self, state: RowState) -> Decision:
        """Blocked ids for the next token of the sequence described by ``state``."""
        if state.exempt:
            return _EXEMPT
        if state.pending:
            return self._pending_decision(state.pending)
        return self._base_decision


__all__ = ["Decision", "RowState", "ScriptMask"]
