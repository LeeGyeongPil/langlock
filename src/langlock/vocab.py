"""Read a byte-level BPE tokenizer and expose the real bytes of every token."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._bytes import decode_piece, is_continuation, split_utf8
from .errors import LanglockError, UnsupportedTokenizerError
from .policy import Policy, resolve_policy
from .scripts import COMMON, SCRIPT_NAMES, script_of

if TYPE_CHECKING:  # pragma: no cover
    from .mask import ScriptMask

# Token kinds used by the classification (policy independent).
SPECIAL, COMPLETE, TAIL, CONTINUATION, INVALID, EMPTY = (
    "special",
    "complete",
    "tail",
    "continuation",
    "invalid",
    "empty",
)


@dataclass(frozen=True)
class ScanStats:
    """Summary of a vocabulary under one policy (what ``langlock scan`` prints)."""

    vocab_size: int
    special_tokens: int
    scripts: dict[str, int]
    complete_tokens: int
    tail_tokens: int
    continuation_tokens: int
    invalid_tokens: int
    static_banned: int
    dynamic_candidates: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "scripts": dict(self.scripts),
            "complete_tokens": self.complete_tokens,
            "tail_tokens": self.tail_tokens,
            "continuation_tokens": self.continuation_tokens,
            "invalid_tokens": self.invalid_tokens,
            "static_banned": self.static_banned,
            "dynamic_candidates": self.dynamic_candidates,
        }


@dataclass
class VocabIndex:
    """Token id -> real bytes for a byte-level BPE vocabulary."""

    size: int
    _raw: list[bytes]
    special_ids: frozenset[int]
    _masks: dict[Policy, "ScriptMask"] = field(default_factory=dict, repr=False)

    def token_bytes(self, token_id: int) -> bytes:
        """Real bytes of a token (the literal text for special tokens, ``b""`` if unused)."""
        if 0 <= token_id < self.size:
            return self._raw[token_id]
        return b""

    def is_special(self, token_id: int) -> bool:
        return token_id in self.special_ids

    @cached_property
    def _by_bytes(self) -> dict[bytes, int]:
        table: dict[bytes, int] = {}
        for token_id, data in enumerate(self._raw):
            if data and token_id not in self.special_ids:
                table.setdefault(data, token_id)
        return table

    def token_for_bytes(self, data: bytes) -> int | None:
        """Id of the (non-special) token that is exactly ``data``, if there is one."""
        return self._by_bytes.get(data)

    @cached_property
    def kinds(self) -> list[str]:
        """Per-token kind: special / complete / tail / continuation / invalid / empty."""
        out: list[str] = []
        for token_id, data in enumerate(self._raw):
            if token_id in self.special_ids:
                out.append(SPECIAL)
            elif not data:
                out.append(EMPTY)
            elif is_continuation(data[0]):
                out.append(CONTINUATION)
            else:
                _, tail, ok = split_utf8(data)
                out.append(INVALID if not ok else (TAIL if tail else COMPLETE))
        return out

    def script_mask(self, policy: "Policy | str") -> "ScriptMask":
        """Cached :class:`~langlock.mask.ScriptMask` for a policy."""
        from .mask import ScriptMask

        policy = resolve_policy(policy)
        mask = self._masks.get(policy)
        if mask is None:
            mask = self._masks[policy] = ScriptMask(self, policy)
        return mask

    def stats(self, policy: "Policy | str" = "ko") -> ScanStats:
        """Count tokens by script and by byte-fragment kind, and what the policy bans."""
        mask = self.script_mask(policy)
        counts = {name: 0 for name in (*SCRIPT_NAMES, COMMON)}
        for token_id, data in enumerate(self._raw):
            if self.kinds[token_id] in (COMPLETE, TAIL):
                cps, _, _ = split_utf8(data)
                for name in {script_of(cp) for cp in cps}:
                    counts[name] += 1
        kinds = self.kinds
        return ScanStats(
            vocab_size=self.size,
            special_tokens=len(self.special_ids),
            scripts=counts,
            complete_tokens=kinds.count(COMPLETE),
            tail_tokens=kinds.count(TAIL),
            continuation_tokens=kinds.count(CONTINUATION),
            invalid_tokens=kinds.count(INVALID),
            static_banned=mask.static_banned_count,
            dynamic_candidates=len(mask.dynamic_ids),
        )


def _find_type(node: Any, wanted: str) -> bool:
    """True if a tokenizers component (or a Sequence of them) contains ``wanted``."""
    if isinstance(node, dict):
        if node.get("type") == wanted:
            return True
        return any(_find_type(child, wanted) for child in node.get("pretokenizers", []) + node.get("decoders", []))
    return False


def _tokenizer_dict(source: Any) -> dict[str, Any]:
    if isinstance(source, dict):
        return source
    if isinstance(source, (str, Path)):
        path = Path(source)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise LanglockError(f"tokenizer file not found: {path}") from None
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise LanglockError(f"{path} is not a valid tokenizer.json: {exc}") from None
    backend = getattr(source, "backend_tokenizer", source)
    to_str = getattr(backend, "to_str", None)
    if callable(to_str):  # transformers fast tokenizer or tokenizers.Tokenizer
        return json.loads(to_str())
    raise UnsupportedTokenizerError(
        "load_vocab() needs a tokenizer.json path, a parsed dict, or a Hugging Face fast "
        f"tokenizer (with backend_tokenizer); got {type(source).__name__}."
    )


def load_vocab(source: Any) -> VocabIndex:
    """Build a :class:`VocabIndex` from a tokenizer.json path, dict, or HF tokenizer."""
    data = _tokenizer_dict(source)
    model = data.get("model") or {}
    if model.get("type") != "BPE" or not isinstance(model.get("vocab"), dict):
        raise UnsupportedTokenizerError(
            f"only byte-level BPE tokenizers are supported (model type is {model.get('type')!r}). "
            "WordPiece and Unigram vocabularies do not expose raw UTF-8 bytes."
        )
    if not (_find_type(data.get("pre_tokenizer"), "ByteLevel") or _find_type(data.get("decoder"), "ByteLevel")):
        raise UnsupportedTokenizerError(
            "this BPE tokenizer is not byte-level (no ByteLevel pre-tokenizer/decoder). "
            "SentencePiece byte-fallback tokenizers (Llama 2, Gemma ...) are not supported yet."
        )
    added = {int(t["id"]): str(t["content"]) for t in data.get("added_tokens") or []}
    pieces: dict[int, bytes] = {}
    for piece, token_id in model["vocab"].items():
        if int(token_id) in added:
            continue
        raw = decode_piece(piece)
        if raw is None:
            raise UnsupportedTokenizerError(
                f"token {piece!r} (id {token_id}) has characters outside the byte-level alphabet; "
                "this is not a GPT-2 style byte-level vocabulary."
            )
        pieces[int(token_id)] = raw
    size = max([*pieces, *added], default=-1) + 1
    if size == 0:
        raise UnsupportedTokenizerError("the tokenizer has an empty vocabulary")
    raw_list = [b""] * size
    for token_id, raw in pieces.items():
        raw_list[token_id] = raw
    for token_id, content in added.items():
        raw_list[token_id] = content.encode("utf-8")
    return VocabIndex(size=size, _raw=raw_list, special_ids=frozenset(added))
