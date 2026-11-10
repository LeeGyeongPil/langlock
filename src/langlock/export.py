"""Write the same policy in formats other engines understand."""

from __future__ import annotations

from .errors import LanglockError, PolicyError
from .policy import Policy, resolve_policy
from .scripts import ranges_for
from .vocab import VocabIndex


def _hard_only(policy: Policy, what: str) -> None:
    if policy.mode != "hard":
        raise LanglockError(
            f"{what} can only express a hard ban; use a policy with mode='hard' "
            "(or export a logit_bias list, which supports a soft bias)."
        )


def to_logit_bias(
    vocab: VocabIndex,
    policy: "Policy | str",
    style: str = "llamacpp",
    bias: float | None = None,
) -> "list[list[object]] | dict[str, float]":
    """Static ``logit_bias`` for the tokens that are blocked in *every* state.

    ``style="llamacpp"`` gives ``[[id, false], ...]`` (llama.cpp server: ``false``
    means never) or ``[[id, bias], ...]`` when ``bias`` is given.
    ``style="openai"`` gives ``{"id": bias}`` with values in -100..100
    (default -100). This is weaker than the stateful processor: it cannot
    track half-written characters.
    """
    policy = resolve_policy(policy)
    ids = vocab.script_mask(policy).static_ids()
    if style == "llamacpp":
        value: object = False if bias is None else bias
        return [[token_id, value] for token_id in ids]
    if style == "openai":
        value_f = -100.0 if bias is None else float(bias)
        if not -100.0 <= value_f <= 100.0:
            raise PolicyError(f"OpenAI logit_bias values must be between -100 and 100, got {value_f}")
        return {str(token_id): value_f for token_id in ids}
    raise PolicyError(f"unknown style {style!r}; use 'llamacpp' or 'openai'")


def _class_body(policy: Policy, syntax: str) -> str:
    def esc(cp: int) -> str:
        if syntax == "gbnf":
            return f"\\u{cp:04X}" if cp <= 0xFFFF else f"\\U{cp:08X}"
        return f"\\u{cp:04x}" if cp <= 0xFFFF else f"\\U{cp:08x}"

    parts = []
    for lo, hi in ranges_for(policy.ban_scripts):
        parts.append(esc(lo) if lo == hi else f"{esc(lo)}-{esc(hi)}")
    return "".join(parts)


def to_gbnf(policy: "Policy | str") -> str:
    """A llama.cpp GBNF grammar that accepts any text without banned characters."""
    policy = resolve_policy(policy)
    _hard_only(policy, "GBNF")
    note = ""
    if policy.exempt_code_blocks or policy.exempt_think:
        note = "# note: code-block / <think> exemptions cannot be expressed in a grammar\n"
    return (
        f"# langlock: no {', '.join(sorted(policy.ban_scripts))} characters\n"
        f"{note}"
        "root ::= char*\n"
        f"char ::= [^{_class_body(policy, 'gbnf')}]\n"
    )


def to_regex(policy: "Policy | str") -> str:
    """A regular expression (full match) for text without banned characters.

    Written with ``\\uXXXX`` escapes so it can be pasted into vLLM
    ``structured_outputs.regex`` or xgrammar.
    """
    policy = resolve_policy(policy)
    _hard_only(policy, "a regex")
    return f"[^{_class_body(policy, 'regex')}]*"

