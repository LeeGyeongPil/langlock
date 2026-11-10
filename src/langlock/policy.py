"""Policy: which writing systems to keep out, and how hard."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from .errors import PolicyError, UnknownPresetError
from .scripts import SCRIPT_NAMES

PRESETS: dict[str, frozenset[str]] = {
    "ko": frozenset({"Han", "Kana"}),
    "en": frozenset({"Han", "Kana", "Hangul", "Cyrillic"}),
    "ru": frozenset({"Han", "Kana", "Hangul"}),
}


@dataclass(frozen=True)
class Policy:
    """What to block while decoding.

    ``mode="hard"`` sets blocked tokens to ``-inf``; ``mode="soft"`` adds
    ``bias`` (a negative number) to their logits instead.
    """

    ban_scripts: frozenset[str] = field(default_factory=lambda: PRESETS["ko"])
    mode: Literal["hard", "soft"] = "hard"
    bias: float = -8.0
    exempt_code_blocks: bool = False
    exempt_think: bool = False
    tail_ban_ratio: float = 0.9

    def __post_init__(self) -> None:
        scripts = frozenset(self.ban_scripts)
        unknown = sorted(scripts - set(SCRIPT_NAMES))
        if unknown:
            raise PolicyError(
                f"unknown script(s) {unknown}; choose from {sorted(SCRIPT_NAMES)}"
            )
        object.__setattr__(self, "ban_scripts", scripts)
        if self.mode not in ("hard", "soft"):
            raise PolicyError(f"mode must be 'hard' or 'soft', got {self.mode!r}")
        if not isinstance(self.bias, (int, float)) or math.isnan(self.bias) or self.bias > 0:
            raise PolicyError(f"bias must be a number <= 0 (for example -8.0), got {self.bias!r}")
        if not 0.0 < self.tail_ban_ratio <= 1.0:
            raise PolicyError(f"tail_ban_ratio must be in (0, 1], got {self.tail_ban_ratio!r}")

    @classmethod
    def preset(cls, name: str, **overrides: Any) -> "Policy":
        """Build a policy from a preset name (``ko``, ``en``, ``ru``) plus overrides."""
        try:
            scripts = PRESETS[name]
        except KeyError:
            raise UnknownPresetError(
                f"unknown preset {name!r}; available presets: {', '.join(sorted(PRESETS))}"
            ) from None
        overrides.setdefault("ban_scripts", scripts)
        return cls(**overrides)

    def with_changes(self, **changes: Any) -> "Policy":
        """Return a copy with some fields replaced (re-validated)."""
        return replace(self, **changes)


def resolve_policy(policy: "Policy | str") -> Policy:
    """Accept a :class:`Policy` or a preset name."""
    if isinstance(policy, Policy):
        return policy
    if isinstance(policy, str):
        return Policy.preset(policy)
    raise PolicyError(f"expected a Policy or a preset name, got {type(policy).__name__}")
