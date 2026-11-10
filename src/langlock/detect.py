"""Measure how much of a finished text is in a banned writing system."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .policy import Policy, resolve_policy
from .scripts import COMMON, script_of

_THINK = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL)
_FENCE = re.compile(r"```.*?(?:```|\Z)", re.DOTALL)


@dataclass(frozen=True)
class DetectReport:
    """Result of :func:`detect`. Positions are 1-based; ``col`` counts characters."""

    banned_chars: int
    letters: int
    ratio: float
    spans: tuple[tuple[int, int, str, str], ...]  # (line, col, script, text)
    replacement_chars: int
    line_pass_rate: float
    lines: int

    @property
    def ok(self) -> bool:
        """True when nothing banned and no U+FFFD replacement character was found."""
        return self.banned_chars == 0 and self.replacement_chars == 0

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "banned_chars": self.banned_chars,
            "letters": self.letters,
            "ratio": self.ratio,
            "replacement_chars": self.replacement_chars,
            "line_pass_rate": self.line_pass_rate,
            "lines": self.lines,
            "spans": [
                {"line": line, "col": col, "script": script, "text": text}
                for line, col, script, text in self.spans
            ],
        }


def _exempt_flags(text: str, policy: Policy) -> list[bool]:
    flags = [False] * len(text)
    patterns = []
    if policy.exempt_think:
        patterns.append(_THINK)
    if policy.exempt_code_blocks:
        patterns.append(_FENCE)
    for pattern in patterns:
        for match in pattern.finditer(text):
            flags[match.start() : match.end()] = [True] * (match.end() - match.start())
    return flags


def detect(text: str, policy: "Policy | str" = "ko") -> DetectReport:
    """Find characters of banned scripts in ``text``.

    ``ratio`` is banned characters divided by letters (characters of any known
    script; digits, punctuation, emoji and spaces do not count).
    """
    policy = resolve_policy(policy)
    exempt = _exempt_flags(text, policy)
    spans: list[tuple[int, int, str, str]] = []
    banned = letters = 0
    line = col = 1
    open_span: list[object] | None = None  # [line, col, script, text]
    bad_lines: set[int] = set()
    for index, char in enumerate(text):
        if char == "\n":
            line, col, open_span = line + 1, 1, None
            continue
        script = script_of(ord(char))
        if script != COMMON and not exempt[index]:
            letters += 1
        if script in policy.ban_scripts and not exempt[index]:
            banned += 1
            bad_lines.add(line)
            if open_span is not None and open_span[2] == script and open_span[0] == line:
                open_span[3] += char  # type: ignore[operator]
            else:
                open_span = [line, col, script, char]
                spans.append(open_span)  # type: ignore[arg-type]
        else:
            open_span = None
        col += 1
    checked = [n for n, raw in enumerate(text.split("\n"), 1) if raw.strip()]
    passing = [n for n in checked if n not in bad_lines]
    return DetectReport(
        banned_chars=banned,
        letters=letters,
        ratio=(banned / letters) if letters else 0.0,
        spans=tuple((int(s[0]), int(s[1]), str(s[2]), str(s[3])) for s in spans),  # type: ignore[call-overload]
        replacement_chars=text.count("�"),
        line_pass_rate=(len(passing) / len(checked)) if checked else 1.0,
        lines=len(checked),
    )
