"""Exception types raised by langlock."""

from __future__ import annotations


class LanglockError(Exception):
    """Base class for every error langlock raises on purpose."""


class UnsupportedTokenizerError(LanglockError):
    """The tokenizer is not a byte-level BPE tokenizer that langlock can read."""


class UnknownPresetError(LanglockError, KeyError):
    """The requested policy preset name does not exist."""

    def __str__(self) -> str:  # KeyError would otherwise repr() the message
        return str(self.args[0])


class PolicyError(LanglockError, ValueError):
    """A policy value is invalid (unknown script, bad bias, bad mode ...)."""
