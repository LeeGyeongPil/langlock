"""Shared fixtures. Tests never touch the network or a real model unless marked ``real``."""

from __future__ import annotations

from pathlib import Path

import pytest

from langlock import Policy, VocabIndex, load_vocab
from langlock.demo import bundled_vocab

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "src" / "langlock" / "_data" / "tiny_bytebpe.json"


@pytest.fixture(scope="session")
def vocab() -> VocabIndex:
    return bundled_vocab()


@pytest.fixture(scope="session")
def fixture_path() -> Path:
    return FIXTURE


@pytest.fixture(scope="session")
def ko() -> Policy:
    return Policy.preset("ko")


@pytest.fixture(scope="session")
def mask(vocab, ko):
    return vocab.script_mask(ko)


def byte_ids(vocab: VocabIndex, data: bytes) -> list[int]:
    """Ids of the single-byte tokens spelling ``data`` (the 'fragment' spelling of a character)."""
    ids = [vocab.token_for_bytes(bytes([b])) for b in data]
    assert None not in ids
    return ids  # type: ignore[return-value]
