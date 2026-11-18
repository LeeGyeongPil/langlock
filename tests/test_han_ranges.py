"""Regression tests: the Han table must match Unicode Scripts.txt (Script=Han)."""

import re

import pytest

from langlock import detect
from langlock.export import to_regex
from langlock.scripts import COMMON, script_of


@pytest.mark.parametrize(
    "cp",
    [0x3005, 0x3007, 0x3021, 0x3029, 0x3038, 0x303A, 0x303B, 0x16FE2, 0x16FE3, 0x16FF0,
     0x16FF6, 0x2EBF0, 0x2EE5D, 0x323B0, 0x33479, 0x31350, 0x6F22],
)
def test_han_code_points(cp):
    assert script_of(cp) == "Han"


@pytest.mark.parametrize("cp", [0x3006, 0x3190, 0x319F, 0x31C0, 0x31EF, 0x3020, 0x3030])
def test_neighbours_stay_common(cp):
    assert script_of(cp) == COMMON


def test_iteration_mark_is_banned_in_ko():
    assert detect("々", "ko").banned_chars == 1


def test_regex_export_blocks_iteration_mark():
    pattern = re.compile(to_regex("ko"))
    assert not pattern.fullmatch("々")
    assert not pattern.fullmatch("漢")
