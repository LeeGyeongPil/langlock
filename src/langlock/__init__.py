"""langlock: keep stray Chinese characters out of Korean LLM output at decode time.

Public API::

    from langlock import load_vocab, Policy, ScriptMask, detect
    from langlock.hf import LanglockLogitsProcessor          # needs langlock[hf]
    from langlock.llamacpp import make_logits_processor      # needs numpy
    from langlock.export import to_logit_bias, to_gbnf, to_regex
"""

from .detect import DetectReport, detect
from .errors import LanglockError, PolicyError, UnknownPresetError, UnsupportedTokenizerError
from .mask import Decision, RowState, ScriptMask
from .policy import Policy
from .vocab import ScanStats, VocabIndex, load_vocab

__version__ = "0.1.2"

__all__ = [
    "Decision",
    "DetectReport",
    "LanglockError",
    "Policy",
    "PolicyError",
    "RowState",
    "ScanStats",
    "ScriptMask",
    "UnknownPresetError",
    "UnsupportedTokenizerError",
    "VocabIndex",
    "__version__",
    "detect",
    "load_vocab",
]
