import subprocess
import sys

CODE = """
import sys
import langlock
from langlock import cli
assert cli.main(['demo']) == 0
heavy = [name for name in ('torch', 'numpy', 'transformers', 'tokenizers') if name in sys.modules]
assert not heavy, heavy
print('light')
"""


def test_import_does_not_load_heavy_modules(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", CODE], capture_output=True, text=True, cwd=tmp_path, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.rstrip().endswith("light")
