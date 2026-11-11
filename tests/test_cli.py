import json
from pathlib import Path

from langlock import load_vocab
from langlock.cli import main

SAMPLE = Path(__file__).resolve().parents[1] / "examples" / "sample_mixed_ko.txt"


def test_demo_table_teaches_the_lesson(capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert "연습용 목(mock)" in lines[0]
    rows = {line.split("|")[0].strip(): line for line in lines if "|" in line}
    assert "漢" in rows["none"] and "뷁" in rows["none"]
    assert "漢" not in rows["naive"] and "뷁" not in rows["naive"]
    assert "漢" not in rows["langlock"] and "뷁" in rows["langlock"].split("|", 1)[1]


def test_check_exit_codes(tmp_path, capsys):
    clean = tmp_path / "clean.txt"
    clean.write_text("한국어 문장만 있습니다.\n", encoding="utf-8")
    assert main(["check", str(clean), "--policy", "ko"]) == 0
    assert main(["check", str(SAMPLE), "--policy", "ko"]) == 1
    capsys.readouterr()
    assert main(["check", str(SAMPLE), "--policy", "ko", "--exit-zero"]) == 0
    out = capsys.readouterr().out
    assert "FOUND" in out and "2:" in out and "Han" in out and "漢字" in out and "%" in out
    assert main(["check", str(tmp_path / "missing.txt")]) == 2
    assert "cannot read" in capsys.readouterr().err


def test_check_json(capsys):
    assert main(["check", str(SAMPLE), "--json", "--exit-zero"]) == 0
    data = json.loads(capsys.readouterr().out)[str(SAMPLE)]
    assert data["banned_chars"] == 4 and data["spans"][0]["script"] == "Han"


def test_scan_json(fixture_path, capsys):
    assert main(["scan", str(fixture_path), "--policy", "ko", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == load_vocab(fixture_path).stats("ko").as_dict()
    assert main(["scan", str(fixture_path)]) == 0
    out = capsys.readouterr().out
    for word in ("vocab:", "Han:", "Hangul:", "static banned:", "dynamic candidates:"):
        assert word in out


def test_export_commands(fixture_path, capsys):
    assert main(["export", "--policy", "ko", "--format", "gbnf"]) == 0
    assert "root ::= char*" in capsys.readouterr().out
    assert main(["export", str(fixture_path), "--format", "logit-bias"]) == 0
    assert json.loads(capsys.readouterr().out)[0][1] is False
    assert main(["export", "--format", "openai-bias"]) == 2
    assert "needs a tokenizer.json" in capsys.readouterr().err


def test_bad_tokenizer_exits_2(tmp_path, capsys):
    path = tmp_path / "wp.json"
    path.write_text(json.dumps({"model": {"type": "WordPiece", "vocab": {}}}), encoding="utf-8")
    assert main(["scan", str(path)]) == 2
    assert "byte-level" in capsys.readouterr().err


def test_out_of_range_openai_bias_is_one_line_error(fixture_path, capsys):
    code = main(["export", str(fixture_path), "--format", "openai-bias", "--mode", "soft", "--bias", "-200"])
    err = capsys.readouterr().err
    assert code == 2
    assert err.startswith("langlock: error:")
    assert "-100" in err and "Traceback" not in err


# --- early-closed stdout pipe (e.g. `langlock export ... | head`) -----------------------------
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

import pytest  # noqa: E402

TINY = Path(__file__).resolve().parents[1] / "src" / "langlock" / "_data" / "tiny_bytebpe.json"
posix_only = pytest.mark.skipif(os.name != "posix", reason="pipe semantics checked on POSIX only")


def _byte_to_unicode() -> dict[int, str]:
    # GPT-2 byte -> printable unicode table, used to spell CJK tokens in a byte-level vocab.
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def _big_tokenizer(tmp_path: Path, extra: int = 8000) -> Path:
    """Bundled tiny tokenizer plus many Han tokens, so the export output exceeds a 64 KiB pipe."""
    data = json.loads(TINY.read_text(encoding="utf-8"))
    vocab = data["model"]["vocab"]
    table = _byte_to_unicode()
    next_id = max(vocab.values()) + 1
    for cp in range(0x4E00, 0x4E00 + extra):
        vocab["".join(table[b] for b in chr(cp).encode("utf-8"))] = next_id
        next_id += 1
    path = tmp_path / "big_tokenizer.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def _run_langlock(args: list[str], read_bytes: int | None):
    """Start the CLI, read `read_bytes` (None = all), close stdout, and always reap the process."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "langlock.cli", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert proc.stdout is not None and proc.stderr is not None
        if read_bytes is None:
            out = proc.stdout.read()
        else:
            out = proc.stdout.read(read_bytes)
        proc.stdout.close()  # reader goes away early -> writer gets EPIPE
        err = proc.stderr.read().decode("utf-8", "replace")
        code = proc.wait(timeout=60)
        return code, out, err
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        if proc.stderr:
            proc.stderr.close()


@posix_only
def test_export_closed_pipe_is_quiet(tmp_path):
    tok = _big_tokenizer(tmp_path)
    code, out, err = _run_langlock(["export", str(tok), "--format", "logit-bias"], 50)
    assert len(out) == 50
    assert "Traceback" not in err and "BrokenPipeError" not in err and "Exception ignored" not in err
    assert err == ""
    assert code == 141


@posix_only
def test_check_closed_pipe_is_quiet(tmp_path):
    text = tmp_path / "big.txt"
    text.write_text("漢字가 섞인 문장 漢字\n" * 20000, encoding="utf-8")
    code, out, err = _run_langlock(["check", str(text)], 50)
    assert len(out) == 50
    assert "Traceback" not in err and "BrokenPipeError" not in err and "Exception ignored" not in err
    assert err == ""
    assert code == 141


@posix_only
def test_export_full_read_is_unchanged(tmp_path):
    tok = _big_tokenizer(tmp_path)
    code, out, err = _run_langlock(["export", str(tok), "--format", "logit-bias"], None)
    assert code == 0 and err == ""
    assert len(out) > 65536
    assert isinstance(json.loads(out), list)
