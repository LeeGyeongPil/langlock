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
