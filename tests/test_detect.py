from langlock import Policy, detect

TEXT = "안녕하세요 漢字입니다\n정상 문장\n漢 and かな"


def test_detect_report():
    report = detect(TEXT, "ko")
    assert report.banned_chars == 5 and report.letters == 20
    assert report.ratio == 0.25
    assert report.spans == ((1, 7, "Han", "漢字"), (3, 1, "Han", "漢"), (3, 7, "Kana", "かな"))
    assert report.replacement_chars == 0
    assert report.lines == 3 and abs(report.line_pass_rate - 1 / 3) < 1e-9
    assert not report.ok


def test_clean_text_and_replacement_character():
    assert detect("깨끗한 한국어 문장입니다. 123 😀", "ko").ok
    broken = detect("깨진 글자 � 포함", "ko")
    assert broken.replacement_chars == 1 and not broken.ok
    assert detect("", "ko").ratio == 0.0 and detect("", "ko").line_pass_rate == 1.0


def test_exempt_regions_are_not_counted():
    text = "본문입니다\n```\n# 中文 주석\n```\n<think>漢字 생각</think>끝"
    assert detect(text, "ko").banned_chars == 4
    relaxed = detect(text, Policy.preset("ko", exempt_code_blocks=True, exempt_think=True))
    assert relaxed.banned_chars == 0 and relaxed.ok
