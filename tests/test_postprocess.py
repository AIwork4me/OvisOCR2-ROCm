"""Post-processing tests (P0-3): visual-region filter, repeat cleaning, UTF-8."""
from adapter import postprocess as P


def test_drops_visual_region_img_tags():
    src = '<img src="images/bbox_1_2_3_4.jpg" />\n\nreal text'
    assert P.postprocess(src).strip() == "real text"


def test_keeps_real_img_outside_bbox_namespace():
    # only bbox_ visual-region tags are dropped; other markdown survives
    src = "intro\n\n![photo](images/photo.jpg)\n\noutro"
    assert "![photo](images/photo.jpg)" in P.postprocess(src)


def test_clean_truncated_repeats_short_text_untouched():
    assert P.clean_truncated_repeats("short") == "short"


def test_clean_truncated_repeats_trims_tail():
    tail = "abcd" * 50  # 200 chars, repeat unit 4, >5 times, >100 chars
    text = "HEAD" + tail
    out = P.clean_truncated_repeats(text, min_text_len=10, min_repeat_chars=20, min_repeat_times=5)
    assert "HEAD" in out and len(out) < len(text)


def test_utf8_roundtrip():
    s = "中文公式 $x = \\frac{1}{2}$ — ✓"
    assert P.postprocess(s) == s
