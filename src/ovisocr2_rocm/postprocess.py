"""OvisOCR2 post-processing (upstream model card recipe, verbatim).

Relocated from ``adapter/postprocess.py`` so config/postprocess/sharding and the
pipeline share one home (``ovisocr2_rocm``); ``adapter/postprocess.py`` is now a
thin re-export so existing callers/tests keep working.
"""
from __future__ import annotations


def clean_truncated_repeats(
    text,
    min_text_len=8000,
    max_period=200,
    min_period=1,
    min_repeat_chars=100,
    min_repeat_times=5,
):
    n = len(text)
    if n < min_text_len:
        return text
    max_period = min(max_period, n - 1)
    for unit_len in range(min_period, max_period + 1):
        if text[n - 1] != text[n - 1 - unit_len]:
            continue
        match_len, idx = 1, n - 2
        while idx >= unit_len and text[idx] == text[idx - unit_len]:
            match_len += 1
            idx -= 1
        total_len = match_len + unit_len
        repeat_times = total_len // unit_len
        tail_len = total_len % unit_len
        if repeat_times >= min_repeat_times and total_len >= min_repeat_chars:
            return text[: n - total_len + unit_len] + text[n - tail_len:]
    return text


def postprocess(text: str) -> str:
    """Drop visual-region ``<img>`` tags (card default) then clean repeat tails."""
    text = "\n\n".join(
        b for b in text.split("\n\n") if not b.strip().startswith('<img src="images/bbox_'))
    return clean_truncated_repeats(text)
