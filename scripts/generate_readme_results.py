#!/usr/bin/env python3
"""Generate the README results block from ``model_card_v2.json``.

Single source of truth: every score in the README comes from the v2 card, never
hand-typed. Emits the markdown between the GENERATED markers; ``--check`` exits
non-zero if the README block is stale (CI drift detection).

    python scripts/generate_readme_results.py README.md            # write
    python scripts/generate_readme_results.py README.md --check    # CI gate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BEGIN = "<!-- BEGIN GENERATED RESULTS -->"
END = "<!-- END GENERATED RESULTS -->"

_HEADERS = {
    "en": (("| result_id | platform | backend | precision | overall | text_edit_dist | "
            "reading_order | table_teds % | formula_cdm % | assurance | status |"),
           "|---|---|---|---|---|---|---|---|---|---|---|",
           "_Last generated from `model_card_v2.json`. Cross-model comparison lives in the "
           "[central hub](https://github.com/AIwork4me/OmniDocBench-ROCm), not in this repo._"),
    "zh": (("| result_id | 平台 | 后端 | 精度 | Overall | 文本编辑距 | 阅读顺序 | "
            "表格 TEDS % | 公式 CDM % | assurance | 状态 |"),
           "|---|---|---|---|---|---|---|---|---|---|---|",
           "_由 `model_card_v2.json` 自动生成，请勿手改。跨模型对比见 "
           "[中央 hub](https://github.com/AIwork4me/OmniDocBench-ROCm)，不在本子仓。_"),
}


def _fmt(v, ndigits=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{ndigits}f}"
    return str(v)


def build_block(card: dict, lang: str) -> str:
    hdr, sep, note = _HEADERS[lang]
    rows = [hdr, sep]
    for r in card.get("results", []):
        if r.get("status") not in ("valid", "superseded"):
            continue
        m = r.get("metrics", {}) or {}
        impl = r.get("implementation", {}) or {}
        cov = r.get("coverage", {}) or {}
        rows.append(
            "| {rid} | {plat} | {be} | {prec} | {overall} | {te} | {ro} | {tt} | {fc} | {ass} | {st} |".format(
                rid=r.get("result_id", ""),
                plat=cov.get("platform", ""),
                be=impl.get("backend", ""),
                prec=impl.get("precision") or "—",
                overall=_fmt(m.get("overall"), 2),
                te=_fmt(m.get("text_edit_dist")),
                ro=_fmt(m.get("reading_order_edit_dist")),
                tt=_fmt(m.get("table_teds_percent"), 2),
                fc=_fmt(m.get("formula_cdm_percent"), 2),
                ass=r.get("assurance", ""),
                st=r.get("status", ""),
            ))
    src = "<!-- Source: model_card_v2.json — do not edit by hand; run scripts/generate_readme_results.py -->"
    return "\n".join([BEGIN, src, "", *rows, "", note, END])


def render(readme: Path, card: dict, lang: str) -> str:
    text = readme.read_text(encoding="utf-8")
    block = build_block(card, lang)
    if BEGIN not in text or END not in text:
        # insert the markers right after the first H1 if absent
        lines = text.splitlines()
        idx = next((i for i, ln in enumerate(lines) if ln.startswith("# ")), 0)
        lines.insert(idx + 1, "", block, "")
        return "\n".join(lines) + ("\n" if not text.endswith("\n") else "")
    pre = text[: text.index(BEGIN)]
    post = text[text.index(END) + len(END):]
    return pre + block + post


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("readme")
    p.add_argument("--card", default="model_card_v2.json")
    p.add_argument("--lang", choices=("en", "zh"), default="en")
    p.add_argument("--check", action="store_true", help="exit 1 if the block is stale")
    a = p.parse_args(argv)

    readme = Path(a.readme)
    card = json.loads(Path(a.card).read_text(encoding="utf-8"))
    new = render(readme, card, a.lang)
    if a.check:
        current = readme.read_text(encoding="utf-8")
        if current != new:
            print(f"STALE: {readme} results block does not match {a.card} "
                  "(run scripts/generate_readme_results.py)", file=sys.stderr)
            return 1
        print(f"OK: {readme} up to date")
        return 0
    readme.write_text(new, encoding="utf-8")
    print(f"wrote {readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
