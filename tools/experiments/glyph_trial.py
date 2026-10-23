"""Trial: choose among candidate transcriptions of each math span by matching the PDF's glyphs.

usage: uv run python tools/experiments/glyph_trial.py <primary> <other1>,<other2>,...
"""

import collections
import json
import re
import sys

import pymupdf

from tbook import books
from tbook.evals import extract_eval as ee
from tbook.extract import consensus, glyphs, mdutil


def region_for(item, blocks, page):
    box = consensus.locate(item, blocks)
    if box is None:
        return None
    r = page.rect
    x0, y0, x1, y1 = box
    return pymupdf.Rect((x0 - 0.01) * r.width, (y0 - 0.006) * r.height, (x1 + 0.01) * r.width, (y1 + 0.006) * r.height)


def main() -> None:
    primary, others = sys.argv[1], sys.argv[2].split(",")
    for b in ("mr", "fob"):
        bk = books.load(b)
        con = bk.connect()
        doc = pymupdf.open(bk.pdf)
        math_re = re.compile(bk.cfg["math_fonts"])
        tot = collections.Counter()
        for p, ref in ee.ref_pages(bk).items():
            md = lambda m: (con.execute("SELECT md FROM b_page_md WHERE page_idx=? AND method=?", (p, m)).fetchone() or [None])[0]
            prim = md(primary)
            oth = {m: md(m) for m in others if md(m) is not None}
            meta = con.execute("SELECT meta FROM b_page_md WHERE page_idx=? AND method='mineru'", (p,)).fetchone()
            blocks = json.loads(meta[0]).get("blocks", []) if meta else []
            page = doc[p]
            page_toks = glyphs.pdf_tokens(page, math_re)
            spans, _ = consensus.disputes(prim, oth)
            # every distinct form any voter produced for this span
            texts_by_canon = {}
            for src in [prim] + list(oth.values()):
                for t, _ in mdutil.math_spans(src):
                    texts_by_canon.setdefault(mdutil.canon_math(t), t)
            chosen = {}
            for i, s in enumerate(spans):
                forms = [f for f in s["votes"] if "$" not in texts_by_canon.get(f, "")]
                if len(forms) < 2:
                    continue
                opts = [texts_by_canon.get(f, s["tex"]) for f in forms]
                bad = mdutil.katex_errors([(o, s["display"]) for o in opts])
                keep = [k for k, e in enumerate(bad) if e is None]
                forms, opts = [forms[k] for k in keep], [opts[k] for k in keep]
                if len(forms) < 2:
                    continue
                votes = [len(s["votes"][f]) for f in forms]
                region = region_for({"span": s, "options": opts}, blocks, page) if s["display"] else None
                if region is not None:
                    ref_toks = glyphs.pdf_tokens(page, math_re, region)
                    k, scores = glyphs.best_candidate(opts, ref_toks, local=False)
                    top = [j for j, sc in enumerate(scores) if sc >= scores[k] - 1e-6]
                    k = max(top, key=lambda j: votes[j])          # ties broken by votes
                else:
                    k = max(range(len(forms)), key=lambda j: votes[j])
                    if votes.count(votes[k]) > 1:
                        continue                                   # inline tie: keep primary
                if forms[k] != s["canon"]:
                    chosen[i] = opts[k]
            fixed = consensus.apply(prim, spans, chosen)
            s0, s1 = ee.score_page(ref, prim), ee.score_page(ref, fixed)
            tot["n"] += s0["math_n"]
            tot["before"] += round(s0["math_exact"] * s0["math_n"])
            tot["after"] += round(s1["math_exact"] * s1["math_n"])
            print(f"{b} p{p:3d}: replaced {len(chosen):2d}  math {s0['math_exact']:.2f} -> {s1['math_exact']:.2f}")
        print(f"== {b}: {tot['before']} -> {tot['after']} of {tot['n']}")


if __name__ == "__main__":
    main()
