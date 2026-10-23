"""Score converter output against hand-checked reference pages.

Reference pages live in evals/extract/ref/<book>/p<idx>.md. Math is compared
after canonicalization (spacing, \\left/\\right, font aliases, \\ldots/\\cdots,
equation-number styles, single-token braces), so only material differences
count.
"""

import collections
import difflib
import json
import re

from .. import config
from ..books import Book
from ..extract import mdutil
from ..extract.mdutil import canon_math
from ..extract.pages import words

REF_DIR = config.EVALS_DIR / "extract" / "ref"


def tokens(tex: str) -> list[str]:
    return re.findall(r"\\[a-zA-Z]+|\\.|.", canon_math(tex))


def _text_words(md: str) -> list[str]:
    stripped = mdutil.MATH_RE.sub(" ", md)
    return words(mdutil.md_to_text(stripped))


def score_page(ref_md: str, cand_md: str) -> dict:
    rw, cw = collections.Counter(_text_words(ref_md)), collections.Counter(_text_words(cand_md))
    text_recall = 1.0 if not rw else sum((rw & cw).values()) / sum(rw.values())
    text_extra = sum((cw - rw).values())
    ref_spans = [canon_math(t) for t, _ in mdutil.math_spans(ref_md)]
    cand_spans = [canon_math(t) for t, _ in mdutil.math_spans(cand_md)]
    cand_pool = collections.Counter(cand_spans)
    # "$A, B$" vs "$A$, $B$" is not a material difference: accept comma parts.
    parts = collections.Counter(p for c in cand_spans if "," in c and "{" not in c for p in c.split(",") if p)
    exact, sims, wrong = 0, [], []
    for r in ref_spans:
        if cand_pool[r] > 0 or parts[r] > 0:
            (cand_pool if cand_pool[r] > 0 else parts)[r] -= 1
            exact += 1
            sims.append(1.0)
            continue
        best, best_c = 0.0, ""
        rt = re.findall(r"\\[a-zA-Z]+|\\.|.", r)
        for c in set(cand_spans):
            s = difflib.SequenceMatcher(None, rt, re.findall(r"\\[a-zA-Z]+|\\.|.", c), autojunk=False).ratio()
            if s > best:
                best, best_c = s, c
        sims.append(best)
        wrong.append({"ref": r[:160], "best": best_c[:160], "sim": round(best, 3)})
    n = len(ref_spans)
    return {
        "text_recall": round(text_recall, 4), "text_extra": text_extra,
        "math_n": n, "math_exact": round(exact / n, 4) if n else 1.0,
        "math_sim": round(sum(sims) / n, 4) if n else 1.0, "math_wrong": wrong,
    }


def ref_pages(book: Book) -> dict[int, str]:
    d = REF_DIR / book.id
    return {int(p.stem[1:]): p.read_text(encoding="utf-8") for p in sorted(d.glob("p*.md"))} if d.exists() else {}


def evaluate(book: Book, methods: list[str]) -> list[dict]:
    con = book.connect()
    refs = ref_pages(book)
    rows = []
    for method in methods:
        q = ("SELECT page_idx, md FROM b_page_final" if method == "final"
             else "SELECT page_idx, md FROM b_page_md WHERE method=?")
        cands = {r["page_idx"]: r["md"] for r in con.execute(q, [] if method == "final" else [method])}
        checks = {r["page_idx"]: r for r in con.execute(
            "SELECT * FROM b_page_checks WHERE method=?", (method,))}
        for p, ref in refs.items():
            if p not in cands:
                continue
            s = score_page(ref, cands[p])
            c = checks.get(p)
            s.update({
                "method": method, "page_idx": p,
                "auto_pass": None if c is None else bool(c["passed"]),
                "judge_n": None if c is None or c["judge"] in (None, "null") else len(json.loads(c["judge"])),
            })
            rows.append(s)
    return rows
