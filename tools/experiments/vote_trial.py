"""Trial: full plurality vote over several transcriptions, plus the oracle ceiling.

usage: uv run python tools/experiments/vote_trial.py <primary> <other1>,<other2>,...
For each primary math span, the canonical form with the most votes across all
transcriptions replaces it. 'ceiling' = reference spans that at least one voter got right.
"""

import collections
import sys

from tbook import books
from tbook.evals import extract_eval as ee
from tbook.extract import consensus, mdutil


def main() -> None:
    primary, others = sys.argv[1], sys.argv[2].split(",")
    for b in ("mr", "fob"):
        bk = books.load(b)
        con = bk.connect()
        tot = collections.Counter()
        for p, ref in ee.ref_pages(bk).items():
            md = lambda m: (con.execute("SELECT md FROM b_page_md WHERE page_idx=? AND method=?", (p, m)).fetchone() or [None])[0]
            prim = md(primary)
            oth = {m: md(m) for m in others if md(m) is not None}
            spans, _ = consensus.disputes(prim, oth)
            chosen = {}
            for i, s in enumerate(spans):
                ranked = sorted(s["votes"].items(), key=lambda kv: -len(kv[1]))
                top_canon, voters = ranked[0]
                tie = len(ranked) > 1 and len(ranked[1][1]) == len(voters)
                if top_canon != s["canon"] and not tie and len(voters) > len(s["votes"][s["canon"]]):
                    for m in [primary] + others:
                        if m in voters:
                            src = prim if m == "primary" else oth[m]
                            for t, _ in mdutil.math_spans(src):
                                if mdutil.canon_math(t) == top_canon:
                                    chosen[i] = t
                                    break
                            if i in chosen:
                                break
            fixed = consensus.apply(prim, spans, chosen)
            s0, s1 = ee.score_page(ref, prim), ee.score_page(ref, fixed)
            ref_c = [mdutil.canon_math(t) for t, _ in mdutil.math_spans(ref)]
            pool = set()
            for src in [prim] + list(oth.values()):
                pool |= {mdutil.canon_math(t) for t, _ in mdutil.math_spans(src)}
            tot["n"] += s0["math_n"]
            tot["before"] += round(s0["math_exact"] * s0["math_n"])
            tot["after"] += round(s1["math_exact"] * s1["math_n"])
            tot["ceiling"] += sum(1 for r in ref_c if r in pool)
            print(f"{b} p{p:3d}: replaced {len(chosen):2d}  math {s0['math_exact']:.2f} -> {s1['math_exact']:.2f}")
        print(f"== {b}: {tot['before']} -> {tot['after']} of {tot['n']}  (ceiling {tot['ceiling']})")


if __name__ == "__main__":
    main()
