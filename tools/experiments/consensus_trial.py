"""Trial: does consensus + adjudication beat the primary transcription on the reference pages?

usage: uv run python tools/experiments/consensus_trial.py <primary> <other1>[,<other2>] [--seed N]
"""

import json
import sys

from tbook import books
from tbook.evals import extract_eval as ee
from tbook.extract import consensus


def _tex_of(prim, oth, item, method):
    """The LaTeX that `method` wrote for this disputed span (its first option with that canonical form)."""
    target = [c for c, voters in item["span"]["votes"].items() if method in voters][0]
    for o in item["options"]:
        if consensus.mdutil.canon_math(o) == target:
            return o
    return item["options"][0]


def main() -> None:
    primary, others = sys.argv[1], sys.argv[2].split(",")
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0
    for b in ("mr", "fob"):
        bk = books.load(b)
        con = bk.connect()
        refs = ee.ref_pages(bk)
        before = after = n = 0
        for p, ref in refs.items():
            md = lambda m: (con.execute("SELECT md FROM b_page_md WHERE page_idx=? AND method=?", (p, m)).fetchone() or [None])[0]
            prim = md(primary)
            oth = {m: md(m) for m in others if md(m) is not None}
            png = con.execute("SELECT png FROM b_pages WHERE page_idx=?", (p,)).fetchone()[0]
            spans, items = consensus.disputes(prim, oth)
            chosen = {}
            if "--plurality" in sys.argv:
                # the canonical form with the most votes wins; ties are adjudicated
                rest = []
                for it in items:
                    ranked = sorted(it["span"]["votes"].items(), key=lambda kv: -len(kv[1]))
                    if len(ranked) == 1 or len(ranked[0][1]) > len(ranked[1][1]):
                        if "primary" not in ranked[0][1] and len(ranked[0][1]) >= 2:
                            chosen[it["index"]] = _tex_of(prim, oth, it, ranked[0][1][0])
                        elif len(ranked[0][1]) < 2:
                            rest.append(it)
                    else:
                        rest.append(it)
                items = rest
            if "--majority" in sys.argv:
                # two other transcriptions agreeing outvote the primary without asking the VLM
                for it in [i for i in items if i["span"]["status"] == "outvoted"]:
                    winner = max(it["span"]["votes"].items(), key=lambda kv: len(kv[1]))[1][0]
                    chosen[it["index"]] = _tex_of(prim, oth, it, winner)
                items = [i for i in items if i["span"]["status"] != "outvoted"]
            tag = f"trial:{b}:{p}:{primary}:{'+'.join(others)}:{seed}"
            if "--zoom" in sys.argv:
                meta = con.execute("SELECT meta FROM b_page_md WHERE page_idx=? AND method='mineru'", (p,)).fetchone()
                blocks = json.loads(meta[0]).get("blocks", []) if meta else []
                chosen.update(consensus.adjudicate_zoomed(bk.pdf, p, png, items, blocks, tag=tag + ":zoom", seed=seed))
            else:
                chosen.update(consensus.adjudicate(png, items, tag=tag, seed=seed))
            fixed = consensus.apply(prim, spans, chosen)
            s0, s1 = ee.score_page(ref, prim), ee.score_page(ref, fixed)
            before += s0["math_exact"] * s0["math_n"]
            after += s1["math_exact"] * s1["math_n"]
            n += s0["math_n"]
            print(f"{b} p{p:3d}: {consensus.summarize(spans)} adjudicated {len(chosen)}/{len(items)} "
                  f"math {s0['math_exact']:.2f} -> {s1['math_exact']:.2f}")
        print(f"== {b}: math exact {before / n:.3f} -> {after / n:.3f} ({round(before)} -> {round(after)} of {n})")


if __name__ == "__main__":
    main()
