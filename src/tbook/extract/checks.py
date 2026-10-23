"""Stage 1c: reference-free checks on every extracted page.

  coverage  bag-of-words recall of the text-layer oracle (prose + code words)
  order     share of oracle words matched in reading order
  katex     every math span parses in KaTeX
  math_ok   pages with math-font text contain math spans
  hygiene   no running heads left in the output
  judge     VLM compares page image and transcription (material discrepancies)
"""

import collections
import difflib
import json
import time
from concurrent.futures import ThreadPoolExecutor

from .. import config, llm
from ..books import Book
from . import mdutil
from .pages import head_key, is_word, tokens, words

MATH_PAGE_CHARS = 20
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "discrepancies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": [
                        "missing_text", "extra_text", "wrong_math", "math_as_text", "garbled", "table", "code"]},
                    "where": {"type": "string"},
                    "detail": {"type": "string"},
                },
                "required": ["type", "where", "detail"],
            },
        },
    },
    "required": ["discrepancies"],
}


def retokenize(oracle: list[str], text: str) -> list[str]:
    """The oracle with the text layer's lost or spurious word breaks taken from the output.

    Some PDFs set justified spaces as glyph offsets with no space character, so
    the text layer glues words ("andwhichtries"); URLs broken across lines split
    one word in two ("revolu" "tion"). A token equal to consecutive output words
    is split like them; adjacent tokens that join into an output word are merged.
    """
    have = set(words(text))
    toks = tokens(text)                   # with one-letter words: "alinearsystem"
    joined = "".join(toks)
    bounds, pos = {0: 0}, 0
    for k, w in enumerate(toks):
        pos += len(w)
        bounds[pos] = k + 1
    oracle = [w for w in oracle if is_word(w)]
    res, i = [], 0
    while i < len(oracle):
        t = oracle[i]
        if t in have:
            res.append(t)
            i += 1
            continue
        merged = next((n for n in (2, 3) if i + n <= len(oracle) and "".join(oracle[i:i + n]) in have), None)
        if merged:
            res.append("".join(oracle[i:i + merged]))
            i += merged
            continue
        split = None
        if len(t) >= 6:
            s = joined.find(t)
            while s >= 0 and split is None:
                if s in bounds and s + len(t) in bounds and bounds[s + len(t)] - bounds[s] >= 2:
                    split = [w for w in toks[bounds[s]:bounds[s + len(t)]] if is_word(w)]
                s = joined.find(t, s + 1)
        res.extend(split or [t])
        i += 1
    return res


def coverage(oracle: list[str], md: str) -> tuple[float, float, list[str]]:
    """Recall of oracle words, in-order share, and the missing words.

    A missing word with a near-identical unmatched output word counts as found:
    converters silently fix the book's own typos ("satisifies" -> "satisfies").
    """
    text = mdutil.md_to_text(md)
    out = words(text)
    oracle = retokenize(oracle, text)
    need, have = collections.Counter(oracle), collections.Counter(out)
    extra = list((have - need).elements())
    missing = []
    for w in (need - have).elements():
        near = difflib.get_close_matches(w, extra, n=1, cutoff=0.8) if len(w) >= 5 else []
        if near:
            extra.remove(near[0])
        else:
            missing.append(w)
    recall = 1.0 if not oracle else 1 - len(missing) / len(oracle)
    oset = set(oracle)
    sm = difflib.SequenceMatcher(None, oracle, [w for w in out if w in oset], autojunk=False)
    in_order = 1.0 if not oracle else sum(b.size for b in sm.get_matching_blocks()) / len(oracle)
    return recall, in_order, missing


def hygiene(md: str, running_heads: list[str]) -> list[str]:
    keys = {head_key(h) for h in running_heads if sum(c.isalpha() for c in h) >= 4}
    issues = []
    for line in md.split("\n"):
        bare = line.strip("#* \t")
        if bare and head_key(bare) in keys:
            issues.append(f"remove the running header/footer: {bare[:60]}")
    issues += [f"prose typeset as math (write it as text with inline math): {t}" for t in mdutil.prose_in_math(md)]
    issues += [f"the transcription loops, repeating {r[:30]!r} over and over: transcribe what is on the page once"
               for r in mdutil.loops(md)]
    return issues


def judge(png: str, md: str, tag: str) -> list[dict]:
    msgs = [
        {"role": "system", "content": llm.load_prompt("extract_judge")},
        {"role": "user", "content": [
            llm.image_part(png),
            {"type": "text", "text": "Transcription to check:\n\n<transcription>\n" + md + "\n</transcription>"},
        ]},
    ]
    res = llm.chat(msgs, config.VISION_MODEL, schema=JUDGE_SCHEMA, max_tokens=3000, tag=tag)
    return res["discrepancies"]


def check(book: Book, method: str, pages: list[int] | None = None, *, use_judge: bool = True,
          workers: int = config.VISION_PARALLEL, log=print) -> list[dict]:
    """Check stored Markdown for `method` (or 'final') and record results in b_page_checks."""
    con = book.connect()
    if method == "final":
        q = "SELECT f.page_idx, f.md, p.* FROM b_page_final f JOIN b_pages p USING(page_idx)"
        args = []
    else:
        q = "SELECT m.page_idx, m.md, p.* FROM b_page_md m JOIN b_pages p USING(page_idx) WHERE m.method=?"
        args = [method]
    rows = [r for r in con.execute(q, args) if pages is None or r["page_idx"] in set(pages)]

    spans_by_page = {r["page_idx"]: mdutil.math_spans(r["md"]) for r in rows}
    flat = [(p, s) for p, spans in spans_by_page.items() for s in spans]
    errs = mdutil.katex_errors([s for _, s in flat])
    bad_by_page = collections.defaultdict(list)
    for (p, (tex, _)), e in zip(flat, errs):
        if e:
            bad_by_page[p].append({"tex": tex[:200], "error": e})

    def one(r):
        p = r["page_idx"]
        oracle = json.loads(r["oracle_words"])
        recall, order, missing = coverage(oracle, r["md"])
        prose = words(mdutil.md_to_text(mdutil.MATH_RE.sub(" ", r["md"])))
        extra = sum((collections.Counter(prose) - collections.Counter(oracle)).values())
        hyg = hygiene(r["md"], json.loads(r["running_heads"]))
        math_ok = r["math_chars"] < MATH_PAGE_CHARS or bool(spans_by_page[p])
        disc = judge(r["png"], r["md"], f"{book.id}:judge:{method}:{p}") if use_judge else None
        n_oracle = len(json.loads(r["oracle_words"]))
        cov_ok = recall >= config.COVERAGE_MIN or len(missing) <= 1 or n_oracle < 30 and len(missing) <= 2
        passed = cov_ok and order >= 0.97 and not bad_by_page[p] and math_ok and not hyg and (disc is None or not disc)
        return {
            "page_idx": p, "method": method, "coverage": round(recall, 4), "order": round(order, 4),
            "missing": missing[:40], "extra": extra, "katex_total": len(spans_by_page[p]), "katex_bad": bad_by_page[p],
            "math_ok": math_ok, "hygiene": hyg, "judge": disc, "passed": passed,
        }

    with ThreadPoolExecutor(workers if use_judge else 1) as ex:
        results = list(ex.map(one, rows))
    con.executemany(
        "INSERT OR REPLACE INTO b_page_checks VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        [(r["page_idx"], method, r["coverage"], r["katex_total"], len(r["katex_bad"]),
          json.dumps(r["katex_bad"]), int(r["math_ok"]),
          json.dumps({"issues": r["hygiene"], "order": r["order"], "missing": r["missing"]}),
          json.dumps(r["judge"]), int(r["passed"]), time.time()) for r in results],
    )
    con.commit()
    n_pass = sum(r["passed"] for r in results)
    log(f"[{book.id}] check {method}: {n_pass}/{len(results)} pages pass")
    return sorted(results, key=lambda r: r["page_idx"])
