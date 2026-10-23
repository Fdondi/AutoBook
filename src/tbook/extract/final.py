"""Stage 1d: assemble the final per-page Markdown (local only).

1. Primary: of the glyph-hinted transcriptions, the one that passes the
   deterministic checks best (coverage, KaTeX, math presence, hygiene).
2. Math consensus against every other transcription of the page:
   - display equations located on the page: the candidate whose symbols,
     scripts and typefaces align best with the PDF's own glyphs wins
     (ties go to the most votes);
   - inline math: a form with strictly more votes than the primary's wins;
   - an expression no two transcriptions agree on, or whose chosen form the
     PDF glyphs contradict, is flagged for manual review (not guessed).
3. Pages failing deterministic checks get a targeted repair (the VLM sees
   the page, the glyph layer, its previous transcription and the exact
   problems), then consensus runs again, up to REPAIR_ATTEMPTS times.
4. b_page_final + reports/extraction.md (flagged pages with their spans).
"""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

import pymupdf

from .. import config, llm
from ..books import Book
from . import checks, consensus, convert, glyphs, mdutil

PRIMARY_CANDIDATES = ("vlm_ml", "vlm_ml_fast")
VOTERS = ("vlm_ml", "vlm_ml_fast", "vlm_hint", "vlm_fast", "mineru", "marker")
GLYPH_MIN = 0.55          # chosen display form must explain this share of the best possible score


def _check_rank(r: dict) -> tuple:
    return (int(r["passed"]), -len(r["katex_bad"]), r["coverage"], -r.get("extra", 0), r["order"])


def _region(item: dict, blocks: list[dict], page) -> pymupdf.Rect | None:
    box = consensus.locate(item, blocks)
    if box is None:
        return None
    r = page.rect
    x0, y0, x1, y1 = box
    return pymupdf.Rect((x0 - 0.01) * r.width, (y0 - 0.006) * r.height, (x1 + 0.01) * r.width, (y1 + 0.006) * r.height)


def reconcile(book: Book, page_idx: int, primary_md: str, others: dict[str, str], blocks: list[dict]) -> tuple[str, list[dict]]:
    """Primary Markdown with consensus math applied, and the spans flagged for review."""
    doc = pymupdf.open(book.pdf)
    page = doc[page_idx]
    math_re = re.compile(book.cfg["math_fonts"])
    spans, _ = consensus.disputes(primary_md, others)
    forms_tex: dict[str, str] = {}
    for src in [primary_md, *others.values()]:
        for t, _ in mdutil.math_spans(src):
            forms_tex.setdefault(mdutil.canon_math(t), t)
    chosen, flagged = {}, []
    for i, s in enumerate(spans):
        forms = [f for f in s["votes"] if "$" not in forms_tex.get(f, s["tex"])]
        if s["canon"] not in forms:
            forms.append(s["canon"])
        opts = [forms_tex.get(f, s["tex"]) for f in forms]
        ok = mdutil.katex_errors([(o, s["display"]) for o in opts])
        cand = [(f, o, len(s["votes"].get(f, []))) for f, o, e in zip(forms, opts, ok) if e is None]
        if not cand:
            flagged.append({"tex": s["tex"], "why": "no candidate parses"})
            continue
        region = _region({"span": s, "options": [o for _, o, _ in cand]}, blocks, page) if s["display"] else None
        if region is not None:
            ref = glyphs.pdf_tokens(page, math_re, region)
            scores = [glyphs.align_score(glyphs.latex_tokens(o), ref) for _, o, _ in cand]
            best = max(scores)
            top = [k for k, sc in enumerate(scores) if sc >= best - 1e-6]
            k = max(top, key=lambda j: cand[j][2])
            perfect = glyphs.align_score(glyphs.latex_tokens(cand[k][1]), glyphs.latex_tokens(cand[k][1]))
            if ref and best < GLYPH_MIN * max(perfect, 1e-6):
                flagged.append({"tex": cand[k][1], "why": "PDF glyphs disagree with every candidate"})
        else:
            k = max(range(len(cand)), key=lambda j: cand[j][2])
            mine = next((v for f, _, v in cand if f == s["canon"]), 0)
            if cand[k][2] <= mine:
                k = next((j for j, c in enumerate(cand) if c[0] == s["canon"]), k)
            if cand[k][2] < 2 and s["status"] not in ("trivial",) and len(others) >= 2:
                flagged.append({"tex": cand[k][1], "why": "no two transcriptions agree"})
        if cand[k][0] != s["canon"]:
            chosen[i] = cand[k][1]
    return consensus.apply(primary_md, spans, chosen), flagged


def _feedback(r: dict) -> str:
    lines = []
    if r["missing"]:
        lines.append("- Words from the page are missing from your transcription (check these passages): "
                     + ", ".join(sorted(set(r["missing"]))[:30]))
    if r["order"] < 0.97:
        lines.append("- Parts of the page are out of reading order.")
    for k in r["katex_bad"][:8]:
        lines.append(f"- This math does not parse in KaTeX ({k['error'][:120]}): `{k['tex'][:160]}`")
    if not r["math_ok"]:
        lines.append("- The page contains typeset math but your transcription has no LaTeX math.")
    for h in r["hygiene"]:
        lines.append(f"- {h}")
    return "\n".join(lines)


def repair_one(book: Book, page: dict, prev_md: str, problems: str, attempt: int) -> str:
    extra = ("A previous transcription of this page had these problems:\n" + problems +
             "\n\nPrevious transcription:\n<previous>\n" + mdutil.collapse_loops(prev_md) + "\n</previous>\n\n"
             "Look at the page image again and produce a corrected, complete transcription of the whole page. "
             "Fix exactly these problems and keep everything that was already right.")
    heads = json.loads(page["running_heads"])
    msgs = convert.vlm_messages(page["png"], None, extra=extra,
                                math_layer=convert.math_layer(book, page["page_idx"], heads))
    try:
        # a later round with an unchanged page would otherwise replay the cached first attempt
        md = llm.chat(msgs, config.VISION_MODEL, temperature=0.2 * (attempt - 1), max_tokens=8000,
                      tag=f"{book.id}:repair{attempt}:{page['page_idx']}")
    except llm.LLMError:
        return prev_md                    # no repair this round; the page stays as it was
    return convert.postprocess(convert._strip_outer_fence(md), heads, book)


def finalize(book: Book, pages: list[int] | None = None, *, attempts: int = config.REPAIR_ATTEMPTS,
             log=print) -> dict:
    con = book.connect()
    page_rows = {r["page_idx"]: dict(r) for r in con.execute("SELECT * FROM b_pages")}
    pages = sorted(page_rows) if pages is None else pages

    def md_of(p, method):
        row = con.execute("SELECT md, meta FROM b_page_md WHERE page_idx=? AND method=?", (p, method)).fetchone()
        return (row["md"], json.loads(row["meta"] or "{}")) if row else (None, {})

    # 1. primary per page
    results = {m: {r["page_idx"]: r for r in checks.check(book, m, pages, use_judge=False, log=log)}
               for m in PRIMARY_CANDIDATES}
    primary = {}
    for p in pages:
        options = [(m, results[m][p]) for m in PRIMARY_CANDIDATES if p in results[m]]
        if not options:
            raise SystemExit(f"[{book.id}] page {p}: no glyph-hinted transcription; run convert first")
        primary[p] = max(options, key=lambda mr: _check_rank(mr[1]))[0]

    # 2. consensus
    def reconcile_page(p, prim_md):
        others = {m: md_of(p, m)[0] for m in VOTERS if m != primary[p] and md_of(p, m)[0] is not None}
        blocks = md_of(p, "mineru")[1].get("blocks", [])
        return reconcile(book, p, prim_md, others, blocks)

    final, flags = {}, {}
    for p in pages:
        final[p], flags[p] = reconcile_page(p, md_of(p, primary[p])[0])
    _store(con, final, method="consensus")
    rechecked = {r["page_idx"]: r for r in checks.check(book, "consensus", pages, use_judge=False, log=log)}

    # 3. repair pages that fail deterministic checks
    failing = [p for p in pages if not rechecked[p]["passed"]]
    attempts_used = {p: 0 for p in pages}
    for attempt in range(1, attempts + 1):
        if not failing:
            break
        log(f"[{book.id}] finalize: repair round {attempt} on {len(failing)} pages")
        with ThreadPoolExecutor(config.VISION_PARALLEL) as ex:
            repaired = dict(ex.map(lambda p: (p, repair_one(book, page_rows[p], final[p], _feedback(rechecked[p]), attempt)),
                                   failing))
        _store(con, repaired, method=f"repair{attempt}")
        rr = {r["page_idx"]: r for r in checks.check(book, f"repair{attempt}", failing, use_judge=False, log=log)}
        for p in failing:
            if _check_rank(rr[p]) > _check_rank(rechecked[p]):
                final[p], flags[p] = reconcile_page(p, repaired[p])
                rechecked[p] = rr[p]
                attempts_used[p] = attempt
        failing = [p for p in failing if not rechecked[p]["passed"]]

    # 4. final rows + report
    rows = []
    for p in pages:
        r = rechecked[p]
        status = ("flagged" if flags[p] or not r["passed"] else
                  "repaired" if attempts_used[p] else "auto_pass")
        note = {"primary": primary[p], "flags": flags[p], "problems": None if r["passed"] else _feedback(r)}
        rows.append((p, final[p], primary[p], status, attempts_used[p], json.dumps(note, ensure_ascii=False)))
    con.executemany("INSERT OR REPLACE INTO b_page_final VALUES (?,?,?,?,?,?)", rows)
    con.commit()
    counts = {s: sum(1 for r in rows if r[3] == s) for s in ("auto_pass", "repaired", "flagged")}
    log(f"[{book.id}] finalize: {counts}, {sum(len(f) for f in flags.values())} expressions flagged for review")
    apply_review(book, log=log)
    write_report(book)
    return counts


def apply_review(book: Book, log=print) -> dict:
    """Apply books/<id>.review.toml: exact fixes and accepted pages -> status 'reviewed'."""
    import tomllib
    path = config.BOOKS_DIR / f"{book.id}.review.toml"
    if not path.exists():
        return {}
    review = tomllib.loads(path.read_text(encoding="utf-8"))
    con = book.connect()
    done, problems = 0, []
    touched = set()
    for fx in review.get("fix", []):
        row = con.execute("SELECT md FROM b_page_final WHERE page_idx=?", (fx["page"],)).fetchone()
        if row is None:
            continue
        md = row["md"]
        if md.count(fx["old"]) == 1:
            con.execute("UPDATE b_page_final SET md=? WHERE page_idx=?", (md.replace(fx["old"], fx["new"]), fx["page"]))
            done += 1
            touched.add(fx["page"])
        elif fx["new"] in md:
            touched.add(fx["page"])           # already applied
        else:
            problems.append(f"page {fx['page']}: fix not applicable ({fx['old'][:50]!r})")
    for acc in review.get("accept", []):
        touched.add(acc["page"])
    for p in touched:
        row = con.execute("SELECT note FROM b_page_final WHERE page_idx=?", (p,)).fetchone()
        if row is not None:
            note = json.loads(row["note"] or "{}") | {"reviewed": True}
            con.execute("UPDATE b_page_final SET status='reviewed', note=? WHERE page_idx=?", (json.dumps(note), p))
    con.commit()
    for msg in problems:
        log(f"[{book.id}] review: {msg}")
    log(f"[{book.id}] review: {done} fixes applied, {len(touched)} pages reviewed")
    return {"fixes": done, "reviewed": len(touched), "problems": problems}


def write_report(book: Book) -> None:
    """reports/extraction.md from every finalized page of the book."""
    con = book.connect()
    rows = con.execute("SELECT f.page_idx, f.status, f.note, p.label FROM b_page_final f "
                       "JOIN b_pages p USING(page_idx) ORDER BY f.page_idx").fetchall()
    counts: dict[str, int] = {}
    body, n_flags = [], 0
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        note = json.loads(r["note"] or "{}")
        n_flags += len(note.get("flags") or [])
        if r["status"] == "flagged":
            lines = [f"## Page {r['label']} (pdf index {r['page_idx']})", ""]
            if note.get("problems"):
                lines += [note["problems"], ""]
            lines += [f"- {f['why']}: `{f['tex'][:200]}`" for f in note.get("flags") or []]
            body.append("\n".join(lines) + "\n")
    head = [f"# Extraction report: {book.cfg['title']}", "", f"{len(rows)} pages finalized.", "", "```json",
            json.dumps({**counts, "flagged_expressions": n_flags}, indent=2), "```", ""]
    (book.reports_dir / "extraction.md").write_text("\n".join(head + body), encoding="utf-8")


def _store(con, mds: dict[int, str], method: str) -> None:
    con.executemany("INSERT OR REPLACE INTO b_page_md VALUES (?,?,?,?,?)",
                    [(p, method, md, "{}", time.time()) for p, md in mds.items()])
    con.commit()
