"""Digest orchestrator: runs the 12 stages per book, skipping stages whose
inputs (upstream state + prompts) are unchanged. LLM calls are cached, so an
interrupted run resumes where it stopped.
"""

import hashlib
import json
import time

from .. import config, lmstudio, llm
from ..books import Book
from . import concepts, embed, generate, package, segment

PROMPTS = {
    "segment": [], "embed": [],
    "concepts_extract": ["concept_extract"], "concepts_resolve": ["concept_merge"],
    "concepts_edges": ["concept_edges"], "concepts_external": ["concept_external"],
    "summaries": ["summary", "leak_semantic"], "qa": ["qa_generate", "grounding_judge", "leak_semantic"],
    "verify": [], "exercises": [], "package": [],
}
DIGEST_STAGES = list(PROMPTS)


def _stage_fn(name: str):
    return {
        "segment": segment.segment,
        "embed": embed.embed_units,
        "concepts_extract": concepts.extract,
        "concepts_resolve": concepts.resolve,
        "concepts_edges": concepts.edges,
        "concepts_external": concepts.external,
        "summaries": generate.summaries,
        "qa": generate.qa,
        "verify": generate.verify_report,
        "exercises": package.exercises,
    }[name]


def _source(fn) -> str:
    """Source of the module implementing a stage: editing the code re-runs the stage."""
    import inspect
    return inspect.getsource(inspect.getmodule(fn))


def _needs_llm(name: str) -> bool:
    return name.startswith("concepts_") or name in ("summaries", "qa")


def run(book: Book, *, start: str | None = None, only: str | None = None, force: bool = False, log=print) -> None:
    con = book.connect()
    stages = DIGEST_STAGES
    if only:
        stages = [only]
    elif start:
        stages = stages[stages.index(start):]
    upstream = _upstream_hash(con, stages[0])
    for name in stages:
        code = "" if name == "package" else _source(_stage_fn(name))
        cfg = json.dumps(book.cfg, sort_keys=True, default=str)       # e.g. recap_titles
        h = hashlib.sha256((upstream + name + code + cfg + "".join(llm.load_prompt(p) for p in PROMPTS[name])
                            ).encode()).hexdigest()
        if not force and con.execute("SELECT 1 FROM b_stage_runs WHERE stage=? AND input_hash=?", (name, h)).fetchone():
            log(f"[{book.id}] {name}: up to date")
            upstream = h
            continue
        lmstudio.ensure_embedder()
        if _needs_llm(name):
            lmstudio.ensure_loaded(config.DIGEST_MODEL, config.DIGEST_CTX, config.DIGEST_PARALLEL)
        t0 = time.time()
        u0 = llm.usage.summary()
        if name == "package":
            stats = package.package(book, {"digest": config.DIGEST_MODEL, "vision": config.VISION_MODEL,
                                           "embed": config.EMBED_MODEL}, log=log)
        else:
            stats = _stage_fn(name)(book, log=log)
        u1 = llm.usage.summary()
        stats = dict(stats or {})
        stats.update({"seconds": round(time.time() - t0, 1),
                      "llm_calls": u1["calls"] - u0["calls"], "llm_cached": u1["cached"] - u0["cached"],
                      "completion_tokens": u1["completion_tokens"] - u0["completion_tokens"]})
        con.execute("INSERT OR REPLACE INTO b_stage_runs VALUES (?,?,?,?)", (name, h, time.time(), str(stats)))
        con.commit()
        log(f"[{book.id}] {name}: done in {stats['seconds']}s ({stats['llm_calls']} calls, "
            f"{stats['completion_tokens']} tokens out)")
        upstream = h


def _upstream_hash(con, first: str) -> str:
    """Hash of the stage before `first` (or of the extracted pages for segment)."""
    i = DIGEST_STAGES.index(first)
    if i == 0:
        row = con.execute("SELECT count(*), coalesce(sum(length(md)), 0), coalesce(max(attempts), 0) FROM b_page_final").fetchone()
        return f"pages:{row[0]}:{row[1]}:{row[2]}"
    row = con.execute("SELECT input_hash FROM b_stage_runs WHERE stage=?", (DIGEST_STAGES[i - 1],)).fetchone()
    return row[0] if row else "missing"


def status(book: Book) -> list[tuple]:
    con = book.connect()
    rows = {r["stage"]: r for r in con.execute("SELECT * FROM b_stage_runs")}
    out = []
    pages = con.execute("SELECT count(*) FROM b_pages").fetchone()[0]
    final = con.execute("SELECT status, count(*) FROM b_page_final GROUP BY status").fetchall()
    out.append(("extract", f"{pages} pages; final: " + ", ".join(f"{s}={n}" for s, n in final) if final else f"{pages} pages; no final pages yet"))
    for s in DIGEST_STAGES:
        r = rows.get(s)
        out.append((s, "pending" if r is None else time.strftime("%Y-%m-%d %H:%M", time.localtime(r["finished_at"])) + "  " + r["stats"]))
    return out
