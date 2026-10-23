"""Stages 8-10: summaries, pre-generated Q&A + misconceptions, verification.

Every generated item is leak-checked (lexical + semantic) against the concepts
introduced up to its scope, and every answer is checked for grounding in the
passages it cites. A failing item is regenerated once with explicit feedback,
then dropped. Stage 10 audits what was kept and writes reports/verify.md.
"""

import collections
import json
import threading
from concurrent.futures import ThreadPoolExecutor

from .. import config, leak, llm, retrieval
from ..books import Book

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"one_line": {"type": "string"}, "paragraph": {"type": "string"}},
    "required": ["one_line", "paragraph"],
}
QA_KINDS = ["definition", "why", "how", "contrast", "misconception"]
GROUND_SCHEMA = {
    "type": "object",
    "properties": {"grounded": {"type": "boolean"}, "unsupported": {"type": "array", "items": {"type": "string"}}},
    "required": ["grounded", "unsupported"],
}


class Scope:
    """Concepts allowed after reading through a given unit ordinal (sequential reading)."""

    def __init__(self, con):
        self.con = con
        self.lex = leak.Lexicon.from_db(con)

    def allowed(self, ordinal: int, sources=()) -> set[str]:
        """Introduced up to `ordinal`, plus whatever the source passages themselves name:
        a summary or answer may name what the passage it is about names."""
        return ({c.concept_id for c in self.lex.concepts
                 if c.introduced_ordinal is not None and c.introduced_ordinal <= ordinal}
                | self.lex.externals | self.lex.named_in(sources))

    def blocked_near(self, text: str, allowed: set[str], k: int = 8) -> list[leak.Concept]:
        pool = [c for c in self.lex.concepts if c.concept_id not in allowed]
        return leak._nearest(self.lex, text, pool, k)

    def check(self, text: str, allowed: set[str], tag: str, sources: str | None = None) -> leak.LeakReport:
        return leak.check(self.lex, text, allowed, model=config.DIGEST_MODEL, sources=sources, tag=tag)

    def names(self, ids) -> str:
        return ", ".join(self.lex.by_id[i].name for i in ids if i in self.lex.by_id)


_DB_LOCK = threading.Lock()          # worker threads share one sqlite connection


def _record(con, kind: str, item_id: str, attempt: int, rep: leak.LeakReport | None, grounded, verdict: str):
    with _DB_LOCK:
        con.execute("INSERT OR REPLACE INTO b_verify VALUES (?,?,?,?,?,?)", (
            kind, item_id, attempt,
            json.dumps({"lexical": rep.lexical, "semantic": rep.semantic} if rep else None),
            None if grounded is None else int(grounded), verdict))


def _blocked_note(blocked: list[leak.Concept]) -> str:
    if not blocked:
        return "Not yet introduced: (none nearby)."
    return "Not yet introduced (do not use): " + "; ".join(
        c.name + (f" ({', '.join(c.notation[:2])})" if c.notation else "") for c in blocked)


# ---------------------------------------------------------------- stage 8

def summaries(book: Book, log=print, workers: int = config.DIGEST_PARALLEL) -> dict:
    con = book.connect()
    scope = Scope(con)
    con.execute("DELETE FROM b_verify WHERE item_kind='summary'")
    con.commit()
    system = llm.load_prompt("summary")
    units = con.execute("SELECT * FROM units WHERE kind='content' ORDER BY ordinal").fetchall()
    verdicts = collections.Counter()

    def gen(scope_kind, scope_id, title, source, ordinal, sources):
        allowed = scope.allowed(ordinal, sources)
        blocked = scope.blocked_near(source, allowed)
        base = (f"Book: {book.cfg['title']}\nScope: {scope_kind} {title}\n{_blocked_note(blocked)}\n\n"
                f"<passage>\n{source}\n</passage>")
        feedback = ""
        for attempt in (1, 2):
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": base + feedback}]
            res = llm.chat(msgs, config.DIGEST_MODEL, schema=SUMMARY_SCHEMA, max_tokens=800,
                           tag=f"{book.id}:summary:{scope_kind}:{scope_id}:{attempt}")
            text = res["one_line"] + "\n\n" + res["paragraph"]
            rep = scope.check(text, allowed, f"{book.id}:leak:summary:{scope_id}:{attempt}", sources=source)
            if rep.ok():
                _record(con, "summary", f"{scope_kind}:{scope_id}", attempt, rep, None, "pass")
                return scope_kind, scope_id, res["one_line"], res["paragraph"], f"pass{attempt}"
            _record(con, "summary", f"{scope_kind}:{scope_id}", attempt, rep, None, "leak")
            feedback = (f"\n\nYour previous summary used concepts the student has not learned yet: "
                        f"{scope.names(rep.violations)}. Rewrite it without them or their ideas.")
        return None

    def safe_gen(job):
        try:
            return gen(*job)
        except llm.LLMError as e:                    # one bad call drops one item, not the stage
            log(f"[{book.id}] summary {job[0]} {job[1]} dropped: {e}")
            return None

    def run(jobs):
        with ThreadPoolExecutor(workers) as ex:
            out = list(ex.map(safe_gen, jobs))
        for r in out:
            verdicts[r[4] if r else "dropped"] += 1
        return {(r[0], r[1]): r for r in out if r}

    unit_rows = run([("unit", u["unit_id"], u["title"], u["body"], u["ordinal"], [u["unit_id"]]) for u in units])

    sections = collections.OrderedDict()
    for u in units:
        sections.setdefault(u["section"], []).append(u)
    sec_jobs = []
    for sec, us in sections.items():
        src = "\n\n".join(f"[{u['unit_id']}] {unit_rows[('unit', u['unit_id'])][3]}"
                          for u in us if ("unit", u["unit_id"]) in unit_rows)
        if src:
            sec_jobs.append(("section", sec, us[0]["title"], src, us[-1]["ordinal"], [u["unit_id"] for u in us]))
    sec_rows = run(sec_jobs)

    chapters = collections.OrderedDict()
    for u in units:
        chapters.setdefault(u["chapter"], []).append(u)
    ch_jobs = []
    for ch, us in chapters.items():
        secs = list(dict.fromkeys(u["section"] for u in us))
        src = "\n\n".join(f"[{s}] {sec_rows[('section', s)][3]}" for s in secs if ("section", s) in sec_rows)
        if src:
            title = json.loads(us[0]["section_path"])[0]
            ch_jobs.append(("chapter", ch, title, src, us[-1]["ordinal"], [u["unit_id"] for u in us]))
    ch_rows = run(ch_jobs)

    con.execute("DELETE FROM summaries")
    con.executemany("INSERT INTO summaries VALUES (?,?,?,?)",
                    [r[:4] for r in list(unit_rows.values()) + list(sec_rows.values()) + list(ch_rows.values())])
    con.commit()
    log(f"[{book.id}] summaries: {dict(verdicts)}")
    return dict(verdicts)


# ---------------------------------------------------------------- stage 9

def _qa_schema(unit_ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "items": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"}, "answer": {"type": "string"},
                    "kind": {"type": "string", "enum": QA_KINDS},
                    "cites": {"type": "array", "items": {"type": "string", "enum": unit_ids}},
                },
                "required": ["question", "answer", "kind", "cites"]}},
            "misconceptions": {"type": "array", "items": {
                "type": "object",
                "properties": {"statement": {"type": "string"}, "correction": {"type": "string"}},
                "required": ["statement", "correction"]}},
        },
        "required": ["items", "misconceptions"],
    }


def grounded(book_id: str, answer: str, passages: dict[str, str], tag: str) -> tuple[bool, list[str]]:
    src = "\n\n".join(f"[{k}]\n{v}" for k, v in passages.items())
    msgs = [{"role": "system", "content": llm.load_prompt("grounding_judge")},
            {"role": "user", "content": f"<passages>\n{src}\n</passages>\n\n<answer>\n{answer}\n</answer>"}]
    res = llm.chat(msgs, config.DIGEST_MODEL, schema=GROUND_SCHEMA, max_tokens=600, tag=tag)
    return res["grounded"], res["unsupported"]


def qa(book: Book, log=print, workers: int = config.DIGEST_PARALLEL) -> dict:
    con = book.connect()
    scope = Scope(con)
    index = retrieval.Index(con)
    con.execute("DELETE FROM b_verify WHERE item_kind='qa'")
    con.commit()
    system = llm.load_prompt("qa_generate")
    units = con.execute("SELECT * FROM units WHERE kind='content' ORDER BY ordinal").fetchall()
    ordinal = {r["unit_id"]: r["ordinal"] for r in con.execute("SELECT unit_id, ordinal FROM units")}
    bodies = {r["unit_id"]: r["body"] for r in con.execute("SELECT unit_id, body FROM units")}
    verdicts = collections.Counter()

    def one(u):
        earlier = {uid for uid, o in ordinal.items() if o < u["ordinal"]}
        hits = index.search(u["title"] + "\n" + u["body"][:600], earlier, k=2, token_budget=900) if earlier else []
        passages = {u["unit_id"]: u["body"]}
        for h in hits:
            passages.setdefault(h.unit_id, h.text)
        allowed = scope.allowed(u["ordinal"], list(passages))
        blocked = scope.blocked_near(u["body"], allowed)
        src = "\n\n".join(f'<passage id="{k}">\n{v}\n</passage>' for k, v in passages.items())
        user = (f"Book: {book.cfg['title']}\nMain passage: {u['unit_id']} ({u['title']})\n"
                f"{_blocked_note(blocked)}\n\n{src}")
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        res = llm.chat(msgs, config.DIGEST_MODEL, schema=_qa_schema(list(passages)), max_tokens=3000,
                       tag=f"{book.id}:qa:{u['unit_id']}")
        kept, miscon = [], []
        for k, item in enumerate(res["items"][:4]):
            qid = f"{u['unit_id']}#q{k}"
            for attempt in (1, 2):
                text = item["question"] + "\n\n" + item["answer"]
                rep = scope.check(text, allowed, f"{book.id}:leak:qa:{qid}:{attempt}", sources=src)
                cites = [c for c in item["cites"] if c in passages] or [u["unit_id"]]
                ok_g, unsupported = grounded(book.id, item["answer"], {c: passages[c] for c in cites},
                                             f"{book.id}:ground:{qid}:{attempt}")
                _record(con, "qa", qid, attempt, rep, ok_g, "pass" if rep.ok() and ok_g else "fail")
                if rep.ok() and ok_g:
                    kept.append((qid, item, cites, attempt))
                    break
                if attempt == 2:
                    verdicts["qa_dropped"] += 1
                    break
                fb = []
                if not rep.ok():
                    fb.append(f"it used concepts the student has not learned yet: {scope.names(rep.violations)}")
                if not ok_g:
                    fb.append("it contains claims the passages do not support: " + "; ".join(unsupported[:3]))
                retry = msgs + [{"role": "assistant", "content": json.dumps({"items": [item], "misconceptions": []})},
                                {"role": "user", "content": "Rewrite that question/answer pair: " + " and ".join(fb)
                                 + ". Return it as the only item."}]
                r2 = llm.chat(retry, config.DIGEST_MODEL, schema=_qa_schema(list(passages)), max_tokens=1500,
                              tag=f"{book.id}:qa-retry:{qid}")
                if not r2["items"]:
                    verdicts["qa_dropped"] += 1
                    break
                item = r2["items"][0]
        for m in res["misconceptions"][:2]:
            rep = scope.check(m["statement"] + "\n" + m["correction"], allowed, f"{book.id}:leak:mis:{u['unit_id']}",
                              sources=src)
            if rep.ok():
                miscon.append((u["unit_id"], m["statement"], m["correction"]))
            else:
                verdicts["misconception_dropped"] += 1
        return u, kept, miscon

    def safe_one(u):
        try:
            return one(u)
        except llm.LLMError as e:
            log(f"[{book.id}] qa {u['unit_id']} dropped: {e}")
            verdicts["unit_failed"] += 1
            return u, [], []

    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(safe_one, units))

    qa_rows, mis_rows = [], []
    for u, kept, miscon in results:
        for qid, item, cites, attempt in kept:
            verdicts[f"qa_pass{attempt}"] += 1
            req = sorted(scope.lex.named_concepts(item["question"] + "\n" + item["answer"]))
            needed = [scope.lex.by_id[c].introduced_unit for c in req
                      if c in scope.lex.by_id and scope.lex.by_id[c].introduced_unit] + cites
            earliest = max(needed, key=lambda x: ordinal.get(x, 0))
            qa_rows.append([qid, u["unit_id"], item["question"], item["answer"], json.dumps(cites),
                            json.dumps(req), earliest, item["kind"]])
        mis_rows.extend(miscon)
    vecs = llm.embed([r[2] for r in qa_rows], kind="query") if qa_rows else []
    con.execute("DELETE FROM qa")
    con.execute("DELETE FROM misconceptions")
    con.executemany("INSERT INTO qa VALUES (?,?,?,?,?,?,?,?,?)",
                    [tuple(r) + (v.tobytes(),) for r, v in zip(qa_rows, vecs)])
    con.executemany("INSERT INTO misconceptions VALUES (?,?,?)", mis_rows)
    con.commit()
    log(f"[{book.id}] qa: {len(qa_rows)} Q&A, {len(mis_rows)} misconceptions; {dict(verdicts)}")
    return dict(verdicts)


# ---------------------------------------------------------------- stage 10

def verify_report(book: Book, log=print) -> dict:
    """Audit what was kept (deterministic lexical pass) and summarize verification outcomes."""
    con = book.connect()
    scope = Scope(con)
    ordinal = {r["unit_id"]: r["ordinal"] for r in con.execute("SELECT unit_id, ordinal FROM units")}
    residual = []
    bodies = {r["unit_id"]: r["body"] for r in con.execute("SELECT unit_id, body FROM units")}

    def lexical_violations(text, allowed, units):
        return leak.check(scope.lex, text, allowed, model=config.DIGEST_MODEL, semantic=False,
                          sources="\n\n".join(bodies.get(u, "") for u in units)).violations

    for r in con.execute("SELECT qa_id, unit_id, question, answer, cited_units FROM qa").fetchall():
        cited = json.loads(r["cited_units"])
        bad = lexical_violations(r["question"] + "\n" + r["answer"], scope.allowed(ordinal[r["unit_id"]], cited), cited)
        if bad:
            residual.append(f"qa {r['qa_id']}: {scope.names(bad)}")
    for r in con.execute("SELECT scope_id, one_line, paragraph FROM summaries WHERE scope='unit'").fetchall():
        bad = lexical_violations(r["one_line"] + "\n" + r["paragraph"],
                                 scope.allowed(ordinal[r["scope_id"]], [r["scope_id"]]), [r["scope_id"]])
        if bad:
            residual.append(f"summary {r['scope_id']}: {scope.names(bad)}")
    rows = con.execute("SELECT item_kind, attempt, verdict, leak, grounded FROM b_verify").fetchall()
    first = [r for r in rows if r["attempt"] == 1]
    leak_first = sum(1 for r in first if r["leak"] and (json.loads(r["leak"])["lexical"] or json.loads(r["leak"])["semantic"]))
    ground_fail = sum(1 for r in first if r["grounded"] == 0)
    n_qa = sum(1 for r in first if r["item_kind"] == "qa")
    stats = {
        "items_checked": len(first),
        "leak_rate_first_attempt": round(leak_first / max(1, len(first)), 4),
        "grounding_fail_rate_first_attempt": round(ground_fail / max(1, n_qa), 4),
        "kept_qa": con.execute("SELECT count(*) FROM qa").fetchone()[0],
        "kept_summaries": con.execute("SELECT count(*) FROM summaries").fetchone()[0],
        "residual_lexical_leaks": len(residual),
    }
    lines = [f"# Verification: {book.cfg['title']}", "", "```json", json.dumps(stats, indent=2), "```", ""]
    if residual:
        lines += ["## Residual lexical leaks (should be empty)", ""] + [f"- {x}" for x in residual]
    (book.reports_dir / "verify.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"[{book.id}] verify: {stats}")
    return stats
