"""Kill gate (plan §4, Phase 2): leak rate of the reader on trap questions.

traps    for sampled concepts, the digest model writes a question a student
         would ask just before the concept is introduced, whose best answer
         uses it without naming it (evals/<book>/traps.jsonl, editable)
answer   the reader (small model, same pipeline as `tbook ask`) answers each
         trap with the seen set ending just before the concept's unit
grade    the digest model, an independent grader with a wider candidate set
         (all concepts introduced later, nearest first), lists leaked concepts
report   leak rate vs the threshold fixed in config before running
"""

import json
import random

import numpy as np

from .. import config, llm, lmstudio
from ..books import Book
from ..db import connect
from ..leak import Lexicon

GRADE_CANDIDATES = 60
PASSAGE_CHARS = 3500


def traps_path(book: Book):
    d = config.EVALS_DIR / book.id
    d.mkdir(parents=True, exist_ok=True)
    return d / "traps.jsonl"


def build_traps(book: Book, n: int = 60, seed: int = 7, log=print) -> list[dict]:
    con = connect(book.package, readonly=True)
    lex = Lexicon.from_db(con)
    units = {r["unit_id"]: dict(r) for r in con.execute("SELECT unit_id, ordinal, title, body, kind, chapter FROM units")}
    by_ord = {u["ordinal"]: u for u in units.values()}
    content = sorted(u["ordinal"] for u in units.values() if u["kind"] == "content")
    defs = {r["concept_id"]: r["definition"] for r in con.execute("SELECT concept_id, definition FROM concepts")}
    pool = [c for c in lex.concepts if c.introduced_ordinal and c.introduced_ordinal > content[0] + 2
            and c.concept_id not in lex.externals]
    rng = random.Random(seed)
    rng.shuffle(pool)
    system = llm.load_prompt("trap_question")
    traps = []
    for c in pool[:n]:
        prev = max((o for o in content if o < c.introduced_ordinal), default=None)
        if prev is None:
            continue
        before = by_ord[prev]
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": f"Concept: {c.name}\nAlso called: {', '.join(c.aliases[:4]) or '-'}\n"
                                            f"Book's definition: {defs.get(c.concept_id, '')}\n\n"
                                            f"Passage the student just read ({before['title']}):\n<passage>\n{before['body'][:3000]}\n</passage>"}]
        q = llm.chat(msgs, config.DIGEST_MODEL, max_tokens=200, tag=f"{book.id}:trap:{c.concept_id}").strip().strip('"')
        if any(t.lower() in q.lower() for t in [c.name] + c.aliases if len(t) >= 4):
            continue                                      # the question gives the concept away
        traps.append({"concept": c.concept_id, "name": c.name, "seen_through": before["unit_id"], "question": q})
    with open(traps_path(book), "w", encoding="utf-8") as f:
        for t in traps:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    log(f"[{book.id}] traps: {len(traps)} written to {traps_path(book)}")
    return traps


def run(book: Book, log=print) -> dict:
    from ..reader.answer import Reader
    traps = [json.loads(l) for l in traps_path(book).read_text(encoding="utf-8").splitlines() if l.strip()]
    # answer phase: the small reader model
    lmstudio.ensure_embedder()
    lmstudio.ensure_loaded(config.READER_MODEL, config.READER_CTX, 1)
    reader = Reader(book.id)
    answers = []
    for t in traps:
        cut = reader.units[t["seen_through"]]["ordinal"]
        seen = {u for u, r in reader.units.items() if r["ordinal"] <= cut}
        ans = reader.ask(t["question"], seen, mode="ask")
        answers.append({**t, "kind": ans.kind, "text": ans.text, "checker": ans.leak, "attempts": ans.attempts,
                        "ttft_s": ans.stats.get("ttft_s"), "seen_ordinal": cut,
                        "drew_on": list(dict.fromkeys(ans.retrieved + [c["unit"] for c in ans.citations]))})
        log(f"  [{ans.kind:9s}] {t['question'][:80]}")
    # grade phase: the larger digest model as independent grader
    lmstudio.ensure_loaded(config.DIGEST_MODEL, config.DIGEST_CTX, config.DIGEST_PARALLEL)
    lex = reader.lex
    defs = {r["concept_id"]: r["definition"] for r in reader.con.execute("SELECT concept_id, definition FROM concepts")}
    system = llm.load_prompt("leak_grade")
    leaks = 0
    graded = []
    for a in answers:
        if a["kind"] in ("forward", "fallback") or not a["text"]:
            graded.append({**a, "leaked": []})
            continue
        later = [c for c in lex.concepts if c.introduced_ordinal and c.introduced_ordinal > a["seen_ordinal"]
                 and c.concept_id not in lex.externals]
        if lex.embeddings is not None and later:
            q = llm.embed([a["text"]], kind="query")[0]
            idx = {c.concept_id: i for i, c in enumerate(lex.concepts)}
            later.sort(key=lambda c: -float(lex.embeddings[idx[c.concept_id]] @ q))
        cands = later[:GRADE_CANDIDATES]
        if a["concept"] not in {c.concept_id for c in cands} and a["concept"] in lex.by_id:
            cands.append(lex.by_id[a["concept"]])
        listing = "\n".join(f"- {c.concept_id}: {c.name} — {defs.get(c.concept_id, '')[:140]}" for c in cands)
        # what the student has read and the tutor drew on: restating it is not a leak
        read = [u for u in a["drew_on"] if reader.units.get(u, {}).get("ordinal", 10**9) <= a["seen_ordinal"]]
        bodies = {r["unit_id"]: r["body"] for r in reader.con.execute(
            f"SELECT unit_id, body FROM units WHERE unit_id IN ({','.join('?' * len(read))})", read)}
        passages = "\n\n".join(f'<passage id="{u}">\n{bodies[u][:PASSAGE_CHARS]}\n</passage>' for u in read if u in bodies)
        schema = {"type": "object", "properties": {"leaked": {"type": "array", "items": {
            "type": "object", "properties": {"id": {"type": "string", "enum": [c.concept_id for c in cands]},
                                             "evidence": {"type": "string"}}, "required": ["id", "evidence"]}}},
                  "required": ["leaked"]}
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": f"Concepts introduced later:\n{listing}\n\n"
                                            f"Passages the student has read that the tutor drew on:\n{passages or '(none)'}\n\n"
                                            f"Answer:\n<answer>\n{a['text']}\n</answer>"}]
        res = llm.chat(msgs, config.DIGEST_MODEL, schema=schema, max_tokens=800, tag=f"{book.id}:grade:{a['concept']}")
        graded.append({**a, "leaked": res["leaked"]})
        leaks += bool(res["leaked"])
    n = len(graded)
    kinds = {k: sum(1 for g in graded if g["kind"] == k) for k in ("generated", "qa", "forward", "fallback")}
    stats = {
        "traps": n, "leaked_answers": leaks, "leak_rate": round(leaks / max(1, n), 4),
        "threshold": config.MAX_LEAK_RATE, "pass": leaks / max(1, n) <= config.MAX_LEAK_RATE,
        "answer_kinds": kinds,
        "checker_caught": sum(1 for g in graded if g.get("attempts", 0) > 1 or g["kind"] == "fallback"),
        "median_ttft_s": float(np.median([g["ttft_s"] for g in graded if g.get("ttft_s")])) if any(g.get("ttft_s") for g in graded) else None,
    }
    out = config.EVALS_DIR / book.id / "leak_eval.json"
    out.write_text(json.dumps({"stats": stats, "answers": graded}, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"[{book.id}] leak eval: {stats}")
    return stats
