"""Expected outputs of the desktop reader's deterministic steps, for the Android port's
parity test (app/app/src/test/.../ParityTest.kt).

    .venv/Scripts/python.exe tools/reader_parity.py mr

Questions are the leak gate's trap questions (evals/<book>/traps.jsonl), so they cover
the whole book; leak cases are the answers the gate graded. Query embeddings come from
the desktop embedder (LM Studio must serve it) and are stored, so retrieval is compared
apart from the phone's embedder; a few texts check the phone's embedder itself.
"""

import json
import sys
from pathlib import Path

import numpy as np

from tbook import config, leak, llm
from tbook.reader.answer import TAG_SIM_MIN, Reader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "app" / "app" / "src" / "test" / "resources"
EMBED_TEXTS = ["What is a twist?", "The screw axis of a rigid body.", "How fast does the body rotate?"]   # embedded as documents


def main(book_id: str) -> None:
    reader = Reader(book_id)
    lex = reader.lex
    traps = [json.loads(line) for line in (config.EVALS_DIR / book_id / "traps.jsonl").read_text(encoding="utf-8").splitlines()]
    graded = json.loads((config.EVALS_DIR / book_id / "leak_eval.json").read_text(encoding="utf-8"))["answers"]

    questions = []
    for t in traps:
        seen = reader.seen_through(t["seen_through"])
        seen_ord = {reader.units[u]["ordinal"] for u in seen}
        allowed = lex.allowed(seen_ord)
        qvec = llm.embed([t["question"]], kind="query")[0]
        named = lex.named_concepts(t["question"])
        tagged = set(named)
        sims = lex.embeddings @ qvec
        for i in np.argsort(-sims)[:3]:
            if sims[i] >= TAG_SIM_MIN:
                tagged.add(lex.concepts[int(i)].concept_id)
        needed = sorted((tagged - allowed) - lex.externals, key=lambda c: (lex.by_id[c].introduced_ordinal or 10**9, c))
        hits = reader.index.search(t["question"], seen, qvec=qvec)
        questions.append({
            "question": t["question"], "seen": sorted(seen), "qvec": [round(float(x), 6) for x in qvec],
            "allowed": sorted(allowed), "named": sorted(named), "needed_unseen": needed,
            "hits": [h.unit_id + "|" + h.chunk_id for h in hits],
        })

    leaks = []
    for a in graded:
        if not a["text"]:
            continue
        seen = reader.seen_through(a["seen_through"])
        allowed = lex.allowed({reader.units[u]["ordinal"] for u in seen})
        mentioned = lex.named_in(seen) - allowed
        rep = leak.check(lex, a["text"], allowed, model=config.READER_MODEL, mentioned=mentioned, semantic=False)
        leaks.append({"text": a["text"], "seen": sorted(seen), "lexical_all": lex.lexical(a["text"]), "violations": rep.lexical})

    embeds = [{"text": t, "vec": [round(float(x), 6) for x in llm.embed([t], kind="document")[0]]} for t in EMBED_TEXTS]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"parity-{book_id}.json"
    path.write_text(json.dumps({"book": book_id, "questions": questions, "leaks": leaks, "embeds": embeds}), encoding="utf-8")
    print(f"{path}: {len(questions)} questions, {len(leaks)} leak cases")


if __name__ == "__main__":
    main(sys.argv[1])
