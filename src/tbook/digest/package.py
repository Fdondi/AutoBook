"""Stage 11 (exercises) and stage 12 (package)."""

import json
import re
import sqlite3
import time

from .. import config, leak
from ..books import Book

EXERCISE_START = re.compile(
    r"^\s*(?:\*\*)?(?:Exercise|Question|Problem)\s+(\d+(?:\.\d+)*)\.?(?:\*\*)?|^\s*(?:\*\*)?(\d+)\.(?:\*\*)?\s", re.M)


def exercises(book: Book, log=print) -> dict:
    """Split exercise units into individual exercises and link them to concepts (lexical tagger)."""
    con = book.connect()
    lex = leak.Lexicon.from_db(con)
    ordinal = {r["unit_id"]: r["ordinal"] for r in con.execute("SELECT unit_id, ordinal FROM units")}
    rows = []
    for u in con.execute("SELECT unit_id, body FROM units WHERE kind='exercise' ORDER BY ordinal"):
        body = u["body"]
        starts = [(m.start(), m.group(1) or m.group(2)) for m in EXERCISE_START.finditer(body)]
        if not starts:
            starts = [(0, None)]
        for k, (a, label) in enumerate(starts):
            b = starts[k + 1][0] if k + 1 < len(starts) else len(body)
            text = body[a:b].strip()
            if len(text.split()) < 5:
                continue
            found = sorted(lex.named_concepts(text))
            needed = [lex.by_id[c].introduced_unit for c in found if lex.by_id[c].introduced_unit]
            earliest = max(needed, key=lambda x: ordinal.get(x, 0)) if needed else None
            rows.append((f"{u['unit_id']}#e{k}", u["unit_id"], label, text, json.dumps(found), earliest, None))
    con.execute("DELETE FROM exercises")
    con.executemany("INSERT INTO exercises VALUES (?,?,?,?,?,?,?)", rows)
    con.commit()
    log(f"[{book.id}] exercises: {len(rows)}")
    return {"exercises": len(rows)}


def package(book: Book, stage_models: dict, log=print) -> dict:
    con = book.connect()
    manifest = {
        "book_id": book.id, "title": book.cfg["title"], "authors": book.cfg.get("authors", ""),
        "source": book.cfg["pdf"], "license": book.cfg.get("license", ""), "digest_version": "0.1",
        "embedder_id": config.EMBED_MODEL, "embedder_dim": str(config.EMBED_DIM),
        "llm_ids": json.dumps(stage_models), "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    con.execute("DELETE FROM manifest")
    con.executemany("INSERT INTO manifest VALUES (?,?)", manifest.items())
    con.commit()
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    book.package.unlink(missing_ok=True)
    con.execute(f"VACUUM INTO '{book.package.as_posix()}'")
    out = sqlite3.connect(book.package)
    for (name,) in out.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'b\\_%' ESCAPE '\\'").fetchall():
        out.execute(f"DROP TABLE {name}")
    out.commit()
    out.execute("VACUUM")
    out.close()
    size = book.package.stat().st_size
    log(f"[{book.id}] package: {book.package} ({size / 1e6:.1f} MB)")
    return {"bytes": size}
