"""Stage 3: chunk units, embed chunks (shared embedder), build the BM25 index."""

import re

from .. import config, llm
from ..books import Book
from ..extract import mdutil

WORDS_PER_CHUNK = int(config.CHUNK_TOKENS / 1.4)


def _chunks(body: str) -> list[tuple[int, int, str]]:
    """Paragraph-aligned spans of about CHUNK_TOKENS tokens."""
    paras = [(m.start(), m.end()) for m in re.finditer(r"(?:(?!\n\s*\n).)+", body, re.S)]
    out, start, end, n = [], None, None, 0
    for a, b in paras:
        w = len(body[a:b].split())
        if start is not None and n + w > WORDS_PER_CHUNK and n > WORDS_PER_CHUNK // 3:
            out.append((start, end, body[start:end]))
            start, n = None, 0
        start = a if start is None else start
        end = b
        n += w
    if start is not None:
        out.append((start, end, body[start:end]))
    return out


def embed_units(book: Book, log=print) -> dict:
    con = book.connect()
    units = con.execute("SELECT unit_id, title, section_path, body FROM units WHERE kind != 'front' ORDER BY ordinal").fetchall()
    rows = []
    for u in units:
        for k, (a, b, text) in enumerate(_chunks(u["body"])):
            rows.append((f"{u['unit_id']}#{k}", u["unit_id"], k, a, b, text, u["title"]))
    # Embed plain text with the section title for context; store Markdown for display.
    vecs = llm.embed([f"{r[6]}\n{mdutil.md_to_text(r[5])}" for r in rows], kind="document")
    con.execute("DELETE FROM chunks")
    con.execute("DELETE FROM chunk_vec")
    con.execute("DELETE FROM chunk_fts")
    con.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?)", [r[:6] for r in rows])
    con.executemany("INSERT INTO chunk_vec VALUES (?,?)", [(r[0], v.tobytes()) for r, v in zip(rows, vecs)])
    con.executemany("INSERT INTO chunk_fts VALUES (?,?)",
                    [(r[0], r[6] + "\n" + mdutil.md_to_text(r[5])) for r in rows])
    con.commit()
    log(f"[{book.id}] embed: {len(rows)} chunks from {len(units)} units")
    return {"chunks": len(rows)}
