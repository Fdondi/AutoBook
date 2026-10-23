"""Hybrid retrieval (FTS5 BM25 + vectors, reciprocal-rank fusion), hard-filtered
to an allowed set of units. Shared by the digester (Q&A grounding) and the reader.
"""

import re
import sqlite3
import threading
from dataclasses import dataclass

import numpy as np

from . import config, llm
from .extract import mdutil

RRF_K = 60
STOP = {"the", "a", "an", "of", "and", "or", "to", "in", "is", "are", "what", "how", "why", "does", "do",
        "can", "i", "it", "this", "that", "for", "with", "on", "be", "by", "as", "at", "from", "which"}


@dataclass
class Hit:
    chunk_id: str
    unit_id: str
    text: str
    score: float


class Index:
    def __init__(self, con: sqlite3.Connection):
        self.con = con
        self.lock = threading.Lock()                 # the connection may be shared by worker threads
        rows = con.execute(
            "SELECT c.chunk_id, c.unit_id, c.text, v.embedding FROM chunks c JOIN chunk_vec v USING(chunk_id) "
            "ORDER BY c.rowid").fetchall()
        self.ids = [r["chunk_id"] for r in rows]
        self.units = [r["unit_id"] for r in rows]
        self.texts = [r["text"] for r in rows]
        self.pos = {cid: i for i, cid in enumerate(self.ids)}
        self.vecs = (np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
                     if rows else np.zeros((0, config.EMBED_DIM), np.float32))

    def search(self, query: str, allowed_units: set[str] | None, *, k: int = config.RETRIEVE_K,
               token_budget: int = config.RETRIEVE_TOKEN_BUDGET, qvec: np.ndarray | None = None) -> list[Hit]:
        if not self.ids:
            return []
        mask = np.array([allowed_units is None or u in allowed_units for u in self.units])
        if not mask.any():
            return []
        ranks: dict[int, float] = {}
        # vector
        q = qvec if qvec is not None else llm.embed([query], kind="query")[0]
        sims = self.vecs @ q
        sims[~mask] = -np.inf
        for r, i in enumerate(np.argsort(-sims)[:50]):
            if np.isfinite(sims[i]):
                ranks[int(i)] = ranks.get(int(i), 0) + 1 / (RRF_K + r)
        # bm25
        terms = [t for t in re.findall(r"[A-Za-z][A-Za-z0-9]+", mdutil.md_to_text(query).lower()) if t not in STOP]
        if terms:
            fts = " OR ".join(f'"{t}"' for t in dict.fromkeys(terms))
            with self.lock:
                rows = self.con.execute(
                    "SELECT chunk_id FROM chunk_fts WHERE chunk_fts MATCH ? ORDER BY rank LIMIT 200", (fts,)).fetchall()
            r = 0
            for row in rows:
                i = self.pos.get(row[0])
                if i is None or not mask[i]:
                    continue
                ranks[i] = ranks.get(i, 0) + 1 / (RRF_K + r)
                r += 1
                if r >= 50:
                    break
        hits, used = [], 0
        for i, s in sorted(ranks.items(), key=lambda kv: -kv[1]):
            toks = int(len(self.texts[i].split()) * 1.4)
            if hits and used + toks > token_budget:
                continue
            hits.append(Hit(self.ids[i], self.units[i], self.texts[i], s))
            used += toks
            if len(hits) >= k:
                break
        return hits
