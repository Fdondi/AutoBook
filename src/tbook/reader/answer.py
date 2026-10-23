"""Answer pipeline (plan §3.4): scope -> tag -> branch -> retrieve -> generate ->
check -> cite -> log. Deterministic steps decide; the small model only phrases.
"""

import json
import re
import sqlite3
import time
from dataclasses import dataclass, field

import numpy as np

from .. import config, leak, llm, retrieval
from ..books import load as load_book
from ..db import connect

CITE_RE = re.compile(r"\[([a-z]+:[^\]\s]+)\]")
TAG_SIM_MIN = 0.80          # embedding match of a question to a concept name (lexical match always counts)


@dataclass
class Answer:
    kind: str                                   # forward | qa | generated | fallback
    text: str
    citations: list[dict] = field(default_factory=list)
    needed_unseen: list[dict] = field(default_factory=list)
    leak: dict | None = None
    attempts: int = 0
    stats: dict = field(default_factory=dict)
    scope: dict = field(default_factory=dict)
    retrieved: list[str] = field(default_factory=list)


class Reader:
    def __init__(self, book_id: str, model: str = config.READER_MODEL):
        book = load_book(book_id)
        if not book.package.exists():
            raise SystemExit(f"no package for {book_id}: run `tbook digest {book_id}` first")
        self.book = book
        self.model = model
        self.con: sqlite3.Connection = connect(book.package, readonly=True)
        self.lex = leak.Lexicon.from_db(self.con)
        self.index = retrieval.Index(self.con)
        self.units = {r["unit_id"]: dict(r) for r in self.con.execute(
            "SELECT unit_id, ordinal, section, title, printed_page_label, kind FROM units")}
        self.edges = {}
        for r in self.con.execute("SELECT from_concept, to_concept FROM concept_edges"):
            self.edges.setdefault(r[0], []).append(r[1])
        qa = self.con.execute("SELECT qa_id, question, answer, cited_units, earliest_unit_needed, embedding FROM qa").fetchall()
        self.qa = [dict(r) for r in qa]
        self.qa_vecs = (np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in qa])
                        if qa else np.zeros((0, config.EMBED_DIM), np.float32))

    # -- scope -------------------------------------------------------------

    def seen_through(self, section: str) -> set[str]:
        """All units up to and including the last unit of `section` (sequential reading)."""
        match = [u for u in self.units.values() if u["section"] == section or u["section"].startswith(section + ".")
                 or u["unit_id"] == section]
        if not match:
            raise SystemExit(f"unknown section {section}")
        cut = max(u["ordinal"] for u in match)
        return {u["unit_id"] for u in self.units.values() if u["ordinal"] <= cut}

    def where(self, concept_id: str) -> dict:
        c = self.lex.by_id[concept_id]
        u = self.units.get(c.introduced_unit or "", {})
        return {"concept": c.name, "unit": c.introduced_unit, "section": u.get("section"),
                "title": u.get("title"), "page": u.get("printed_page_label")}

    def _closure(self, ids: set[str]) -> set[str]:
        out, stack = set(), list(ids)
        while stack:
            x = stack.pop()
            if x not in out:
                out.add(x)
                stack.extend(self.edges.get(x, []))
        return out

    # -- pipeline ----------------------------------------------------------

    def ask(self, question: str, seen: set[str], *, mode: str = "ask", on_token=None) -> Answer:
        t0 = time.time()
        seen_ord = {self.units[u]["ordinal"] for u in seen if u in self.units}
        allowed = self.lex.allowed(seen_ord)
        mentioned = self.lex.named_in(seen) - allowed        # named by the book so far: nameable, not explainable
        qvec = llm.embed([question], kind="query")[0]

        # 2. tag the question
        tagged = self.lex.named_concepts(question)
        if self.lex.embeddings is not None:
            sims = self.lex.embeddings @ qvec
            for i in np.argsort(-sims)[:3]:
                if sims[i] >= TAG_SIM_MIN:
                    tagged.add(self.lex.concepts[int(i)].concept_id)
        needed_unseen = sorted((tagged - allowed) - self.lex.externals,
                               key=lambda c: (self.lex.by_id[c].introduced_ordinal or 10**9, c))
        if mode == "preview" and needed_unseen:
            allowed |= self._closure(set(needed_unseen))
            seen = seen | {u for u in self.units if self.units[u]["ordinal"] <=
                           max(self.lex.by_id[c].introduced_ordinal or 0 for c in needed_unseen)}
        scope = {"seen_units": len(seen), "allowed_concepts": len(allowed),
                 "tagged": sorted(tagged), "mode": mode}

        # 3a. needs unseen concepts: point forward (the reader chooses spoilers)
        if needed_unseen and mode == "ask":
            ans = Answer("forward", "", needed_unseen=[self.where(c) for c in needed_unseen], scope=scope)
            ans.stats = {"total_s": round(time.time() - t0, 2)}
            return ans
        # 3b. pre-generated Q&A
        if len(self.qa) and mode != "preview":
            sims = self.qa_vecs @ qvec
            order = np.argsort(-sims)
            for i in order[:5]:
                q = self.qa[int(i)]
                if sims[i] < config.QA_MATCH_MIN:
                    break
                if q["earliest_unit_needed"] in seen:
                    rep = leak.check(self.lex, q["answer"], allowed, model=self.model, mentioned=mentioned,
                                     semantic=False)
                    if rep.ok():
                        cites = [self._cite(u) for u in json.loads(q["cited_units"])]
                        ans = Answer("qa", q["answer"], citations=cites, scope=scope,
                                     leak={"lexical": {}, "semantic": []})
                        ans.stats = {"total_s": round(time.time() - t0, 2), "qa_sim": round(float(sims[i]), 3),
                                     "qa_question": q["question"]}
                        return ans

        # 4. retrieve from seen units only
        hits = self.index.search(question, seen, qvec=qvec)
        if not hits:
            return Answer("fallback", "Nothing you have read so far covers this yet.", scope=scope)
        # mode "within": the question names unseen concepts; they lead the off-limits list
        blocked = [self.lex.by_id[c] for c in needed_unseen if c not in allowed]
        blocked += [c for c in leak._nearest(self.lex, question + "\n" + hits[0].text,
                                             [c for c in self.lex.concepts if c.concept_id not in allowed], 8)
                    if c not in blocked]
        passages = "\n\n".join(f'<passage id="{h.unit_id}">\n{h.text}\n</passage>' for h in hits)
        note = "Not yet covered (do not use): " + ("; ".join(c.name for c in blocked) or "(none)")
        msgs = [{"role": "system", "content": llm.load_prompt("reader_answer")},
                {"role": "user", "content": f"{note}\n\n{passages}\n\nQuestion: {question}"}]

        # 5-7. generate, check, regenerate once, fall back
        text, stats, rep = "", {}, None
        for attempt in (1, 2):
            text, stats = self._generate(msgs, on_token if attempt == 1 else None)
            rep = leak.check(self.lex, text, allowed, model=self.model, mentioned=mentioned,
                             sources="\n\n".join(h.text for h in hits), tag="")
            if rep.ok():
                break
            if attempt == 1:
                names = ", ".join(self.lex.by_id[c].name for c in rep.violations)
                msgs = msgs + [{"role": "assistant", "content": text},
                               {"role": "user", "content": f"Your answer used concepts I have not read yet: {names}. "
                                                           f"Answer again without them or their ideas."}]
        leak_info = {"lexical": rep.lexical, "semantic": rep.semantic}
        if not rep.ok():
            pointers = [self.where(c) for c in rep.violations]
            ans = Answer("fallback", "", needed_unseen=pointers, leak=leak_info, attempts=2, scope=scope,
                         retrieved=[h.unit_id for h in hits])
            ans.stats = {**stats, "total_s": round(time.time() - t0, 2)}
            return ans
        cited = list(dict.fromkeys(m for m in CITE_RE.findall(text) if m in {h.unit_id for h in hits}))
        ans = Answer("generated", text, citations=[self._cite(u) for u in cited], leak=leak_info,
                     attempts=attempt, scope=scope, retrieved=[h.unit_id for h in hits])
        ans.stats = {**stats, "total_s": round(time.time() - t0, 2)}
        return ans

    def _generate(self, msgs, on_token):
        parts, stats = [], {}
        for ev in llm.chat_stream(msgs, self.model, temperature=0.2, max_tokens=700):
            if isinstance(ev, dict):
                stats = ev
            else:
                parts.append(ev)
                if on_token:
                    on_token(ev)
        return "".join(parts).strip(), stats

    def _cite(self, unit_id: str) -> dict:
        u = self.units.get(unit_id, {})
        return {"unit": unit_id, "section": u.get("section"), "title": u.get("title"), "page": u.get("printed_page_label")}
