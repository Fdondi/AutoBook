"""The concept-leak checker, shared by the digester and the reader (plan §2.3).

  allowed set   concepts introduced in seen units ∪ prereqs_external
  mentioned     concepts the seen text itself names: they may be named, not explained
  lexical pass  names/aliases that are distinctive by spelling (multiword, notation-like
                or rare words) and multi-symbol notation, matched deterministically; a
                term is fine when any allowed concept owns it (duplicates share terms)
  semantic pass closed-set tagging by an LLM over the blocked concepts nearest to the
                text, minus near-duplicates of allowed ones (same prompt in digest and reader)

Rules live in the package's concept table, not in code.
"""

import collections
import json
import re
import sqlite3
from dataclasses import dataclass, field

import numpy as np

from . import config, llm
from .extract import mdutil
from .terms import normalize, spelling_distinctive

_SINGLE_SYMBOL = re.compile(r"^[A-Za-z]$")
NEAR_DUPLICATE = 0.90             # a blocked concept this similar to an allowed one is not a semantic candidate


def plural(word: str) -> str:
    w = word.lower()
    if w.endswith("ix") or w.endswith("ex"):
        return word[:-2] + "ices"                   # matrix -> matrices, vertex -> vertices
    if w.endswith("is"):
        return word[:-2] + "es"                     # axis -> axes
    if w.endswith("y") and len(w) > 2 and w[-2] not in "aeiou":
        return word[:-1] + "ies"
    if w.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    return word + "s"


def term_regex(term: str) -> str:
    """Regex for a term: case-insensitive words, flexible hyphen/space, plural last word."""
    words = re.findall(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]", term.strip())
    if not words:
        return r"(?!)"
    parts = []
    for i, w in enumerate(words):
        if i == len(words) - 1 and re.match(r"[A-Za-z]{3,}$", w):
            parts.append(f"(?:{re.escape(w)}|{re.escape(plural(w))})")
        else:
            parts.append(re.escape(w))
    body = r"[\s\-]*".join(parts)
    return r"(?<![A-Za-z0-9])" + body + r"(?![A-Za-z0-9])"


def notation_key(tex: str) -> str | None:
    """Canonical notation usable for matching, or None when too generic.

    Distinctive: a multi-letter operator or group name (SO(3), se(3), Ad_T, Rot) or a
    calligraphic/fraktur/blackboard letter. Instance symbols (T_{ab}, \\omega_s, {c}, R^T)
    are how a book writes particular objects, not concepts, and are never matched.
    """
    c = mdutil.canon_math(tex)
    if len(c) < 3 or _SINGLE_SYMBOL.match(c):
        return None
    plain = re.sub(r"\\[a-zA-Z]+", " ", c)
    if re.search(r"\\(mathcal|mathfrak|mathbb)", c) or re.search(r"[A-Za-z]{2,}", plain):
        return c
    return None


@dataclass
class Concept:
    concept_id: str
    name: str
    aliases: list[str]
    notation: list[str]
    introduced_ordinal: int | None
    introduced_unit: str | None
    lexical_ok: bool = True
    patterns: list[tuple[str, re.Pattern]] = field(default_factory=list)   # (term key, regex), distinctive terms
    loose: list[re.Pattern] = field(default_factory=list)                   # every term, for tagging questions
    notation_keys: list[str] = field(default_factory=list)


class Lexicon:
    """All concepts of one book with compiled matchers."""

    def __init__(self, concepts: list[Concept], externals: set[str], embeddings: np.ndarray | None = None,
                 uses: dict[str, set[str]] | None = None):
        self.concepts = concepts
        self.by_id = {c.concept_id: c for c in concepts}
        self.externals = externals
        self.embeddings = embeddings
        self.uses = uses or {}                       # unit_id -> concept ids the unit names
        self.term_owners: dict[str, set[str]] = collections.defaultdict(set)
        for c in concepts:
            terms = [c.name] + [a for a in c.aliases if len(a) >= 3]
            for t in terms:
                self.term_owners[normalize(t)].add(c.concept_id)
            c.patterns = [(normalize(t), re.compile(term_regex(t), re.I)) for t in terms if spelling_distinctive(t)]
            c.loose = [re.compile(term_regex(t), re.I) for t in terms]
            c.notation_keys = [k for k in (notation_key(n) for n in c.notation) if k]
            for k in c.notation_keys:
                self.term_owners["$" + k].add(c.concept_id)

    @classmethod
    def from_db(cls, con: sqlite3.Connection) -> "Lexicon":
        ordinals = {r["unit_id"]: r["ordinal"] for r in con.execute("SELECT unit_id, ordinal FROM units")}
        rows = con.execute("SELECT * FROM concepts ORDER BY concept_id").fetchall()
        concepts = [
            Concept(r["concept_id"], r["name"], json.loads(r["aliases"] or "[]"), json.loads(r["notation"] or "[]"),
                    ordinals.get(r["introduced_in_unit"]), r["introduced_in_unit"], bool(r["lexical_ok"]))
            for r in rows
        ]
        emb = None
        if rows and rows[0]["embedding"] is not None:
            emb = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        externals = {r[0] for r in con.execute("SELECT concept_id FROM prereqs_external")}
        uses: dict[str, set[str]] = collections.defaultdict(set)
        for r in con.execute("SELECT concept_id, unit_id FROM concept_uses"):
            uses[r[1]].add(r[0])
        return cls(concepts, externals, emb, uses)

    def allowed(self, seen_ordinals: set[int]) -> set[str]:
        out = set(self.externals)
        for c in self.concepts:
            if c.introduced_ordinal is not None and c.introduced_ordinal in seen_ordinals:
                out.add(c.concept_id)
        return out

    def named_in(self, units) -> set[str]:
        """Concepts the given units name (any role): they may be named, not explained."""
        out = set()
        for u in units:
            out |= self.uses.get(u, set())
        return out

    def lexical(self, md: str, *, only_lexical_ok: bool = True) -> dict[str, str]:
        """{concept_id: matched text} for distinctive names/aliases and notation found in Markdown."""
        text = mdutil.md_to_text(mdutil.MATH_RE.sub(" ", md))
        maths = [mdutil.canon_math(t) for t, _ in mdutil.math_spans(md)]
        hits = {}
        for c in self.concepts:
            if c.lexical_ok or not only_lexical_ok:
                for _, p in c.patterns:
                    m = p.search(text)
                    if m:
                        hits[c.concept_id] = m.group(0)
                        break
            if c.concept_id not in hits and c.notation_keys:
                for k in c.notation_keys:
                    if any(k in s for s in maths):
                        hits[c.concept_id] = f"${k}$"
                        break
        return hits

    def named_concepts(self, question: str) -> set[str]:
        """Concepts a student's question names, by any of their terms (a question names
        concepts on purpose, so common words count here)."""
        text = mdutil.md_to_text(question)
        maths = [mdutil.canon_math(t) for t, _ in mdutil.math_spans(question)]
        return {c.concept_id for c in self.concepts
                if any(p.search(text) for p in c.loose) or any(k in s for k in c.notation_keys for s in maths)}

    def term_allowed(self, matched: str, allowed: set[str]) -> bool:
        """A matched term or notation is fine if any allowed concept owns it."""
        key = "$" + matched.strip("$") if matched.startswith("$") else normalize(matched)
        return bool(self.term_owners.get(key, set()) & allowed)


@dataclass
class LeakReport:
    lexical: dict[str, str]
    semantic: list[str]

    @property
    def violations(self) -> list[str]:
        return sorted(set(self.lexical) | set(self.semantic))

    def ok(self) -> bool:
        return not self.violations


def check(lex: Lexicon, md: str, allowed: set[str], *, model: str, mentioned: set[str] = frozenset(),
          sources: str | None = None, semantic: bool = True, tag: str = "") -> LeakReport:
    """Leak check of generated text against an allowed concept set.

    `mentioned`: concepts the reader has seen named; they may be named, not explained.
    `sources`: the passages the text is built from (the unit being summarized, the passages
    an answer was retrieved from). Whatever they name or present may be used: the leak we
    look for is reliance on later material *beyond* the sources.
    """
    in_sources = lex.named_concepts(sources) if sources else set()
    lexical = {cid: hit for cid, hit in lex.lexical(md).items()
               if cid not in allowed and cid not in mentioned and cid not in in_sources
               and not lex.term_allowed(hit, allowed | in_sources)}
    sem = []
    if semantic:
        blocked = [c for c in lex.concepts
                   if c.concept_id not in allowed and c.concept_id not in lexical and c.concept_id not in in_sources]
        blocked = _not_near_allowed(lex, blocked, allowed)
        cands = _nearest(lex, md, blocked, config.SEMANTIC_CANDIDATES)
        if cands:
            known = _nearest(lex, md, [lex.by_id[a] for a in allowed if a in lex.by_id], 12)
            sem = semantic_tags(md, cands, known=[c.name for c in known], sources=sources, model=model, tag=tag)
    return LeakReport(lexical, sem)


def _not_near_allowed(lex: Lexicon, pool: list[Concept], allowed: set[str]) -> list[Concept]:
    if lex.embeddings is None or not allowed:
        return pool
    idx = {c.concept_id: i for i, c in enumerate(lex.concepts)}
    a = lex.embeddings[[idx[x] for x in allowed if x in idx]]
    if not len(a):
        return pool
    keep = []
    for c in pool:
        if float((a @ lex.embeddings[idx[c.concept_id]]).max()) < NEAR_DUPLICATE:
            keep.append(c)
    return keep


def _nearest(lex: Lexicon, md: str, pool: list[Concept], k: int) -> list[Concept]:
    if not pool or lex.embeddings is None:
        return pool[:k]
    q = llm.embed([mdutil.md_to_text(md)[:4000]], kind="query")[0]
    idx = {c.concept_id: i for i, c in enumerate(lex.concepts)}
    scores = [(float(lex.embeddings[idx[c.concept_id]] @ q), c) for c in pool]
    scores.sort(key=lambda s: -s[0])
    return [c for _, c in scores[:k]]


def semantic_tags(md: str, candidates: list[Concept], *, known: list[str] = (), sources: str | None = None,
                  model: str, tag: str = "") -> list[str]:
    listing = "\n".join(f"- {c.concept_id}: {c.name}" + (f" (also: {', '.join(c.aliases[:3])})" if c.aliases else "")
                        for c in candidates)
    schema = {
        "type": "object",
        "properties": {"relies_on": {"type": "array", "items": {
            "type": "object",
            "properties": {"id": {"type": "string", "enum": [c.concept_id for c in candidates]},
                           "quote": {"type": "string"}, "why": {"type": "string"}},
            "required": ["id", "quote", "why"]}}},
        "required": ["relies_on"],
    }
    knows = ("The student HAS already learned (using these is fine): " + "; ".join(known) + "\n\n") if known else ""
    src = (f"Source passages the text is based on (whatever they present is fine to use):\n<sources>\n"
           f"{sources[:9000]}\n</sources>\n\n") if sources else ""
    msgs = [
        {"role": "system", "content": llm.load_prompt("leak_semantic")},
        {"role": "user", "content": f"{knows}{src}Concepts the student has NOT learned yet:\n{listing}\n\n"
                                    f"Text to check:\n<text>\n{md}\n</text>"},
    ]
    res = llm.chat(msgs, model, schema=schema, max_tokens=1500, tag=tag)
    ids = {c.concept_id for c in candidates}
    text = _squash(md)
    # a tag counts only with a real quote from the text as evidence
    return [t["id"] for t in res["relies_on"]
            if t["id"] in ids and len(t["quote"].split()) >= 3 and _squash(t["quote"]) in text]


def _squash(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", mdutil.md_to_text(s).lower()).strip()
