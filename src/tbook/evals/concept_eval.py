"""Concept-extraction oracles.

anchors   every bold "Definition/Proposition/Theorem/Lemma/Corollary X.Y" must
          produce a concept defined in that unit, with evidence from that paragraph
bold      every bold term in body text must match a concept (name or alias)
          introduced in that unit or earlier
listing   concepts introduced per chapter, for a by-hand precision check
"""

import collections
import json
import re

from ..books import Book
from ..terms import normalize

ANCHOR_RE = re.compile(r"\*\*((?:Definition|Proposition|Theorem|Lemma|Corollary)\s+\d+(?:\.\d+)*)\.?\*\*")
BOLD_RE = re.compile(r"(?<!\*)\*\*([^*\n]{2,60}?)\*\*(?!\*)")
LABEL_RE = re.compile(r"^(Definition|Proposition|Theorem|Lemma|Corollary|Example|Exercise|Figure|Table|Proof|Algorithm|Require|Chapter|Part|Fig\.)\b", re.I)


def evaluate(book: Book, chapters: set[str] | None = None) -> dict:
    con = book.connect()
    units = [dict(r) for r in con.execute("SELECT * FROM units WHERE kind='content' ORDER BY ordinal")]
    if chapters:
        units = [u for u in units if u["chapter"] in chapters]
    ordinal = {r["unit_id"]: r["ordinal"] for r in con.execute("SELECT unit_id, ordinal FROM units")}
    concepts = [dict(r) for r in con.execute("SELECT * FROM concepts")]
    keys = {}
    for c in concepts:
        for n in [c["name"]] + json.loads(c["aliases"] or "[]"):
            keys.setdefault(normalize(n), []).append(c)
    defined_in = collections.defaultdict(list)
    for r in con.execute("SELECT * FROM b_concept_candidates WHERE role='define'"):
        defined_in[r["unit_id"]].append(dict(r))

    overview = set(book.cfg.get("overview_chapters", []))
    anchors, anchor_hits, anchor_miss = 0, 0, []
    bold, bold_hits, bold_miss = 0, 0, []
    for u in units:
        body = u["body"]
        paras = re.split(r"\n\s*\n", body)
        for k, para in enumerate(paras):
            for m in ANCHOR_RE.finditer(para):
                anchors += 1
                # the statement may start in the paragraph after a bare label
                plain = re.sub(r"[*_$\\{}]", "", para + "\n" + (paras[k + 1] if k + 1 < len(paras) else ""))
                hit = any(d["evidence"] and _fuzzy_in(d["evidence"], plain) for d in defined_in[u["unit_id"]])
                if hit:
                    anchor_hits += 1
                else:
                    anchor_miss.append(f"{u['unit_id']}: {m.group(1)}")
            if u["chapter"] in overview:
                continue                      # previews bold terms they define later
            for m in BOLD_RE.finditer(para):
                term = m.group(1).strip().strip(":.,")
                if LABEL_RE.match(term) or len(term) < 3 or re.fullmatch(r"[\d.\s]+", term):
                    continue
                bold += 1
                cands = keys.get(normalize(term), [])
                ok = any(c["introduced_in_unit"] and ordinal.get(c["introduced_in_unit"], 10**9) <= u["ordinal"]
                         for c in cands)
                if ok:
                    bold_hits += 1
                else:
                    bold_miss.append(f"{u['unit_id']}: {term}" + (" (introduced later)" if cands else ""))
    by_chapter = collections.defaultdict(list)
    unit_chapter = {u["unit_id"]: u["chapter"] for u in con.execute("SELECT unit_id, chapter FROM units")}
    for c in concepts:
        ch = unit_chapter.get(c["introduced_in_unit"], "-")
        if not chapters or ch in chapters:
            by_chapter[ch].append(c)
    return {
        "anchor_recall": round(anchor_hits / anchors, 3) if anchors else None, "anchors": anchors,
        "anchor_missed": anchor_miss,
        "bold_recall": round(bold_hits / bold, 3) if bold else None, "bold_terms": bold, "bold_missed": bold_miss,
        "concepts_by_chapter": {ch: sorted((c["name"], c["introduced_in_unit"], c["kind"]) for c in cs)
                                for ch, cs in sorted(by_chapter.items())},
    }


def _fuzzy_in(evidence: str, para: str) -> bool:
    ev = re.sub(r"[*_$\\{}]", "", evidence).strip()
    words = ev.split()
    if not words:
        return False
    probe = " ".join(words[: min(6, len(words))])
    return probe.lower() in para.lower() or ev[:40].lower() in para.lower()
