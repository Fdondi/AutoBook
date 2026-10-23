"""Stages 4-7: concept extraction, book-wide resolution, prerequisite edges,
external prerequisites.

4 extract   per unit, LLM with a JSON schema -> b_concept_candidates
5 resolve   merge candidates into concepts; introduced_in_unit = first `define`;
            concept_uses; lexical_ok; forward-reference report
6 edges     for each concept, which earlier concepts its definition relies on
7 external  used-but-never-defined concepts: background / book concept / noise
"""

import collections
import json
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .. import config, leak, llm
from ..books import Book
from ..terms import normalize

ROLE_RANK = {"define": 5, "example": 4, "use": 3, "exercise": 2, "mention": 1, "background": 0}
EMPHASIS_RE = re.compile(r"(?<![*\w])(\*\*|\*|__|_)(?!\s)([^*_\n$]{2,60}?)(?<!\s)\1(?![*\w])")
LABEL_RE = re.compile(r"\b((?:Definition|Proposition|Theorem|Lemma|Corollary|Algorithm)\s+\d+(?:\.\d+)*)")


def checklist(body: str) -> list[str]:
    """Emphasized terms and numbered labels of a unit: where textbooks define things."""
    items = []
    for m in EMPHASIS_RE.finditer(body):
        t = m.group(2).strip().strip(":.,")
        if len(t) >= 3 and not re.match(r"(Definition|Proposition|Theorem|Lemma|Corollary|Example|Exercise|Figure|"
                                        r"Table|Proof|Algorithm|Fig\.|Note)\b", t) and not re.fullmatch(r"[\d.\s]+", t):
            items.append(t)
    items += [m.group(1) for m in LABEL_RE.finditer(body)]
    return list(dict.fromkeys(items))
KINDS = ["definition", "theorem", "method", "notation", "property", "example", "other"]
EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "concepts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                    "notation": {"type": "array", "items": {"type": "string"}},
                    "kind": {"type": "string", "enum": KINDS},
                    "role": {"type": "string", "enum": list(ROLE_RANK)},
                    "evidence": {"type": "string"},
                },
                "required": ["name", "aliases", "notation", "kind", "role", "evidence"],
            },
        },
    },
    "required": ["concepts"],
}


def _units(con, kinds=("content", "exercise")):
    q = f"SELECT * FROM units WHERE kind IN ({','.join('?' * len(kinds))}) ORDER BY ordinal"
    return con.execute(q, kinds).fetchall()


# ---------------------------------------------------------------- stage 4

def extract(book: Book, log=print, workers: int = config.DIGEST_PARALLEL) -> dict:
    con = book.connect()
    units = _units(con)
    overview = set(book.cfg.get("overview_chapters", []))
    recap = re.compile(book.cfg["recap_titles"]) if book.cfg.get("recap_titles") else None
    system = llm.load_prompt("concept_extract")

    def one(u):
        path = " > ".join(json.loads(u["section_path"]))
        items = checklist(u["body"])
        check = ("\n\nChecklist (emphasized terms and numbered labels in this unit): " + "; ".join(items)) if items else ""
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": f"Book: {book.cfg['title']}\nSection: {path}{check}\n\n<unit>\n{u['body']}\n</unit>"}]
        res = llm.chat(msgs, config.DIGEST_MODEL, schema=EXTRACT_SCHEMA, max_tokens=4000,
                       tag=f"{book.id}:extract:{u['unit_id']}")
        rows = []
        for c in res["concepts"]:
            name = c["name"].strip()
            if not name:
                continue
            role = c["role"]
            if (u["chapter"] in overview or recap and recap.match(u["title"])) and role == "define":
                role = "mention"                      # previews and recaps do not define
            if u["kind"] == "exercise" and role in ("define", "use", "example"):
                role = "exercise"
            rows.append((u["unit_id"], name, json.dumps(c["aliases"]), json.dumps(c["notation"]),
                         c["kind"], role, c["evidence"][:300]))
        return rows

    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(one, units))
    con.execute("DELETE FROM b_concept_candidates")
    con.executemany("INSERT INTO b_concept_candidates VALUES (?,?,?,?,?,?,?)", [r for rs in results for r in rs])
    con.commit()
    n = sum(len(rs) for rs in results)
    log(f"[{book.id}] extract: {n} candidate mentions from {len(units)} units")
    return {"mentions": n, "units": len(units)}


# ---------------------------------------------------------------- stage 5

def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", normalize(name)).strip("-")[:60] or "concept"


MERGE_SCHEMA = {
    "type": "object",
    "properties": {"groups": {"type": "array", "items": {
        "type": "object",
        "properties": {"group": {"type": "integer"}, "sets": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}}},
        "required": ["group", "sets"]}}},
    "required": ["groups"],
}


def clean_alias(a: str) -> bool:
    """Aliases are names: a real word, no LaTeX or symbols, no numbered labels."""
    a = a.strip()
    return (len(a) <= 60 and bool(re.search(r"[A-Za-z]{3,}", a)) and not re.search(r"[\\^_{}$=]", a)
            and not LABEL_RE.fullmatch(a) and not re.match(r"(Equation|Eq\.|Figure|Example|Exercise)\s", a))


BOLD_TERM_RE = re.compile(r"(?<!\*)\*\*([^*\n]{2,60}?)\*\*(?!\*)")


def _apply_bold_rule(book: Book, con, mentions: list[dict]) -> int:
    """A concept set in bold in a unit is defined there (how these books mark definitions)."""
    units = {r["unit_id"]: r for r in con.execute("SELECT unit_id, body, chapter, kind, title FROM units")}
    overview = set(book.cfg.get("overview_chapters", []))
    recap = re.compile(book.cfg["recap_titles"]) if book.cfg.get("recap_titles") else None
    bold = {uid: {normalize(t) for t in BOLD_TERM_RE.findall(u["body"])} for uid, u in units.items()}
    n = 0
    for m in mentions:
        u = units.get(m["unit_id"])
        if u is None or u["kind"] != "content" or u["chapter"] in overview or recap and recap.match(u["title"]):
            continue
        names = [m["name"]] + [a for a in json.loads(m["aliases"]) if clean_alias(a)]
        if any(normalize(x) in bold[m["unit_id"]] for x in names):
            m["bold"] = True
            if m["role"] in ("use", "mention", "background"):
                m["role"] = "define"
                n += 1
    return n


def resolve(book: Book, log=print) -> dict:
    con = book.connect()
    ordinal = {r["unit_id"]: r["ordinal"] for r in con.execute("SELECT unit_id, ordinal FROM units")}
    bodies = {r["unit_id"]: r["body"] for r in con.execute("SELECT unit_id, body FROM units")}
    mentions = []
    for r in con.execute("SELECT * FROM b_concept_candidates"):
        m = dict(r)
        aliases = [a for a in json.loads(m["aliases"]) if clean_alias(a)]
        if LABEL_RE.fullmatch(m["name"].strip()) or not clean_alias(m["name"]):
            if not aliases:
                continue                                      # "Proposition 3.3" with no real name
            m["name"], aliases = aliases[0], aliases[1:]
        m["aliases"] = json.dumps(aliases)
        mentions.append(m)
    promoted = _apply_bold_rule(book, con, mentions)

    # 1. clusters by normalized name
    clusters: dict[str, list[dict]] = collections.defaultdict(list)
    for m in mentions:
        clusters[normalize(m["name"])].append(m)
    keys = sorted(clusters)

    # 2. candidate merges: alias/name collisions + embedding near-duplicates, confirmed by the LLM
    parent = {k: k for k in keys}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    pairs = set()
    key_set = set(keys)
    for k in keys:
        for m in clusters[k]:
            for a in json.loads(m["aliases"]):
                ak = normalize(a)
                if ak in key_set and ak != k:
                    pairs.add(tuple(sorted((k, ak))))
    emb = llm.embed(keys, kind="clustering") if keys else np.zeros((0, config.EMBED_DIM))
    if len(keys) > 1:
        sims = emb @ emb.T
        np.fill_diagonal(sims, 0)
        for i, j in zip(*np.where(np.triu(sims) >= 0.94)):
            pairs.add((keys[i], keys[j]))
    merged = _confirm_merges(book, pairs, clusters)
    for a, b in merged:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for k in keys:
        groups[find(k)].extend(clusters[k])

    # 3. build concepts
    concepts, uses, used_ids = [], [], set()
    for root, ms in groups.items():
        units = {m["unit_id"] for m in ms}
        n_background = sum(1 for m in ms if m["role"] == "background")
        defines = sorted((m for m in ms if m["role"] == "define"), key=lambda m: ordinal[m["unit_id"]])
        # background only when nothing in the book defines it: a wrong "background" is a leak
        background = not defines and n_background * 2 >= len(ms)
        # the book's emphasized definition beats an earlier informal one (e.g. a chapter preview)
        emphasized = [m for m in defines if m.get("bold")]
        if emphasized and emphasized[0] is not defines[0]:
            defines.remove(emphasized[0])
            defines.insert(0, emphasized[0])
        if not defines and len(units) < 2 and not background:
            continue                                          # one-off mention: noise
        surface = collections.Counter(m["name"] for m in (defines or ms)).most_common(1)[0][0]
        name_key = normalize(surface)
        aliases = sorted({a for m in ms for a in [m["name"]] + json.loads(m["aliases"])
                          if normalize(a) != name_key and a.strip()}, key=str.lower)
        notation = sorted({n for m in ms for n in json.loads(m["notation"]) if n.strip()})
        kind = collections.Counter(m["kind"] for m in (defines or ms)).most_common(1)[0][0]
        cid = _slug(surface)
        while cid in used_ids:
            cid += "-x"
        used_ids.add(cid)
        intro = defines[0]["unit_id"] if defines else None
        concepts.append({
            "concept_id": cid, "name": surface, "aliases": aliases[:12], "notation": notation[:8], "kind": kind,
            "introduced_in_unit": intro, "definition": defines[0]["evidence"] if defines else ms[0]["evidence"],
            "mentions": ms, "background": background,
        })
        best = {}
        for m in ms:
            if ROLE_RANK[m["role"]] > ROLE_RANK.get(best.get(m["unit_id"]), 0):
                best[m["unit_id"]] = m["role"]
        uses.extend((cid, u, r) for u, r in best.items())

    # 4. lexical_ok: the name already occurs as an ordinary word before its introduction
    lex = leak.Lexicon([leak.Concept(c["concept_id"], c["name"], c["aliases"], c["notation"], None, None)
                        for c in concepts], set())
    tagged = {(cid, u) for cid, u, _ in uses}
    ordered_units = sorted(bodies, key=lambda u: ordinal[u])
    for c, lc in zip(concepts, lex.concepts):
        c["lexical_ok"] = 1
        if not c["introduced_in_unit"]:
            continue
        cut = ordinal[c["introduced_in_unit"]]
        plain = 0
        for u in ordered_units:
            if ordinal[u] >= cut:
                break
            if (c["concept_id"], u) in tagged:
                continue
            if any(p.search(bodies[u]) for _, p in lc.patterns[:1]):
                plain += 1
        c["lexical_ok"] = int(plain <= 3)

    cemb = llm.embed([f"{c['name']}: {c['definition']}" for c in concepts], kind="document") if concepts else []
    con.execute("DELETE FROM concepts")
    con.execute("DELETE FROM concept_uses")
    con.executemany(
        "INSERT INTO concepts VALUES (?,?,?,?,?,?,?,?,?)",
        [(c["concept_id"], c["name"], json.dumps(c["aliases"]), json.dumps(c["notation"]), c["kind"],
          c["introduced_in_unit"], c["definition"], c["lexical_ok"], cemb[i].tobytes())
         for i, c in enumerate(concepts)],
    )
    con.executemany("INSERT OR IGNORE INTO concept_uses VALUES (?,?,?)", uses)
    con.execute("DELETE FROM prereqs_external")
    con.executemany("INSERT INTO prereqs_external VALUES (?, 'background')",
                    [(c["concept_id"],) for c in concepts if c["background"]])
    con.commit()
    _report_concepts(book, con, concepts, ordinal)
    stats = {
        "concepts": len(concepts),
        "defined": sum(1 for c in concepts if c["introduced_in_unit"]),
        "merges": len(merged), "lexical_off": sum(1 for c in concepts if not c["lexical_ok"]),
        "bold_promoted": promoted,
    }
    log(f"[{book.id}] resolve: {stats}")
    return stats


def _confirm_merges(book: Book, pairs: set, clusters: dict) -> list[tuple[str, str]]:
    """Group candidate pairs into small components and let the LLM partition them."""
    adj = collections.defaultdict(set)
    for a, b in pairs:
        adj[a].add(b)
        adj[b].add(a)
    seen, comps = set(), []
    for k in sorted(adj):
        if k in seen:
            continue
        comp, stack = [], [k]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.append(x)
            stack.extend(adj[x] - seen)
        for i in range(0, len(comp), 8):
            if len(comp[i:i + 8]) > 1:
                comps.append(sorted(comp[i:i + 8]))
    merged = []
    system = llm.load_prompt("concept_merge")
    for start in range(0, len(comps), 10):
        batch = comps[start:start + 10]
        lines = []
        for gi, comp in enumerate(batch):
            lines.append(f"Group {gi}:")
            for ci, k in enumerate(comp):
                ev = clusters[k][0]["evidence"][:120].replace("\n", " ")
                lines.append(f"  {ci}. {clusters[k][0]['name']} — \"{ev}\"")
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(lines)}]
        res = llm.chat(msgs, config.DIGEST_MODEL, schema=MERGE_SCHEMA, max_tokens=2000,
                       tag=f"{book.id}:merge:{start}")
        for g in res["groups"]:
            if not 0 <= g["group"] < len(batch):
                continue
            comp = batch[g["group"]]
            for s in g["sets"]:
                ids = [comp[i] for i in s if 0 <= i < len(comp)]
                merged.extend((ids[0], other) for other in ids[1:])
    return merged


def _report_concepts(book: Book, con, concepts: list[dict], ordinal: dict) -> None:
    units = {r["unit_id"]: r for r in con.execute("SELECT unit_id, title, printed_page_label FROM units")}
    lines = [f"# Concepts: {book.cfg['title']}", "", f"{len(concepts)} concepts.", ""]
    for c in sorted(concepts, key=lambda c: ordinal.get(c["introduced_in_unit"], 10**9)):
        u = units.get(c["introduced_in_unit"])
        where = f"{c['introduced_in_unit']} ({u['title']}, p. {u['printed_page_label']})" if u else "never defined"
        flag = "" if c["lexical_ok"] else " [semantic only]"
        lines.append(f"- **{c['name']}**{flag} — {where}; aliases: {', '.join(c['aliases'][:5]) or '—'}")
    (book.reports_dir / "concepts.md").write_text("\n".join(lines), encoding="utf-8")

    fwd = ["# Forward references", "", "Concepts used or mentioned before the unit that defines them.", ""]
    for c in concepts:
        if not c["introduced_in_unit"]:
            continue
        cut = ordinal[c["introduced_in_unit"]]
        early = sorted({m["unit_id"] for m in c["mentions"] if ordinal[m["unit_id"]] < cut and m["role"] != "mention"},
                       key=lambda u: ordinal[u])
        if early:
            fwd.append(f"- **{c['name']}** defined in {c['introduced_in_unit']}, used earlier in: {', '.join(early[:6])}")
    (book.reports_dir / "forward_refs.md").write_text("\n".join(fwd), encoding="utf-8")


# ---------------------------------------------------------------- stage 6

def edges(book: Book, log=print, workers: int = config.DIGEST_PARALLEL) -> dict:
    con = book.connect()
    ordinal = {r["unit_id"]: r["ordinal"] for r in con.execute("SELECT unit_id, ordinal FROM units")}
    bodies = {r["unit_id"]: r["body"] for r in con.execute("SELECT unit_id, body FROM units")}
    lex = leak.Lexicon.from_db(con)
    defs = {r["concept_id"]: r["definition"] for r in con.execute("SELECT concept_id, definition FROM concepts")}

    def pos(c):
        body = bodies[c.introduced_unit]
        p = body.find((defs.get(c.concept_id) or "")[:40])
        return (ordinal[c.introduced_unit], p if p >= 0 else len(body))

    defined = [c for c in lex.concepts if c.introduced_unit]
    order = {c.concept_id: pos(c) for c in defined}
    idx = {c.concept_id: i for i, c in enumerate(lex.concepts)}
    system = llm.load_prompt("concept_edges")

    def one(c):
        earlier = [d for d in defined if order[d.concept_id] < order[c.concept_id]]
        if not earlier:
            return []
        body = bodies[c.introduced_unit]
        present = lex.named_concepts(body)
        cands = [d for d in earlier if d.concept_id in present]
        if lex.embeddings is not None:
            q = lex.embeddings[idx[c.concept_id]]
            ranked = sorted(earlier, key=lambda d: -float(lex.embeddings[idx[d.concept_id]] @ q))
            cands += [d for d in ranked[:10] if d not in cands]
        cands = cands[:30]
        listing = "\n".join(f"- {d.concept_id}: {d.name}" for d in cands)
        schema = {"type": "object", "properties": {"depends_on": {"type": "array", "items": {
            "type": "string", "enum": [d.concept_id for d in cands]}}}, "required": ["depends_on"]}
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": f"Concept: {c.name}\n\nDefining passage:\n<passage>\n{body}\n</passage>\n\n"
                                            f"Earlier concepts:\n{listing}"}]
        res = llm.chat(msgs, config.DIGEST_MODEL, schema=schema, max_tokens=600, tag=f"{book.id}:edges:{c.concept_id}")
        ok = {d.concept_id for d in cands}
        return [(c.concept_id, t, "depends_on") for t in res["depends_on"] if t in ok]

    with ThreadPoolExecutor(workers) as ex:
        rows = [e for es in ex.map(one, defined) for e in es]
    con.execute("DELETE FROM concept_edges")
    con.executemany("INSERT OR IGNORE INTO concept_edges VALUES (?,?,?)", rows)
    con.commit()
    log(f"[{book.id}] edges: {len(rows)} depends_on edges over {len(defined)} defined concepts")
    return {"edges": len(rows)}


# ---------------------------------------------------------------- stage 7

def external(book: Book, log=print) -> dict:
    con = book.connect()
    ordinal = {r["unit_id"]: r["ordinal"] for r in con.execute("SELECT unit_id, ordinal FROM units")}
    rows = con.execute("SELECT concept_id, name, definition FROM concepts WHERE introduced_in_unit IS NULL "
                       "AND concept_id NOT IN (SELECT concept_id FROM prereqs_external WHERE reason='background')"
                       ).fetchall()
    system = llm.load_prompt("concept_external")
    verdicts = {}
    for start in range(0, len(rows), 40):
        batch = rows[start:start + 40]
        listing = "\n".join(f"- {r['concept_id']}: {r['name']} — \"{(r['definition'] or '')[:100]}\"" for r in batch)
        schema = {"type": "object", "properties": {"items": {"type": "array", "items": {
            "type": "object", "properties": {
                "id": {"type": "string", "enum": [r["concept_id"] for r in batch]},
                "class": {"type": "string", "enum": ["background", "book_concept", "noise"]}},
            "required": ["id", "class"]}}}, "required": ["items"]}
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": f"Book: {book.cfg['title']}\n\n{listing}"}]
        res = llm.chat(msgs, config.DIGEST_MODEL, schema=schema, max_tokens=3000, tag=f"{book.id}:external:{start}")
        verdicts.update({it["id"]: it["class"] for it in res["items"]})

    con.execute("DELETE FROM prereqs_external WHERE reason != 'background'")
    report = ["# External prerequisites and undefined concepts", ""]
    counts = collections.Counter()
    for r in rows:
        v = verdicts.get(r["concept_id"], "background")
        counts[v] += 1
        if v == "background":
            con.execute("INSERT INTO prereqs_external VALUES (?,?)", (r["concept_id"], "assumed background"))
            report.append(f"- background: **{r['name']}**")
        elif v == "noise":
            con.execute("DELETE FROM concepts WHERE concept_id=?", (r["concept_id"],))
            con.execute("DELETE FROM concept_uses WHERE concept_id=?", (r["concept_id"],))
        else:
            first = con.execute(
                "SELECT unit_id FROM concept_uses WHERE concept_id=? AND role IN ('use','example') ", (r["concept_id"],)
            ).fetchall()
            first = sorted((f["unit_id"] for f in first), key=lambda u: ordinal[u])
            if first:
                con.execute("UPDATE concepts SET introduced_in_unit=? WHERE concept_id=?", (first[0], r["concept_id"]))
                report.append(f"- book concept, no explicit definition: **{r['name']}** → first use {first[0]}")
            else:
                con.execute("INSERT INTO prereqs_external VALUES (?,?)", (r["concept_id"], "mentioned only"))
    con.commit()
    (book.reports_dir / "external_prereqs.md").write_text("\n".join(report), encoding="utf-8")
    log(f"[{book.id}] external: {dict(counts)}")
    return dict(counts)
