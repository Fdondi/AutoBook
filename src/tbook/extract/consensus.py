"""Math consensus between independent transcriptions of a page.

Each math span of the primary transcription is aligned with the spans of the
other transcriptions (same order, canonical LaTeX). A span is confirmed when
at least two transcriptions agree exactly after canonicalization. Otherwise
the candidates go to a focused adjudication: the VLM sees the page, the words
before the expression and the shuffled candidates, and picks or corrects.
The primary Markdown is then rewritten with the adjudicated LaTeX.
"""

import difflib
import json
import random
import re

from .. import config, llm
from . import mdutil

MIN_SIM = 0.55            # below this, spans are not counterparts
TRIVIAL_TOKENS = 3        # "$n$", "$x_0$": agreement not required

ADJ_SCHEMA = {
    "type": "object",
    "properties": {"verdicts": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "integer"}, "choice": {"type": "string"}, "corrected": {"type": "string"}},
        "required": ["id", "choice", "corrected"]}}},
    "required": ["verdicts"],
}


def _toks(canon: str) -> list[str]:
    return re.findall(r"\\[a-zA-Z]+|\\.|.", canon)


def _spans_with_pos(md: str) -> list[dict]:
    """Math spans with their character offsets and the preceding words (outside code)."""
    out = []
    code = [(m.start(), m.end()) for m in mdutil.CODE_FENCE_RE.finditer(md)]
    for m in mdutil.MATH_RE.finditer(md):
        if any(a <= m.start() < b for a, b in code):
            continue
        tex = m.group(1) if m.group(1) is not None else m.group(2)
        before = mdutil.md_to_text(mdutil.MATH_RE.sub(" ", md[max(0, m.start() - 300):m.start()]))
        out.append({
            "start": m.start(), "end": m.end(), "tex": tex, "display": m.group(1) is not None,
            "canon": mdutil.canon_math(tex), "context": " ".join(before.split()[-12:]),
        })
    return out


def _align(primary: list[dict], other: list[dict]) -> list[dict | None]:
    """Order-preserving alignment of primary spans to the other transcription's spans."""
    sm = difflib.SequenceMatcher(None, [s["canon"] for s in primary], [s["canon"] for s in other], autojunk=False)
    match: list[dict | None] = [None] * len(primary)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                match[i1 + k] = other[j1 + k]
        elif tag == "replace":
            used = set()
            for i in range(i1, i2):
                best, best_j = 0.0, None
                for j in range(j1, j2):
                    if j in used:
                        continue
                    s = difflib.SequenceMatcher(None, _toks(primary[i]["canon"]), _toks(other[j]["canon"]),
                                                autojunk=False).ratio()
                    if s > best:
                        best, best_j = s, j
                if best_j is not None and best >= MIN_SIM:
                    used.add(best_j)
                    match[i] = other[best_j]
    return match


def disputes(primary_md: str, others: dict[str, str]) -> tuple[list[dict], list[dict]]:
    """(spans of the primary, disputed spans with their candidate LaTeX)."""
    spans = _spans_with_pos(primary_md)
    aligned = {name: _align(spans, _spans_with_pos(md)) for name, md in others.items()}
    out = []
    for i, s in enumerate(spans):
        cands = {"primary": s}
        for name, al in aligned.items():
            if al[i] is not None:
                cands[name] = al[i]
        votes: dict[str, list[str]] = {}
        for name, c in cands.items():
            votes.setdefault(c["canon"], []).append(name)
        s["votes"] = votes
        mine, best = len(votes[s["canon"]]), max(len(v) for v in votes.values())
        if mine >= 2:
            s["status"] = "agreed"
        elif best >= 2:
            s["status"] = "outvoted"           # two other transcriptions agree on something else
        elif len(_toks(s["canon"])) <= TRIVIAL_TOKENS:
            s["status"] = "trivial"
        elif len(cands) == 1:
            s["status"] = "unconfirmed"        # no counterpart anywhere
        else:
            s["status"] = "disputed"
        if s["status"] in ("outvoted", "disputed", "unconfirmed"):
            options = []
            seen = set()
            for name, c in cands.items():
                if c["canon"] not in seen:
                    seen.add(c["canon"])
                    options.append(c["tex"])
            out.append({"index": i, "span": s, "options": options})
    return spans, out


def adjudicate(png: str, items: list[dict], *, tag: str, seed: int = 0) -> dict[int, str]:
    """{span index: chosen LaTeX} for disputed spans, judged on the full page (batched)."""
    return _adjudicate_images([png], items, tag=tag, seed=seed, zoomed=False)


def _adjudicate_images(images: list, items: list[dict], *, tag: str, seed: int, zoomed: bool) -> dict[int, str]:
    if not items:
        return {}
    rng = random.Random(seed)
    lines, keymap = [], {}
    for k, it in enumerate(items):
        opts = list(it["options"])
        rng.shuffle(opts)
        letters = [chr(ord("A") + n) for n in range(len(opts))]
        keymap[k] = dict(zip(letters, opts))
        kind = "displayed equation" if it["span"]["display"] else "inline expression"
        lines.append(f"Dispute {k} ({kind}), after the words: \"{it['span']['context']}\"")
        for letter, o in zip(letters, opts):
            lines.append(f"  {letter}: {o}")
    where = ("The image is an enlarged crop of the page region containing the expression."
             if zoomed else "The image is the whole page.")
    content = [*(llm.image_part(i) for i in images), {"type": "text", "text": where + "\n\n" + "\n".join(lines) +
               "\n\nFor each dispute give `choice` (a letter, or \"none\") and `corrected` (the exact LaTeX "
               "when choice is none, else empty)."}]
    msgs = [{"role": "system", "content": llm.load_prompt("extract_adjudicate")},
            {"role": "user", "content": content}]
    res = llm.chat(msgs, config.VISION_MODEL, schema=ADJ_SCHEMA, max_tokens=4000, tag=tag)
    out = {}
    for v in res["verdicts"]:
        k = v["id"]
        if k not in keymap:
            continue
        choice = v["choice"].strip().upper()[:1]
        if choice in keymap[k]:
            out[items[k]["index"]] = keymap[k][choice]
        elif v["corrected"].strip():
            out[items[k]["index"]] = v["corrected"].strip()
    return out


def _sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _toks(a), _toks(b), autojunk=False).ratio()


def _with_groups(blocks: list[dict]) -> list[dict]:
    """Layout blocks plus merged runs of vertically adjacent equation blocks
    (a multi-line display is often split into one block per line)."""
    eqs = sorted((b for b in blocks if b["type"] == "equation" and b.get("latex")), key=lambda b: b["bbox"][1])
    groups, run = [], []
    for b in eqs:
        if run and b["bbox"][1] - run[-1]["bbox"][3] < 0.012:
            run.append(b)
        else:
            if len(run) > 1:
                groups.append(run)
            run = [b]
    if len(run) > 1:
        groups.append(run)
    merged = [{"type": "equation", "latex": " \\\\ ".join(b["latex"] for b in g),
               "bbox": [min(b["bbox"][0] for b in g), g[0]["bbox"][1], max(b["bbox"][2] for b in g), g[-1]["bbox"][3]]}
              for g in groups]
    return blocks + merged


def locate(item: dict, blocks: list[dict]) -> list[float] | None:
    """Bounding box (page fractions) of the disputed expression, from layout blocks."""
    span = item["span"]
    cands = [mdutil.canon_math(o) for o in item["options"]]
    ctx = set(span["context"].lower().split()[-6:])
    best, best_box = 0.0, None
    for b in _with_groups(blocks):
        score = 0.0
        if span["display"] and b["type"] == "equation" and b.get("latex"):
            score = max(_sim(mdutil.canon_math(b["latex"]), c) for c in cands)
        elif not span["display"] and b.get("text") is not None:
            inl = [mdutil.canon_math(x) for x in b.get("inline") or [] if isinstance(x, str)]
            score = max((_sim(i, c) for i in inl for c in cands), default=0.0)
            words = set(b["text"].lower().split())
            score = 0.7 * score + 0.3 * (len(ctx & words) / max(1, len(ctx)))
        if score > best:
            best, best_box = score, b["bbox"]
    return best_box if best >= 0.5 else None


def crop(pdf_path, page_idx: int, bbox: list[float], dpi: int = 360) -> bytes:
    """Re-render just this region of the PDF page at high resolution."""
    import pymupdf
    page = pymupdf.open(pdf_path)[page_idx]
    r = page.rect
    x0, y0, x1, y1 = bbox
    pad_x, pad_y = 0.02, 0.012
    clip = pymupdf.Rect(max(0, x0 - pad_x) * r.width, max(0, y0 - pad_y) * r.height,
                        min(1, x1 + pad_x) * r.width, min(1, y1 + pad_y) * r.height)
    return page.get_pixmap(dpi=dpi, clip=clip).tobytes("png")


def adjudicate_zoomed(pdf_path, page_idx: int, png: str, items: list[dict], blocks: list[dict], *,
                      tag: str, seed: int = 0) -> dict[int, str]:
    """Adjudicate each dispute on a high-resolution crop of its region (full page if not located)."""
    out = {}
    for k, it in enumerate(items):
        box = locate(it, blocks)
        image = crop(pdf_path, page_idx, box) if box else None
        chosen = _adjudicate_images([image] if image else [png], [it], tag=f"{tag}:{k}", seed=seed + k,
                                    zoomed=image is not None)
        out.update(chosen)
    return out


def apply(md: str, spans: list[dict], chosen: dict[int, str]) -> str:
    """Rewrite the primary Markdown with adjudicated LaTeX (back to front keeps offsets valid)."""
    for i in sorted(chosen, reverse=True):
        s = spans[i]
        delim = "$$" if s["display"] else "$"
        md = md[:s["start"]] + delim + chosen[i] + delim + md[s["end"]:]
    return md


def summarize(spans: list[dict]) -> dict:
    out = {}
    for s in spans:
        out[s["status"]] = out.get(s["status"], 0) + 1
    return out


def dump(spans: list[dict]) -> str:
    return json.dumps([{k: s[k] for k in ("canon", "status", "votes")} for s in spans], ensure_ascii=False)
