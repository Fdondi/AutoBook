"""Stage 2: split the verified page Markdown into reading units.

Section starts come from the bookmark tree: each entry's heading is located in
its page's Markdown (fuzzy title match), falling back to the bookmark's y
position mapped through the text layer. Each section's own text is then cut
into 300-500-word units at paragraph boundaries; math, code and tables are
never split.
"""

import json
import re

from .. import config
from ..books import Book
from ..extract.mdutil import CODE_FENCE_RE
from ..extract.pages import norm

WORD_RE = re.compile(r"\w+")


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", norm(text).lower())


def _strip_heading(line: str) -> str:
    line = re.sub(r"^\s*#+\s*", "", line)
    line = line.strip("*_ ")
    line = re.sub(r"^(chapter|appendix)\s+[a-z0-9]+[.:]?\s*", "", line, flags=re.I)
    line = re.sub(r"^[A-Z]?\d+(\.\d+)*\.?\s+", "", line)
    return line


def _find_heading(md: str, title: str, start: int = 0) -> int | None:
    """Char offset of the line holding `title` as a heading, at or after `start`."""
    tkey = _key(title)
    if not tkey:
        return None
    best = None
    pos = 0
    for line in md.split("\n"):
        line_start = pos
        pos += len(line) + 1
        if line_start < start or not line.strip():
            continue
        lkey = _key(_strip_heading(line))
        if not lkey:
            continue
        exact = lkey == tkey
        loose = len(line) < len(title) + 25 and (lkey.endswith(tkey) or tkey.endswith(lkey) and len(lkey) > 0.7 * len(tkey))
        if exact or loose:
            score = (2 if line.lstrip().startswith("#") else 0) + (1 if exact else 0)
            if best is None or score > best[0]:
                best = (score, line_start)
            if score == 3:
                break
    return None if best is None else best[1]


def _y_fallback(md: str, text_layer: str, page_h: float, y: float) -> int:
    """Map a y position to a Markdown offset by word fraction (approximate)."""
    frac = min(max(y / page_h, 0.0), 1.0)
    words = list(WORD_RE.finditer(md))
    if not words:
        return 0
    return words[min(len(words) - 1, int(frac * len(words)))].start()


def _blocks(text: str) -> list[str]:
    """Blank-line-separated blocks; code fences and display math are never cut."""
    protected = []

    def keep(m):
        protected.append(m.group(0))
        return f"\x00{len(protected) - 1}\x00"

    t = CODE_FENCE_RE.sub(keep, text)
    t = re.sub(r"\$\$.+?\$\$", keep, t, flags=re.S)
    parts = [p for p in re.split(r"\n\s*\n", t) if p.strip()]
    return [re.sub(r"\x00(\d+)\x00", lambda m: protected[int(m.group(1))], p) for p in parts]


def _can_break_before(block: str, prev: str) -> bool:
    """A unit may start here only at a new paragraph, never mid-thought."""
    b = block.lstrip()
    if b.startswith(("$$", "|", "```", "- ", "* ", "+ ")) or re.match(r"\d+[.)] ", b):
        return False                      # equation, table, code, list item
    if b[:1].islower():
        return False                      # "where ..." after an equation
    return not prev.rstrip().endswith((":", ","))


def _units_from_text(text: str) -> list[str]:
    blocks = _blocks(text)
    units, cur, n = [], [], 0
    for b in blocks:
        w = len(WORD_RE.findall(b))
        if cur and n + w > config.UNIT_MAX_WORDS and n >= config.UNIT_MIN_WORDS and _can_break_before(b, cur[-1]):
            units.append("\n\n".join(cur))
            cur, n = [], 0
        cur.append(b)
        n += w
    if cur:
        if units and n < config.UNIT_MIN_WORDS // 3 and len(WORD_RE.findall(units[-1])) + n <= config.UNIT_MAX_WORDS * 1.3:
            units[-1] += "\n\n" + "\n\n".join(cur)
        else:
            units.append("\n\n".join(cur))
    return units


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", norm(title).lower()).strip("-")[:40]


def segment(book: Book, log=print) -> dict:
    con = book.connect()
    pages = {r["page_idx"]: r for r in con.execute(
        "SELECT p.page_idx, p.label, p.height, p.text_layer, f.md FROM b_pages p "
        "LEFT JOIN b_page_final f USING(page_idx)")}
    if not any(r["md"] for r in pages.values()):
        raise SystemExit(f"[{book.id}] no final page Markdown: run extraction first")
    toc = [dict(r) for r in con.execute("SELECT * FROM b_toc ORDER BY idx")]

    # Locate each entry: (page_idx, offset in that page's md).
    located, approx = [], []
    last = (-1, -1)
    for e in toc:
        p = e["page_idx"]
        md = pages[p]["md"] or ""
        off = _find_heading(md, e["title"], start=last[1] + 1 if last[0] == p else 0)
        if off is None and p + 1 in pages:
            nxt = pages[p + 1]["md"] or ""
            off2 = _find_heading(nxt[:600], e["title"])
            if off2 is not None:
                p, off = p + 1, off2
        if off is None and e["level"] == 1:
            off = 0 if last[0] != p else last[1] + 1      # chapters open at the top of their page
        if off is None:
            off = _y_fallback(md, pages[p]["text_layer"], pages[p]["height"], e["y"])
            if (p, off) <= last:
                off = last[1] + 1 if last[0] == p else off
            approx.append(f"{e['number'] or ''} {e['title']} (page {pages[p]['label']})")
        last = (p, off)
        located.append((p, off))

    # Section text runs from one entry's start to the next entry's start.
    order = sorted(pages)
    def text_between(a, b):
        (pa, oa), (pb, ob) = a, b
        parts = []
        for p in order:
            if p < pa or p > pb:
                continue
            md = pages[p]["md"] or ""
            s = oa if p == pa else 0
            e = ob if p == pb else len(md)
            if e > s:
                parts.append((p, md[s:e]))
        return parts

    last_page = max(order)
    end_of_book = (last_page, len(pages[last_page]["md"] or ""))
    rows, ordinal = [], 0
    used: set[str] = set()
    path: list[str] = []
    numbers: list[str | None] = []
    for i, e in enumerate(toc):
        lvl = e["level"]
        path = path[: lvl - 1] + [e["title"]]
        numbers = numbers[: lvl - 1] + [e["number"]]
        if e["kind"] in ("back", "part") or e["kind"] == "front" and e["title"].lower() == "contents":
            continue
        nxt = located[i + 1] if i + 1 < len(toc) else end_of_book
        parts = text_between(located[i], nxt)
        text = "\n\n".join(t for _, t in parts).strip()
        if len(WORD_RE.findall(text)) < 12:
            continue                                   # heading only
        number = e["number"]
        if number is None:
            # titles below the nearest numbered ancestor: FoB ch. 10 has two unnumbered "Singularities"
            anc = max((j for j, n in enumerate(numbers[:-1]) if n), default=-1)
            number = (numbers[anc] if anc >= 0 else "x") + "/" + "/".join(_slug(t) for t in path[anc + 1:])
        chapter = (numbers[0] or "") if numbers else ""
        if e["kind"] == "front":
            chapter, number = "front", f"front/{_slug(e['title'])}"
        while number in used:
            number += "+"
        used.add(number)
        page_start, page_end = parts[0][0], parts[-1][0]
        for k, body in enumerate(_units_from_text(text), 1):
            ordinal += 1
            wc = len(WORD_RE.findall(body))
            # page of this unit: locate its first words among the section's pages
            probe = body[:80]
            up = next((p for p, t in parts if probe[:40] in t), page_start)
            rows.append((
                f"{book.id}:{number}-{k:02d}", ordinal, chapter, number, e["title"],
                json.dumps(path), e["kind"], up, page_end, pages[up]["label"], body, wc, int(wc * 1.5),
            ))
    con.execute("DELETE FROM units")
    con.executemany("INSERT INTO units VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.commit()
    sizes = [r[11] for r in rows if r[6] == "content"]
    stats = {
        "units": len(rows), "content_units": len(sizes),
        "words_median": sorted(sizes)[len(sizes) // 2] if sizes else 0,
        "over_max": sum(s > config.UNIT_MAX_WORDS * 1.3 for s in sizes),
        "approx_headings": approx,
    }
    log(f"[{book.id}] segment: {stats['units']} units ({stats['content_units']} content), "
        f"median {stats['words_median']} words, {stats['over_max']} oversized, "
        f"{len(approx)} headings placed by position")
    return stats
