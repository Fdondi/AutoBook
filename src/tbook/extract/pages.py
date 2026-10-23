"""Stage 1a: page images, the text-layer coverage oracle, and the bookmark tree.

The PDF text layer is exact for prose but destroys math, so it is never used
as content. It provides:
  * oracle_words: prose/code words any correct extraction must contain,
  * math_chars: whether the page has math (math-font spans),
  * running heads to strip, and
  * section start positions for segmentation.
"""

import collections
import json
import re
import unicodedata

import pymupdf

from .. import config
from ..books import Book

FLAGS = pymupdf.TEXT_DEHYPHENATE | pymupdf.TEXT_MEDIABOX_CLIP | pymupdf.TEXT_PRESERVE_WHITESPACE
WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*")
HEAD_EDGE_LINES = 3        # first/last lines of a page that may be running heads
HEAD_MIN_PAGES = 5         # generic repeats; numbered section running heads need only 2
EQ_NUMBER_RE = re.compile(r"^\([a-z]?#(\.#)*[a-z]?\)$")   # "(3.12)"
ORACLE_BLOCK_WORDS = 12    # shorter blocks are figure labels / table cells: judged, not counted
ORACLE_LINE_WORDS = 5
# Operator names set in text fonts inside math ("det R"): not prose.
MATH_WORDS = {"det", "tr", "sin", "cos", "tan", "exp", "log", "ln", "min", "max", "rank", "dim", "arg",
              "sgn", "diag", "span", "atan", "rot", "trans", "mod", "lim", "sup", "inf", "arctan",
              "ith", "jth", "kth", "nth"}          # "$i$th": the letter is math, the ordinal glued to it
SPACING_ACCENTS = set("¨´`ˆ˜¸˘ˇ˙˚")
# TeX sets accents as a separate glyph before the letter ("Gr¨ubler"): compose them.
_COMBINING = {"¨": "̈", "´": "́", "ˆ": "̂", "˜": "̃",
              "¸": "̧", "˘": "̆", "ˇ": "̌", "˙": "̇", "˚": "̊"}
_ACCENT_RE = re.compile("([" + "".join(_COMBINING) + "])([A-Za-z])")


def norm(text: str) -> str:
    text = _ACCENT_RE.sub(lambda m: unicodedata.normalize("NFC", m.group(2) + _COMBINING[m.group(1)]), text)
    return unicodedata.normalize("NFKC", text).replace(" ", " ")


def fold(text: str) -> str:
    """ASCII-fold for word matching: Gr¨ubler, Grübler -> Grubler; ’ -> '."""
    t = unicodedata.normalize("NFKD", norm(text)).replace("’", "'")
    return "".join(c for c in t if not unicodedata.combining(c) and c not in SPACING_ACCENTS)


def head_key(text: str) -> str:
    t = norm(text).lower().strip()
    t = re.sub(r"\d+", "#", t)
    t = re.sub(r"\b[ivxlc]+\b", "#", t)
    return re.sub(r"\s+", " ", t)


def tokens(text: str) -> list[str]:
    return [m.lower() for m in WORD_RE.findall(fold(text))]


def is_word(w: str) -> bool:
    return len(w) >= 2 and w not in MATH_WORDS


def words(text: str) -> list[str]:
    return [w for w in tokens(text) if is_word(w)]


def book_vocab(con) -> tuple[set[str], set[str]]:
    """(words, hyphenated compounds) seen anywhere in the book's text layer."""
    plain, compounds = set(), set()
    for r in con.execute("SELECT oracle_words, text_layer FROM b_pages"):
        plain.update(json.loads(r["oracle_words"]))
        for line in (r["text_layer"] or "").split("\n"):
            body = line[:-1] if line.endswith("-") else line          # a line-end hyphen is not a compound
            compounds.update(c.lower() for c in re.findall(r"[A-Za-z]+-[A-Za-z]+", fold(body)))
    return plain, compounds


def _running_heads(head_pages: dict, toc_titles: list[str]) -> set[str]:
    titles = {re.sub(r"[^a-z]+", "", norm(t).lower()) for t in toc_titles}
    heads = set()
    for key, pages in head_pages.items():
        n = len(pages)
        if EQ_NUMBER_RE.match(key.replace(" ", "")):
            continue                       # equation numbers are content
        if key.strip("# .") == "":
            heads.add(key)                 # page numbers, dot leaders
        elif n >= HEAD_MIN_PAGES:
            heads.add(key)
        elif n >= 2 and re.sub(r"[^a-z]+", "", key) in titles and (
            key.startswith("#") or key.startswith("appendix")
        ):
            heads.add(key)                 # "3.2. Rotations and Angular Velocities"
    return heads


def _page_lines(page, math_re, code_re):
    """Lines as dicts: text, bbox, block index, and the folded text with a per-char kind mask."""
    out = []
    data = page.get_text("dict", flags=FLAGS)
    for bi, block in enumerate(data["blocks"]):
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            text, ftext, kinds = "", "", ""
            top = max((s["size"] for s in line["spans"] if s["text"].strip()), default=0)
            for span in line["spans"]:
                t = norm(span["text"])
                k = "m" if math_re.match(span["font"]) else "c" if code_re.match(span["font"]) else "t"
                if k == "t" and span["size"] < 0.86 * top:
                    k = "s"                       # script-size text, e.g. the "a" of a roman x-hat sub a
                text += t
                for ch in t:
                    f = fold(ch)
                    ftext += f
                    kinds += k * len(f)
            if text.strip():
                out.append({"text": text, "ftext": ftext, "kinds": kinds, "bbox": line["bbox"], "block": bi,
                            "size": top})
    return out


def _vocab(all_lines) -> tuple[collections.Counter, collections.Counter]:
    """Book-wide words and hyphenated compounds, for line-end dehyphenation."""
    plain, compounds = collections.Counter(), collections.Counter()
    for lines in all_lines:
        for ln in lines:
            plain.update(w.lower() for w in WORD_RE.findall(ln["ftext"]))
            compounds.update(c.lower() for c in re.findall(r"[A-Za-z]+-[A-Za-z]+", ln["ftext"]))
    return plain, compounds


def _block_words(blines, vocab) -> list[list[str]]:
    """Per-line prose/code words of a block, joining words hyphenated across lines."""
    plain, compounds = vocab
    text, kinds, line_of = "", "", []
    for k, ln in enumerate(blines):
        t = ln["ftext"].rstrip()
        kd = ln["kinds"][: len(t)]
        if text:
            m_left = re.search(r"([A-Za-z]+)-$", text)
            m_right = re.match(r"[A-Za-z]+", t)
            if m_left and m_right and t[0].islower():
                left, right = m_left.group(1).lower(), m_right.group().lower()
                if plain[left + right] or not compounds[f"{left}-{right}"]:
                    text, kinds, line_of = text[:-1], kinds[:-1], line_of[:-1]
            else:
                text, kinds = text + " ", kinds + "t"
                line_of.append(k)
        text += t
        kinds += kd
        line_of.extend([k] * len(t))
    per_line = [[] for _ in blines]
    for m in WORD_RE.finditer(text):
        a, b = m.start(), m.end()
        if "m" in kinds[a:b] or ("s" in kinds[a:b] and kinds[a:b].strip("s")):
            continue                          # math, or glued across a script boundary ("xa")
        if (a > 0 and kinds[a - 1] == "m" and text[a - 1] != " ") or (b < len(text) and kinds[b] == "m"):
            continue                      # glued to math, e.g. "RTR"
        w = m.group().lower()
        if len(w) >= 2 and w not in MATH_WORDS:
            per_line[line_of[a]].append(w)
    return per_line


def extract_pages(book: Book, *, render: bool = True, log=print) -> None:
    con = book.connect()
    doc = pymupdf.open(book.pdf)
    math_re = re.compile(book.cfg["math_fonts"])
    code_re = re.compile(book.cfg["code_fonts"])
    book.pages_dir.mkdir(parents=True, exist_ok=True)

    all_lines = []
    head_pages = collections.defaultdict(set)
    for i, page in enumerate(doc):
        lines = _page_lines(page, math_re, code_re)
        by_y = sorted(range(len(lines)), key=lambda j: lines[j]["bbox"][1])
        edge = set(by_y[:HEAD_EDGE_LINES] + by_y[-HEAD_EDGE_LINES:])
        sizes = collections.Counter(round(ln["size"], 1) for ln in lines)
        body = sizes.most_common(1)[0][0] if sizes else 0
        for j, ln in enumerate(lines):
            # a chapter title on its opener page is large; running heads never are
            ln["band"] = j in edge and ln["size"] <= 1.15 * body
            if ln["band"]:
                head_pages[head_key(ln["text"])].add(i)
        all_lines.append(lines)
        if render:
            png = book.pages_dir / f"p{i:04d}.png"
            if not png.exists():
                page.get_pixmap(dpi=config.PAGE_DPI).save(png)
    heads = _running_heads(head_pages, [t for _, t, _ in doc.get_toc()])
    vocab = _vocab(all_lines)

    rows = []
    for i, page in enumerate(doc):
        removed, kept = [], []
        for ln in all_lines[i]:
            (removed if ln["band"] and head_key(ln["text"]) in heads else kept).append(ln)
        blocks = collections.defaultdict(list)
        for ln in kept:
            blocks[ln["block"]].append(ln)
        oracle, math_chars, code_chars, paras = [], 0, 0, []
        pending = None                      # word fragment hyphenated at the end of the previous block
        for _, blines in sorted(blocks.items()):
            per_line = _block_words(blines, vocab)
            total = sum(len(w) for w in per_line)
            first = next((w for w in per_line if w), None)
            if pending and first and vocab[0][pending + first[0]] and oracle and oracle[-1] == pending:
                oracle[-1] = pending + first.pop(0)          # "coor-" | "dinates" -> "coordinates"
            for ln, w in zip(blines, per_line):
                math_chars += ln["kinds"].count("m")
                code_chars += ln["kinds"].count("c")
                if total >= ORACLE_BLOCK_WORDS or len(w) >= ORACLE_LINE_WORDS:
                    oracle.extend(w)
            last = blines[-1]["ftext"].rstrip()
            m = re.search(r"([A-Za-z]+)-$", last)
            pending = m.group(1).lower() if m else None
            paras.append("\n".join(ln["text"].rstrip() for ln in blines))
        rows.append((
            i, page.get_label() or str(i + 1), page.rect.width, page.rect.height,
            str(book.pages_dir / f"p{i:04d}.png"), "\n\n".join(paras), json.dumps(oracle),
            math_chars, code_chars, json.dumps([ln["text"] for ln in removed]),
        ))
    con.execute("DELETE FROM b_pages")
    con.executemany("INSERT INTO b_pages VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    con.commit()
    log(f"[{book.id}] pages: {len(rows)}, running-head patterns removed: {len(heads)}")
    extract_toc(book, doc, con, all_lines, log=log)


# ---------------------------------------------------------------- bookmark tree

NUM_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(.*)$")


def _title_key(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", norm(t).lower())


def _find_heading_y(lines, title: str) -> list[float]:
    key = _title_key(NUM_RE.sub(r"\2", norm(title)))
    if len(key) < 3:
        return []
    ys = []
    for ln in lines:
        lk = _title_key(ln["text"])
        if lk and (lk.endswith(key) or key.startswith(lk) and len(lk) > 0.6 * len(key)):
            ys.append(ln["bbox"][1])
    return ys


def extract_toc(book: Book, doc, con, all_lines, log=print) -> None:
    toc = doc.get_toc(simple=False)
    # Bookmark y can be top-down or bottom-up depending on the producer: vote.
    votes = collections.Counter()
    for lvl, title, pno, dest in toc:
        to = dest.get("to") if isinstance(dest, dict) else None
        if to is None or pno < 1:
            continue
        h = doc[pno - 1].rect.height
        for y in _find_heading_y(all_lines[pno - 1], title):
            votes["direct"] += abs(y - to.y) < 40
            votes["flip"] += abs(y - (h - to.y)) < 40
    flip = votes["flip"] > votes["direct"]

    cfg = book.cfg
    front = {norm(t) for t in cfg.get("front_titles", [])}
    back = {norm(t) for t in cfg.get("back_titles", [])}
    appendix = [norm(t) for t in cfg.get("appendix_titles", [])]
    part_re = re.compile(cfg["part_titles"]) if cfg.get("part_titles") else None
    ex_re = re.compile(cfg["exercise_titles"])
    ref_re = re.compile(cfg["reference_titles"])

    rows = []
    counters: list[int] = []
    chapter_no = 0
    appendix_no = 0
    parent_kind: dict[int, str] = {}
    number_stack: list[str] = []
    for idx, (lvl, title, pno, dest) in enumerate(toc):
        title = norm(title).strip()
        to = dest.get("to") if isinstance(dest, dict) else None
        page_idx = max(0, pno - 1)
        h = doc[page_idx].rect.height
        y = 0.0 if to is None else (h - to.y if flip else to.y)
        y = min(max(y - 2, 0.0), h)

        number, kind = None, "content"
        bare = title
        if lvl == 1 and title in front:
            kind = "front"
        elif lvl == 1 and title in back:
            kind = "back"
        elif lvl == 1 and part_re and part_re.match(title):
            kind = "part"
        if cfg["numbering"] == "title":
            m = NUM_RE.match(title)
            if m:
                number, bare = m.group(1), m.group(2)
        elif kind == "content":
            if lvl == 1:
                if title in appendix:
                    appendix_no += 1
                    number = chr(ord("A") + appendix_no - 1)
                else:
                    chapter_no += 1
                    number = str(chapter_no)
                counters = [0]
                number_stack = [number]
            else:
                while len(counters) < lvl:
                    counters.append(0)
                counters = counters[:lvl]
                counters[lvl - 1] += 1
                number_stack = number_stack[: lvl - 1]
                number = ".".join([number_stack[0]] + [str(c) for c in counters[1:lvl]])
                number_stack.append(number)
        inherited = parent_kind.get(lvl - 1)
        if kind == "content":
            if inherited in ("exercise", "reference", "front", "back"):
                kind = inherited
            elif ex_re.match(bare):
                kind = "exercise"
            elif ref_re.match(bare):
                kind = "reference"
        parent_kind[lvl] = kind
        for deeper in [k for k in parent_kind if k > lvl]:
            del parent_kind[deeper]
        rows.append((idx, lvl, bare, page_idx, y, number, kind))
    con.execute("DELETE FROM b_toc")
    con.executemany("INSERT INTO b_toc VALUES (?,?,?,?,?,?,?)", rows)
    con.commit()
    log(f"[{book.id}] toc: {len(rows)} entries, y-origin {'bottom' if flip else 'top'} "
        f"(votes {dict(votes)})")
