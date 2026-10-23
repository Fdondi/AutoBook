"""Annotated text layer: the PDF's own glyphs with script levels and math typefaces.

The plain text layer flattens math ("RTR = I"). A glyph's size relative to its
line (tier 0 = base, 1 = script, 2 = scriptscript) and its baseline relative to
the glyph it attaches to tell superscript from subscript, nested as deep as
TeX goes. Its font tells the typeface (calligraphic, fraktur, blackboard,
bold). Rendering ^{..}/_{..} groups and \\cal{..}/\\frak{..}/\\bb{..}/\\bf{..}
runs gives the transcriber exact symbol evidence, in printed order.
"""

import re
import unicodedata

import pymupdf

from .pages import head_key

TIER1 = 0.86                # size ratio below which a glyph is a script
TIER2 = 0.62                # ... a script of a script
FONT_CLASS = [
    (re.compile(r"^(CMSY|MTSY|CMBSY)"), "cal"),
    (re.compile(r"^(EUFM|EUFB|EUSM)"), "frak"),
    (re.compile(r"^(MSBM)"), "bb"),
    (re.compile(r"^(CMBX|CMMIB|CMSSBX|RMTMIB|Times-Bold|MTMIB)|Bold"), "bf"),
]


def _klass(font: str, ch: str, math: bool) -> str | None:
    for rx, k in FONT_CLASS:
        if rx.search(font):
            if k in ("cal", "frak", "bb"):
                return k if math and ch.isalpha() else None
            return k if ch.strip() else None
    return None


def _junk(ch: str) -> bool:
    return ch == "�" or unicodedata.category(ch) in ("Co", "Cn", "Cc")


def _tier(size: float, base: float) -> int:
    r = size / base
    return 0 if r >= TIER1 else 1 if r >= TIER2 else 2


def annotate_line(chars: list[tuple[dict, dict]], math_re: re.Pattern) -> str:
    """chars: (rawdict char, its span). Returns the line with scripts and typefaces marked."""
    sizes = [s["size"] for c, s in chars if c["c"].strip()]
    if not sizes:
        return ""
    base = max(sizes)
    pieces = []                                   # (path, class, char)
    last_y = {0: None}                            # baseline of the latest glyph per tier
    path: tuple[str, ...] = ()
    for c, s in chars:
        ch = c["c"]
        y = c["origin"][1]
        if ch.strip():
            t = _tier(s["size"], base)
            if t == 0:
                path = ()
            else:
                parent_y = last_y.get(t - 1)
                if parent_y is None:
                    parent_y = y
                parent_size = base if t == 1 else base * TIER1
                dy = y - parent_y
                mark = "^" if dy < -0.12 * parent_size else "_" if dy > 0.06 * parent_size else (path[t - 1] if len(path) >= t else "_")
                path = path[: t - 1] + (mark,)
                if len(path) < t:
                    path = ("_",) * (t - len(path)) + path
            last_y[min(_tier(s["size"], base), 2)] = y
            for deeper in [k for k in last_y if k > _tier(s["size"], base)]:
                last_y[deeper] = None
        pieces.append((path, _klass(s["font"], ch, bool(math_re.match(s["font"]))), ch))
    return _render(pieces)


def _render(pieces: list[tuple[tuple[str, ...], str | None, str]]) -> str:
    out: list[str] = []
    cur: tuple[str, ...] = ()
    klass = None
    for path, k, ch in pieces:
        if not ch.strip():
            path, k = cur, klass                  # spaces stay in the current group
        if path != cur or k != klass:
            if klass:
                out.append("}")
            common = 0
            while common < min(len(cur), len(path)) and cur[common] == path[common]:
                common += 1
            out.append("}" * (len(cur) - common))
            for m in path[common:]:
                out.append(m + "{")
            if k:
                out.append(f"\\{k}{{")
            cur, klass = path, k
        out.append(ch)
    if klass:
        out.append("}")
    out.append("}" * len(cur))
    text = "".join(out)
    return re.sub(r"\s+(\}+)", r"\1 ", text).rstrip()


def annotate_page(page, math_re: re.Pattern, running_heads: list[str] = ()) -> str:
    raw = page.get_text("rawdict", flags=pymupdf.TEXT_MEDIABOX_CLIP)
    heads = {head_key(h) for h in running_heads}
    blocks_out = []
    for block in raw["blocks"]:
        if block.get("type") != 0:
            continue
        lines_out: list[tuple[str, bool]] = []          # (text, is prose)
        for line in block["lines"]:
            chars = [(c, s) for s in line["spans"] for c in s["chars"] if not _junk(c["c"])]
            plain = "".join(c["c"] for c, _ in chars)
            if not plain.strip() or head_key(plain) in heads:
                continue
            text = annotate_line(chars, math_re)
            words = re.findall(r"[A-Za-z]{2,}", plain)
            prose = len(words) >= 4 and "^{" not in text and "_{" not in text
            lines_out.append((text, prose))
        if lines_out:
            blocks_out.append(_join_prose(lines_out))
    return "\n\n".join(blocks_out)


def _join_prose(lines: list[tuple[str, bool]]) -> str:
    """Join consecutive prose lines into one paragraph line, dropping line-end hyphens,
    so transcribers do not copy the PDF's line breaks ("spa-\\ntial")."""
    out: list[str] = []
    prev_prose = False
    for text, prose in lines:
        if out and prose and prev_prose:
            if re.search(r"[A-Za-z]-$", out[-1]) and re.match(r"[a-z]", text):
                out[-1] = out[-1][:-1] + text
            else:
                out[-1] = out[-1] + " " + text
        else:
            out.append(text)
        prev_prose = prose
    return "\n".join(out)
