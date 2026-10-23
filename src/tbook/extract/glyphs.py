"""Deterministic math check: compare a LaTeX candidate with the glyphs the PDF prints.

Both sides become token sequences (symbol, is_script, typeface). The PDF side
comes from the text layer of the page region (scripts from glyph size and
baseline, typeface from font); the LaTeX side from a small math parser. A
global alignment scores how well a candidate explains the printed glyphs, so
among several transcriptions of an expression the one that matches the PDF
wins, and agreed transcriptions that contradict the PDF are caught.
"""

import re
import unicodedata
from dataclasses import dataclass

import pymupdf
from pylatexenc.latex2text import LatexNodes2Text

from . import mathlayer

_l2t = LatexNodes2Text(math_mode="verbatim")
FONTS = {
    "mathcal": "cal", "mathscr": "cal", "mathfrak": "frak", "mathbb": "bb",
    "mathbf": "bf", "boldsymbol": "bf", "bm": "bf", "textbf": "bf",
    "mathrm": None, "text": None, "textrm": None, "operatorname": None, "mathit": None, "textit": None,
    "mathsf": None, "texttt": None, "mathtt": None,
}
ACCENTS = {"dot": "˙", "ddot": "¨", "hat": "ˆ", "widehat": "ˆ", "bar": "¯", "overline": "¯", "tilde": "˜",
           "widetilde": "˜", "vec": "→", "overrightarrow": "→", "check": "ˇ", "breve": "˘"}
SKIP = {"left", "right", "big", "Big", "bigg", "Bigg", "bigl", "bigr", "Bigl", "Bigr", "biggl", "biggr",
        "quad", "qquad", "displaystyle", "textstyle", "scriptstyle", "nonumber", "notag", "label",
        "limits", "nolimits", "middle", "phantom", "hspace", "vspace", "mathstrut", "strut"}
BRACKETS = set("()[]{}|‖⟨⟩")
_ROW_OPS = {"&", "\\\\"}


@dataclass(frozen=True)
class Tok:
    sym: str
    script: bool
    font: str | None


def _norm_sym(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    return {"−": "-", "–": "-", "′": "'", "’": "'", "∗": "*", "⋅": "·", "∶": ":", "ϕ": "φ", "ϑ": "θ",
            "ϵ": "ε", "∣": "|", "∥": "‖", "‖": "‖", "⋯": "…", "…": "…", "→": "→"}.get(s, s)


def _collapse_dots(toks: list[Tok]) -> list[Tok]:
    out = []
    for t in toks:
        if t.sym in ("·", ".") and len(out) >= 1 and out[-1].sym in ("·", ".", "…") and \
                (out[-1].sym == "…" or (len(out) >= 2 and out[-2].sym in ("·", "."))):
            if out[-1].sym != "…":
                out[-2:] = [Tok("…", out[-1].script, None)]
            continue
        out.append(t)
    return [t for t in out]


# ------------------------------------------------------------------ LaTeX side

_TOKEN_RE = re.compile(r"\\[a-zA-Z]+\*?|\\.|[{}^_]|[^\s]")


def latex_tokens(tex: str) -> list[Tok]:
    toks = _TOKEN_RE.findall(tex)
    out: list[Tok] = []

    def atom(i, script, font):
        """Parse one atom starting at i; return new index."""
        if i >= len(toks):
            return i
        if toks[i] == "{":
            return group(i + 1, script, font)
        return single(i, script, font)

    def group(i, script, font):
        while i < len(toks) and toks[i] != "}":
            i = single(i, script, font)
        return i + 1

    def single(i, script, font):
        t = toks[i]
        if t == "{":
            return group(i + 1, script, font)
        if t in ("^", "_"):
            return atom(i + 1, True, font)
        if t in ("&",) or t == "\\\\" or t == "}":
            return i + 1
        if t.startswith("\\") and len(t) > 1 and t[1].isalpha():
            name = t[1:].rstrip("*")
            if name in FONTS:
                return atom(i + 1, script, FONTS[name] or font if FONTS[name] else None)
            if name in ACCENTS:
                out.append(Tok(ACCENTS[name], script, None))
                return atom(i + 1, script, font)
            if name in ("frac", "dfrac", "tfrac", "binom"):
                return atom(atom(i + 1, script, font), script, font)
            if name == "sqrt":
                j = i + 1
                if j < len(toks) and toks[j] == "[":
                    while j < len(toks) and toks[j] != "]":
                        j += 1
                    j += 1
                out.append(Tok("√", script, None))
                return atom(j, script, font)
            if name in ("begin", "end"):
                j = i + 1
                if j < len(toks) and toks[j] == "{":
                    while j < len(toks) and toks[j] != "}":
                        j += 1
                    j += 1
                if name == "begin" and j < len(toks) and toks[j] == "{":   # array column spec
                    while j < len(toks) and toks[j] != "}":
                        j += 1
                    j += 1
                return j
            if name in ("underbrace", "overbrace", "underset", "overset", "stackrel", "mathop", "mathbin",
                        "mathrel", "boxed", "hbox", "mbox"):
                return atom(i + 1, script, font)
            if name == "tag":
                j = i + 1
                start = len(out)
                j = atom(j, False, None)
                out[start:start] = [Tok("(", False, None)]
                out.append(Tok(")", False, None))
                return j
            if name in SKIP:
                return i + 1
            sym = _l2t.latex_to_text(t).strip()
            if name in ("sin", "cos", "tan", "exp", "log", "ln", "det", "tr", "min", "max", "lim", "arctan", "Pr"):
                out.extend(Tok(ch, script, None) for ch in name)
                return i + 1
            if not sym or sym == t:
                return i + 1                       # unknown or spacing macro
            out.extend(Tok(_norm_sym(ch), script, font) for ch in sym if ch.strip())
            return i + 1
        if t.startswith("\\"):
            ch = t[1:]
            if ch in "{}|":
                out.append(Tok({"|": "‖"}.get(ch, ch), script, None))
            return i + 1
        sym = _norm_sym(t)
        out.append(Tok(sym, True if sym == "'" else script, font if t.isalnum() else None))   # primes are set raised
        return i + 1

    i = 0
    while i < len(toks):
        i = single(i, False, None)
    return _collapse_dots(out)


# ------------------------------------------------------------------ PDF side

def pdf_tokens(page, math_re: re.Pattern, region: pymupdf.Rect | None = None) -> list[Tok]:
    raw = page.get_text("rawdict", flags=pymupdf.TEXT_MEDIABOX_CLIP)
    out: list[Tok] = []
    for block in raw["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            chars = [(c, s) for s in line["spans"] for c in s["chars"] if not mathlayer._junk(c["c"])]
            if region is not None:
                chars = [(c, s) for c, s in chars
                         if region.contains(pymupdf.Point((c["bbox"][0] + c["bbox"][2]) / 2, (c["bbox"][1] + c["bbox"][3]) / 2))]
            if not chars:
                continue
            sizes = [s["size"] for c, s in chars if c["c"].strip()]
            if not sizes:
                continue
            base = max(sizes)
            for c, s in chars:
                ch = c["c"]
                if not ch.strip():
                    continue
                script = s["size"] < mathlayer.TIER1 * base
                font = mathlayer._klass(s["font"], ch, bool(math_re.match(s["font"])))
                out.append(Tok(_norm_sym(ch), script, font))
    return _collapse_dots(out)


# ------------------------------------------------------------------ alignment

def align_score(cand: list[Tok], ref: list[Tok], *, local: bool = False) -> float:
    """Alignment score of candidate against printed glyphs (higher is better).

    Global (Needleman-Wunsch) for a located region, local (Smith-Waterman) when
    the reference is a larger stretch of text such as a paragraph.
    """
    n, m = len(cand), len(ref)
    if n == 0:
        return 0.0
    gap_c = lambda t: -0.3 if t.sym in BRACKETS else -1.0      # noqa: E731
    prev = [0.0] * (m + 1)
    if not local:
        for j in range(1, m + 1):
            prev[j] = prev[j - 1] + gap_c(ref[j - 1]) * 0.6
    best = 0.0
    for i in range(1, n + 1):
        cur = [0.0] * (m + 1)
        cur[0] = 0.0 if local else prev[0] + gap_c(cand[i - 1])
        a = cand[i - 1]
        for j in range(1, m + 1):
            b = ref[j - 1]
            if a.sym == b.sym:
                s = 2.0 + (0.6 if a.script == b.script else -0.8) + (0.4 if a.font == b.font else -0.6)
            else:
                s = -1.2
            v = max(prev[j - 1] + s, prev[j] + gap_c(a), cur[j - 1] + gap_c(b) * 0.6)
            if local:
                v = max(v, 0.0)
                best = max(best, v)
            cur[j] = v
        prev = cur
    return best if local else prev[m]


def best_candidate(cands: list[str], ref: list[Tok], *, local: bool) -> tuple[int, list[float]]:
    scores = [align_score(latex_tokens(c), ref, local=local) for c in cands]
    return max(range(len(cands)), key=lambda k: scores[k]), scores
