"""Markdown + LaTeX helpers shared by extraction checks, segmentation and the reader."""

import json
import re
import subprocess

from pylatexenc.latex2text import LatexNodes2Text

from .. import config

# $$...$$ (display) first, then $...$ (inline). Escaped \$ is not a delimiter.
MATH_RE = re.compile(r"(?<!\\)\$\$(.+?)(?<!\\)\$\$|(?<![\\$])\$(?!\$)(.+?)(?<![\\$])\$(?!\$)", re.S)
CODE_FENCE_RE = re.compile(r"^```.*?^```[ \t]*$", re.S | re.M)


def normalize_delimiters(md: str) -> str:
    """\\[...\\] -> $$...$$ and \\(...\\) -> $...$ (outside code fences)."""
    def fix(text: str) -> str:
        text = re.sub(r"\\\[(.+?)\\\]", lambda m: "$$" + m.group(1) + "$$", text, flags=re.S)
        return re.sub(r"\\\((.+?)\\\)", lambda m: "$" + m.group(1) + "$", text, flags=re.S)
    return _outside_code(md, fix)


def _outside_code(md: str, fn) -> str:
    out, last = [], 0
    for m in CODE_FENCE_RE.finditer(md):
        out.append(fn(md[last:m.start()]))
        out.append(m.group(0))
        last = m.end()
    out.append(fn(md[last:]))
    return "".join(out)


_HYPH_RE = re.compile(r"([A-Za-z]{2,})-(?:[ \t]*\n[ \t]*|[ \t]+)([a-z][A-Za-z]*)")


def dehyphenate(md: str, vocab: set[str], compounds: set[str] = frozenset()) -> str:
    """Join words split by a hyphen at a line end ("spa-\\ntial" -> "spatial") when the
    joined word is in the book's vocabulary and the hyphenated form is not a known compound."""
    def join(m):
        a, b = m.group(1), m.group(2)
        if (a + b).lower() in vocab and f"{a}-{b}".lower() not in compounds:
            return a + b
        return f"{a}-{b}" if f"{a}-{b}".lower() in compounds else m.group(0)

    def join_inline(m):                       # "rota-tion" written on one line
        a, b = m.group(1), m.group(2)
        whole = (a + b).lower()
        if whole in vocab and f"{a}-{b}".lower() not in compounds and (a.lower() not in vocab or b.lower() not in vocab):
            return a + b
        return m.group(0)

    def fix(t):
        t = _HYPH_RE.sub(join, t)
        return re.sub(r"(?<![A-Za-z\\])([A-Za-z]{2,})-([a-z]{2,})(?![A-Za-z])", join_inline, t)
    return _outside_code(md, fix)


def unwrap_text_math(md: str) -> str:
    """'$\\text{for example}$' / '$\\textit{ourselves}$' is prose wrapped in math by mistake: unwrap it
    (a single lowercase word becomes *italic*, the usual reason it was set apart)."""
    pat = re.compile(r"(?<![\\$])\$\\(text|textit|textrm|mathrm|mathit|textbf|mathbf)\{([^{}$]*)\}\$(?!\$)")

    def sub(m):
        bold = m.group(1) in ("textbf", "mathbf")
        body = m.group(2)
        if re.search(r"[A-Za-z]{3,}\s+[A-Za-z]", body) or re.fullmatch(r"[a-z]{3,}", body):
            return f"**{body}**" if bold else f"*{body}*" if " " not in body else body
        return m.group(0)
    return _outside_code(md, lambda t: pat.sub(sub, t))


def prose_in_math(md: str) -> list[str]:
    """Inline math spans that are really sentences: six or more words inside \\text{}
    making up most of the span. (Displays with text, like "dof = (sum of freedoms) - ...",
    are how some books typeset formulas in words, so they are left alone.)"""
    out = []
    for tex, display in math_spans(md):
        if display:
            continue
        texts = re.findall(r"\\text\{([^{}]*)\}", tex)
        words = re.findall(r"[A-Za-z]{2,}", " ".join(texts))
        if len(words) >= 6 and sum(len(t) for t in texts) > 0.6 * len(tex):
            out.append(tex[:80])
    return out


LOOP_RE = re.compile(r"(.{1,40}?)\1{19,}", re.S)
LOOP_MIN_CHARS = 600       # a TOC dot leader line is ~100 chars; a VLM loop runs to the token limit


def loops(md: str) -> list[str]:
    """Patterns the transcription degenerated into repeating (a VLM generation loop)."""
    return [m.group(1) for m in LOOP_RE.finditer(md) if len(m.group(0)) >= LOOP_MIN_CHARS]


def collapse_loops(md: str) -> str:
    return LOOP_RE.sub(lambda m: m.group(1) + " [...]" if len(m.group(0)) >= LOOP_MIN_CHARS else m.group(0), md)


def math_spans(md: str) -> list[tuple[str, bool]]:
    """(tex, is_display) for every math span outside code fences."""
    spans = []

    def collect(text: str) -> str:
        for m in MATH_RE.finditer(text):
            spans.append((m.group(1), True) if m.group(1) is not None else (m.group(2), False))
        return text

    _outside_code(md, collect)
    return spans


def katex_errors(spans: list[tuple[str, bool]]) -> list[str | None]:
    if not spans:
        return []
    proc = subprocess.run(
        ["node", str(config.TOOLS_DIR / "katex_check.js")],
        input=json.dumps([{"tex": t, "display": d} for t, d in spans]),
        capture_output=True, text=True, encoding="utf-8", cwd=config.TOOLS_DIR,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"katex_check failed: {proc.stderr[:500]}")
    return json.loads(proc.stdout)


_l2t = LatexNodes2Text(math_mode="text", strict_latex_spaces=False)


def latex_to_text(tex: str) -> str:
    try:
        return _l2t.latex_to_text(tex)
    except Exception:
        return tex


def md_to_text(md: str) -> str:
    """Plain text for word matching: math rendered to Unicode, markup removed."""
    def math_sub(m):
        return " " + latex_to_text(m.group(1) if m.group(1) is not None else m.group(2)) + " "
    text = _outside_code(md, lambda t: MATH_RE.sub(math_sub, t))
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)           # images
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)        # links
    text = text.replace(r"\_", "_").replace("<", " ").replace(">", " ")  # keep words in XML/HTML
    text = re.sub(r"[#*_`|>]+", " ", text)
    return text


def md_for_terminal(md: str) -> str:
    """Math spans converted to Unicode, rest untouched (reader display)."""
    def math_sub(m):
        if m.group(1) is not None:
            return "\n    " + latex_to_text(m.group(1)).strip() + "\n"
        return latex_to_text(m.group(2))
    return _outside_code(md, lambda t: MATH_RE.sub(math_sub, t))


def canon_math(tex: str) -> str:
    """Canonical LaTeX for comparison: drops spacing, sizing, font aliases and layout."""
    t = tex.strip()
    # array with delimiters is a matrix; a bare array is alignment layout
    t = re.sub(r"\\left\s*[\[(|]\s*\\begin\{array\}\{[^}]*\}(.*?)\\end\{array\}\s*\\right\s*[\])|]",
               r"\\begin{matrix}\1\\end{matrix}", t, flags=re.S)
    t = re.sub(r"\\begin\{array\}\{[^}]*\}|\\end\{array\}", "", t)
    t = re.sub(r"\\quad\s*\((\d+(?:\.\d+)*[a-z]?)\)\s*$", r"\\tag{\1}", t)       # "\quad (5.5)"
    t = re.sub(r"\\(?:left|right|bigl|bigr|Bigl|Bigr|biggl|biggr|big|Big|bigg|Bigg)(?![a-zA-Z])", "", t)
    t = re.sub(r"\\(?:,|;|:|!|quad|qquad|displaystyle|textstyle|nonumber|notag)(?![a-zA-Z])", "", t)
    t = re.sub(r"(?<!\\)\\ ", "", t)                               # "\ " explicit space (not "\\ ")
    t = re.sub(r"~", "", t)
    t = re.sub(r"\\(?:ldots|cdots|dots|dotsc|dotsb|cdotp)(?![a-zA-Z])", r"\\cdots", t)
    t = re.sub(r"\\(?:mathrm|text|textrm|operatorname|mathit|textit)\s*\{([^{}]*)\}", r"\1", t)
    t = re.sub(r"\\(?:boldsymbol|bm|mathbf|textbf)(?![a-zA-Z])", r"\\mathbf", t)
    t = re.sub(r"\\[dt]frac(?![a-zA-Z])", r"\\frac", t)
    t = re.sub(r"\\var(phi|theta|epsilon|rho|sigma|pi)(?![a-zA-Z])", r"\\\1", t)
    t = re.sub(r"\\ensuremath(?![a-zA-Z])", "", t)
    t = re.sub(r"\\vec(?![a-zA-Z])", r"\\overrightarrow", t)
    t = re.sub(r"\\(?:ge|geq|geqslant)(?![a-zA-Z])", r"\\geq", t)
    t = re.sub(r"\\(?:le|leq|leqslant)(?![a-zA-Z])", r"\\leq", t)
    t = re.sub(r"\\(?:ne|neq)(?![a-zA-Z])", r"\\neq", t)
    t = re.sub(r"\\(?:to|rightarrow)(?![a-zA-Z])", r"\\to", t)
    t = re.sub(r"\\(?:lVert|Vert)(?![a-zA-Z])|\\rVert(?![a-zA-Z])", r"\\|", t)
    t = re.sub(r"\\(?:lvert|rvert)(?![a-zA-Z])", "|", t)
    t = re.sub(r"\\begin\{(?:aligned|align\*?|split|gathered|eqnarray\*?)\}|\\end\{(?:aligned|align\*?|split|gathered|eqnarray\*?)\}", "", t)
    t = re.sub(r"\\(begin|end)\{[bpvBV]?matrix\}", r"\\\1{matrix}", t)
    t = re.sub(r"[\[(]\s*\\begin\{matrix\}", r"\\begin{matrix}", t)     # explicit brackets around a matrix
    t = re.sub(r"\\end\{matrix\}\s*[\])]", r"\\end{matrix}", t)
    # Line breaks matter inside matrices only; elsewhere they are layout.
    mats = []

    def protect(m):
        mats.append(m.group(0))
        return f"\x00{len(mats) - 1}\x00"

    t = re.sub(r"\\begin\{matrix\}.*?\\end\{matrix\}", protect, t, flags=re.S)
    t = t.replace("\\\\", "").replace("&", "")
    t = re.sub(r"\x00(\d+)\x00", lambda m: mats[int(m.group(1))], t)
    t = re.sub(r"\s+", "", t)
    prev = None
    while prev != t:
        prev = t
        t = re.sub(r"\{(\\[a-zA-Z]+|[^{}\\])\}", r"\1", t)
    t = t.rstrip(".,;")
    return t
