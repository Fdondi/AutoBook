"""Stage 1b: run a converter on a set of pages and store per-page Markdown.

External converters (Marker, Docling, MinerU) run in their own venvs as
subprocesses on a sub-PDF of the requested pages; the VLM methods run here.
"""

import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import pymupdf

from .. import config, llm
from ..books import Book
from . import mathlayer, mdutil
from .pages import book_vocab, head_key

CONVERTERS_DIR = config.TOOLS_DIR / "converters"
EXTERNAL = {
    "marker": ("marker", ["run_marker.py"]),
    "marker_llm": ("marker", ["run_marker.py", "--llm", config.LMSTUDIO_URL, config.VISION_MODEL]),
    "docling": ("docling", ["run_docling.py"]),
    "mineru": ("mineru", ["run_mineru.py"]),
}
# vlm: image only; vlm_hint: image + text layer; vlm_ml: image + annotated glyph layer;
# *_fast: reasoning off
VLM_METHODS = ("vlm", "vlm_hint", "vlm_fast", "vlm_ml", "vlm_ml_fast")
METHODS = tuple(EXTERNAL) + VLM_METHODS


def convert(book: Book, method: str, pages: list[int], *, log=print, workers: int = config.VISION_PARALLEL) -> None:
    con = book.connect()
    t0 = time.time()
    if method in EXTERNAL:
        results = _run_external(book, method, pages, log)
    elif method in VLM_METHODS:
        results = _run_vlm(book, method, pages, con, workers)
    else:
        raise SystemExit(f"unknown method {method}; choose from {', '.join(METHODS)}")
    heads = _heads_by_page(con, pages)
    rows = []
    for p, res in results.items():
        md, blocks = (res["md"], res.get("blocks")) if isinstance(res, dict) else (res, None)
        meta = {"raw_len": len(md)} | ({"blocks": blocks} if blocks is not None else {})
        rows.append((p, method, postprocess(md, heads.get(p, []), book), json.dumps(meta, ensure_ascii=False), time.time()))
    con.executemany("INSERT OR REPLACE INTO b_page_md VALUES (?,?,?,?,?)", rows)
    con.commit()
    missing = sorted(set(pages) - set(results))
    log(f"[{book.id}] {method}: {len(rows)} pages in {time.time() - t0:.0f}s"
        + (f"; no output for {missing}" if missing else ""))


def _heads_by_page(con, pages: list[int]) -> dict[int, list[str]]:
    q = f"SELECT page_idx, running_heads FROM b_pages WHERE page_idx IN ({','.join('?' * len(pages))})"
    return {r["page_idx"]: json.loads(r["running_heads"]) for r in con.execute(q, pages)}


_VOCAB: dict[str, tuple[set[str], set[str]]] = {}


def vocab(book: Book) -> tuple[set[str], set[str]]:
    if book.id not in _VOCAB:
        _VOCAB[book.id] = book_vocab(book.connect())
    return _VOCAB[book.id]


def postprocess(md: str, running_heads: list[str], book: Book | None = None) -> str:
    md = md.replace("\r\n", "\n")
    md = mdutil.normalize_delimiters(md)
    md = mdutil.unwrap_text_math(md)
    if book is not None:
        md = mdutil.dehyphenate(md, *vocab(book))
    head_keys = {head_key(h) for h in running_heads if re.search(r"[A-Za-z]", h)}
    lines = []
    for line in md.split("\n"):
        bare = re.sub(r"^[#*\s]+|[*\s]+$", "", line)
        if bare and head_key(bare) in head_keys:
            continue
        lines.append(line.rstrip())
    md = "\n".join(lines)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def _sub_pdf(book: Book, method: str, pages: list[int]):
    tmp = book.build_dir / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(book.pdf)
    if pages == list(range(doc.page_count)):
        return book.pdf, pages
    sub = pymupdf.open()
    for p in pages:
        sub.insert_pdf(doc, from_page=p, to_page=p)
    path = tmp / f"sub_{method}.pdf"
    sub.save(path)
    return path, pages


def _run_external(book: Book, method: str, pages: list[int], log) -> dict[int, str]:
    venv, script_args = EXTERNAL[method]
    py = config.TOOLS_DIR / "venvs" / venv / "Scripts" / "python.exe"
    pdf, mapping = _sub_pdf(book, method, pages)
    out = book.build_dir / "tmp" / f"out_{method}.jsonl"
    out.unlink(missing_ok=True)
    logf = book.build_dir / "tmp" / f"log_{method}.txt"
    cmd = [str(py), str(CONVERTERS_DIR / script_args[0]), str(pdf), str(out), *script_args[1:]]
    with open(logf, "w", encoding="utf-8") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=book.build_dir / "tmp",
                              env=_converter_env())
    if proc.returncode != 0 or not out.exists():
        tail = logf.read_text(encoding="utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"{method} failed (exit {proc.returncode}); log {logf}:\n{tail}")
    results = {}
    for line in out.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if 0 <= rec["page"] < len(mapping):
            results[mapping[rec["page"]]] = rec if "blocks" in rec else rec["md"]
    return results


def _converter_env() -> dict:
    import os
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["DO_NOT_TRACK"] = "1"
    return env


MATH_LAYER_NOTE = (
    "\n\nThe PDF's own glyphs for this page follow, annotated: `^{..}` superscript, `_{..}` subscript (nested as "
    "printed), `\\cal{X}` calligraphic, `\\frak{X}` fraktur, `\\bb{X}` blackboard bold, `\\bf{..}` bold. Within a "
    "line the glyphs are in printed order; stacked layouts (fractions, matrices, limits, braces) are split across "
    "lines, and accents such as ˙ ¨ ˆ appear just before the letter they sit on. Big brackets are omitted. The words, "
    "symbols, scripts and typefaces here are exact: use them to get every symbol right, and use the image for "
    "layout, structure (fractions, matrices, braces) and reading order. Running heads may still be present.\n\n"
    "<glyphs>\n{layer}\n</glyphs>"
)


def vlm_messages(png: str, text_layer: str | None, extra: str = "", math_layer: str | None = None) -> list:
    content = [llm.image_part(png)]
    note = "Transcribe this page."
    if math_layer is not None:
        note += MATH_LAYER_NOTE.replace("{layer}", math_layer)
    elif text_layer is not None:
        note += (
            "\n\nThe PDF's raw text layer for this page follows. Its words and their spelling are exact, "
            "but math is garbled and running heads may be present. Use it to get the wording exactly right; "
            "use the image for math, layout and reading order.\n\n<text_layer>\n" + text_layer + "\n</text_layer>"
        )
    if extra:
        note += "\n\n" + extra
    content.append({"type": "text", "text": note})
    return [{"role": "system", "content": llm.load_prompt("extract_transcribe")},
            {"role": "user", "content": content}]


def math_layer(book: Book, page_idx: int, running_heads: list[str]) -> str:
    doc = pymupdf.open(book.pdf)
    return mathlayer.annotate_page(doc[page_idx], re.compile(book.cfg["math_fonts"]), running_heads)


def _run_vlm(book: Book, method: str, pages: list[int], con, workers: int) -> dict[int, str]:
    rows = {r["page_idx"]: r for r in con.execute(
        f"SELECT page_idx, png, text_layer, running_heads FROM b_pages WHERE page_idx IN ({','.join('?' * len(pages))})",
        pages)}

    def one(p):
        r = rows[p]
        if method in ("vlm_ml", "vlm_ml_fast"):
            msgs = vlm_messages(r["png"], None, math_layer=math_layer(book, p, json.loads(r["running_heads"])))
        else:
            msgs = vlm_messages(r["png"], r["text_layer"] if method in ("vlm_hint", "vlm_fast") else None)
        reasoning = "none" if method.endswith("_fast") else None
        try:
            md = llm.chat(msgs, config.VISION_MODEL, max_tokens=8000, reasoning=reasoning, tag=f"{book.id}:{method}:{p}")
        except llm.LLMError:
            return p, None                # reported by convert() as "no output"
        return p, _strip_outer_fence(md)

    with ThreadPoolExecutor(workers) as ex:
        return {p: md for p, md in ex.map(one, pages) if md is not None}



def _strip_outer_fence(md: str) -> str:
    m = re.match(r"^\s*```(?:markdown|md)?\s*\n(.*)\n```\s*$", md, re.S)
    return m.group(1) if m else md
