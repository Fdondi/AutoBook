"""Browser reader (`tbook open`): the book's own PDF pages, with a panel to ask
questions or be quizzed.

Reading is tracked by page: a page counts as read once it has been on screen for
a while (the browser decides and reports it), and a unit once all its pages are
read. The seen set the answer pipeline uses is the learner log's open_unit events,
the same as the terminal reader's.
"""

import json
import random
import re
import threading
import urllib.request
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .. import config, llm, lmstudio
from ..books import load as load_book
from ..db import connect
from .learner import Learner

PORT = 8765
STATIC = Path(__file__).parent / "static"
ROUTES = {                                       # URL prefix -> directory served under it
    "/static/": STATIC,
    "/vendor/pdfjs/": config.TOOLS_DIR / "node_modules" / "pdfjs-dist",
    "/vendor/katex/": config.TOOLS_DIR / "node_modules" / "katex" / "dist",
}
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript",
         ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml", ".wasm": "application/wasm",
         ".woff2": "font/woff2", ".woff": "font/woff", ".ttf": "font/ttf", ".pdf": "application/pdf"}
GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["correct", "partly", "incorrect"]},
        "feedback": {"type": "string"},
    },
    "required": ["verdict", "feedback"],
}
QUIZ_ORDER = {None: 0, "incorrect": 1, "partly": 2, "skipped": 3, "correct": 4}   # unasked first, then misses
CITE_MATH_RE = re.compile(r"\$\\text\{([a-z]+:[^}\s]+)\}\$")       # Q&A answers cite as "$\text{mr:1-01}$"
CITE_RE = re.compile(r"\s*\[[a-z]+:[^\]\s]+\]")
PASSAGE_RE = re.compile(r",?\s+(?:according to|as described in|as stated in|as explained in|described in|in|from) "
                        r"the (?:passage|text)\b", re.I)
# The digest's JSON decoding turned some LaTeX commands into control characters: "\text" -> TAB + "ext".
CONTROL_RE = re.compile(r"[\t\x08\x0c\r](?=[A-Za-z])")
CONTROL_CMD = {"\t": "\\t", "\x08": "\\b", "\x0c": "\\f", "\r": "\\r"}
INLINE_MATH_RE = re.compile(r"(?<![\\$])\$(?!\$)([^$]+?)\$")


def clean_text(text: str, book_id: str) -> str:
    """Repair LaTeX broken into control characters (inside inline math, a newline before
    letters was "\\n...": "\\neq", "\\nu") and make every citation a [unit] link:
    "$\\text{mr:1-01}$", "`mr:1-01`" and bare "mr:1-01"."""
    text = CONTROL_RE.sub(lambda m: CONTROL_CMD[m[0]], text)
    text = INLINE_MATH_RE.sub(lambda m: "$" + re.sub(r"\n(?=[a-z])", lambda _: "\\n", m[1]) + "$", text)
    text = CITE_MATH_RE.sub(r"[\1]", text)
    text = re.sub(rf"`({book_id}:[^`\s]+)`", r"[\1]", text)
    return re.sub(rf"(?<![\[\w:]){book_id}:[\w.+/-]*\w", lambda m: f"[{m[0]}]", text)


def clean_question(q: str, book_id: str) -> str:
    """Generated questions refer to "the passage" they were written from; the reader sees a book."""
    return re.sub(r"\bthe passage\b", "the book", PASSAGE_RE.sub("", clean_text(q, book_id)))


class Session:
    """One open book. SQLite objects stay on the thread that made them: learner and
    package reads run on `db`, the Reader (one model slot) on `llm`."""

    def __init__(self, book_id: str):
        self.book = load_book(book_id)
        if not self.book.package.exists():
            raise LookupError(f"{book_id} has no package yet: run `tbook digest {book_id}`")
        self.db = ThreadPoolExecutor(1, thread_name_prefix=f"{book_id}-db")
        self.llm = ThreadPoolExecutor(1, thread_name_prefix=f"{book_id}-llm")
        self.on_db(self._open)
        self.reader = self.llm.submit(self._open_reader)
        self.model_ok = False

    def on_db(self, fn, *args):
        return self.db.submit(fn, *args).result()

    def _open(self) -> None:
        self.con = connect(self.book.package, readonly=True)
        self.learner = Learner(self.book.id)
        self.units = [dict(r) for r in self.con.execute(
            "SELECT unit_id, ordinal, chapter, section, title, page_start, page_end, printed_page_label "
            "FROM units ORDER BY ordinal")]
        # A unit runs from its first page to the page before the next unit starts (or that
        # page itself when the next one starts on it).
        for u, nxt in zip(self.units, self.units[1:] + [None]):
            end = nxt["page_start"] if nxt else (u["page_end"] or u["page_start"]) + 1
            u["last_page"] = max(u["page_start"], end - 1)
        self.by_id = {u["unit_id"]: u for u in self.units}
        self.qa = {r["qa_id"]: dict(r) for r in self.con.execute("SELECT qa_id, unit_id, question, answer, cited_units FROM qa")}

    def _open_reader(self):
        from .answer import Reader
        return Reader(self.book.id)

    def info(self) -> dict:
        return {"id": self.book.id, "title": self.book.cfg.get("title", self.book.id),
                "authors": self.book.cfg.get("authors", "")}

    def place(self, unit_id: str | None) -> dict:
        u = self.by_id.get(unit_id or "")
        if not u:
            return {}
        return {"unit": unit_id, "section": u["section"], "title": u["title"], "page": u["printed_page_label"],
                "page_idx": u["page_start"]}

    # -- reading state (db thread) ------------------------------------------

    def state(self) -> dict:
        seen = self.learner.seen_units()
        read = [u for u in self.units if u["unit_id"] in seen]
        furthest = max(read, key=lambda u: u["ordinal"], default=None)
        positions = self.learner.events("position")
        return {"book": self.info(), "features": {"ask": True, "check": True}, "pages_read": sorted(self._pages_read()),
                "last_page": positions[-1]["page"] if positions else None,
                "units_read": len(read), "units": len(self.units),
                "furthest": self.place(furthest["unit_id"]) if furthest else None}

    def _pages_read(self) -> set[int]:
        return {p for e in self.learner.events("page_read") for p in e["pages"]}

    def read_pages(self, pages: list[int], via: str) -> dict:
        have = self._pages_read()
        new = sorted(set(pages) - have)
        if new:
            self.learner.log("page_read", None, pages=new, via=via)
            have.update(new)
            seen = self.learner.seen_units()
            for u in self.units:
                if u["unit_id"] not in seen and all(p in have for p in range(u["page_start"], u["last_page"] + 1)):
                    self.learner.log("open_unit", u["unit_id"], via=via)
        return self.state()

    def position(self, page: int) -> None:
        positions = self.learner.events("position")
        if not positions or positions[-1]["page"] != page:
            self.learner.log("position", None, page=page)

    def reset(self) -> dict:
        self.learner.reset()
        return self.state()

    def unit_at(self, page: int) -> dict:
        return max((u for u in self.units if u["page_start"] <= page), key=lambda u: u["ordinal"], default=self.units[0])

    # -- quiz (db thread) ----------------------------------------------------

    def quiz(self, scope: str, page: int | None) -> dict:
        """A pre-generated question from a unit the reader has read: unasked ones first, then misses."""
        seen = self.learner.seen_units()
        chapter = self.unit_at(page)["chapter"] if page is not None else None
        pool = [q for q in self.qa.values() if q["unit_id"] in seen
                and (scope == "all" or self.by_id[q["unit_id"]]["chapter"] == chapter)]
        last = {e["qa_id"]: e["result"] for e in self.learner.events("quiz_result")}
        if not pool:
            return {"empty": True, "scope": scope, "any_read": bool(seen)}
        best = min(QUIZ_ORDER.get(last.get(q["qa_id"]), 0) for q in pool)
        q = random.choice([q for q in pool if QUIZ_ORDER.get(last.get(q["qa_id"]), 0) == best])
        answer = clean_text(q["answer"], self.book.id)
        units = json.loads(q["cited_units"] or "[]") + re.findall(r"\[([a-z]+:[^\]\s]+)\]", answer) + [q["unit_id"]]
        return {"qa_id": q["qa_id"], "question": clean_question(q["question"], self.book.id), "answer": answer,
                "source": self.place(q["unit_id"]),
                "citations": [self.place(u) for u in dict.fromkeys(units) if u in self.by_id],
                "pool": len(pool), "unasked": sum(1 for x in pool if x["qa_id"] not in last)}

    def quiz_result(self, qa_id: str, result: str, **data) -> None:
        self.learner.log("quiz_result", self.qa[qa_id]["unit_id"], qa_id=qa_id, result=result, **data)

    # -- model calls (llm thread) ----------------------------------------------

    def _ensure_model(self, emit) -> None:
        if self.model_ok:
            return
        if not any((m.get("modelKey") or m.get("identifier")) == config.READER_MODEL for m in lmstudio.loaded()):
            emit({"type": "status", "text": "Loading the reader model…"})
        lmstudio.ensure_embedder()
        lmstudio.ensure_loaded(config.READER_MODEL, config.READER_CTX, 1, exclusive=False)
        self.model_ok = True

    def _with_model(self, fn, emit):
        """Run `fn` on the llm thread; reload the model once if it went away (another tool unloaded it)."""
        def work():
            self._ensure_model(emit)
            try:
                return fn()
            except Exception:
                self.model_ok = False
                self._ensure_model(emit)
                return fn()
        return self.llm.submit(work).result()

    def ask(self, question: str, mode: str, page: int | None, emit) -> dict:
        if page is not None:
            self.on_db(self.read_pages, [page], "ask")          # the page on screen is being read
        seen = self.on_db(self.learner.seen_units)
        self.on_db(lambda: self.learner.log("question_asked", None, question=question, mode=mode, page=page))
        reader = self.reader.result()
        tokens = 0

        def on_token(_):
            nonlocal tokens
            tokens += 1
            if tokens % 5 == 0:
                emit({"type": "progress", "tokens": tokens})

        emit({"type": "status", "text": "Looking through the pages you've read…"})
        ans = self._with_model(lambda: reader.ask(question, seen, mode=mode, on_token=on_token), emit)
        self.on_db(lambda: self.learner.log("answer_given", None, answer_kind=ans.kind, source="local",
                                            text=ans.text[:2000]))
        return {"type": "answer", "kind": ans.kind, "mode": mode, "text": clean_text(ans.text, self.book.id),
                "citations": [self.place(c["unit"]) for c in ans.citations],
                "pointers": [{**w, **self.place(w.get("unit"))} for w in ans.needed_unseen],
                "stats": ans.stats}

    def check(self, qa_id: str, answer: str) -> dict:
        q = self.qa[qa_id]
        msgs = [{"role": "system", "content": llm.load_prompt("reader_grade")},
                {"role": "user", "content": f"Question: {clean_question(q['question'], self.book.id)}\n\n"
                                            f"Reference answer: {CITE_RE.sub('', clean_text(q['answer'], self.book.id))}\n\n"
                                            f"Student's answer: {answer}"}]
        res = self._with_model(lambda: llm.chat(msgs, config.READER_MODEL, schema=GRADE_SCHEMA, max_tokens=300,
                                                use_cache=False), lambda e: None)
        self.on_db(lambda: self.quiz_result(qa_id, res["verdict"], answer=answer[:2000], via="model"))
        return {**res, "feedback": clean_text(res["feedback"], self.book.id)}


_sessions: dict[str, Session] = {}
_sessions_lock = threading.Lock()


def session(book_id: str) -> Session:
    with _sessions_lock:
        if book_id not in _sessions:
            _sessions[book_id] = Session(book_id)
        return _sessions[book_id]


def library() -> list[dict]:
    out = []
    for path in sorted(config.BOOKS_DIR.glob("*.toml")):
        b = load_book(path.stem)
        if "pdf" not in b.cfg:                      # books/<id>.review.toml holds manual fixes, not a book
            continue
        out.append({"id": b.id, "title": b.cfg.get("title", b.id), "authors": b.cfg.get("authors", ""),
                    "ready": b.package.exists(), "units_read": len(Learner(b.id).seen_units())})
    return out


def _int(v) -> int | None:
    return int(v) if v not in (None, "") else None


class Handler(BaseHTTPRequestHandler):
    server_version = "tbook"

    def log_message(self, fmt, *args) -> None:
        pass

    def do_GET(self) -> None:
        url = urlparse(self.path)
        path = unquote(url.path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if path == "/":
                return self._file(STATIC / "library.html")
            if re.fullmatch(r"/book/\w+", path):
                return self._file(STATIC / "reader.html")
            if m := re.fullmatch(r"/pdf/(\w+)", path):
                return self._file(load_book(m[1]).pdf)
            if path == "/api/books":
                return self._json(library())
            if m := re.fullmatch(r"/api/(\w+)/state", path):
                s = session(m[1])
                return self._json(s.on_db(s.state))
            if m := re.fullmatch(r"/api/(\w+)/quiz", path):
                s = session(m[1])
                return self._json(s.on_db(s.quiz, query.get("scope", "chapter"), _int(query.get("page"))))
            for prefix, base in ROUTES.items():
                if path.startswith(prefix):
                    return self._file(base / path[len(prefix):], base=base)
            self.send_error(404)
        except (LookupError, SystemExit) as e:
            self.send_error(404, str(e))

    def do_POST(self) -> None:
        m = re.fullmatch(r"/api/(\w+)/([\w/]+)", urlparse(self.path).path)
        if not m:
            return self.send_error(404)
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        try:
            s = session(m[1])
        except (LookupError, SystemExit) as e:
            return self.send_error(404, str(e))
        action = m[2]
        if action == "ask":
            return self._stream(lambda emit: s.ask(body["question"], body.get("mode", "ask"), body.get("page"), emit))
        if action == "pages":
            return self._json(s.on_db(s.read_pages, body["pages"], "pdf"))
        if action == "mark":
            return self._json(s.on_db(s.read_pages, list(range(body["through"] + 1)), "mark"))
        if action == "position":
            s.on_db(s.position, body["page"])
            return self._json({"ok": True})
        if action == "reset":
            return self._json(s.on_db(s.reset))
        if action == "quiz/check":
            return self._stream(lambda emit: {"type": "grade", **s.check(body["qa_id"], body["answer"])})
        if action == "quiz/result":
            s.on_db(s.quiz_result, body["qa_id"], body["result"])
            return self._json({"ok": True})
        self.send_error(404)

    def _json(self, obj) -> None:
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _stream(self, fn) -> None:
        """Newline-delimited JSON events: status/progress while working, then the result (or an error)."""
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        lock = threading.Lock()

        def emit(event: dict) -> None:
            try:
                with lock:
                    self.wfile.write((json.dumps(event) + "\n").encode())
                    self.wfile.flush()
            except OSError:
                pass                                    # the page was closed: finish quietly

        try:
            emit(fn(emit))
        except Exception as e:
            emit({"type": "error", "text": f"{type(e).__name__}: {e}"})

    def _file(self, path: Path, base: Path | None = None) -> None:
        path = path.resolve()
        if (base is not None and not path.is_relative_to(base.resolve())) or not path.is_file():
            return self.send_error(404)
        size = path.stat().st_size
        start, end = 0, size - 1
        rng = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range") or "")
        if rng and (rng[1] or rng[2]):
            if rng[1]:
                start, end = int(rng[1]), min(int(rng[2]) if rng[2] else size - 1, size - 1)
            else:
                start = max(0, size - int(rng[2]))
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", TYPES.get(path.suffix.lower(), "application/octet-stream"))
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            with path.open("rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0 and (chunk := f.read(min(1 << 20, left))):
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except ConnectionError:
            pass                                        # PDF.js aborts the full download once ranges work


class Server(ThreadingHTTPServer):
    allow_reuse_address = False         # on Windows reuse lets a second server bind the same port
    daemon_threads = True


def _running(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/books", timeout=2) as r:
            return r.headers.get("Server", "").startswith("tbook")
    except OSError:
        return False


def serve(book_id: str | None = None, port: int = PORT, open_browser: bool = True) -> None:
    if book_id:
        load_book(book_id)
    url = f"http://127.0.0.1:{port}/" + (f"book/{book_id}" if book_id else "")
    if _running(port):
        print(f"The reader is already running: {url}")
        if open_browser:
            webbrowser.open(url)
        return
    try:
        server = Server(("127.0.0.1", port), Handler)
    except OSError as e:
        raise SystemExit(f"port {port} is taken ({e}); try --port {port + 1}")
    print(f"Reader running at {url}\nClose this window (or press Ctrl+C) to stop it.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
