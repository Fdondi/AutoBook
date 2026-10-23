"""tbook command line: digest, extraction evals, reader."""

import argparse
import sys
import tomllib

from . import config


def _hard_pages(book_id: str) -> list[int]:
    data = tomllib.loads((config.EVALS_DIR / "extract" / "pages.toml").read_text(encoding="utf-8"))
    return data[book_id]["pages"]


def _page_arg(book_id: str, spec: str | None) -> list[int] | None:
    """'hard' -> hard-page set; '10-20,33' -> explicit; None -> all."""
    if spec is None:
        return None
    if spec == "hard":
        return _hard_pages(book_id)
    pages = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        pages.extend(range(int(a), int(b or a) + 1))
    return pages


def cmd_pages(args) -> None:
    from .books import load
    from .extract import pages
    for b in args.books:
        pages.extract_pages(load(b), render=not args.no_render)


def cmd_convert(args) -> None:
    import pymupdf
    from .books import load
    from .extract import convert
    from . import lmstudio
    for b in args.books:
        book = load(b)
        sel = _page_arg(b, args.pages) or list(range(pymupdf.open(book.pdf).page_count))
        for method in args.methods.split(","):
            if method in convert.VLM_METHODS or method == "marker_llm":
                lmstudio.ensure_loaded(config.VISION_MODEL, config.VISION_CTX, config.VISION_PARALLEL)
            convert.convert(book, method, sel)


def cmd_check(args) -> None:
    from .books import load
    from .extract import checks
    from . import lmstudio
    if not args.no_judge:
        lmstudio.ensure_loaded(config.VISION_MODEL, config.VISION_CTX, config.VISION_PARALLEL)
    for b in args.books:
        for method in args.methods.split(","):
            checks.check(load(b), method, _page_arg(b, args.pages), use_judge=not args.no_judge)


def cmd_extract_eval(args) -> None:
    import collections
    from rich.console import Console
    from rich.table import Table
    from .books import load
    from .evals import extract_eval
    t = Table("book", "method", "pages", "text recall", "math exact", "auto pass", "judge flags")
    for b in args.books:
        rows = extract_eval.evaluate(load(b), args.methods.split(","))
        by = collections.defaultdict(list)
        for r in rows:
            by[r["method"]].append(r)
        for m, rs in by.items():
            n = sum(r["math_n"] for r in rs)
            ex = sum(r["math_exact"] * r["math_n"] for r in rs)
            judged = [r["judge_n"] for r in rs if r["judge_n"] is not None]
            t.add_row(b, m, str(len(rs)), f"{sum(r['text_recall'] for r in rs) / len(rs):.3f}",
                      f"{ex / max(n, 1):.3f} ({round(ex)}/{n})",
                      f"{sum(bool(r['auto_pass']) for r in rs)}/{len(rs)}",
                      f"{sum(judged)} on {sum(1 for j in judged if j)} pages" if judged else "-")
            if args.verbose:
                for r in rs:
                    for w in r["math_wrong"]:
                        print(f"  {b} {m} p{r['page_idx']}  REF {w['ref'][:110]}\n{'':>16}GOT {w['best'][:110]}")
    Console().print(t)


def cmd_finalize(args) -> None:
    from .books import load
    from .extract import final
    from . import lmstudio
    lmstudio.ensure_loaded(config.VISION_MODEL, config.VISION_CTX, config.VISION_PARALLEL)
    for b in args.books:
        final.finalize(load(b), _page_arg(b, args.pages), attempts=args.attempts)


def cmd_concept_eval(args) -> None:
    from .books import load
    from .evals import concept_eval
    res = concept_eval.evaluate(load(args.book), set(args.chapters.split(",")) if args.chapters else None)
    print(f"anchor recall: {res['anchor_recall']} of {res['anchors']}   bold-term recall: {res['bold_recall']} of {res['bold_terms']}")
    for m in res["anchor_missed"]:
        print("  missed anchor:", m)
    for m in res["bold_missed"][: args.show]:
        print("  missed bold:", m)
    if args.list:
        for ch, cs in res["concepts_by_chapter"].items():
            print(f"== chapter {ch}: {len(cs)} concepts")
            for name, unit, kind in cs:
                print(f"   {unit:18s} {kind:10s} {name}")


def cmd_eval(args) -> None:
    from .books import load
    from .evals import leak_eval
    from . import lmstudio
    book = load(args.book)
    if args.rebuild or not leak_eval.traps_path(book).exists():
        lmstudio.ensure_embedder()
        lmstudio.ensure_loaded(config.DIGEST_MODEL, config.DIGEST_CTX, config.DIGEST_PARALLEL)
        leak_eval.build_traps(book, n=args.n)
    stats = leak_eval.run(book)
    print(("PASS" if stats["pass"] else "FAIL") + f": leak rate {stats['leak_rate']:.1%} "
          f"(threshold {stats['threshold']:.0%}) on {stats['traps']} trap questions")


def cmd_digest(args) -> None:
    from .books import load
    from .digest import pipeline
    for b in args.books:
        pipeline.run(load(b), start=args.start, only=args.only, force=args.force)


def cmd_status(args) -> None:
    from .books import load
    from .digest import pipeline
    for b in args.books:
        print(f"== {b}")
        for name, s in pipeline.status(load(b)):
            print(f"  {name:18s} {s}")


def _reader(args):
    from .reader.answer import Reader
    from . import lmstudio
    lmstudio.ensure_embedder()
    lmstudio.ensure_loaded(config.READER_MODEL, config.READER_CTX, 1, exclusive=args.exclusive)
    return Reader(args.book)


def cmd_toc(args) -> None:
    from rich.console import Console
    from .books import load
    from .db import connect
    from .reader.learner import Learner
    con = connect(load(args.book).package, readonly=True)
    seen = Learner(args.book).seen_units()
    console = Console()
    last = None
    for u in con.execute("SELECT unit_id, section, title, printed_page_label, kind FROM units ORDER BY ordinal"):
        if u["section"] == last:
            continue
        last = u["section"]
        units = [r[0] for r in con.execute("SELECT unit_id FROM units WHERE section=?", (u["section"],))]
        mark = "[green]●[/green]" if all(x in seen for x in units) else "[yellow]◐[/yellow]" if any(x in seen for x in units) else "○"
        depth = u["section"].count(".")
        console.print(f"{mark} {'  ' * depth}§{u['section']} {u['title']} [dim]p. {u['printed_page_label']}[/dim]")


def cmd_show(args) -> None:
    from .books import load
    from .db import connect
    from .reader import render
    from .reader.learner import Learner
    con = connect(load(args.book).package, readonly=True)
    rows = con.execute("SELECT * FROM units WHERE section=? OR unit_id=? ORDER BY ordinal", (args.section, args.section)).fetchall()
    if not rows:
        raise SystemExit(f"no section {args.section}")
    learner = Learner(args.book)
    for r in rows:
        render.unit(r)
        learner.log("open_unit", r["unit_id"], via="show")


def cmd_read(args) -> None:
    from .books import load
    from .db import connect
    from .reader.learner import Learner
    con = connect(load(args.book).package, readonly=True)
    learner = Learner(args.book)
    if args.reset:
        learner.reset()
    if args.section:
        if args.only:
            units = [r[0] for r in con.execute("SELECT unit_id FROM units WHERE section=? OR section LIKE ?",
                                                (args.section, args.section + ".%"))]
        else:
            match = con.execute("SELECT max(ordinal) FROM units WHERE section=? OR section LIKE ? OR unit_id=?",
                                (args.section, args.section + ".%", args.section)).fetchone()[0]
            if match is None:
                raise SystemExit(f"no section {args.section}")
            units = [r[0] for r in con.execute("SELECT unit_id FROM units WHERE ordinal <= ?", (match,))]
        for u in units:
            learner.log("open_unit", u, via="read")
    print(f"{len(learner.seen_units())} units read")


def cmd_ask(args) -> None:
    from .reader import render
    from .reader.learner import Learner
    reader = _reader(args)
    learner = Learner(args.book)
    seen = reader.seen_through(args.seen_through) if args.seen_through else learner.seen_units()
    mode = "preview" if args.preview else "within" if args.within else "ask"
    learner.log("question_asked", None, question=args.question, mode=mode)
    ans = reader.ask(args.question, seen, mode=mode)
    learner.log("answer_given", None, answer_kind=ans.kind, source="local", text=ans.text[:2000])
    render.answer(ans, reader)


def cmd_chat(args) -> None:
    from rich.console import Console
    from .reader import render
    from .reader.learner import Learner
    console = Console()
    reader = _reader(args)
    learner = Learner(args.book)
    mode = "ask"
    console.print("[dim]/read <section>  /seen  /mode ask|preview|within  /quit[/dim]")
    while True:
        try:
            q = console.input("[bold cyan]you>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q in ("/quit", "/exit"):
            break
        if q.startswith("/read "):
            for u in reader.seen_through(q.split(maxsplit=1)[1]):
                learner.log("open_unit", u, via="chat")
            console.print(f"[dim]{len(learner.seen_units())} units read[/dim]")
            continue
        if q == "/seen":
            seen = learner.seen_units()
            last = max((reader.units[u] for u in seen), key=lambda u: u["ordinal"], default=None)
            console.print(f"[dim]{len(seen)} units read; furthest: §{last['section'] if last else '-'}[/dim]")
            continue
        if q.startswith("/mode "):
            mode = q.split()[1]
            continue
        learner.log("question_asked", None, question=q, mode=mode)
        ans = reader.ask(q, learner.seen_units(), mode=mode)
        learner.log("answer_given", None, answer_kind=ans.kind, source="local", text=ans.text[:2000])
        render.answer(ans, reader)


def cmd_open(args) -> None:
    from .reader import web
    web.serve(args.book, port=args.port, open_browser=not args.no_browser)


def cmd_export(args) -> None:
    from .books import load
    from .export import export
    for b in args.books:
        path = export(load(b))
        print(f"{path} ({path.stat().st_size / 1e6:.1f} MB): copy it to the phone and add it in the reader app")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="tbook")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pages", help="render pages, build text-layer oracle and TOC")
    p.add_argument("books", nargs="+")
    p.add_argument("--no-render", action="store_true")
    p.set_defaults(fn=cmd_pages)

    p = sub.add_parser("convert", help="run converters on pages")
    p.add_argument("books", nargs="+")
    p.add_argument("--methods", required=True, help="comma list: marker,marker_llm,docling,mineru,vlm,vlm_hint")
    p.add_argument("--pages", help="'hard', '10-20,33', or omit for all")
    p.set_defaults(fn=cmd_convert)

    p = sub.add_parser("check", help="run extraction checks")
    p.add_argument("books", nargs="+")
    p.add_argument("--methods", required=True)
    p.add_argument("--pages")
    p.add_argument("--no-judge", action="store_true")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("extract-eval", help="score converters against reference pages")
    p.add_argument("books", nargs="+")
    p.add_argument("--methods", required=True)
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fn=cmd_extract_eval)

    p = sub.add_parser("finalize", help="consensus + checks + repair -> final page Markdown")
    p.add_argument("books", nargs="+")
    p.add_argument("--pages")
    p.add_argument("--attempts", type=int, default=config.REPAIR_ATTEMPTS)
    p.set_defaults(fn=cmd_finalize)

    p = sub.add_parser("concept-eval", help="concept extraction oracles")
    p.add_argument("book")
    p.add_argument("--chapters", help="comma list of chapter numbers")
    p.add_argument("--list", action="store_true", help="list concepts per chapter")
    p.add_argument("--show", type=int, default=30)
    p.set_defaults(fn=cmd_concept_eval)

    p = sub.add_parser("digest", help="run digest stages 2-12 (after extraction)")
    p.add_argument("books", nargs="+")
    p.add_argument("--start", help="first stage to run")
    p.add_argument("--only", help="run a single stage")
    p.add_argument("--force", action="store_true", help="rerun even if up to date")
    p.set_defaults(fn=cmd_digest)

    p = sub.add_parser("eval", help="leak-rate kill gate on trap questions")
    p.add_argument("book")
    p.add_argument("--n", type=int, default=60, help="number of trap questions to generate")
    p.add_argument("--rebuild", action="store_true", help="regenerate evals/<book>/traps.jsonl")
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("status", help="digest progress")
    p.add_argument("books", nargs="+")
    p.set_defaults(fn=cmd_status)

    for name, fn, hlp in (("toc", cmd_toc, "table of contents with reading progress"),
                          ("show", cmd_show, "print a section (marks it read)")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("book")
        if name == "show":
            p.add_argument("section")
        p.set_defaults(fn=fn)

    p = sub.add_parser("read", help="mark everything up to a section as read")
    p.add_argument("book")
    p.add_argument("section", nargs="?")
    p.add_argument("--only", action="store_true", help="mark only this section")
    p.add_argument("--reset", action="store_true", help="forget reading history first")
    p.set_defaults(fn=cmd_read)

    for name, fn in (("ask", cmd_ask), ("chat", cmd_chat)):
        p = sub.add_parser(name, help="ask the book a question" if name == "ask" else "interactive session")
        p.add_argument("book")
        if name == "ask":
            p.add_argument("question")
            p.add_argument("--seen-through", help="override reading history: everything up to this section")
            g = p.add_mutually_exclusive_group()
            g.add_argument("--preview", action="store_true", help="allow spoilers for the concepts the question needs")
            g.add_argument("--within", action="store_true", help="answer using only what has been read")
        p.add_argument("--exclusive", action="store_true", help="unload other models (phone-like memory budget)")
        p.set_defaults(fn=fn)

    p = sub.add_parser("open", help="read a book in the browser, with questions and quizzes")
    p.add_argument("book", nargs="?", help="open this book directly (default: the library)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(fn=cmd_open)

    p = sub.add_parser("export", help="phone-ready book file (package + PDF) in out/phone/")
    p.add_argument("books", nargs="+")
    p.set_defaults(fn=cmd_export)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
