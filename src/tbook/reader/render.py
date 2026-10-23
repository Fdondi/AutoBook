"""Terminal rendering for the reader (rich)."""

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from ..extract.mdutil import md_for_terminal

console = Console()


def answer(ans, reader) -> None:
    if ans.kind == "forward":
        lines = ["This needs material you haven't read yet:"]
        for w in ans.needed_unseen:
            lines.append(f"  • **{w['concept']}**: §{w['section']} *{w['title']}*, p. {w['page']}")
        lines.append("")
        lines.append("Read it first, ask again with `--preview` for a spoiler, or `--within` for an answer "
                     "that uses only what you've read.")
        console.print(Panel(Markdown("\n".join(lines)), title="forward pointer", border_style="yellow"))
    elif ans.kind == "fallback":
        lines = ["I couldn't answer this without using material you haven't read."]
        for w in ans.needed_unseen:
            lines.append(f"  • **{w['concept']}**: §{w['section']} *{w['title']}*, p. {w['page']}")
        console.print(Panel(Markdown("\n".join(lines)), title="fallback", border_style="red"))
    else:
        title = "answer (pre-generated Q&A)" if ans.kind == "qa" else "answer"
        console.print(Panel(Markdown(md_for_terminal(ans.text)), title=title, border_style="green"))
    if ans.citations:
        t = Table(show_header=False, box=None, padding=(0, 1))
        for c in ans.citations:
            t.add_row("↳", f"§{c['section']}", c["title"] or "", f"p. {c['page']}", f"[dim]{c['unit']}[/dim]")
        console.print(t)
    footer = [f"seen units: {ans.scope.get('seen_units')}", f"allowed concepts: {ans.scope.get('allowed_concepts')}"]
    if ans.leak is not None:
        v = (ans.leak.get("lexical") or {}) | {k: "(semantic)" for k in ans.leak.get("semantic") or []}
        footer.append("leak check: " + ("clean" if not v else "violations " + ", ".join(v)))
    if ans.attempts > 1:
        footer.append(f"attempts: {ans.attempts}")
    s = ans.stats
    if "ttft_s" in s:
        footer.append(f"ttft {s['ttft_s']}s · {s['tok_s']} tok/s")
    if "qa_sim" in s:
        footer.append(f"matched Q&A ({s['qa_sim']}): {s['qa_question'][:60]}")
    footer.append(f"total {s.get('total_s', '?')}s")
    console.print("[dim]" + " · ".join(str(x) for x in footer) + "[/dim]")


def unit(row) -> None:
    console.print(Panel(Markdown(md_for_terminal(row["body"])),
                        title=f"§{row['section']} {row['title']} · p. {row['printed_page_label']} · {row['unit_id']}"))
