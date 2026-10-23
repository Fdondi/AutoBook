"""Runs inside tools/venvs/mineru. Converts a PDF to per-page Markdown (JSONL).

Uses MinerUParser in-process (no document-library server, no telemetry).
usage: run_mineru.py <pdf> <out.jsonl> [tier]
Each output line: {"page": <0-based page in this pdf>, "md": "...",
                   "blocks": [{"type", "bbox" (page fractions x0,y0,x1,y1), "latex"|"text"}]}
The blocks locate equations on the page so disputed math can be cropped and re-read.
"""

import copy
import json
import sys

from mineru.parser.base import ParseResult
from mineru.parser.mineru_parser import MinerUParser


def _blocks(page: dict) -> list[dict]:
    out = []
    for b in page.get("blocks", []):
        c = b.get("content")
        rec = {"type": b["type"], "bbox": b["bbox"]}
        if b["type"] == "equation" and isinstance(c, str):
            rec["latex"] = c
        elif isinstance(c, list):
            rec["inline"] = [x.get("content") for x in c if isinstance(x, dict) and x.get("type") == "equation_inline"]
            rec["text"] = "".join(x.get("content", "") if isinstance(x.get("content"), str) else ""
                                  for x in c if isinstance(x, dict))[:400]
        out.append(rec)
    return out


def main() -> None:
    pdf, out = sys.argv[1], sys.argv[2]
    tier = sys.argv[3] if len(sys.argv) > 3 else "advanced"
    result = MinerUParser(tier=tier).parse(pdf)
    middle = result.middle_json
    pages = result.to_dict()["pages"]
    with open(out, "w", encoding="utf-8") as f:
        for i, page in enumerate(middle.pages):
            one = copy.copy(middle)
            one.pages = [page]
            md = ParseResult(middle_json=one).markdown()
            rec = {"page": i, "md": md.strip(), "blocks": _blocks(pages[i]) if i < len(pages) else []}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
