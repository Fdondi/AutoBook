"""Dump MinerU's blocks (type, bbox, content) for one page (runs in tools/venvs/mineru)."""

import json
import sys

from mineru.parser.mineru_parser import MinerUParser

if __name__ == "__main__":
    r = MinerUParser(tier="advanced").parse(sys.argv[1], page_range=sys.argv[2])
    d = r.to_dict()
    pg = d["pages"][0]
    print("page keys", [k for k in pg if k != "blocks"], {k: pg[k] for k in pg if k not in ("blocks",)})
    for b in pg["blocks"]:
        print(b["type"], b["bbox"], json.dumps(b.get("content"), ensure_ascii=False)[:230])
