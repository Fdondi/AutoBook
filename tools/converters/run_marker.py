"""Runs inside tools/venvs/marker. Converts a PDF to per-page Markdown (JSONL).

usage: run_marker.py <pdf> <out.jsonl> [--llm <openai_base_url> <model>]
Each output line: {"page": <0-based page in this pdf>, "md": "..."}
"""

import json
import re
import sys

from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict

SEP = "-" * 48


def main() -> None:
    pdf, out = sys.argv[1], sys.argv[2]
    cfg = {
        "output_format": "markdown",
        "paginate_output": True,
        "page_separator": SEP,
        "disable_image_extraction": True,
    }
    if "--llm" in sys.argv:
        i = sys.argv.index("--llm")
        cfg.update({
            "use_llm": True,
            "redo_inline_math": True,
            "llm_service": "marker.services.openai.OpenAIService",
            "openai_base_url": sys.argv[i + 1],
            "openai_model": sys.argv[i + 2],
            "openai_api_key": "lm-studio",
        })
    parser = ConfigParser(cfg)
    converter = PdfConverter(
        config=parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=parser.get_processors(),
        renderer=parser.get_renderer(),
        llm_service=parser.get_llm_service(),
    )
    md = converter(pdf).markdown
    parts = re.split(r"\n*\{(\d+)\}" + re.escape(SEP) + r"\n*", "\n\n" + md)
    with open(out, "w", encoding="utf-8") as f:
        for k in range(1, len(parts) - 1, 2):
            f.write(json.dumps({"page": int(parts[k]), "md": parts[k + 1].strip()}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
