"""Runs inside tools/venvs/docling. Converts a PDF to per-page Markdown (JSONL).

usage: run_docling.py <pdf> <out.jsonl>
Each output line: {"page": <0-based page in this pdf>, "md": "..."}
"""

import json
import sys

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


def main() -> None:
    pdf, out = sys.argv[1], sys.argv[2]
    opts = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=True,
        do_formula_enrichment=True,
        do_code_enrichment=True,
        generate_page_images=False,
    )
    conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    doc = conv.convert(pdf).document
    with open(out, "w", encoding="utf-8") as f:
        for page_no in sorted(doc.pages):
            md = doc.export_to_markdown(page_no=page_no, image_placeholder="")
            f.write(json.dumps({"page": page_no - 1, "md": md.strip()}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
