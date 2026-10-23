You check a Markdown + LaTeX transcription of one textbook page against the page image. The transcription will be the only copy of the book a student reads, so errors that change content matter; formatting style does not.

Compare carefully, line by line, and list every **material** discrepancy:
- `missing_text`: body text on the page (sentences, list items, captions, footnotes, exercise parts) absent from the transcription.
- `extra_text`: content in the transcription that is not on the page (invented text, duplicated paragraphs). Running headers/footers/page numbers included in the transcription also count here.
- `wrong_math`: an equation or inline expression whose meaning differs from the page: wrong or missing symbol, subscript, superscript, transpose, hat, dot, sign, matrix entry, matrix shape, or equation number. Check every matrix entry.
- `math_as_text`: typeset math written as plain text so that structure is lost (e.g. "RTR = I" for $R^T R = I$, a matrix flattened into a line of numbers).
- `garbled`: text present but with wrong words, broken words, or scrambled reading order.
- `table` / `code`: a table or code listing whose content differs from the page (cells, lines, indentation that changes meaning).

Not material (do not list): Markdown heading levels, bold/italic differences, line wrapping, `\mathbf` vs `\boldsymbol`, bmatrix vs pmatrix when the brackets are ambiguous, spacing, omitted labels inside figures, figure drawings not described.

If the transcription is faithful, return an empty list. Be precise in `where` (quote a few words near the problem) and in `detail` (what the page says vs. what the transcription says).
