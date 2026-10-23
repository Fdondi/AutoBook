# Talking Textbook (AutoBook)

A textbook you can talk to: answers stay inside what you've read, cite it, and never leak concepts the book introduces later. Design: [`talking-textbook-plan.md`](talking-textbook-plan.md).

Everything runs locally: LM Studio on `localhost:1234` serves Qwen3-30B-A3B (digest), Gemma 4 26B-A4B (page vision), Qwen3-4B (reader) and nomic-embed-text v1.5 (retrieval). `tbook` loads the right model for each step.

## Reader

Double-click **`AutoBook Reader.cmd`** (or run `uv run tbook open [book]`). It opens the book's own PDF in your browser, with a normal page view: contents, page numbers, zoom and search. **Ask** and **Quiz me** open a side panel:

- **Ask** answers from the pages you've read and links each claim to its page. Select text on a page to ask about it. A question that needs later material gets pointers to where the book covers it, plus a choice: answer with what you've read, or explain anyway (spoilers).
- **Quiz me** asks the book's pre-written questions from sections you've read, in this chapter or everything so far. You can type an answer for the reader model to check, or reveal the book's answer and grade yourself.

A page counts as read after about 20 seconds on screen, and a section once all its pages are read. The `…` menu in the panel can mark everything up to the current page as read, or forget your history. The PDF.js viewer comes from `tools/package.json` (`npm install` in `tools/`).

### Android

`app/` is the Android reader (Kotlin, Compose, minSdk 34). The library screen is native; the reader screen runs the same web reader in a WebView. Reading progress and quizzes are handled in Kotlin (`BookSession.kt` mirrors `web.py`). Asking questions runs fully on the phone: llama.cpp (pinned in `app/app/src/main/cpp/CMakeLists.txt`) with the same Qwen3-4B and nomic-embed models as the desktop, and a Kotlin port of the answer pipeline (`pipeline/`).

1. `uv run tbook export mr` writes `out/phone/mr.tbook`: the package with the PDF inside, plus the concept-term patterns the phone's leak checker uses.
2. Copy that file to the phone over USB, for example into Download.
3. In the app, tap **Add book** and pick it. Alternatively, copy it into `Android/data/com.example.autoreader/files/` and it's added the next time the app opens.
4. For asking questions, add the two model files the same way: `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` and `nomic-embed-text-v1.5.Q4_K_M.gguf`.

Build with `app/gradlew assembleDebug` (run `npm install` in `tools/` first: the build copies PDF.js, KaTeX and the reader prompts into the app). `app/gradlew testDebugUnitTest` includes a parity test of the Kotlin pipeline against the desktop's decisions, from `tools/reader_parity.py`.

Terminal reader:

```bash
uv run tbook toc mr                       # chapters with your reading progress
uv run tbook read mr 3.2.3                # mark everything up to §3.2.3 as read
uv run tbook show mr 3.3.2                # read a section (marks it read)
uv run tbook ask mr "How do I describe how fast a body is rotating?"
uv run tbook ask mr "What is a twist?"            # unread concept -> forward pointer
uv run tbook ask mr "What is a twist?" --preview  # ...or a spoiler, if you ask for it
uv run tbook chat mr                      # interactive; /read 3.3, /seen, /mode preview
uv run tbook read mr --reset              # forget reading history
```

Each answer shows its citations (§ and printed page), how many concepts were allowed, the leak-checker verdict, and time to first token.

## Digest (once per book, on the 3090)

```bash
uv run tbook pages mr fob                 # page images, text-layer oracle, bookmark tree
bash tools/setup_converters.sh            # MinerU / Marker / Docling venvs (once)
bash tools/run_full_extraction.sh         # 4 transcriptions per page -> consensus -> checks -> repair
uv run tbook digest mr fob                # stages 2-12 -> out/<book>.tbook
uv run tbook status mr
```

Extraction quality is measured, not assumed:

- `uv run tbook extract-eval mr fob --methods vlm_ml,mineru` scores transcriptions against hand-checked pages in `evals/extract/ref/`.
- `build/<book>/reports/extraction.md` lists every page or expression the checks could not confirm. Fixes and sign-offs go in `books/<book>.review.toml` and are re-applied on every run.
- `uv run tbook concept-eval mr --chapters 2,3 --list` checks concept extraction (numbered definitions, bold terms).
- `uv run tbook eval mr` is the leak-rate kill gate on auto-generated trap questions (`evals/<book>/traps.jsonl`, editable).

Other reports in `build/<book>/reports/`: `concepts.md`, `forward_refs.md`, `external_prereqs.md`, `verify.md`.
