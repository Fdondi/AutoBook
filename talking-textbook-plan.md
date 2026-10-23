# Talking Textbook — Implementation Plan

A textbook you can talk to. Explanations stay inside what you've already read, link back to it, and build on your history over weeks.

Two parts:

1. **Digester (desktop, offline).** Turns a book into a self-contained package: structure, concept graph, summaries, retrieval index and pre-generated Q&A. Heavy compute happens once, here.
2. **Reader (Fedora + Android).** Reads the package, tracks the learner, and answers with a small local model. The cloud is an optional escalation path.

Target hardware:

| Role | Hardware | Budget |
|---|---|---|
| Digest | RTX 3090 (24 GB), 13th-gen Intel CPU, 64 GB RAM | Unlimited time, local first |
| Reader, phone | 8 GB+ Android (dev device: Galaxy Fold 4) | ~3.5 GB for the AI stack |
| Reader, laptop | Fedora, Ryzen 7 7735HS + Radeon 680M iGPU (4 GB UMA) or CPU | ~3.5–4 GB |

---

## 0. Core principles

- **Deterministic first, model second.** The app computes learner state, retrieval scope and drill-down choice. The model reads, phrases and explains.
- **Do the expensive thinking at publish time.** Anything that doesn't depend on the reader is precomputed on the 3090.
- **Seen-set enforcement is the product.** Every generated answer is checked against what the reader has seen. If this doesn't hold, nothing else matters, so it gets tested first.
- **One tool hop at most.** Small models handle single calls well and multi-step chains poorly.
- **The same checker runs in both places.** The concept-leak checker used to validate digest output ships unchanged in the reader.

---

## 1. Book package format (`.tbook`)

A single SQLite file (plus optional asset directory), read-only on the client. Versioned and self-describing.

```
manifest          book id, title, source, license, digest version,
                  embedder id + dim, LLM ids used per stage, created_at
units             unit_id, chapter, section, ordinal, title,
                  printed_page_label, body (rich text), token_count
assets            asset_id, unit_id, kind (math-svg | figure | table), blob/path
chunks            chunk_id, unit_id, start, end, text
chunk_vec         chunk_id, embedding            -- sqlite-vec virtual table
chunk_fts         FTS5 over chunk text            -- BM25
concepts          concept_id, name, aliases, notation, kind
                  (definition | theorem | method | notation | example),
                  introduced_in_unit
concept_edges     from_concept, to_concept, relation (depends_on | generalizes | contrasts)
concept_uses      concept_id, unit_id, role (introduce | use | example | exercise)
prereqs_external  concepts the book assumes (e.g. high-school algebra); always allowed
summaries         scope (unit | section | chapter), scope_id, one_line, paragraph
qa                qa_id, question, answer, cited_units, required_concepts,
                  earliest_unit_needed, kind (definition | why | how | contrast | misconception)
exercises         exercise_id, unit_id, text, concepts, solution (optional, gated)
misconceptions    unit_id, statement, correction
```

**"Page" becomes "reading unit."** Digital text has no real pages. A unit is a section or roughly a 300–500-word block with a stable ID. Keep the printed page label for citations only.

**Target size:** under 100 MB per book, excluding models.

---

## 2. Part 1 — Digester (desktop)

### 2.1 Stack

- Python orchestrator. Each stage writes to the package DB and is re-runnable on its own, with content hashes for caching.
- **Local LLM on the 3090:** llama.cpp server or vLLM.
  - Default: Qwen3-30B-A3B-Instruct-2507 (MoE, fast, ~18 GB at Q4).
  - Quality fallback: Qwen3-32B dense at Q4.
  - Constrained JSON output (grammar or JSON schema) for every extraction step.
  - These are Dec 2025-era picks; re-check for newer models before starting.
- **Embedder:** must be the same model the reader runs for queries. Pick small, at most ~150M params, so it fits next to the 4B on the phone. Pre-embed all chunks on the desktop.
- **Math:** pre-render all LaTeX to SVG at digest time (MathJax/KaTeX in Node), so readers never need a math engine or WebView.
- **Cloud API:** only behind a per-stage flag, with a cost estimate printed before running (see 2.4).

### 2.2 Pipeline stages

| # | Stage | Engine | Output | Check |
|---|---|---|---|---|
| 1 | **Ingest.** Prefer structured sources (OpenStax publishes CNXML/HTML). PDF via a math-aware converter (e.g. Marker or Docling; evaluate on a hard chapter). | Python | raw text + LaTeX + figures | Diff a chapter by eye; count broken equations |
| 2 | **Segment** into chapters, sections, units with stable IDs | Python | `units`, `assets` | Unit sizes within bounds |
| 3 | **Chunk + embed** | Embedder | `chunks`, `chunk_vec`, `chunk_fts` | Retrieval sanity queries |
| 4 | **Concept extraction**, per unit: defined concepts, used concepts, notation | Local LLM, JSON | candidate concepts per unit | Spot-check 2 chapters by hand |
| 5 | **Concept resolution**, book-wide: merge aliases and notation variants, set `introduced_in_unit` to the first definition | LLM + rules | `concepts`, `concept_uses` | List of uses before definition (forward refs) for review |
| 6 | **Dependency edges:** for each concept, which earlier concepts its definition and proofs rely on | Local LLM | `concept_edges` | No cycles; edges point backward in the book |
| 7 | **External prerequisites:** concepts used but never defined | LLM + review | `prereqs_external` | Short human review |
| 8 | **Summaries:** unit, then section, then chapter (hierarchical), each limited to concepts introduced so far | Local LLM | `summaries` | Leak checker (stage 10) |
| 9 | **Q&A + misconceptions:** anticipated questions per unit, answered only within the allowed set, with citations | Local LLM | `qa`, `misconceptions` | Leak checker + grounding judge |
| 10 | **Verification:** leak checker on all generated text; grounding check that answers are supported by cited units | Checker + LLM judge | pass/fail per item; regenerate failures once, then drop | Leak rate, grounding rate |
| 11 | **Exercises:** link to concepts; gate solutions | LLM | `exercises` | Spot-check |
| 12 | **Package:** vacuum, write manifest, compute size | Python | `.tbook` | Load test in reader |

**Throughput.** A textbook is roughly 300k–600k tokens. With an MoE model on the 3090, a full digest should take hours, not days. Measure this on the pilot chapters before assuming.

### 2.3 The leak checker (shared component)

Input: generated text plus an allowed concept set. Output: violations.

1. **Lexical pass.** Match concept names, aliases and notation patterns from `concepts` against the text. Fast and deterministic.
2. **Semantic pass.** For text that passes, a small LLM tags which concepts it actually relies on (catches paraphrased leaks). It's a closed-set classification over the book's concept list, which small models handle well.
3. **Allowed set** = concepts introduced in seen units ∪ `prereqs_external`.

Write it once as a library: Python for the digester, plus a Kotlin port or a shared core for the reader. Keep the rules in data (the concept table), not in code.

### 2.4 When to use the cloud during digest

Only where a local model measurably fails. Procedure:

1. Build the gold set (Phase 0) for the pilot chapters.
2. Run stages 4–9 locally, and on the cloud for the same chapters only.
3. Escalate a stage to the cloud only if the quality gap is material on the gold set. Likely candidates: book-wide concept resolution (stage 5, long context) and Q&A on hard proofs.
4. Print the estimated token cost before any cloud run.

---

## 3. Part 2 — Reader app (Fedora + Android)

### 3.1 Stack

- **UI:** Compose Multiplatform. One codebase for Android and desktop JVM on Fedora.
- **Inference:** llama.cpp on both platforms, behind the same OpenAI-compatible localhost API.
  - Android: reuse the MindfulHome design, a foreground Service wrapping llama.cpp via JNI and serving `127.0.0.1:8080`.
  - Fedora: bundled `llama-server`, using the Vulkan backend on the Radeon 680M with CPU fallback.
  - Shared Kotlin client code talks to either.
- **Storage:** the book package (read-only SQLite with sqlite-vec + FTS5) and a learner DB (read-write SQLite).
- **Content rendering:** units as rich-text blocks plus pre-rendered SVG math and figures. No WebView.

### 3.2 Model and memory budget

| Component | Phone / laptop (target) | Low-end fallback |
|---|---|---|
| LLM | Qwen3-4B-Instruct-2507 Q4_K_M, ~2.5 GB | Qwen3-1.7B Q4, ~1.1 GB |
| KV cache | 6k ctx, 8-bit KV, ~0.45 GB | 8k ctx fp16, ~0.9 GB |
| Runtime buffers | ~0.3–0.4 GB | ~0.3 GB |
| Embedder (≤150M) | ~0.1 GB | ~0.1 GB |
| **Total** | **~3.4–3.5 GB** | **~2.4 GB** |

This is tight at 4B. Levers, in order: shrink the context to 6k, use BM25-first retrieval with lazy-loading the embedder, or quantize the KV cache further. Measure the real peak RSS on the Fold 4 and the laptop in Phase 4.

The context budget per turn is about 5–6k tokens:

| Part | Tokens |
|---|---|
| System prompt + tool schema | ~0.5k |
| Learner table + flags | ~1k |
| One chapter drill-down | ~1k |
| Retrieved units/chunks | ~2–3k |
| Answer | ~0.5–1k |

### 3.3 Learner state (deterministic)

**Event log.** Append-only; the source of truth.

```
open_unit, close_unit (active_ms with idle detection), scroll_depth,
mark (understood | review), question_asked, answer_given (local | cloud),
exercise_attempt (correct?), self_rating, model_rating
```

**Derived per chapter.** Recomputed incrementally, never by the model:

- Title, 1-line summary (from the package)
- First and last opened, % units read at least once
- **Median** active time per unit (robust to a tab left open)
- Top re-read units (only if re-read count > 1)
- Understood vs review %
- Questions asked; questions where the reader had the model give the answer
- Self vs model understanding score, **and their gap**

**Flags on top of the table** (computed, so the model doesn't have to compare rows):

- Weakest chapters
- Biggest self-vs-model gap (overconfidence)
- Longest time since last visit
- Due for review

Unopened chapters collapse to their titles only.

**Drill-down block per chapter (~1k tokens):** units marked review, concepts answered well vs. confused (from Q&A and exercise outcomes), and the last few questions asked in that chapter.

### 3.4 Answer pipeline (per question)

1. **Scope.** Seen set → allowed concept set.
2. **Tag the question.** Map it to book concepts (lexical match plus an embedding lookup over concept names).
3. **Branch:**
   - **Needs unseen concepts:** say so plainly and point forward ("This is covered in §5.1. Want a preview, or an answer using only what you've read?"). Spoilers are the reader's choice.
   - **Pre-generated `qa` match** above a confidence threshold: serve it, lightly adapted to the reader. No full generation, so it's instant.
4. **Retrieve.** Hybrid BM25 + vector search, **filtered to seen units**, re-ranked toward units the reader struggled with.
5. **Prefetch drill-downs.** The app chooses chapter drill-downs from the concept graph. The model may make **one** extra `get_chapter_detail(ch)` call; after that, it must answer.
6. **Generate.** Static prefix restored from a saved KV state; only the dynamic part is prefilled.
7. **Check.** Run the leak checker. On a violation, regenerate once with explicit feedback. If it fails again, fall back to the forward-pointer answer.
8. **Cite.** Every answer links to the units it used ("same trick as Example 2.7").
9. **Log.** Record events, including the model's understanding rating.

### 3.5 Latency: KV state caching

Prefill, not RAM, is the bottleneck on a phone. Structure every prompt as:

```
[static: system + tool schema + book structure]   ← saved KV state, restored from disk
[semi-static: learner table]                       ← re-prefilled when it changes
[dynamic: drill-down + retrieved text + question]  ← prefilled every turn
```

Use llama.cpp state save/restore (slot save in llama-server, or the state API via JNI).

### 3.6 Cloud escalation (later phase)

- Triggers: the user asks, local confidence is low, or the question spans many chapters.
- Same prompt contract and the same leak checker on the response.
- Billing: a provider key with a hard spend cap, credit sold at or above retail. The payment flow is the only middleman. Treat the key like a gift card, not a secret.
- Bring-your-own-key option for power users.

### 3.7 Sync between devices (later phase)

The append-only event log makes merging simple: union the events, dedupe by event ID, recompute aggregates. Transport can start as file export/import or the user's own sync folder; decide later.

---

## 4. Phases and exit criteria

| Phase | Scope | Exit criterion |
|---|---|---|
| **0. Groundwork** | Pick one OpenStax book (CC BY). Hand-write a gold set of 60–100 questions over 3 chapters: in-scope, forward-reference traps, cross-chapter, out-of-book. | Gold set frozen |
| **1. Digest MVP** | Stages 1–10 on the 3 pilot chapters, local only | Package builds; concept graph spot-checked; leak rate on generated Q&A measured |
| **2. Leak test (kill gate)** | Minimal Fedora reader (even a CLI): seen-set scoping + retrieval + Qwen3-4B + leak checker | Leak rate on the gold set's trap questions low enough to be credible (set the threshold before running). **If it fails, fix the checker or the scoping before building UI.** |
| **3. Learner state** | Event log, aggregates, flags, drill-downs, pre-generated Q&A serving | Table under ~1k tokens for a typical reader; drill-down choice matches the gold set's expected chapters |
| **4. Real clients** | Compose Multiplatform UI on Fedora + Android; JNI service on Android; Vulkan on the 680M; KV state caching | Peak memory ≤ 3.5 GB measured; time-to-first-token acceptable on the Fold 4 (set a target, e.g. under 3 s) |
| **5. Full book + cloud** | Digest the whole book; local vs cloud comparison per stage; capped-key escalation in the reader | Cost per digested book known; escalation works end to end |
| **6. Polish** | Sync, exercises with gated solutions, spaced review from flags | — |

---

## 5. Evaluation (runs continuously)

- **Leak rate:** answers using concepts outside the allowed set (gold traps + random sample).
- **Grounding:** answers supported by cited units (LLM judge, spot-checked by hand).
- **Citation accuracy:** cited units actually contain the claimed material.
- **Routing:** single tool call correct, and never more than one hop. Write 50 cases against your own tools and run them on the quantized build you ship.
- **Pre-generated hit rate:** share of questions served from `qa` without full generation.
- **Latency and memory:** time to first token, tokens/s, and peak RSS on the Fold 4 and the 7735HS laptop.

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| Model leaks unseen concepts despite instructions | Post-generation checker; regenerate once; forward-pointer fallback. Kill gate in Phase 2. |
| PDF math extraction is poor | Start with structured sources (OpenStax); evaluate converters on the hardest chapter first |
| Concept extraction is noisy or inconsistent | Book-wide resolution stage; forward-ref report; human review of the concept list (it's small) |
| Memory doesn't fit at 4B | 6k context, KV quantization, lazy embedder, 1.7B fallback |
| Prefill too slow on phone | KV state caching of the static prefix; pre-generated Q&A for common questions |
| Small-model explanations are weak on hard material | Escalate to cloud; that's what it's for |
| Licensing beyond open textbooks | Out of scope until the open-book version proves the idea |

---

## 7. Open decisions

1. Definition of "seen": opened, or scrolled past a threshold? (Affects spoiler strictness.)
2. Embedder choice under ~150M params, which must also run on device.
3. How the reader rates "understood": explicit marks only, or inferred from exercises too?
4. Whether pre-generated Q&A is shown verbatim or always adapted by the local model.
5. Sync transport.
6. Which OpenStax book to pilot: math-heavy (Calculus Vol. 1) is the harder and more convincing test.
