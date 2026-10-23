"""SQLite schema for the build DB and the .tbook package (plan §1).

The build DB holds the package tables plus build-only tables (prefixed in
BUILD_ONLY); `package` copies it and drops those.
"""

import json
import sqlite3
import time
from pathlib import Path

PACKAGE_DDL = """
CREATE TABLE IF NOT EXISTS manifest (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS units (
    unit_id TEXT PRIMARY KEY,
    ordinal INTEGER UNIQUE,
    chapter TEXT, section TEXT, title TEXT,
    section_path TEXT,            -- JSON list of ancestor titles
    kind TEXT,                    -- content | exercise | reference | front
    page_start INTEGER, page_end INTEGER,
    printed_page_label TEXT,
    body TEXT,                    -- Markdown + LaTeX
    word_count INTEGER, token_count INTEGER
);

CREATE TABLE IF NOT EXISTS assets (
    asset_id TEXT PRIMARY KEY, unit_id TEXT, kind TEXT, path TEXT, blob BLOB
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY, unit_id TEXT, ordinal INTEGER,
    start INTEGER, "end" INTEGER, text TEXT
);
CREATE INDEX IF NOT EXISTS chunks_unit ON chunks(unit_id);
CREATE TABLE IF NOT EXISTS chunk_vec (chunk_id TEXT PRIMARY KEY, embedding BLOB);
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(chunk_id UNINDEXED, text);

CREATE TABLE IF NOT EXISTS concepts (
    concept_id TEXT PRIMARY KEY,
    name TEXT, aliases TEXT, notation TEXT,   -- aliases/notation: JSON lists
    kind TEXT, introduced_in_unit TEXT,
    definition TEXT,                          -- one-sentence gloss from the defining unit
    lexical_ok INTEGER DEFAULT 1,
    embedding BLOB
);
CREATE TABLE IF NOT EXISTS concept_edges (
    from_concept TEXT, to_concept TEXT, relation TEXT,
    PRIMARY KEY (from_concept, to_concept, relation)
);
CREATE TABLE IF NOT EXISTS concept_uses (
    concept_id TEXT, unit_id TEXT, role TEXT,
    PRIMARY KEY (concept_id, unit_id, role)
);
CREATE TABLE IF NOT EXISTS prereqs_external (concept_id TEXT PRIMARY KEY, reason TEXT);

CREATE TABLE IF NOT EXISTS summaries (
    scope TEXT, scope_id TEXT, one_line TEXT, paragraph TEXT,
    PRIMARY KEY (scope, scope_id)
);
CREATE TABLE IF NOT EXISTS qa (
    qa_id TEXT PRIMARY KEY, unit_id TEXT, question TEXT, answer TEXT,
    cited_units TEXT, required_concepts TEXT, earliest_unit_needed TEXT,
    kind TEXT, embedding BLOB
);
CREATE TABLE IF NOT EXISTS exercises (
    exercise_id TEXT PRIMARY KEY, unit_id TEXT, label TEXT, text TEXT,
    concepts TEXT, earliest_unit_needed TEXT, solution TEXT
);
CREATE TABLE IF NOT EXISTS misconceptions (
    unit_id TEXT, statement TEXT, correction TEXT
);
"""

BUILD_DDL = """
CREATE TABLE IF NOT EXISTS b_pages (
    page_idx INTEGER PRIMARY KEY,
    label TEXT, width REAL, height REAL, png TEXT,
    text_layer TEXT,          -- body text, running heads removed
    oracle_words TEXT,        -- JSON list: prose/code words expected in any extraction
    math_chars INTEGER, code_chars INTEGER,
    running_heads TEXT        -- JSON list of removed header/footer lines
);
CREATE TABLE IF NOT EXISTS b_toc (
    idx INTEGER PRIMARY KEY, level INTEGER, title TEXT,
    page_idx INTEGER, y REAL, number TEXT, kind TEXT
);
CREATE TABLE IF NOT EXISTS b_page_md (
    page_idx INTEGER, method TEXT, md TEXT, meta TEXT, created_at REAL,
    PRIMARY KEY (page_idx, method)
);
CREATE TABLE IF NOT EXISTS b_page_checks (
    page_idx INTEGER, method TEXT,
    coverage REAL, katex_total INTEGER, katex_bad INTEGER, katex_errors TEXT,
    math_ok INTEGER, hygiene TEXT, judge TEXT, passed INTEGER, checked_at REAL,
    PRIMARY KEY (page_idx, method)
);
CREATE TABLE IF NOT EXISTS b_page_final (
    page_idx INTEGER PRIMARY KEY, md TEXT, method TEXT,
    status TEXT,              -- auto_pass | repaired | accepted | flagged
    attempts INTEGER, note TEXT
);
CREATE TABLE IF NOT EXISTS b_concept_candidates (
    unit_id TEXT, name TEXT, aliases TEXT, notation TEXT, kind TEXT, role TEXT, evidence TEXT
);
CREATE INDEX IF NOT EXISTS b_cc_unit ON b_concept_candidates(unit_id);
CREATE TABLE IF NOT EXISTS b_stage_runs (
    stage TEXT PRIMARY KEY, input_hash TEXT, finished_at REAL, stats TEXT
);
CREATE TABLE IF NOT EXISTS b_verify (
    item_kind TEXT, item_id TEXT, attempt INTEGER, leak TEXT, grounded INTEGER,
    verdict TEXT, PRIMARY KEY (item_kind, item_id, attempt)
);
"""


def connect(path: Path, *, readonly: bool = False) -> sqlite3.Connection:
    path = Path(path)
    if readonly:
        con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, check_same_thread=False)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(path, check_same_thread=False, timeout=60)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
    con.row_factory = sqlite3.Row
    return con


def init_build(con: sqlite3.Connection) -> None:
    con.executescript(PACKAGE_DDL)
    con.executescript(BUILD_DDL)
    con.commit()


def stage_done(con: sqlite3.Connection, stage: str, input_hash: str) -> bool:
    row = con.execute("SELECT input_hash FROM b_stage_runs WHERE stage=?", (stage,)).fetchone()
    return row is not None and row["input_hash"] == input_hash


def mark_stage(con: sqlite3.Connection, stage: str, input_hash: str, stats: dict) -> None:
    con.execute(
        "INSERT OR REPLACE INTO b_stage_runs VALUES (?,?,?,?)",
        (stage, input_hash, time.time(), json.dumps(stats)),
    )
    con.commit()
