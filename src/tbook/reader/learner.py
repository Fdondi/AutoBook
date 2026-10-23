"""Learner DB: an append-only event log (plan §3.3). The seen set is derived from it."""

import json
import sqlite3
import time
import uuid

from .. import config

DDL = """
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY, ts REAL, kind TEXT, unit_id TEXT, data TEXT
);
CREATE INDEX IF NOT EXISTS events_kind ON events(kind);
"""


class Learner:
    def __init__(self, book_id: str):
        config.LEARNER_DIR.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(config.LEARNER_DIR / f"{book_id}.sqlite")
        self.con.executescript(DDL)

    def log(self, kind: str, unit_id: str | None = None, **data) -> None:
        self.con.execute("INSERT INTO events VALUES (?,?,?,?,?)",
                         (uuid.uuid4().hex, time.time(), kind, unit_id, json.dumps(data)))
        self.con.commit()

    def seen_units(self) -> set[str]:
        return {r[0] for r in self.con.execute("SELECT DISTINCT unit_id FROM events WHERE kind='open_unit'")}

    def events(self, kind: str) -> list[dict]:
        """Data of every `kind` event, oldest first."""
        return [json.loads(r[0]) for r in self.con.execute("SELECT data FROM events WHERE kind=? ORDER BY ts", (kind,))]

    def reset(self) -> None:
        self.con.execute("DELETE FROM events")
        self.con.commit()
