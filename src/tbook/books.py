"""Book configs (books/<id>.toml) and per-book paths."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from . import config, db


@dataclass
class Book:
    id: str
    cfg: dict

    @property
    def pdf(self) -> Path:
        return config.ROOT / self.cfg["pdf"]

    @property
    def build_dir(self) -> Path:
        return config.BUILD_DIR / self.id

    @property
    def build_db(self) -> Path:
        return self.build_dir / "build.sqlite"

    @property
    def pages_dir(self) -> Path:
        return self.build_dir / "pages"

    @property
    def reports_dir(self) -> Path:
        d = self.build_dir / "reports"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def package(self) -> Path:
        return config.OUT_DIR / f"{self.id}.tbook"

    def connect(self):
        con = db.connect(self.build_db)
        db.init_build(con)
        return con


def load(book_id: str) -> Book:
    path = config.BOOKS_DIR / f"{book_id}.toml"
    if not path.exists():
        known = ", ".join(p.stem for p in config.BOOKS_DIR.glob("*.toml"))
        raise SystemExit(f"unknown book '{book_id}' (known: {known})")
    return Book(book_id, tomllib.loads(path.read_text(encoding="utf-8")))
