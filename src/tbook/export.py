"""Phone-ready book file (`tbook export`): the .tbook package with the book's PDF inside
(assets row 'pdf'), so one file carries everything the Android reader needs. Copy it to
the phone over USB and add it from the app's library.

The phone's leak checker matches concept terms with the patterns the desktop builds
(leak.Lexicon), stored in `reader_terms` so both readers decide alike:

  distinctive  term key + regex of a name/alias distinctive by spelling (lexical pass)
  loose        regex of every name/alias (tagging a question)
  notation     canonical notation key
  owner        term key (or "$" + notation key) -> a concept owning it
"""

import shutil
import sqlite3
from pathlib import Path

from . import config, leak
from .books import Book

EXPORT_DIR = config.OUT_DIR / "phone"


def reader_terms(con: sqlite3.Connection) -> list[tuple]:
    lex = leak.Lexicon.from_db(con)
    rows = []
    for c in lex.concepts:
        rows += [(c.concept_id, "distinctive", key, p.pattern) for key, p in c.patterns]
        rows += [(c.concept_id, "loose", None, p.pattern) for p in c.loose]
        rows += [(c.concept_id, "notation", k, None) for k in c.notation_keys]
    rows += [(cid, "owner", key, None) for key, owners in lex.term_owners.items() for cid in sorted(owners)]
    return rows


def export(book: Book, out_dir: Path = EXPORT_DIR) -> Path:
    if not book.package.exists():
        raise SystemExit(f"no package for {book.id}: run `tbook digest {book.id}` first")
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / book.package.name
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.copyfile(book.package, tmp)
    con = sqlite3.connect(tmp)
    con.row_factory = sqlite3.Row
    con.execute("INSERT OR REPLACE INTO assets (asset_id, unit_id, kind, path, blob) VALUES ('pdf', NULL, 'pdf', ?, ?)",
                (book.pdf.name, book.pdf.read_bytes()))
    con.execute("DROP TABLE IF EXISTS reader_terms")
    con.execute("CREATE TABLE reader_terms (concept_id TEXT, kind TEXT, term_key TEXT, pattern TEXT)")
    con.executemany("INSERT INTO reader_terms VALUES (?,?,?,?)", reader_terms(con))
    con.commit()
    con.execute("VACUUM")
    con.close()
    tmp.replace(dest)
    return dest
