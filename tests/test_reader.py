import json

import numpy as np
import pytest

from tbook import db, llm
from tbook.reader import answer as answer_mod


def _vec(seed):
    v = np.random.default_rng(seed).normal(size=768).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def reader(tmp_path, monkeypatch):
    pkg = tmp_path / "t.tbook"
    con = db.connect(pkg)
    db.init_build(con)
    units = [("t:1-01", 1, "1", "1", "Rotations", "Rotation matrices satisfy R^T R = I."),
             ("t:2-01", 2, "2", "2", "Angular velocity", "Angular velocity is omega."),
             ("t:3-01", 3, "3", "3", "Twists", "A twist combines angular and linear velocity.")]
    for uid, o, ch, sec, title, body in units:
        con.execute("INSERT INTO units (unit_id, ordinal, chapter, section, title, kind, body, printed_page_label) "
                    "VALUES (?,?,?,?,?,?,?,?)", (uid, o, ch, sec, title, "content", body, str(o * 10)))
        con.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?)", (uid + "#0", uid, 0, 0, len(body), body))
        con.execute("INSERT INTO chunk_vec VALUES (?,?)", (uid + "#0", _vec(o).tobytes()))
        con.execute("INSERT INTO chunk_fts VALUES (?,?)", (uid + "#0", body))
    for cid, name, unit, seed in (("rotation-matrix", "rotation matrix", "t:1-01", 11),
                                  ("angular-velocity", "angular velocity", "t:2-01", 12),
                                  ("twist", "twist", "t:3-01", 13)):
        con.execute("INSERT INTO concepts VALUES (?,?,?,?,?,?,?,?,?)",
                    (cid, name, "[]", "[]", "definition", unit, name, 1, _vec(seed).tobytes()))
    con.commit()
    con.close()

    class FakeBook:
        id = "t"
        package = pkg
    monkeypatch.setattr(answer_mod, "load_book", lambda _id: FakeBook())
    monkeypatch.setattr(llm, "embed", lambda texts, kind="document", batch=64: np.stack([_vec(99) for _ in texts]))
    monkeypatch.setattr(answer_mod.leak, "semantic_tags", lambda *a, **k: [])
    return answer_mod.Reader("t", model="fake")


def _stream(text):
    def gen(msgs, model, **kw):
        yield text
        yield {"ttft_s": 0.1, "tokens": 5, "tok_s": 50.0}
    return gen


def test_forward_pointer_for_unseen_concept(reader):
    ans = reader.ask("What is a twist?", reader.seen_through("1"))
    assert ans.kind == "forward" and ans.needed_unseen[0]["concept"] == "twist"
    assert ans.needed_unseen[0]["section"] == "3"


def test_generated_answer_cites_only_seen_units(reader, monkeypatch):
    monkeypatch.setattr(llm, "chat_stream", _stream("Angular velocity is omega [t:2-01]."))
    ans = reader.ask("How fast does it spin?", reader.seen_through("2"))
    assert ans.kind == "generated" and [c["unit"] for c in ans.citations] == ["t:2-01"]
    assert set(ans.retrieved) <= {"t:1-01", "t:2-01"}


def test_leaking_answer_falls_back_to_pointer(reader, monkeypatch):
    monkeypatch.setattr(llm, "chat_stream", _stream("Use a twist [t:2-01]."))
    # "twist" is ordinary English, so the semantic pass is what catches it
    monkeypatch.setattr(answer_mod.leak, "semantic_tags",
                        lambda md, cands, **k: [c.concept_id for c in cands if c.concept_id in md.lower()])
    ans = reader.ask("How do I describe the full velocity?", reader.seen_through("2"))
    assert ans.kind == "fallback" and ans.attempts == 2
    assert ans.needed_unseen[0]["concept"] == "twist"
