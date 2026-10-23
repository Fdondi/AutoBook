import random

import numpy as np

from tbook import db, leak, retrieval


def test_plural_and_term_regex():
    assert leak.plural("matrix") == "matrices" and leak.plural("axis") == "axes"
    import re
    rx = re.compile(leak.term_regex("screw axis"), re.I)
    assert rx.search("two screw axes meet") and rx.search("a screw-axis") and not rx.search("screwdriver axis")


def test_lexicon_respects_lexical_ok_and_notation():
    cs = [leak.Concept("twist", "twist", ["spatial velocity"], [r"\mathcal{V}_s"], 10, "u10"),
          leak.Concept("screw", "screw", [], [], 12, "u12"),
          leak.Concept("jac", "Jacobian", [], [], 20, "u20"),
          leak.Concept("so3", "special orthogonal group", [], ["SO(3)", "R"], 3, "u3")]
    lex = leak.Lexicon(cs, set())
    hits = lex.lexical(r"The spatial velocity $\mathcal{V}_s$ of a screw, a Jacobian, with $R \in SO(3)$.")
    # "screw" is ordinary English: semantic pass only; "Jacobian" is rare enough to match; "R" never matched
    assert set(hits) == {"twist", "so3", "jac"}
    assert lex.allowed({3, 4}) == {"so3"}


def test_terms_owned_by_an_allowed_duplicate_are_fine():
    cs = [leak.Concept("dof", "degrees of freedom", [], [], 2, "u2"),
          leak.Concept("dof-robot", "degrees of freedom of a robot", ["degrees of freedom"], [], 9, "u9")]
    lex = leak.Lexicon(cs, set())
    rep = leak.check(lex, "It has three degrees of freedom.", {"dof"}, model="x", semantic=False)
    assert rep.ok()
    rep = leak.check(lex, "It has three degrees of freedom.", set(), model="x", semantic=False)
    assert not rep.ok()


def test_sources_may_be_used_and_tags_need_real_quotes(monkeypatch):
    cs = [leak.Concept("jac", "Jacobian", [], [], 20, "u20"),
          leak.Concept("twist", "twist", [], [], 30, "u30")]
    lex = leak.Lexicon(cs, set())
    # the source passage itself names the Jacobian: a summary of it may too
    rep = leak.check(lex, "The Jacobian maps joint rates.", set(), model="x", semantic=False,
                     sources="Here we define the Jacobian.")
    assert rep.ok()
    # semantic tags without a verbatim quote from the checked text are discarded
    monkeypatch.setattr(leak.llm, "chat", lambda *a, **k: {"relies_on": [
        {"id": "twist", "quote": "a six-vector of velocities", "why": "x"},
        {"id": "twist", "quote": "words that are not there", "why": "x"}]})
    monkeypatch.setattr(leak.llm, "load_prompt", lambda name: "")
    got = leak.semantic_tags("It is a six-vector of velocities.", [cs[1]], model="x")
    assert got == ["twist"]
    assert leak.semantic_tags("Nothing relevant here.", [cs[1]], model="x") == []


def _db(tmp_path, n_units=40, chunks_per_unit=3):
    con = db.connect(tmp_path / "t.sqlite")
    db.init_build(con)
    rng = np.random.default_rng(0)
    for u in range(n_units):
        uid = f"b:{u}"
        con.execute("INSERT INTO units (unit_id, ordinal, body, kind) VALUES (?,?,?,?)", (uid, u, "x", "content"))
        for k in range(chunks_per_unit):
            cid = f"{uid}#{k}"
            text = f"rotation matrix twist screw unit {u} chunk {k}"
            v = rng.normal(size=768).astype(np.float32)
            v /= np.linalg.norm(v)
            con.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?)", (cid, uid, k, 0, 1, text))
            con.execute("INSERT INTO chunk_vec VALUES (?,?)", (cid, v.tobytes()))
            con.execute("INSERT INTO chunk_fts VALUES (?,?)", (cid, text))
    con.commit()
    return con


def test_retrieval_never_returns_unseen_units(tmp_path):
    con = _db(tmp_path)
    index = retrieval.Index(con)
    rnd = random.Random(1)
    for _ in range(50):
        seen = {f"b:{u}" for u in range(40) if rnd.random() < 0.3}
        q = np.random.default_rng(rnd.randrange(10**6)).normal(size=768).astype(np.float32)
        hits = index.search("rotation matrix twist", seen, qvec=q / np.linalg.norm(q), k=8, token_budget=10**6)
        assert all(h.unit_id in seen for h in hits)
        if seen:
            assert hits
    assert index.search("rotation", set(), qvec=np.ones(768, np.float32)) == []
