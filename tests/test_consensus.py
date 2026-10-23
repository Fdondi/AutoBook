from tbook.extract import consensus, glyphs, mathlayer, mdutil


def test_dehyphenate_uses_vocabulary_and_keeps_compounds():
    vocab = {"spatial", "rotation", "end", "effector"}
    md = "known as spa-\ntial velocities, a rota-tion and the end-\neffector"
    out = mdutil.dehyphenate(md, vocab, {"end-effector"})
    assert out == "known as spatial velocities, a rotation and the end-effector"


def test_unwrap_text_math():
    assert mdutil.unwrap_text_math(r"to ask, $\text{for example}$, why") == "to ask, for example, why"
    assert mdutil.unwrap_text_math(r"examine $\textit{ourselves}$") == "examine *ourselves*"
    assert mdutil.unwrap_text_math(r"$\textbf{not}$ the") == "**not** the"
    assert mdutil.unwrap_text_math(r"$\text{Ad}$") == r"$\text{Ad}$"           # notation stays


def test_prose_in_math_ignores_displays():
    inline = r"the $\text{circle of radius } d_{BC} \text{ centered at } B \text{ these two solutions}$"
    display = r"$$\text{dof} = \text{(sum of freedoms of the points)} - \text{(number of constraints)}$$"
    assert mdutil.prose_in_math(inline) and not mdutil.prose_in_math(display)


def test_canon_math_ignores_layout_and_fonts_aliases():
    a = r"\left[\begin{array}{cc} 1 & 0 \\ 0 & 1 \end{array}\right] \quad (3.5)"
    b = r"\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \tag{3.5}"
    assert mdutil.canon_math(a) == mdutil.canon_math(b)
    assert mdutil.canon_math(r"R^{\mathrm{T}}R") == mdutil.canon_math(r"R^T R")
    assert mdutil.canon_math(r"\hat{\omega}\dot{\theta}") != mdutil.canon_math(r"\dot{\hat{\omega}}\theta")


def test_latex_tokens_scripts_fonts_and_primes():
    toks = glyphs.latex_tokens(r"\mathbf{R}'_x e^{[\mathcal{S}_1]\theta_1}")
    assert [(t.sym, t.script, t.font) for t in toks][:3] == [("R", False, "bf"), ("'", True, None), ("x", True, None)]
    assert ("S", True, "cal") in [(t.sym, t.script, t.font) for t in toks]


def test_glyph_alignment_prefers_the_printed_order():
    printed = glyphs.latex_tokens(r"\mathbf{R}'_x \mathbf{R}'_y \mathbf{R}'_z")
    right = glyphs.align_score(glyphs.latex_tokens(r"\mathbf{R}'_x\mathbf{R}'_y\mathbf{R}'_z"), printed)
    wrong = glyphs.align_score(glyphs.latex_tokens(r"\mathbf{R}'_z\mathbf{R}'_y\mathbf{R}'_x"), printed)
    assert right > wrong


def test_disputes_votes_and_apply():
    primary = r"Let $\dot{\hat{\omega}}\theta$ be and $x$ here."
    others = {"a": r"Let $\hat{\omega}\dot{\theta}$ be and $x$ here.", "b": r"Let $\hat{\omega}\dot{\theta}$ be."}
    spans, items = consensus.disputes(primary, others)
    assert spans[0]["status"] == "outvoted"
    fixed = consensus.apply(primary, spans, {0: r"\hat{\omega}\dot{\theta}"})
    assert fixed.startswith(r"Let $\hat{\omega}\dot{\theta}$ be")


def test_join_prose_drops_line_end_hyphens():
    out = mathlayer._join_prose([("known as spa-", True), ("tial velocities and more words", True), ("R^{T}R = I", False)])
    assert out.split("\n")[0] == "known as spatial velocities and more words"
