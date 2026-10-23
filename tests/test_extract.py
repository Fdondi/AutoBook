from tbook.digest import segment
from tbook.extract import checks, mdutil, pages


def test_fold_composes_tex_accents_and_apostrophes():
    assert pages.words("Gr\u00a8ubler\u2019s formula") == ["grubler's", "formula"]
    assert pages.words("Grübler's") == ["grubler's"]


def test_words_skip_operator_names():
    assert pages.words("det R and tr R") == ["and"]


def _line(text, kinds=None, block=0):
    return {"text": text, "ftext": text, "kinds": kinds or "t" * len(text), "bbox": (0, 0, 0, 0), "block": block}


def test_block_words_dehyphenates_using_vocabulary():
    vocab = pages._vocab([[_line("the motion of a rigid-body is"), _line("motion")]])
    blines = [_line("the mo-"), _line("tion of a rigid-"), _line("body")]
    flat = [w for ws in pages._block_words(blines, vocab) for w in ws]
    assert flat == ["the", "motion", "of", "rigid", "body"]


def test_block_words_skip_words_glued_to_math():
    blines = [_line("RTR = I holds", kinds="mtm" + "t" * 10)]
    assert pages._block_words(blines, pages._vocab([blines]))[0] == ["holds"]


def test_running_heads_keep_equation_numbers_and_content():
    head_pages = {"(#.#)": set(range(30)), "#": set(range(300)), "where": {1, 2, 3},
                  "#.#. wrenches": {5, 6}, "may # preprint of modern robotics": set(range(500))}
    heads = pages._running_heads(head_pages, ["Wrenches"])
    assert heads == {"#", "#.#. wrenches", "may # preprint of modern robotics"}


def test_coverage_counts_dropped_words_and_tolerates_fixed_typos():
    oracle = ["the", "matrix", "satisifies", "this", "property"]
    recall, order, missing = checks.coverage(oracle, "The matrix satisfies this.")
    assert missing == ["property"] and abs(recall - 0.8) < 1e-9


def test_coverage_retokenizes_glued_and_split_text_layer_words():
    # FoB sets justified spaces as glyph offsets: the text layer glues words
    oracle = ["andwhichtries", "toembedthreecharacteristics", "of", "living", "beings", "alinearsystem"]
    md = "and which tries to embed three characteristics of living beings, a linear system"
    assert checks.coverage(oracle, md)[2] == []
    # a URL broken across lines splits one word in two
    assert checks.coverage(["see", "amorphicr", "obotworks", "com"], "see http://amorphicrobotworks.com")[2] == []
    # a glued token whose words are absent stays missing
    assert checks.coverage(["thesecondpart", "of"], "the first part of")[2] == ["thesecondpart"]


def test_math_spans_and_katex():
    md = "Inline $R^T R = I$ and\n\n$$\\begin{bmatrix} 1 & 0 \\\\ 0 & 1 \\end{bmatrix} \\tag{3.1}$$\n\n$\\frac{a}{$"
    spans = mdutil.math_spans(md)
    assert [d for _, d in spans] == [False, True, False]
    errs = mdutil.katex_errors(spans)
    assert errs[0] is None and errs[1] is None and errs[2]


def test_loops_flag_degenerate_repetition_but_not_dot_leaders():
    toc = "2.1 Rigid Bodies " + ". " * 45 + "12"
    looped = "Figure 12.33 " + "\text{---} " * 300
    assert mdutil.loops(toc) == [] and mdutil.loops(looped)
    assert len(mdutil.collapse_loops(looped)) < 40


def test_math_spans_ignore_code_fences():
    assert mdutil.math_spans("```bash\necho $HOME $PATH\n```\n") == []


def test_units_never_split_display_math_or_continuations():
    para = " ".join(["word"] * 200)
    text = f"{para}\n\n{para}\n\n$$x = y$$\n\nwhere x is {para}\n\nNew paragraph {para}"
    units = segment._units_from_text(text)
    assert all(not u.lstrip().startswith(("$$", "where")) for u in units)
    assert "".join(units).count("$$x = y$$") == 1


def test_find_heading_prefers_markdown_headings():
    md = "Text mentioning Rotation Matrices inline.\n\n### 3.2.1 Rotation Matrices\n\nBody."
    off = segment._find_heading(md, "Rotation Matrices")
    assert md[off:].startswith("### 3.2.1")
