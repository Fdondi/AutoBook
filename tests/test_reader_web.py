from tbook.reader.web import clean_question, clean_text


def test_clean_text_repairs_latex_decoded_as_control_characters():
    assert clean_text("$R_{ca}^{\text{T}}$ and $\x0crac{1}{2}$", "mr") == "$R_{ca}^{\\text{T}}$ and $\\frac{1}{2}$"
    assert clean_text("$a \neq b$ but a\nnew paragraph", "mr") == "$a \\neq b$ but a\nnew paragraph"


def test_clean_text_turns_every_citation_form_into_a_link():
    assert clean_text("as in $\\text{mr:1-01}$.", "mr") == "as in [mr:1-01]."
    assert clean_text("see `mr:3.2.1-04` and mr:3.3-02, [mr:2-01]", "mr") == \
        "see [mr:3.2.1-04] and [mr:3.3-02], [mr:2-01]"


def test_clean_question_drops_references_to_the_source_passage():
    assert clean_question("What is a robot mechanism according to the passage?", "mr") == "What is a robot mechanism?"
