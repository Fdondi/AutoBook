package com.example.autoreader

import org.junit.Assert.assertEquals
import org.junit.Test

/** Same cases as tests/test_reader_web.py: the desktop and phone readers must agree. */
class TextCleanupTest {
    @Test
    fun repairsLatexDecodedAsControlCharacters() {
        assertEquals("\$R_{ca}^{\\text{T}}\$ and \$\\frac{1}{2}\$",
            TextCleanup.text("\$R_{ca}^{\text{T}}\$ and \$\u000crac{1}{2}\$", "mr"))
        assertEquals("\$a \\neq b\$ but a\nnew paragraph", TextCleanup.text("\$a \neq b\$ but a\nnew paragraph", "mr"))
    }

    @Test
    fun turnsEveryCitationFormIntoALink() {
        assertEquals("as in [mr:1-01].", TextCleanup.text("as in \$\\text{mr:1-01}\$.", "mr"))
        assertEquals("see [mr:3.2.1-04] and [mr:3.3-02], [mr:2-01]",
            TextCleanup.text("see `mr:3.2.1-04` and mr:3.3-02, [mr:2-01]", "mr"))
    }

    @Test
    fun dropsReferencesToTheSourcePassage() {
        assertEquals("What is a robot mechanism?",
            TextCleanup.question("What is a robot mechanism according to the passage?", "mr"))
    }
}
