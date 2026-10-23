package com.example.autoreader

/** Display fixes for digest text; mirrors clean_text/clean_question in src/tbook/reader/web.py. */
object TextCleanup {
    private val citeMath = Regex("""\$\\text\{([a-z]+:[^}\s]+)\}\$""")          // "$\text{mr:1-01}$"
    // Digest JSON decoding turned some LaTeX commands into control characters: "\text" -> TAB + "ext".
    private val control = Regex("[\t\u0008\u000c\r](?=[A-Za-z])")
    private val controlCmd = mapOf('\t' to "\\t", '\u0008' to "\\b", '\u000c' to "\\f", '\r' to "\\r")
    private val inlineMath = Regex("""(?<![\\$])\$(?!\$)([^$]+?)\$""")
    private val newlineCmd = Regex("\n(?=[a-z])")
    private val passage = Regex(
        """,?\s+(?:according to|as described in|as stated in|as explained in|described in|in|from) the (?:passage|text)\b""",
        RegexOption.IGNORE_CASE,
    )
    private val thePassage = Regex("""\bthe passage\b""")

    /** Repair LaTeX broken into control characters and make every citation a [unit] link. */
    fun text(s: String, bookId: String): String {
        var t = control.replace(s) { controlCmd.getValue(it.value[0]) }
        t = inlineMath.replace(t) { m -> "$" + newlineCmd.replace(m.groupValues[1]) { "\\n" } + "$" }
        t = citeMath.replace(t) { "[${it.groupValues[1]}]" }
        t = Regex("`($bookId:[^`\\s]+)`").replace(t) { "[${it.groupValues[1]}]" }
        return Regex("(?<![\\[\\w:])$bookId:[\\w.+/-]*\\w").replace(t) { "[${it.value}]" }
    }

    /** Generated questions refer to "the passage" they were written from; the reader sees a book. */
    fun question(q: String, bookId: String): String =
        thePassage.replace(passage.replace(text(q, bookId), ""), "the book")
}
