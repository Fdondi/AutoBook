package com.example.autoreader.pipeline

/** Markdown + LaTeX helpers; mirrors src/tbook/extract/mdutil.py (what the reader needs of it). */
object Markdown {
    // $$...$$ (display) first, then $...$ (inline). Escaped \$ is not a delimiter.
    val MATH = Regex("""(?<!\\)\$\$(.+?)(?<!\\)\$\$|(?<![\\$])\$(?!\$)(.+?)(?<![\\$])\$(?!\$)""", RegexOption.DOT_MATCHES_ALL)
    private val codeFence = Regex("^```.*?^```[ \\t]*$", setOf(RegexOption.DOT_MATCHES_ALL, RegexOption.MULTILINE))

    private fun outsideCode(md: String, fn: (String) -> String): String {
        val out = StringBuilder()
        var last = 0
        for (m in codeFence.findAll(md)) {
            out.append(fn(md.substring(last, m.range.first))).append(m.value)
            last = m.range.last + 1
        }
        return out.append(fn(md.substring(last))).toString()
    }

    /** (tex, isDisplay) for every math span outside code fences. */
    fun mathSpans(md: String): List<Pair<String, Boolean>> {
        val spans = mutableListOf<Pair<String, Boolean>>()
        outsideCode(md) { t ->
            for (m in MATH.findAll(t)) {
                val display = m.groups[1]
                spans += if (display != null) display.value to true else m.groups[2]!!.value to false
            }
            t
        }
        return spans
    }

    /** Rough Unicode-free rendering of LaTeX for word matching (the desktop uses pylatexenc). */
    fun latexToText(tex: String): String = tex.replace(Regex("""\\[a-zA-Z]+\*?"""), " ").replace(Regex("[{}]"), " ")

    /** Plain text for word matching: math rendered to text, markup removed. */
    fun mdToText(md: String): String {
        var text = outsideCode(md) { t ->
            MATH.replace(t) { m -> " " + latexToText(m.groups[1]?.value ?: m.groups[2]!!.value) + " " }
        }
        text = Regex("""!\[[^\]]*]\([^)]*\)""").replace(text, " ")
        text = Regex("""\[([^\]]*)]\([^)]*\)""").replace(text) { it.groupValues[1] }
        text = text.replace("\\_", "_").replace("<", " ").replace(">", " ")
        return Regex("[#*_`|>]+").replace(text, " ")
    }

    /** Lower-case words only, for comparing a quote with the text it came from. */
    fun squash(s: String): String = Regex("[^a-z0-9]+").replace(mdToText(s).lowercase(), " ").trim()

    private fun Regex.sub(t: String, f: (MatchResult) -> String) = replace(t, f)

    /** Canonical LaTeX for comparison: drops spacing, sizing, font aliases and layout. */
    fun canonMath(tex: String): String {
        var t = tex.trim()
        // array with delimiters is a matrix; a bare array is alignment layout
        t = Regex("""\\left\s*[\[(|]\s*\\begin\{array\}\{[^}]*\}(.*?)\\end\{array\}\s*\\right\s*[\])|]""",
            RegexOption.DOT_MATCHES_ALL).sub(t) { "\\begin{matrix}" + it.groupValues[1] + "\\end{matrix}" }
        t = Regex("""\\begin\{array\}\{[^}]*\}|\\end\{array\}""").sub(t) { "" }
        t = Regex("""\\quad\s*\((\d+(?:\.\d+)*[a-z]?)\)\s*$""").sub(t) { "\\tag{" + it.groupValues[1] + "}" }
        t = Regex("""\\(?:left|right|bigl|bigr|Bigl|Bigr|biggl|biggr|big|Big|bigg|Bigg)(?![a-zA-Z])""").sub(t) { "" }
        t = Regex("""\\(?:,|;|:|!|quad|qquad|displaystyle|textstyle|nonumber|notag)(?![a-zA-Z])""").sub(t) { "" }
        t = Regex("""(?<!\\)\\ """).sub(t) { "" }
        t = t.replace("~", "")
        t = Regex("""\\(?:ldots|cdots|dots|dotsc|dotsb|cdotp)(?![a-zA-Z])""").sub(t) { "\\cdots" }
        t = Regex("""\\(?:mathrm|text|textrm|operatorname|mathit|textit)\s*\{([^{}]*)\}""").sub(t) { it.groupValues[1] }
        t = Regex("""\\(?:boldsymbol|bm|mathbf|textbf)(?![a-zA-Z])""").sub(t) { "\\mathbf" }
        t = Regex("""\\[dt]frac(?![a-zA-Z])""").sub(t) { "\\frac" }
        t = Regex("""\\var(phi|theta|epsilon|rho|sigma|pi)(?![a-zA-Z])""").sub(t) { "\\" + it.groupValues[1] }
        t = Regex("""\\ensuremath(?![a-zA-Z])""").sub(t) { "" }
        t = Regex("""\\vec(?![a-zA-Z])""").sub(t) { "\\overrightarrow" }
        t = Regex("""\\(?:ge|geq|geqslant)(?![a-zA-Z])""").sub(t) { "\\geq" }
        t = Regex("""\\(?:le|leq|leqslant)(?![a-zA-Z])""").sub(t) { "\\leq" }
        t = Regex("""\\(?:ne|neq)(?![a-zA-Z])""").sub(t) { "\\neq" }
        t = Regex("""\\(?:to|rightarrow)(?![a-zA-Z])""").sub(t) { "\\to" }
        t = Regex("""\\(?:lVert|Vert)(?![a-zA-Z])|\\rVert(?![a-zA-Z])""").sub(t) { "\\|" }
        t = Regex("""\\(?:lvert|rvert)(?![a-zA-Z])""").sub(t) { "|" }
        t = Regex("""\\begin\{(?:aligned|align\*?|split|gathered|eqnarray\*?)\}|\\end\{(?:aligned|align\*?|split|gathered|eqnarray\*?)\}""")
            .sub(t) { "" }
        t = Regex("""\\(begin|end)\{[bpvBV]?matrix\}""").sub(t) { "\\" + it.groupValues[1] + "{matrix}" }
        t = Regex("""[\[(]\s*\\begin\{matrix\}""").sub(t) { "\\begin{matrix}" }
        t = Regex("""\\end\{matrix\}\s*[\])]""").sub(t) { "\\end{matrix}" }
        // Line breaks matter inside matrices only; elsewhere they are layout.
        val mats = mutableListOf<String>()
        t = Regex("""\\begin\{matrix\}.*?\\end\{matrix\}""", RegexOption.DOT_MATCHES_ALL).sub(t) {
            mats += it.value
            "\u0000${mats.size - 1}\u0000"
        }
        t = t.replace("\\\\", "").replace("&", "")
        t = Regex("\u0000(\\d+)\u0000").sub(t) { mats[it.groupValues[1].toInt()] }
        t = Regex("""\s+""").sub(t) { "" }
        val braces = Regex("""\{(\\[a-zA-Z]+|[^{}\\])\}""")
        while (true) {
            val next = braces.sub(t) { it.groupValues[1] }
            if (next == t) break
            t = next
        }
        return t.trimEnd('.', ',', ';')
    }
}
