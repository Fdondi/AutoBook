package com.example.autoreader.pipeline

import java.text.Normalizer

/** Term normalization; mirrors normalize/singular in src/tbook/terms.py. */
object Terms {
    private val uninflected = setOf("atlas", "alias", "bias", "gas", "canvas", "lens", "chaos", "series", "species", "means")

    fun singular(w: String): String = when {
        w in uninflected -> w
        w.endsWith("ices") && w.length > 5 -> w.dropLast(4) + "ix"
        w == "axes" || w.endsWith(" axes") -> w.dropLast(2) + "is"
        w.endsWith("ies") && w.length > 4 -> w.dropLast(3) + "y"
        Regex("(ses|xes|zes|ches|shes)$").containsMatchIn(w) && !(w.endsWith("bases") || w.endsWith("cases") || w.endsWith("phases")) ->
            w.dropLast(2)
        w.endsWith("s") && !(w.endsWith("ss") || w.endsWith("us") || w.endsWith("is") || w.endsWith("ics")) && w.length > 3 ->
            w.dropLast(1)
        else -> w
    }

    fun normalize(name: String): String {
        var t = Normalizer.normalize(name, Normalizer.Form.NFKD).filter { it.code < 128 }.lowercase()
        t = Regex("""^(the|a|an)\s+""").replace(t.trim(), "")
        t = Regex("[‐-―\\-_/]+").replace(t, " ")
        t = Regex("""[^a-z0-9()\[\]\s']+""").replace(t, "")
        t = Regex("""\s+""").replace(t, " ").trim()
        val parts = t.split(" ").toMutableList()
        if (parts.isNotEmpty() && parts.last().isNotEmpty()) parts[parts.size - 1] = singular(parts.last())
        return parts.joinToString(" ")
    }
}
