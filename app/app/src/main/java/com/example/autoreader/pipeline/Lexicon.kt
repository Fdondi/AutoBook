package com.example.autoreader.pipeline

import org.json.JSONArray

class Concept(
    val id: String,
    val name: String,
    val aliases: List<String>,
    val introducedOrdinal: Int?,
    val introducedUnit: String?,
    val lexicalOk: Boolean,
) {
    val patterns = mutableListOf<Pair<String, Regex>>()     // (term key, regex), distinctive terms
    val loose = mutableListOf<Regex>()                      // every term, for tagging questions
    val notationKeys = mutableListOf<String>()
}

/**
 * All concepts of one book with their matchers; mirrors leak.Lexicon. The term patterns
 * come precomputed from the desktop (`reader_terms`, written by `tbook export`).
 */
class Lexicon(
    val concepts: List<Concept>,
    val externals: Set<String>,
    val embeddings: Array<FloatArray>?,
    private val uses: Map<String, Set<String>>,              // unit id -> concept ids the unit names
    private val termOwners: Map<String, Set<String>>,
) {
    val byId = concepts.associateBy { it.id }
    val index = concepts.withIndex().associate { (i, c) -> c.id to i }

    fun allowed(seenOrdinals: Set<Int>): Set<String> =
        externals + concepts.filter { it.introducedOrdinal != null && it.introducedOrdinal in seenOrdinals }.map { it.id }

    /** Concepts the given units name (any role): they may be named, not explained. */
    fun namedIn(units: Collection<String>): Set<String> = units.flatMapTo(mutableSetOf()) { uses[it].orEmpty() }

    /** {concept id: matched text} for distinctive names/aliases and notation found in Markdown. */
    fun lexical(md: String, onlyLexicalOk: Boolean = true): Map<String, String> {
        val text = Markdown.mdToText(Markdown.MATH.replace(md, " "))
        val maths = Markdown.mathSpans(md).map { Markdown.canonMath(it.first) }
        val hits = linkedMapOf<String, String>()
        for (c in concepts) {
            if (c.lexicalOk || !onlyLexicalOk) {
                for ((_, p) in c.patterns) {
                    val m = p.find(text)
                    if (m != null) {
                        hits[c.id] = m.value
                        break
                    }
                }
            }
            if (c.id !in hits) {
                c.notationKeys.firstOrNull { k -> maths.any { k in it } }?.let { hits[c.id] = "$$it$" }
            }
        }
        return hits
    }

    /** Concepts a question names, by any of their terms (common words count here). */
    fun namedConcepts(question: String): Set<String> {
        val text = Markdown.mdToText(question)
        val maths = Markdown.mathSpans(question).map { Markdown.canonMath(it.first) }
        return concepts.filter { c ->
            c.loose.any { it.containsMatchIn(text) } || c.notationKeys.any { k -> maths.any { k in it } }
        }.mapTo(mutableSetOf()) { it.id }
    }

    /** A matched term or notation is fine if any allowed concept owns it. */
    fun termAllowed(matched: String, allowed: Set<String>): Boolean {
        val key = if (matched.startsWith("$")) "$" + matched.trim('$') else Terms.normalize(matched)
        return termOwners[key].orEmpty().any { it in allowed }
    }

    companion object {
        fun load(db: Db): Lexicon {
            val ordinals = db.query("SELECT unit_id, ordinal FROM units") { it.string(0)!! to it.int(1)!! }.toMap()
            var embeddings: MutableList<FloatArray>? = mutableListOf()
            val concepts = db.query(
                "SELECT concept_id, name, aliases, introduced_in_unit, lexical_ok, embedding FROM concepts ORDER BY concept_id",
            ) { r ->
                val emb = r.blob(5)
                if (emb == null) embeddings = null else embeddings?.add(emb.toFloats())
                val aliases = JSONArray(r.string(2) ?: "[]").let { a -> (0 until a.length()).map { a.getString(it) } }
                Concept(r.string(0)!!, r.string(1)!!, aliases, ordinals[r.string(3)], r.string(3), (r.int(4) ?: 0) != 0)
            }
            val byId = concepts.associateBy { it.id }
            val owners = mutableMapOf<String, MutableSet<String>>()
            db.query("SELECT concept_id, kind, term_key, pattern FROM reader_terms") { r ->
                val c = byId[r.string(0)]
                when (r.string(1)) {
                    "distinctive" -> c?.patterns?.add(r.string(2)!! to Regex(r.string(3)!!, RegexOption.IGNORE_CASE))
                    "loose" -> c?.loose?.add(Regex(r.string(3)!!, RegexOption.IGNORE_CASE))
                    "notation" -> c?.notationKeys?.add(r.string(2)!!)
                    "owner" -> owners.getOrPut(r.string(2)!!) { mutableSetOf() }.add(r.string(0)!!)
                }
            }
            val externals = db.query("SELECT concept_id FROM prereqs_external") { it.string(0)!! }.toSet()
            val uses = mutableMapOf<String, MutableSet<String>>()
            db.query("SELECT concept_id, unit_id FROM concept_uses") { r ->
                uses.getOrPut(r.string(1)!!) { mutableSetOf() }.add(r.string(0)!!)
            }
            return Lexicon(concepts, externals, embeddings?.toTypedArray(), uses, owners)
        }
    }
}
