package com.example.autoreader.pipeline

import org.json.JSONArray
import org.json.JSONObject

class LeakReport(val lexical: Map<String, String>, val semantic: List<String>) {
    val violations: List<String> get() = (lexical.keys + semantic).toSortedSet().toList()
    fun ok() = violations.isEmpty()
}

/**
 * The concept-leak checker; mirrors check/semantic_tags in src/tbook/leak.py (same prompt).
 * Generated text may use only allowed concepts; `mentioned` ones may be named, not explained;
 * whatever the `sources` passages present may be used.
 */
class LeakChecker(
    private val lex: Lexicon,
    private val embedder: Embedder,
    private val model: ChatModel,
    private val prompt: String,                                // prompts/leak_semantic.md
) {
    fun check(md: String, allowed: Set<String>, mentioned: Set<String> = emptySet(), sources: String? = null,
              semantic: Boolean = true): LeakReport {
        val inSources = sources?.let { lex.namedConcepts(it) }.orEmpty()
        val lexical = lex.lexical(md).filter { (cid, hit) ->
            cid !in allowed && cid !in mentioned && cid !in inSources && !lex.termAllowed(hit, allowed + inSources)
        }
        var sem = emptyList<String>()
        if (semantic) {
            val blocked = notNearAllowed(
                lex.concepts.filter { it.id !in allowed && it.id !in lexical && it.id !in inSources }, allowed)
            val qvec by lazy { embedder.query(Markdown.mdToText(md).take(4000)) }
            val cands = nearest(qvec, blocked, SEMANTIC_CANDIDATES)
            if (cands.isNotEmpty()) {
                val known = nearest(qvec, allowed.mapNotNull { lex.byId[it] }, 12)
                sem = semanticTags(md, cands, known.map { it.name }, sources)
            }
        }
        return LeakReport(lexical, sem)
    }

    /** A blocked concept this similar to an allowed one is not a semantic candidate. */
    private fun notNearAllowed(pool: List<Concept>, allowed: Set<String>): List<Concept> {
        val emb = lex.embeddings ?: return pool
        val a = allowed.mapNotNull { lex.index[it] }.map { emb[it] }
        if (a.isEmpty()) return pool
        return pool.filter { c -> val v = emb[lex.index.getValue(c.id)]; a.maxOf { dot(it, v) } < NEAR_DUPLICATE }
    }

    fun nearest(qvec: FloatArray, pool: List<Concept>, k: Int): List<Concept> {
        val emb = lex.embeddings ?: return pool.take(k)
        return pool.sortedByDescending { dot(emb[lex.index.getValue(it.id)], qvec) }.take(k)
    }

    private fun semanticTags(md: String, candidates: List<Concept>, known: List<String>, sources: String?): List<String> {
        val listing = candidates.joinToString("\n") { c ->
            "- ${c.id}: ${c.name}" + if (c.aliases.isNotEmpty()) " (also: ${c.aliases.take(3).joinToString(", ")})" else ""
        }
        val schema = JSONObject()
            .put("type", "object")
            .put("properties", JSONObject().put("relies_on", JSONObject()
                .put("type", "array")
                .put("items", JSONObject()
                    .put("type", "object")
                    .put("properties", JSONObject()
                        .put("id", JSONObject().put("type", "string").put("enum", JSONArray(candidates.map { it.id })))
                        .put("quote", JSONObject().put("type", "string"))
                        .put("why", JSONObject().put("type", "string")))
                    .put("required", JSONArray(listOf("id", "quote", "why"))))))
            .put("required", JSONArray(listOf("relies_on")))
        val knows = if (known.isNotEmpty()) "The student HAS already learned (using these is fine): " + known.joinToString("; ") + "\n\n" else ""
        val src = sources?.let { "Source passages the text is based on (whatever they present is fine to use):\n<sources>\n${it.take(9000)}\n</sources>\n\n" }.orEmpty()
        val msgs = listOf(
            Message("system", prompt),
            Message("user", "${knows}${src}Concepts the student has NOT learned yet:\n$listing\n\nText to check:\n<text>\n$md\n</text>"),
        )
        val res = JSONObject(model.chat(msgs, temperature = 0f, maxTokens = 1500, schema = schema.toString()).text)
        val ids = candidates.map { it.id }.toSet()
        val text = Markdown.squash(md)
        // a tag counts only with a real quote from the text as evidence
        val tags = res.getJSONArray("relies_on")
        return (0 until tags.length()).map { tags.getJSONObject(it) }.filter { t ->
            val quote = t.getString("quote")
            t.getString("id") in ids && quote.trim().split(Regex("\\s+")).size >= 3 && Markdown.squash(quote) in text
        }.map { it.getString("id") }
    }

    companion object {
        const val SEMANTIC_CANDIDATES = 15
        const val NEAR_DUPLICATE = 0.90f
    }
}
