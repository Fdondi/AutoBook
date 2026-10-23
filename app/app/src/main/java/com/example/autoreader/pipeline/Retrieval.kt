package com.example.autoreader.pipeline

import java.text.Normalizer
import kotlin.math.ln

class Hit(val chunkId: String, val unitId: String, val text: String, val score: Double)

/**
 * Hybrid retrieval (BM25 + vectors, reciprocal-rank fusion), hard-filtered to an allowed
 * set of units; mirrors src/tbook/retrieval.py. BM25 is computed here the way SQLite FTS5
 * scores it (unicode61 tokens, k1 1.2, b 0.75) because Android's SQLite has no FTS5.
 */
class Index(db: Db) {
    private val ids: List<String>
    private val units: List<String>
    private val texts: List<String>
    private val vecs: Array<FloatArray>
    private val termFreqs: List<Map<String, Int>>
    private val docLens: IntArray
    private val docFreq = mutableMapOf<String, Int>()
    private val avgLen: Double

    init {
        val rows = db.query(
            "SELECT c.chunk_id, c.unit_id, c.text, v.embedding FROM chunks c JOIN chunk_vec v USING(chunk_id) ORDER BY c.rowid",
        ) { r -> listOf(r.string(0)!!, r.string(1)!!, r.string(2) ?: "", r.blob(3)!!) }
        ids = rows.map { it[0] as String }
        units = rows.map { it[1] as String }
        texts = rows.map { it[2] as String }
        vecs = Array(rows.size) { (rows[it][3] as ByteArray).toFloats() }
        // Score the text the digest indexed (title + plain text): FTS5 keeps it in a plain shadow table.
        val indexed = db.query("SELECT c0, c1 FROM chunk_fts_content") { it.string(0)!! to (it.string(1) ?: "") }.toMap()
        val toks = ids.mapIndexed { i, id -> tokenize(indexed[id] ?: texts[i]) }
        termFreqs = toks.map { t -> t.groupingBy { it }.eachCount() }
        docLens = IntArray(toks.size) { toks[it].size }
        termFreqs.forEach { tf -> tf.keys.forEach { docFreq[it] = (docFreq[it] ?: 0) + 1 } }
        avgLen = if (docLens.isEmpty()) 1.0 else docLens.average()
    }

    fun search(query: String, allowedUnits: Set<String>?, qvec: FloatArray, k: Int = RETRIEVE_K,
               tokenBudget: Int = RETRIEVE_TOKEN_BUDGET): List<Hit> {
        if (ids.isEmpty()) return emptyList()
        val mask = BooleanArray(ids.size) { allowedUnits == null || units[it] in allowedUnits }
        if (mask.none { it }) return emptyList()
        val ranks = mutableMapOf<Int, Double>()
        // vector
        val sims = DoubleArray(ids.size) { if (mask[it]) dot(vecs[it], qvec).toDouble() else Double.NEGATIVE_INFINITY }
        sims.indices.sortedByDescending { sims[it] }.take(50).forEachIndexed { r, i ->
            if (sims[i].isFinite()) ranks[i] = (ranks[i] ?: 0.0) + 1.0 / (RRF_K + r)
        }
        // bm25
        val terms = Regex("[A-Za-z][A-Za-z0-9]+").findAll(Markdown.mdToText(query).lowercase())
            .map { it.value }.filter { it !in STOP }.distinct().toList()
        if (terms.isNotEmpty()) {
            var r = 0
            for (i in bm25(terms).take(200)) {
                if (!mask[i]) continue
                ranks[i] = (ranks[i] ?: 0.0) + 1.0 / (RRF_K + r)
                r++
                if (r >= 50) break
            }
        }
        val hits = mutableListOf<Hit>()
        var used = 0
        for ((i, s) in ranks.entries.sortedByDescending { it.value }) {
            val toks = (texts[i].split(Regex("\\s+")).count { it.isNotEmpty() } * 1.4).toInt()
            if (hits.isNotEmpty() && used + toks > tokenBudget) continue
            hits += Hit(ids[i], units[i], texts[i], s)
            used += toks
            if (hits.size >= k) break
        }
        return hits
    }

    /** Chunks containing any term, best BM25 score first. */
    private fun bm25(terms: List<String>): List<Int> {
        val n = ids.size.toDouble()
        val idf = terms.associateWith { t ->
            val df = (docFreq[t] ?: 0).toDouble()
            ln((n - df + 0.5) / (df + 0.5)).let { if (it <= 0) 1e-6 else it }
        }
        val scores = mutableListOf<Pair<Int, Double>>()
        for (i in ids.indices) {
            var s = 0.0
            for (t in terms) {
                val f = termFreqs[i][t] ?: continue
                s += idf.getValue(t) * f * (K1 + 1) / (f + K1 * (1 - B + B * docLens[i] / avgLen))
            }
            if (s > 0) scores += i to s
        }
        return scores.sortedByDescending { it.second }.map { it.first }
    }

    companion object {
        const val RETRIEVE_K = 6
        const val RETRIEVE_TOKEN_BUDGET = 2500
        private const val RRF_K = 60
        private const val K1 = 1.2
        private const val B = 0.75
        private val STOP = setOf("the", "a", "an", "of", "and", "or", "to", "in", "is", "are", "what", "how", "why", "does",
            "do", "can", "i", "it", "this", "that", "for", "with", "on", "be", "by", "as", "at", "from", "which")
        private val TOKEN = Regex("[\\p{L}\\p{N}]+")

        /** FTS5 unicode61: runs of letters and digits, case-folded, diacritics removed. */
        fun tokenize(text: String): List<String> =
            TOKEN.findAll(Normalizer.normalize(text.lowercase(), Normalizer.Form.NFKD).replace(Regex("\\p{M}+"), ""))
                .map { it.value }.toList()
    }
}
