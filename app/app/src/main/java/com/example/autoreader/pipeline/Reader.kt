package com.example.autoreader.pipeline

import org.json.JSONArray

class Place(val unit: String, val section: String?, val title: String?, val page: String?, val concept: String? = null)

class Answer(
    val kind: String,                                   // forward | qa | generated | fallback
    val text: String,
    val citations: List<Place> = emptyList(),
    val neededUnseen: List<Place> = emptyList(),
    val stats: Map<String, Any> = emptyMap(),
)

/**
 * The answer pipeline (plan §3.4): scope -> tag -> branch -> retrieve -> generate -> check
 * -> cite; mirrors Reader.ask in src/tbook/reader/answer.py. Deterministic steps decide;
 * the model only phrases.
 */
class Reader(
    db: Db,
    private val embedder: Embedder,
    private val model: ChatModel,
    private val prompts: (String) -> String,
) {
    private class UnitInfo(val id: String, val ordinal: Int, val section: String?, val title: String?, val page: String?)
    private class QA(val answer: String, val citedUnits: List<String>, val earliestNeeded: String?, val vec: FloatArray)

    val lex = Lexicon.load(db)
    private val index = Index(db)
    private val leak = LeakChecker(lex, embedder, model, prompts("leak_semantic"))
    private val units = db.query("SELECT unit_id, ordinal, section, title, printed_page_label FROM units") { r ->
        UnitInfo(r.string(0)!!, r.int(1)!!, r.string(2), r.string(3), r.string(4))
    }.associateBy { it.id }
    private val edges = mutableMapOf<String, MutableList<String>>().also { m ->
        db.query("SELECT from_concept, to_concept FROM concept_edges") { r -> m.getOrPut(r.string(0)!!) { mutableListOf() }.add(r.string(1)!!) }
    }
    private val qa = db.query("SELECT answer, cited_units, earliest_unit_needed, embedding FROM qa") { r ->
        QA(r.string(0)!!, JSONArray(r.string(1) ?: "[]").let { a -> (0 until a.length()).map { a.getString(it) } },
            r.string(2), r.blob(3)!!.toFloats())
    }

    private fun where(conceptId: String): Place {
        val c = lex.byId.getValue(conceptId)
        val u = units[c.introducedUnit ?: ""]
        return Place(c.introducedUnit ?: "", u?.section, u?.title, u?.page, c.name)
    }

    private fun cite(unitId: String): Place = units[unitId].let { Place(unitId, it?.section, it?.title, it?.page) }

    private fun closure(ids: Set<String>): Set<String> {
        val out = mutableSetOf<String>()
        val stack = ArrayDeque(ids)
        while (stack.isNotEmpty()) {
            val x = stack.removeLast()
            if (out.add(x)) stack.addAll(edges[x].orEmpty())
        }
        return out
    }

    fun ask(question: String, seenIn: Set<String>, mode: String = "ask", onToken: ((String) -> Unit)? = null): Answer {
        val t0 = System.nanoTime()
        fun total() = Math.round((System.nanoTime() - t0) / 1e7) / 100.0
        var seen = seenIn
        val seenOrd = seen.mapNotNullTo(mutableSetOf()) { units[it]?.ordinal }
        var allowed = lex.allowed(seenOrd)
        val mentioned = lex.namedIn(seen) - allowed             // named by the book so far: nameable, not explainable
        val qvec = embedder.query(question)

        // 2. tag the question
        val tagged = lex.namedConcepts(question).toMutableSet()
        lex.embeddings?.let { emb ->
            emb.indices.sortedByDescending { dot(emb[it], qvec) }.take(3).forEach { i ->
                if (dot(emb[i], qvec) >= TAG_SIM_MIN) tagged += lex.concepts[i].id
            }
        }
        val neededUnseen = ((tagged - allowed) - lex.externals).sortedWith(
            compareBy({ lex.byId[it]?.introducedOrdinal ?: Int.MAX_VALUE }, { it }))
        if (mode == "preview" && neededUnseen.isNotEmpty()) {
            allowed = allowed + closure(neededUnseen.toSet())
            val upTo = neededUnseen.maxOf { lex.byId[it]?.introducedOrdinal ?: 0 }
            seen = seen + units.values.filter { it.ordinal <= upTo }.map { it.id }
        }

        // 3a. needs unseen concepts: point forward (the reader chooses spoilers)
        if (neededUnseen.isNotEmpty() && mode == "ask") {
            return Answer("forward", "", neededUnseen = neededUnseen.map(::where), stats = mapOf("total_s" to total()))
        }
        // 3b. pre-generated Q&A
        if (qa.isNotEmpty() && mode != "preview") {
            val sims = qa.map { dot(it.vec, qvec) }
            for (i in sims.indices.sortedByDescending { sims[it] }.take(5)) {
                if (sims[i] < QA_MATCH_MIN) break
                val q = qa[i]
                if (q.earliestNeeded in seen && leak.check(q.answer, allowed, mentioned, semantic = false).ok()) {
                    return Answer("qa", q.answer, q.citedUnits.map(::cite), stats = mapOf("total_s" to total()))
                }
            }
        }

        // 4. retrieve from seen units only
        val hits = index.search(question, seen, qvec)
        if (hits.isEmpty()) return Answer("fallback", "Nothing you have read so far covers this yet.")
        // mode "within": the question names unseen concepts; they lead the off-limits list
        val blocked = neededUnseen.filter { it !in allowed }.map { lex.byId.getValue(it) }.toMutableList()
        val near = leak.nearest(embedder.query(Markdown.mdToText(question + "\n" + hits[0].text).take(4000)),
            lex.concepts.filter { it.id !in allowed }, 8)
        blocked += near.filter { it !in blocked }
        val passages = hits.joinToString("\n\n") { "<passage id=\"${it.unitId}\">\n${it.text}\n</passage>" }
        val note = "Not yet covered (do not use): " + (blocked.joinToString("; ") { it.name }.ifEmpty { "(none)" })
        var msgs = listOf(Message("system", prompts("reader_answer")), Message("user", "$note\n\n$passages\n\nQuestion: $question"))

        // 5-7. generate, check, regenerate once, fall back
        val sources = hits.joinToString("\n\n") { it.text }
        var gen: Generation? = null
        var report: LeakReport? = null
        var attempt = 0
        for (a in 1..2) {
            attempt = a
            gen = model.chat(msgs, temperature = 0.2f, maxTokens = 700, onToken = if (a == 1) onToken else null)
            val text = gen.text.trim()
            report = leak.check(text, allowed, mentioned, sources)
            if (report.ok()) break
            if (a == 1) {
                val names = report.violations.joinToString(", ") { lex.byId[it]?.name ?: it }
                msgs = msgs + Message("assistant", text) + Message("user",
                    "Your answer used concepts I have not read yet: $names. Answer again without them or their ideas.")
            }
        }
        val stats = buildMap<String, Any> {
            gen?.ttftS?.let { put("ttft_s", Math.round(it * 100) / 100.0) }
            gen?.tokS?.let { put("tok_s", Math.round(it * 10) / 10.0) }
            put("total_s", total())
            put("attempts", attempt)
        }
        if (report?.ok() != true) {
            return Answer("fallback", "", neededUnseen = report!!.violations.map(::where), stats = stats)
        }
        val text = gen!!.text.trim()
        val hitUnits = hits.map { it.unitId }.toSet()
        val cited = CITE.findAll(text).map { it.groupValues[1] }.filter { it in hitUnits }.distinct().toList()
        return Answer("generated", text, cited.map(::cite), stats = stats)
    }

    companion object {
        const val TAG_SIM_MIN = 0.80f        // embedding match of a question to a concept name
        const val QA_MATCH_MIN = 0.86f
        private val CITE = Regex("""\[([a-z]+:[^\]\s]+)]""")
    }
}
