package com.example.autoreader

import android.database.sqlite.SQLiteDatabase
import com.example.autoreader.pipeline.Message
import com.example.autoreader.pipeline.Reader
import org.json.JSONArray
import org.json.JSONObject

/**
 * One open book: reading state and quizzes. Mirrors Session in src/tbook/reader/web.py
 * (same JSON shapes, so the shared reader page works against either).
 *
 * A page counts as read once the page reports it; a unit once all its pages are read.
 */
class BookSession(
    private val book: Book,
    private val learner: Learner,
    private val models: ModelStore,
    private val prompts: (String) -> String,
) : AutoCloseable {
    private class BookUnit(
        val id: String, val ordinal: Int, val chapter: String?, val section: String?, val title: String?,
        val pageStart: Int, val pageEnd: Int?, val pageLabel: String?,
    ) {
        var lastPage = pageStart
    }

    private class QA(val id: String, val unitId: String, val question: String, val answer: String, val citedUnits: String?)

    private val db = SQLiteDatabase.openDatabase(book.packageFile.path, null, SQLiteDatabase.OPEN_READONLY)
    private val units: List<BookUnit>
    private val byId: Map<String, BookUnit>
    private val qa: Map<String, QA>

    init {
        units = db.rawQuery(
            "SELECT unit_id, ordinal, chapter, section, title, page_start, page_end, printed_page_label " +
                "FROM units ORDER BY ordinal", null,
        ).use { c ->
            buildList {
                while (c.moveToNext()) add(BookUnit(c.getString(0), c.getInt(1), c.getString(2), c.getString(3),
                    c.getString(4), c.getInt(5), if (c.isNull(6)) null else c.getInt(6), c.getString(7)))
            }
        }
        // A unit runs from its first page to the page before the next unit starts (or that
        // page itself when the next one starts on it).
        units.forEachIndexed { i, u ->
            val end = units.getOrNull(i + 1)?.pageStart ?: ((u.pageEnd ?: u.pageStart) + 1)
            u.lastPage = maxOf(u.pageStart, end - 1)
        }
        byId = units.associateBy { it.id }
        qa = db.rawQuery("SELECT qa_id, unit_id, question, answer, cited_units FROM qa", null).use { c ->
            buildMap {
                while (c.moveToNext()) put(c.getString(0), QA(c.getString(0), c.getString(1), c.getString(2),
                    c.getString(3), c.getString(4)))
            }
        }
    }

    private val canAsk = models.ready && "reader_terms" in AndroidDb(db).tables()
    private var reader: Reader? = null

    override fun close() = db.close()

    private fun place(unitId: String?): JSONObject {
        val u = byId[unitId] ?: return JSONObject()
        return JSONObject().put("unit", u.id).put("section", u.section).put("title", u.title)
            .put("page", u.pageLabel).put("page_idx", u.pageStart)
    }

    // -- reading state ----------------------------------------------------------

    @Synchronized
    fun state(): JSONObject {
        val seen = learner.seenUnits()
        val read = units.filter { it.id in seen }
        val positions = learner.events("position")
        return JSONObject()
            .put("book", JSONObject().put("id", book.id).put("title", book.title).put("authors", book.authors))
            .put("features", JSONObject().put("ask", canAsk).put("check", models.ready))
            .put("pages_read", JSONArray(pagesRead().sorted()))
            .put("last_page", positions.lastOrNull()?.getInt("page") ?: JSONObject.NULL)
            .put("units_read", read.size)
            .put("units", units.size)
            .put("furthest", read.maxByOrNull { it.ordinal }?.let { place(it.id) } ?: JSONObject.NULL)
    }

    private fun pagesRead(): Set<Int> =
        learner.events("page_read").flatMapTo(mutableSetOf()) { e ->
            val a = e.getJSONArray("pages")
            (0 until a.length()).map { a.getInt(it) }
        }

    @Synchronized
    fun readPages(pages: Collection<Int>, via: String): JSONObject {
        val have = pagesRead().toMutableSet()
        val new = pages.toSortedSet() - have
        if (new.isNotEmpty()) learner.batch {
            learner.log("page_read", null, JSONObject().put("pages", JSONArray(new.toList())).put("via", via))
            have += new
            val seen = learner.seenUnits()
            for (u in units) {
                if (u.id !in seen && (u.pageStart..u.lastPage).all { it in have }) {
                    learner.log("open_unit", u.id, JSONObject().put("via", via))
                }
            }
        }
        return state()
    }

    @Synchronized
    fun position(page: Int) {
        if (learner.events("position").lastOrNull()?.getInt("page") != page) {
            learner.log("position", null, JSONObject().put("page", page))
        }
    }

    @Synchronized
    fun reset(): JSONObject {
        learner.reset()
        return state()
    }

    private fun unitAt(page: Int): BookUnit = units.filter { it.pageStart <= page }.maxByOrNull { it.ordinal } ?: units.first()

    // -- quiz ---------------------------------------------------------------------

    /** A pre-written question from a unit the reader has read: unasked ones first, then misses. */
    @Synchronized
    fun quiz(scope: String, page: Int?): JSONObject {
        val seen = learner.seenUnits()
        val chapter = page?.let { unitAt(it).chapter }
        val pool = qa.values.filter { it.unitId in seen && (scope == "all" || byId[it.unitId]?.chapter == chapter) }
        val last = learner.events("quiz_result").associate { it.getString("qa_id") to it.getString("result") }
        if (pool.isEmpty()) return JSONObject().put("empty", true).put("scope", scope).put("any_read", seen.isNotEmpty())
        val rank = { q: QA -> QUIZ_ORDER[last[q.id]] ?: 0 }
        val best = pool.minOf(rank)
        val q = pool.filter { rank(it) == best }.random()
        val answer = TextCleanup.text(q.answer, book.id)
        val cited = JSONArray(q.citedUnits ?: "[]").let { a -> (0 until a.length()).map { a.getString(it) } }
        val inText = Regex("""\[([a-z]+:[^\]\s]+)]""").findAll(answer).map { it.groupValues[1] }
        val cites = (cited + inText + q.unitId).distinct().filter { it in byId }.map { place(it) }
        return JSONObject()
            .put("qa_id", q.id)
            .put("question", TextCleanup.question(q.question, book.id))
            .put("answer", answer)
            .put("source", place(q.unitId))
            .put("citations", JSONArray(cites))
            .put("pool", pool.size)
            .put("unasked", pool.count { it.id !in last })
    }

    @Synchronized
    fun quizResult(qaId: String, result: String, answer: String? = null) {
        val q = qa[qaId] ?: return
        learner.log("quiz_result", q.unitId, JSONObject().put("qa_id", qaId).put("result", result).put("answer", answer))
    }

    // -- model calls (not synchronized: they take a while and reading goes on meanwhile) ---------

    private fun reader(emit: (JSONObject) -> Unit): Reader {
        reader?.let { return it }
        emit(status("Loading the reader model…"))
        val engine = models.engine() ?: error("The model files aren't on this phone yet.")
        return Reader(AndroidDb(db), engine, engine, prompts).also { reader = it }
    }

    fun ask(question: String, mode: String, page: Int?, emit: (JSONObject) -> Unit): JSONObject {
        if (page != null) readPages(listOf(page), "ask")             // the page on screen is being read
        val seen = learner.seenUnits()
        learner.log("question_asked", null, JSONObject().put("question", question).put("mode", mode).put("page", page))
        val r = reader(emit)
        emit(status("Looking through the pages you've read…"))
        var tokens = 0
        val ans = r.ask(question, seen, mode) {
            tokens++
            if (tokens % 5 == 0) emit(JSONObject().put("type", "progress").put("tokens", tokens))
        }
        learner.log("answer_given", null, JSONObject().put("answer_kind", ans.kind).put("source", "local")
            .put("text", ans.text.take(2000)))
        return JSONObject()
            .put("type", "answer")
            .put("kind", ans.kind)
            .put("mode", mode)
            .put("text", TextCleanup.text(ans.text, book.id))
            .put("citations", JSONArray(ans.citations.map { place(it.unit) }))
            .put("pointers", JSONArray(ans.neededUnseen.map { place(it.unit).put("concept", it.concept) }))
            .put("stats", JSONObject(ans.stats))
    }

    /** The reader model checks a quiz answer against the book's (prompts/reader_grade.md). */
    fun check(qaId: String, answer: String, emit: (JSONObject) -> Unit): JSONObject {
        val q = qa.getValue(qaId)
        val engine = models.engine() ?: error("The model files aren't on this phone yet.")
        emit(status("Checking your answer…"))
        val reference = Regex("""\s*\[[a-z]+:[^\]\s]+]""").replace(TextCleanup.text(q.answer, book.id), "")
        val msgs = listOf(
            Message("system", prompts("reader_grade")),
            Message("user", "Question: ${TextCleanup.question(q.question, book.id)}\n\nReference answer: $reference\n\nStudent's answer: $answer"),
        )
        val res = JSONObject(engine.chat(msgs, temperature = 0f, maxTokens = 300, schema = GRADE_SCHEMA).text)
        quizResult(qaId, res.getString("verdict"), answer.take(2000))
        return JSONObject().put("type", "grade").put("verdict", res.getString("verdict"))
            .put("feedback", TextCleanup.text(res.getString("feedback"), book.id))
    }

    private fun status(text: String) = JSONObject().put("type", "status").put("text", text)

    private companion object {
        val QUIZ_ORDER = mapOf(null to 0, "incorrect" to 1, "partly" to 2, "skipped" to 3, "correct" to 4)
        const val GRADE_SCHEMA = """{"type":"object","properties":{"verdict":{"type":"string","enum":["correct","partly","incorrect"]},"feedback":{"type":"string"}},"required":["verdict","feedback"]}"""
    }
}
