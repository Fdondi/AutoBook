package com.example.autoreader.pipeline

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.BeforeClass
import org.junit.Test
import java.io.File
import java.sql.Connection
import java.sql.DriverManager

/**
 * The phone's answer pipeline must decide like the desktop's. Expected values come from
 * tools/reader_parity.py (the leak gate's trap questions); the book is the exported
 * out/phone/mr.tbook, so this test is skipped where that file isn't built.
 */
class ParityTest {
    private class JdbcDb(private val con: Connection) : Db {
        override fun <T> query(sql: String, vararg args: Any?, row: (Row) -> T): List<T> =
            con.prepareStatement(sql).use { st ->
                args.forEachIndexed { i, a -> st.setObject(i + 1, a) }
                st.executeQuery().use { rs ->
                    val r = object : Row {
                        override fun string(i: Int): String? = rs.getString(i + 1)
                        override fun int(i: Int): Int? = rs.getInt(i + 1).takeUnless { rs.wasNull() }
                        override fun blob(i: Int): ByteArray? = rs.getBytes(i + 1)
                    }
                    buildList { while (rs.next()) add(row(r)) }
                }
            }
    }

    companion object {
        private lateinit var lex: Lexicon
        private lateinit var index: Index
        private lateinit var fixture: JSONObject

        @BeforeClass
        @JvmStatic
        fun load() {
            val pkg = File("../../out/phone/mr.tbook")
            assumeTrue("exported book not found: run `tbook export mr`", pkg.exists())
            val db = JdbcDb(DriverManager.getConnection("jdbc:sqlite:${pkg.absolutePath}"))
            lex = Lexicon.load(db)
            index = Index(db)
            fixture = JSONObject(ParityTest::class.java.getResource("/parity-mr.json")!!.readText())
        }

        private fun JSONArray.strings() = (0 until length()).map { getString(it) }
        private fun JSONArray.floats() = FloatArray(length()) { getDouble(it).toFloat() }
    }

    private fun cases(key: String) = fixture.getJSONArray(key).let { a -> (0 until a.length()).map { a.getJSONObject(it) } }

    private fun allowedFor(seen: Set<String>, ordinals: Map<String, Int>) = lex.allowed(seen.mapNotNull { ordinals[it] }.toSet())

    private val ordinals by lazy { unitOrdinals() }

    private fun unitOrdinals(): Map<String, Int> {
        val pkg = File("../../out/phone/mr.tbook")
        val db = JdbcDb(DriverManager.getConnection("jdbc:sqlite:${pkg.absolutePath}"))
        return db.query("SELECT unit_id, ordinal FROM units") { it.string(0)!! to it.int(1)!! }.toMap()
    }

    @Test
    fun allowedSetsMatch() {
        for (c in cases("questions")) {
            assertEquals(c.getString("question"), c.getJSONArray("allowed").strings().toSet(),
                allowedFor(c.getJSONArray("seen").strings().toSet(), ordinals))
        }
    }

    @Test
    fun questionTaggingMatches() {
        val misses = mutableListOf<String>()
        for (c in cases("questions")) {
            val got = lex.namedConcepts(c.getString("question"))
            val want = c.getJSONArray("named").strings().toSet()
            if (got != want) misses += "${c.getString("question")}\n  only phone: ${got - want}\n  only desktop: ${want - got}"
        }
        println("tagging: ${misses.size} of ${cases("questions").size} differ\n" + misses.joinToString("\n"))
        assertTrue(misses.joinToString("\n"), misses.isEmpty())
    }

    @Test
    fun forwardPointersMatch() {
        for (c in cases("questions")) {
            val qvec = c.getJSONArray("qvec").floats()
            val tagged = lex.namedConcepts(c.getString("question")).toMutableSet()
            val emb = lex.embeddings!!
            emb.indices.sortedByDescending { dot(emb[it], qvec) }.take(3).forEach {
                if (dot(emb[it], qvec) >= Reader.TAG_SIM_MIN) tagged += lex.concepts[it].id
            }
            val allowed = allowedFor(c.getJSONArray("seen").strings().toSet(), ordinals)
            val needed = ((tagged - allowed) - lex.externals).sortedWith(
                compareBy({ lex.byId[it]?.introducedOrdinal ?: Int.MAX_VALUE }, { it }))
            assertEquals(c.getString("question"), c.getJSONArray("needed_unseen").strings(), needed)
        }
    }

    @Test
    fun retrievalMatches() {
        val misses = mutableListOf<String>()
        for (c in cases("questions")) {
            val got = index.search(c.getString("question"), c.getJSONArray("seen").strings().toSet(),
                c.getJSONArray("qvec").floats()).map { it.unitId + "|" + it.chunkId }
            val want = c.getJSONArray("hits").strings()
            if (got != want) misses += "${c.getString("question")}\n  phone:   $got\n  desktop: $want"
        }
        println("retrieval: ${misses.size} of ${cases("questions").size} differ\n" + misses.joinToString("\n"))
        assertTrue(misses.joinToString("\n"), misses.isEmpty())
    }

    @Test
    fun lexicalLeakCheckMatches() {
        val misses = mutableListOf<String>()
        for (c in cases("leaks")) {
            val text = c.getString("text")
            val got = lex.lexical(text)
            val want = c.getJSONObject("lexical_all").let { o -> o.keys().asSequence().associateWith { o.getString(it) } }
            if (got != want) misses += "${text.take(80)}\n  only phone: ${got - want.keys}\n  only desktop: ${want - got.keys}"
            // the verdict: which of those the reader may not use yet
            val seen = c.getJSONArray("seen").strings().toSet()
            val allowed = allowedFor(seen, ordinals)
            val checker = LeakChecker(lex, noEmbedder, noModel, "")
            val violations = checker.check(text, allowed, lex.namedIn(seen) - allowed, semantic = false).lexical
            val wantViolations = c.getJSONObject("violations").let { o -> o.keys().asSequence().associateWith { o.getString(it) } }
            if (violations != wantViolations) misses += "${text.take(80)}\n  violations phone: $violations\n  desktop: $wantViolations"
        }
        println("lexical: ${misses.size} differences in ${cases("leaks").size} cases\n" + misses.joinToString("\n"))
        assertTrue(misses.joinToString("\n"), misses.isEmpty())
    }

    private val noEmbedder = object : Embedder { override fun embed(text: String) = error("not used") }
    private val noModel = object : ChatModel {
        override fun chat(messages: List<Message>, temperature: Float, maxTokens: Int, schema: String?,
                          onToken: ((String) -> Unit)?) = error("not used")
    }
}
