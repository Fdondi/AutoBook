package com.example.autoreader

import android.content.Context
import com.example.autoreader.llm.Engine
import java.io.DataInputStream
import java.io.File
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * The two GGUF model files answers need, copied in like books (file picker or USB drop):
 * the reader (Qwen3-4B-Instruct-2507) and the embedder the digest used (nomic-embed-text
 * v1.5). Which is which comes from the file's own `general.architecture`.
 */
class ModelStore(context: Context) {
    private val dir = File(context.filesDir, "models").apply { mkdirs() }
    val reader = File(dir, "reader.gguf")
    val embedder = File(dir, "embedder.gguf")
    private var engine: Engine? = null

    val ready get() = reader.exists() && embedder.exists()

    @Synchronized
    fun engine(): Engine? {
        if (!ready) return null
        return engine ?: Engine(reader, embedder).also { engine = it }
    }

    /** Copy a GGUF stream in; returns a description of what was added. */
    @Synchronized
    fun import(input: InputStream): String {
        val tmp = File(dir, "import.tmp")
        try {
            tmp.outputStream().use { input.copyTo(it, 1 shl 20) }
            val arch = tmp.inputStream().use(::architecture) ?: throw ImportException("Couldn't read that model file.")
            val isEmbedder = arch.contains("bert")
            engine?.close()
            engine = null
            tmp.renameTo(if (isEmbedder) embedder else reader)
            return if (isEmbedder) "the embedding model ($arch)" else "the reader model ($arch)"
        } finally {
            tmp.delete()
        }
    }

    companion object {
        private val MAGIC = "GGUF".toByteArray()

        fun isGguf(head: ByteArray) = head.size >= 4 && head.copyOf(4).contentEquals(MAGIC)

        /** `general.architecture` from a GGUF header (it is among the first keys). */
        fun architecture(input: InputStream): String? {
            val d = DataInputStream(input.buffered(1 shl 16))
            fun u32() = ByteBuffer.wrap(ByteArray(4).also(d::readFully)).order(ByteOrder.LITTLE_ENDIAN).int.toLong() and 0xFFFFFFFFL
            fun u64() = ByteBuffer.wrap(ByteArray(8).also(d::readFully)).order(ByteOrder.LITTLE_ENDIAN).long
            fun str() = String(ByteArray(u64().toInt()).also(d::readFully), Charsets.UTF_8)
            val sizes = mapOf(0 to 1, 1 to 1, 2 to 2, 3 to 2, 4 to 4, 5 to 4, 6 to 4, 7 to 1, 10 to 8, 11 to 8, 12 to 8)
            fun skip(type: Int) {
                when (type) {
                    8 -> str()
                    9 -> { val t = u32().toInt(); val n = u64(); repeat(n.toInt()) { skip(t) } }
                    else -> d.skipBytes(sizes[type] ?: throw IllegalStateException("bad GGUF value type $type"))
                }
            }
            if (!isGguf(ByteArray(4).also(d::readFully))) return null
            u32()                                       // version
            u64()                                       // tensor count
            val kvs = u64()
            for (i in 0 until minOf(kvs, 64)) {
                val key = str()
                val type = u32().toInt()
                if (key == "general.architecture" && type == 8) return str()
                skip(type)
            }
            return null
        }
    }
}
