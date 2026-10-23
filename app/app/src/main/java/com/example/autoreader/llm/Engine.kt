package com.example.autoreader.llm

import com.example.autoreader.pipeline.ChatModel
import com.example.autoreader.pipeline.Embedder
import com.example.autoreader.pipeline.Generation
import com.example.autoreader.pipeline.Message
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.concurrent.Executors

/**
 * The on-device models: Qwen3-4B for answers and nomic-embed for retrieval, loaded lazily
 * and used from one thread (llama.cpp contexts are not thread-safe). Same models and
 * context budget as the desktop reader (config.READER_CTX).
 */
class Engine(private val readerFile: File, private val embedderFile: File) : ChatModel, Embedder, AutoCloseable {
    private val thread = Executors.newSingleThreadExecutor { Thread(it, "llama") }
    private val threads = (Runtime.getRuntime().availableProcessors() - 2).coerceIn(1, 4)
    private var readerModel = 0L
    private var chatCtx = 0L
    private var embedModel = 0L
    private var embedCtx = 0L

    private fun <T> onThread(block: () -> T): T = thread.submit(block).get()

    override fun embed(text: String): FloatArray = onThread {
        if (embedCtx == 0L) {
            embedModel = LlamaNative.loadModel(embedderFile.path)
            embedCtx = LlamaNative.newEmbedder(embedModel, threads)
        }
        LlamaNative.embed(embedCtx, text.toByteArray())
    }

    override fun chat(messages: List<Message>, temperature: Float, maxTokens: Int, schema: String?,
                      onToken: ((String) -> Unit)?): Generation = onThread {
        if (chatCtx == 0L) {
            readerModel = LlamaNative.loadModel(readerFile.path)
            chatCtx = LlamaNative.newChat(readerModel, READER_CTX, threads)
        }
        val t0 = System.nanoTime()
        var first = 0L
        var count = 0
        val pending = ByteArrayOutputStream()
        val callback = LlamaNative.TokenCallback { piece ->
            if (first == 0L) first = System.nanoTime()
            count++
            if (onToken != null) {
                pending.write(piece)
                val bytes = pending.toByteArray()
                val complete = completeUtf8(bytes)
                if (complete > 0) {
                    onToken(String(bytes, 0, complete, Charsets.UTF_8))
                    pending.reset()
                    pending.write(bytes, complete, bytes.size - complete)
                }
            }
            true
        }
        val out = LlamaNative.chat(chatCtx, messages.map { it.role }.toTypedArray(),
            messages.map { it.content.toByteArray() }.toTypedArray(), schema?.toByteArray(), temperature, maxTokens, callback)
        val end = System.nanoTime()
        Generation(String(out, Charsets.UTF_8),
            if (first > 0) (first - t0) / 1e9 else null,
            if (first > 0 && end > first) count / ((end - first) / 1e9) else null)
    }

    override fun close() {
        onThread {
            if (chatCtx != 0L) LlamaNative.freeChat(chatCtx)
            if (readerModel != 0L) LlamaNative.freeModel(readerModel)
            if (embedCtx != 0L) LlamaNative.freeEmbedder(embedCtx)
            if (embedModel != 0L) LlamaNative.freeModel(embedModel)
            chatCtx = 0; readerModel = 0; embedCtx = 0; embedModel = 0
        }
        thread.shutdown()
    }

    private companion object {
        const val READER_CTX = 6144

        /** Length of the longest prefix of `b` that ends on a whole UTF-8 character. */
        fun completeUtf8(b: ByteArray): Int {
            var i = b.size - 1
            var back = 0
            while (i >= 0 && back < 4 && (b[i].toInt() and 0xC0) == 0x80) { i--; back++ }
            if (i < 0) return 0
            val lead = b[i].toInt() and 0xFF
            val need = when { lead < 0x80 -> 1; lead >= 0xF0 -> 4; lead >= 0xE0 -> 3; lead >= 0xC0 -> 2; else -> 1 }
            return if (b.size - i >= need) b.size else i
        }
    }
}
