package com.example.autoreader.pipeline

/** The embedder the digest used, run on the phone (nomic-embed-text v1.5). */
interface Embedder {
    /** Unit-length embedding of already-prefixed text. */
    fun embed(text: String): FloatArray

    /** Same prefixes and truncation as src/tbook/llm.embed. */
    fun query(text: String): FloatArray = embed("search_query: " + text.ifEmpty { " " }.take(6000))
}

data class Message(val role: String, val content: String)

class Generation(val text: String, val ttftS: Double?, val tokS: Double?)

interface ChatModel {
    /** Reply to `messages`; `schema` (JSON Schema) constrains the output; `onToken` sees each piece. */
    fun chat(messages: List<Message>, temperature: Float, maxTokens: Int, schema: String? = null,
             onToken: ((String) -> Unit)? = null): Generation
}
