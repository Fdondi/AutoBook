package com.example.autoreader.llm

/** JNI surface of src/main/cpp/engine.cpp. Text crosses as UTF-8 bytes. Handles are not thread-safe. */
internal object LlamaNative {
    init {
        System.loadLibrary("autoreader")
        init()
    }

    fun interface TokenCallback {
        /** One generated piece (UTF-8, may end mid-character); return false to stop. */
        fun onToken(piece: ByteArray): Boolean
    }

    @JvmStatic private external fun init()
    @JvmStatic external fun loadModel(path: String): Long
    @JvmStatic external fun freeModel(model: Long)
    @JvmStatic external fun metadata(model: Long, key: String): String?
    @JvmStatic external fun newChat(model: Long, nCtx: Int, nThreads: Int): Long
    @JvmStatic external fun freeChat(chat: Long)
    @JvmStatic external fun chat(chat: Long, roles: Array<String>, contents: Array<ByteArray>, schema: ByteArray?,
                                 temperature: Float, maxTokens: Int, callback: TokenCallback?): ByteArray
    @JvmStatic external fun newEmbedder(model: Long, nThreads: Int): Long
    @JvmStatic external fun freeEmbedder(embedder: Long)
    @JvmStatic external fun embed(embedder: Long, text: ByteArray): FloatArray
}
