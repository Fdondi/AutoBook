package com.example.autoreader.pipeline

/** Read-only access to a book package, so the pipeline runs on Android and in JVM tests alike. */
interface Db {
    fun <T> query(sql: String, vararg args: Any?, row: (Row) -> T): List<T>
    fun tables(): Set<String> = query("SELECT name FROM sqlite_master WHERE type='table'") { it.string(0)!! }.toSet()
}

interface Row {
    fun string(i: Int): String?
    fun int(i: Int): Int?
    fun blob(i: Int): ByteArray?
}

/** Little-endian float32 vector, as numpy writes them. */
fun ByteArray.toFloats(): FloatArray {
    val buf = java.nio.ByteBuffer.wrap(this).order(java.nio.ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
    return FloatArray(buf.remaining()).also { buf.get(it) }
}

fun dot(a: FloatArray, b: FloatArray): Float {
    var s = 0f
    for (i in a.indices) s += a[i] * b[i]
    return s
}
