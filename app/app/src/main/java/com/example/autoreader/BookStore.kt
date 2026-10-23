package com.example.autoreader

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.net.Uri
import java.io.File
import java.io.InputStream

/** A book on this phone: the digest package and the PDF taken out of it. */
data class Book(val id: String, val title: String, val authors: String, val packageFile: File, val pdfFile: File)

class ImportException(message: String) : Exception(message)

/**
 * Books live in the app's private storage as `<id>.tbook` + `<id>.pdf`. They arrive as one
 * file made by `tbook export <book>` (the package with the PDF inside), either picked with
 * the system file picker or dropped over USB into the app's folder (Android/data/<app>/files).
 */
class BookStore(private val context: Context, val models: ModelStore) {
    private val dir = File(context.filesDir, "books").apply { mkdirs() }

    /** Where USB-copied book files are picked up from. */
    val dropDir: File? get() = context.getExternalFilesDir(null)

    fun list(): List<Book> =
        dir.listFiles { f -> f.extension == "tbook" }.orEmpty().mapNotNull { book(it.nameWithoutExtension) }
            .sortedBy { it.title.lowercase() }

    fun book(id: String): Book? {
        val pkg = File(dir, "$id.tbook")
        val pdf = File(dir, "$id.pdf")
        if (!pkg.exists() || !pdf.exists()) return null
        val meta = manifest(pkg)
        return Book(id, meta["title"] ?: id, meta["authors"].orEmpty(), pkg, pdf)
    }

    /** Add a book file or a model file (told apart by their first bytes); returns what was added. */
    fun import(uri: Uri): String =
        context.contentResolver.openInputStream(uri)?.use { importAny(it) } ?: throw ImportException("Couldn't open that file.")

    /** Import every book and model file found in the drop folder, then remove the copies there. */
    fun importDropped(): List<String> =
        dropDir?.listFiles { f -> f.extension == "tbook" || f.extension == "gguf" }.orEmpty().map { f ->
            f.inputStream().use { importAny(it) }.also { f.delete() }
        }

    private fun importAny(raw: InputStream): String {
        val input = raw.buffered(1 shl 20)
        input.mark(16)
        val head = ByteArray(16).also { input.read(it) }
        input.reset()
        return if (ModelStore.isGguf(head)) models.import(input) else import(input).title
    }

    fun delete(book: Book) {
        book.packageFile.delete()
        book.pdfFile.delete()
    }

    private fun import(input: InputStream): Book {
        val tmp = File.createTempFile("import", ".tbook", context.cacheDir)
        try {
            tmp.outputStream().use { input.copyTo(it) }
            val meta = try {
                manifest(tmp)
            } catch (e: Exception) {
                throw ImportException("That isn't a book file made by `tbook export`.")
            }
            val id = meta["book_id"] ?: throw ImportException("The book file has no id.")
            if (!Regex("\\w+").matches(id)) throw ImportException("Unexpected book id '$id'.")
            val pdf = File(dir, "$id.pdf")
            extractPdf(tmp, File(dir, "$id.pdf.tmp"))
            File(dir, "$id.pdf.tmp").renameTo(pdf)
            tmp.copyTo(File(dir, "$id.tbook"), overwrite = true)
            return book(id) ?: throw ImportException("The book couldn't be stored.")
        } finally {
            tmp.delete()
        }
    }

    private fun manifest(pkg: File): Map<String, String> =
        open(pkg).use { db ->
            db.rawQuery("SELECT key, value FROM manifest", null).use { c ->
                buildMap { while (c.moveToNext()) put(c.getString(0), c.getString(1)) }
            }
        }

    /** Copy the PDF out of the package. Blobs are read in slices: a cursor row holds at most ~2 MB. */
    private fun extractPdf(pkg: File, out: File) {
        open(pkg).use { db ->
            val size = db.rawQuery("SELECT length(blob) FROM assets WHERE asset_id='pdf'", null).use { c ->
                if (c.moveToFirst() && !c.isNull(0)) c.getLong(0) else null
            } ?: throw ImportException("This book file has no PDF inside: make it with `tbook export`.")
            out.outputStream().use { os ->
                var offset = 1L
                while (offset <= size) {
                    db.rawQuery("SELECT substr(blob, ?, ?) FROM assets WHERE asset_id='pdf'",
                        arrayOf(offset.toString(), SLICE.toString())).use { c ->
                        c.moveToFirst()
                        os.write(c.getBlob(0))
                    }
                    offset += SLICE
                }
            }
        }
    }

    private fun open(pkg: File): SQLiteDatabase =
        SQLiteDatabase.openDatabase(pkg.path, null, SQLiteDatabase.OPEN_READONLY)

    private companion object {
        const val SLICE = 1_000_000
    }
}
