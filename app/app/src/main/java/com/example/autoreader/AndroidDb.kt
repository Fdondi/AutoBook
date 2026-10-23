package com.example.autoreader

import android.database.Cursor
import android.database.sqlite.SQLiteDatabase
import com.example.autoreader.pipeline.Db
import com.example.autoreader.pipeline.Row

/** The pipeline's package access over Android's SQLite. */
class AndroidDb(private val db: SQLiteDatabase) : Db {
    override fun <T> query(sql: String, vararg args: Any?, row: (Row) -> T): List<T> =
        db.rawQuery(sql, args.map { it?.toString() }.toTypedArray()).use { c ->
            val r = CursorRow(c)
            buildList { while (c.moveToNext()) add(row(r)) }
        }

    private class CursorRow(private val c: Cursor) : Row {
        override fun string(i: Int): String? = if (c.isNull(i)) null else c.getString(i)
        override fun int(i: Int): Int? = if (c.isNull(i)) null else c.getInt(i)
        override fun blob(i: Int): ByteArray? = if (c.isNull(i)) null else c.getBlob(i)
    }
}
