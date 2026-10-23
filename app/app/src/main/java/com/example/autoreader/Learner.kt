package com.example.autoreader

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import org.json.JSONObject
import java.util.UUID

/**
 * Append-only reading log for one book, the same `events` table as the desktop learner DB
 * (src/tbook/reader/learner.py), so the two can later be merged by event id.
 */
class Learner(context: Context, bookId: String) :
    SQLiteOpenHelper(context, "learner-$bookId.sqlite", null, 1) {

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL("CREATE TABLE events (event_id TEXT PRIMARY KEY, ts REAL, kind TEXT, unit_id TEXT, data TEXT)")
        db.execSQL("CREATE INDEX events_kind ON events(kind)")
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) = Unit

    fun log(kind: String, unitId: String? = null, data: JSONObject = JSONObject()) {
        writableDatabase.insertOrThrow("events", null, ContentValues().apply {
            put("event_id", UUID.randomUUID().toString().replace("-", ""))
            put("ts", System.currentTimeMillis() / 1000.0)
            put("kind", kind)
            put("unit_id", unitId)
            put("data", data.toString())
        })
    }

    /** Run several writes as one transaction. */
    fun <T> batch(block: () -> T): T {
        val db = writableDatabase
        db.beginTransaction()
        try {
            return block().also { db.setTransactionSuccessful() }
        } finally {
            db.endTransaction()
        }
    }

    fun seenUnits(): Set<String> =
        readableDatabase.rawQuery("SELECT DISTINCT unit_id FROM events WHERE kind='open_unit'", null).use { c ->
            buildSet { while (c.moveToNext()) add(c.getString(0)) }
        }

    /** Data of every `kind` event, oldest first. */
    fun events(kind: String): List<JSONObject> =
        readableDatabase.rawQuery("SELECT data FROM events WHERE kind=? ORDER BY ts", arrayOf(kind)).use { c ->
            buildList { while (c.moveToNext()) add(JSONObject(c.getString(0))) }
        }

    fun reset() {
        writableDatabase.delete("events", null, null)
    }
}
