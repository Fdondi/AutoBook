package com.example.autoreader

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import com.example.autoreader.ui.theme.AutoReaderTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        val store = BookStore(applicationContext, ModelStore(applicationContext))
        setContent {
            AutoReaderTheme {
                App(store)
            }
        }
    }
}

@Composable
fun App(store: BookStore) {
    var openId by rememberSaveable { mutableStateOf<String?>(null) }
    val book = openId?.let { store.book(it) }
    if (book == null) {
        LibraryScreen(store, onOpen = { openId = it.id })
    } else {
        ReaderScreen(book, store.models, onClose = { openId = null })
    }
}
