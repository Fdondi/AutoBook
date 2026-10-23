package com.example.autoreader

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExtendedFloatingActionButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.repeatOnLifecycle
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun LibraryScreen(store: BookStore, onOpen: (Book) -> Unit) {
    val scope = rememberCoroutineScope()
    val snackbar = remember { SnackbarHostState() }
    var books by remember { mutableStateOf(store.list()) }
    var busy by remember { mutableStateOf(false) }
    var toRemove by remember { mutableStateOf<Book?>(null) }
    var modelsReady by remember { mutableStateOf(store.models.ready) }
    val lifecycle = LocalLifecycleOwner.current.lifecycle

    fun importing(work: () -> List<String>) {
        busy = true
        scope.launch {
            val message = try {
                val added = withContext(Dispatchers.IO) { work() }
                added.joinToString().takeIf { it.isNotEmpty() }?.let { "Added $it" }
            } catch (e: Exception) {
                e.message ?: "Couldn't add the book."
            }
            books = withContext(Dispatchers.IO) { store.list() }
            modelsReady = store.models.ready
            busy = false
            message?.let { snackbar.showSnackbar(it) }
        }
    }

    // Book files copied over USB into the app's folder are added whenever the library shows.
    LaunchedEffect(lifecycle) {
        lifecycle.repeatOnLifecycle(Lifecycle.State.RESUMED) { importing { store.importDropped() } }
    }
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) importing { listOf(store.import(uri)) }
    }

    Scaffold(
        topBar = { TopAppBar(title = { Text("Your books") }) },
        floatingActionButton = {
            ExtendedFloatingActionButton(onClick = { picker.launch(arrayOf("*/*")) }) { Text("Add book") }
        },
        snackbarHost = { SnackbarHost(snackbar) },
    ) { padding ->
        Column(Modifier.padding(padding).fillMaxSize()) {
            if (busy) LinearProgressIndicator(Modifier.fillMaxWidth())
            if (books.isEmpty() && !busy) {
                Column(Modifier.padding(24.dp).widthIn(max = 560.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    Text("No books yet", style = MaterialTheme.typography.titleLarge)
                    Text("On your computer, run `tbook export <book>` and copy the file from out/phone/ to this " +
                        "phone over USB, for example into Download. Then tap Add book and pick it.")
                    store.dropDir?.let {
                        Text("Or copy it into ${it.path.substringAfter("/0/")}: it is added as soon as you open the app.",
                            style = MaterialTheme.typography.bodySmall)
                    }
                }
            }
            if (books.isNotEmpty() && !modelsReady) {
                Text("Reading and quizzes work now. To ask questions, add the two model files the same way: " +
                    "Qwen3-4B-Instruct-2507-Q4_K_M.gguf (2.5 GB) and nomic-embed-text-v1.5.Q4_K_M.gguf (84 MB).",
                    style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.padding(start = 16.dp, end = 16.dp, top = 12.dp))
            }
            LazyColumn(contentPadding = androidx.compose.foundation.layout.PaddingValues(16.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp)) {
                items(books, key = { it.id }) { book ->
                    Card(Modifier.fillMaxWidth().clickable { onOpen(book) }) {
                        Row(Modifier.padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
                            Column(Modifier.weight(1f)) {
                                Text(book.title, style = MaterialTheme.typography.titleMedium)
                                Text(book.authors, style = MaterialTheme.typography.bodyMedium,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant)
                            }
                            TextButton(onClick = { toRemove = book }) { Text("Remove") }
                        }
                    }
                }
            }
        }
    }

    toRemove?.let { book ->
        AlertDialog(
            onDismissRequest = { toRemove = null },
            title = { Text("Remove ${book.title}?") },
            text = { Text("The book file is deleted from this phone. Your reading history stays, so adding it again picks up where you were.") },
            confirmButton = {
                TextButton(onClick = {
                    store.delete(book)
                    books = store.list()
                    toRemove = null
                }) { Text("Remove") }
            },
            dismissButton = { TextButton(onClick = { toRemove = null }) { Text("Cancel") } },
        )
    }
}
