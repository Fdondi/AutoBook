package com.example.autoreader

import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.content.pm.ApplicationInfo
import android.net.Uri
import android.view.ViewGroup
import android.webkit.JavascriptInterface
import android.webkit.WebChromeClient
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.webkit.WebViewAssetLoader
import org.json.JSONObject
import java.io.File
import java.util.concurrent.Executors

private const val HOST = "appassets.androidplatform.net"

/**
 * The reader: the desktop web reader (src/tbook/reader/static, copied into assets/web by the
 * webAssets Gradle task) in a WebView. Pages and libraries load through WebViewAssetLoader;
 * the page's API calls go through [ReaderBridge] instead of tbook's HTTP server.
 */
@SuppressLint("SetJavaScriptEnabled")
@Composable
fun ReaderScreen(book: Book, models: ModelStore, onClose: () -> Unit) {
    val context = LocalContext.current
    val lifecycle = LocalLifecycleOwner.current.lifecycle
    val session = remember(book.id) {
        BookSession(book, Learner(context, book.id), models) { name ->
            context.assets.open("prompts/$name.md").bufferedReader().use { it.readText() }
        }
    }
    val webView = remember(book.id) {
        WebView.setWebContentsDebuggingEnabled(context.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE != 0)
        WebView(context).apply {
            // Fill the screen: sized to its content, a page that is 100% high collapses to nothing.
            layoutParams = ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)
            settings.javaScriptEnabled = true
            settings.domStorageEnabled = true
            settings.allowFileAccess = false
            settings.allowContentAccess = false
            settings.setSupportZoom(false)
            webChromeClient = WebChromeClient()                 // confirm() dialogs
            webViewClient = ReaderClient(context, book, onClose)
            addJavascriptInterface(ReaderBridge(this, session), "AutoBookBridge")
            loadUrl("https://$HOST/book/${book.id}")
        }
    }
    DisposableEffect(book.id) {
        val observer = LifecycleEventObserver { _, event ->
            when (event) {
                Lifecycle.Event.ON_PAUSE -> webView.onPause()
                Lifecycle.Event.ON_RESUME -> webView.onResume()
                else -> Unit
            }
        }
        lifecycle.addObserver(observer)
        onDispose {
            lifecycle.removeObserver(observer)
            webView.destroy()
            session.close()
        }
    }
    BackHandler {
        webView.evaluateJavascript("window.autobookBack ? window.autobookBack() : false") { handled ->
            if (handled != "true") onClose()
        }
    }
    AndroidView(factory = { webView }, modifier = Modifier.fillMaxSize().windowInsetsPadding(WindowInsets.safeDrawing))
}

private class ReaderClient(private val context: Context, book: Book, private val onClose: () -> Unit) : WebViewClient() {
    private val loader = WebViewAssetLoader.Builder()
        .addPathHandler("/static/") { assetResponse("web/static/$it") }
        .addPathHandler("/vendor/") { assetResponse("web/vendor/$it") }
        .addPathHandler("/book/") { assetResponse("web/static/reader.html") }
        .addPathHandler("/pdf/") { id -> if (id == book.id) fileResponse(book.pdfFile) else null }
        .build()

    override fun shouldInterceptRequest(view: WebView, request: WebResourceRequest): WebResourceResponse? =
        loader.shouldInterceptRequest(request.url)

    override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean {
        val url = request.url
        when {
            url.host != HOST -> context.startActivity(Intent(Intent.ACTION_VIEW, url))   // links out of the book
            url.path == "/" -> onClose()                                                // the page's library button
            else -> return false
        }
        return true
    }

    private fun assetResponse(path: String): WebResourceResponse? =
        try {
            WebResourceResponse(mimeType(path), null, context.assets.open(path))
        } catch (e: java.io.IOException) {
            null
        }

    private fun fileResponse(file: File) = WebResourceResponse("application/pdf", null, file.inputStream())

    private fun mimeType(path: String) = when (path.substringAfterLast('.').lowercase()) {
        "html" -> "text/html"
        "js", "mjs" -> "text/javascript"
        "css" -> "text/css"
        "json" -> "application/json"
        "svg" -> "image/svg+xml"
        "png" -> "image/png"
        "gif" -> "image/gif"
        "wasm" -> "application/wasm"
        "woff2" -> "font/woff2"
        "woff" -> "font/woff"
        "ttf" -> "font/ttf"
        else -> "application/octet-stream"
    }
}

/**
 * What the page calls instead of tbook's HTTP API (see api()/stream() in reader.js).
 * Calls run off the UI thread, model calls on their own so reading goes on meanwhile;
 * replies go back through window.__autobook (events while working, then the result).
 */
private class ReaderBridge(private val webView: WebView, private val session: BookSession) {
    private val worker = Executors.newSingleThreadExecutor()
    private val modelWorker = Executors.newSingleThreadExecutor()

    @JavascriptInterface
    fun call(id: String, book: String, path: String, body: String) {
        val slow = path == "ask" || path == "quiz/check"
        (if (slow) modelWorker else worker).execute {
            try {
                val json = if (body.isEmpty()) JSONObject() else JSONObject(body)
                reply("end", id, handle(path, json) { reply("event", id, it.toString()) }.toString())
            } catch (e: Exception) {
                if (slow) reply("end", id, JSONObject().put("type", "error").put("text", e.message ?: e.toString()).toString())
                else reply("fail", id, e.message ?: e.toString())
            }
        }
    }

    private fun handle(path: String, body: JSONObject, emit: (JSONObject) -> Unit): JSONObject {
        val query = Uri.parse("x:/$path")
        return when (query.path?.trimStart('/')) {
            "state" -> session.state()
            "pages" -> body.getJSONArray("pages").let { a -> session.readPages((0 until a.length()).map(a::getInt), "pdf") }
            "mark" -> session.readPages((0..body.getInt("through")).toList(), "mark")
            "position" -> JSONObject().put("ok", true).also { session.position(body.getInt("page")) }
            "reset" -> session.reset()
            "quiz" -> session.quiz(query.getQueryParameter("scope") ?: "chapter",
                query.getQueryParameter("page")?.toIntOrNull())
            "quiz/result" -> JSONObject().put("ok", true).also {
                session.quizResult(body.getString("qa_id"), body.getString("result"))
            }
            "ask" -> session.ask(body.getString("question"), body.optString("mode", "ask"),
                if (body.isNull("page")) null else body.optInt("page"), emit)
            "quiz/check" -> session.check(body.getString("qa_id"), body.getString("answer"), emit)
            else -> throw IllegalArgumentException("unknown call: $path")
        }
    }

    private fun reply(fn: String, id: String, payload: String) {
        val js = "window.__autobook.$fn(${JSONObject.quote(id)}, ${JSONObject.quote(payload)})"
        webView.post { webView.evaluateJavascript(js, null) }
    }
}
