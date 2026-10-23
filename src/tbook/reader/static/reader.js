// Browser reader: the book's PDF (PDF.js viewer components) plus the ask/quiz panel.
// Server side: src/tbook/reader/web.py. A page counts as read after READ_AFTER_S
// seconds on screen; the server turns read pages into read units.

import * as pdfjsLib from "/vendor/pdfjs/legacy/build/pdf.mjs";
import * as pdfjsViewer from "/vendor/pdfjs/legacy/web/pdf_viewer.mjs";
import katex from "/vendor/katex/katex.mjs";

const BOOK = location.pathname.split("/").pop();
const READ_AFTER_S = 20;
const ZOOM_STEPS = [0.5, 0.67, 0.75, 0.9, 1, 1.1, 1.25, 1.5, 1.75, 2, 2.5, 3];
const PRESETS = new Set(["auto", "page-width", "page-fit", "page-actual"]);
const $ = (id) => document.getElementById(id);

pdfjsLib.GlobalWorkerOptions.workerSrc = "/vendor/pdfjs/legacy/build/pdf.worker.mjs";

const eventBus = new pdfjsViewer.EventBus();
const linkService = new pdfjsViewer.PDFLinkService({ eventBus });
const findController = new pdfjsViewer.PDFFindController({ eventBus, linkService });
const viewer = new pdfjsViewer.PDFViewer({
  container: $("viewerContainer"), viewer: $("viewer"), eventBus, linkService, findController,
});
linkService.setViewer(viewer);

let state = null;
let pdf = null;
let labels = null;
const readPages = new Set();
const dwell = new Map();

// -- small helpers -------------------------------------------------------------

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

function button(text, onClick, cls) {
  const b = el("button", cls, text);
  b.type = "button";
  b.addEventListener("click", onClick);
  return b;
}

function pref(key, fallback) {
  try { return localStorage.getItem(`tbook.${key}`) ?? fallback; } catch { return fallback; }
}

function savePref(key, value) {
  try { localStorage.setItem(`tbook.${key}`, value); } catch { /* private mode: not remembered */ }
}

// In the Android app the page runs in a WebView and reaches the app through
// window.AutoBookBridge (ReaderBridge.kt); on the desktop it talks to tbook's HTTP server.
const bridge = window.AutoBookBridge;
const calls = new Map();
let callId = 0;
window.__autobook = {
  event(id, json) { calls.get(id)?.onEvent(JSON.parse(json)); },
  end(id, json) {
    const c = calls.get(id);
    calls.delete(id);
    c?.resolve(json ? JSON.parse(json) : null);
  },
  fail(id, message) {
    const c = calls.get(id);
    calls.delete(id);
    c?.reject(new Error(message));
  },
};

function bridgeCall(path, body, onEvent = () => {}) {
  return new Promise((resolve, reject) => {
    const id = String(++callId);
    calls.set(id, { resolve, reject, onEvent });
    bridge.call(id, BOOK, path, body === undefined ? "" : JSON.stringify(body));
  });
}

async function api(path, body) {
  if (bridge) return bridgeCall(path, body);
  const init = body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  };
  const r = await fetch(`/api/${BOOK}/${path}`, init);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

// POST that answers with newline-delimited JSON events (status/progress, then the result).
async function stream(path, body, onEvent) {
  if (bridge) {
    const last = await bridgeCall(path, body, onEvent);
    if (last) onEvent(last);
    return;
  }
  const r = await fetch(`/api/${BOOK}/${path}`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, i);
      buf = buf.slice(i + 1);
      if (line.trim()) onEvent(JSON.parse(line));
    }
  }
  if (buf.trim()) onEvent(JSON.parse(buf));
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function renderMath(src) {
  const display = src.startsWith("$$");
  const tex = display ? src.slice(2, -2) : src.slice(1, -1);
  try {
    return katex.renderToString(tex, { displayMode: display, throwOnError: false });
  } catch {
    return escapeHtml(src);
  }
}

// Markdown subset the answers use: paragraphs, lists, bold/italic/code, $math$, [unit] citations.
function renderMd(md, cites) {
  const math = [];
  let s = md.replace(/\$\$([\s\S]+?)\$\$|\$([^$\n]+?)\$/g, (m) => `\u0000${math.push(m) - 1}\u0000`);
  s = escapeHtml(s)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\s*\[([a-z]+:[^\]\s]+)\]/g, (m, id) => {
      const c = cites.get(id);
      return c ? ` <button type="button" class="cite" data-page="${c.page_idx}" title="${escapeHtml(c.title || "")}">§${escapeHtml(c.section)} · p. ${escapeHtml(c.page)}</button>` : "";
    });
  s = s.split(/\n{2,}/).map((block) => {
    const lines = block.split("\n").filter((l) => l.trim());
    if (lines.length && lines.every((l) => /^\s*([-*]|\d+\.)\s+/.test(l))) {
      const tag = /^\s*\d+\./.test(lines[0]) ? "ol" : "ul";
      return `<${tag}>${lines.map((l) => `<li>${l.replace(/^\s*([-*]|\d+\.)\s+/, "")}</li>`).join("")}</${tag}>`;
    }
    return lines.length ? `<p>${lines.join("<br>")}</p>` : "";
  }).join("");
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => renderMath(math[+i]));
}

function mdEl(md, cites = new Map(), cls = "body") {
  const d = el("div", cls);
  d.innerHTML = renderMd(md, cites);
  return d;
}

function citeButton(place, text) {
  const b = button(text, () => goToPage(place.page_idx), "cite");
  b.title = place.title || "";
  return b;
}

function goToPage(idx) {
  if (idx == null || !pdf) return;
  viewer.currentPageNumber = idx + 1;
}

function statusEl(text) {
  return el("div", "status", text);
}

function errorEl(text) {
  return el("div", "error", `Something went wrong: ${text}`);
}

// -- the book ------------------------------------------------------------------

async function start() {
  try {
    setState(await api("state"));
  } catch (err) {
    $("loading").textContent = `This book can't be opened yet (${err.message}).`;
    return;
  }
  document.title = state.book.title;
  applyFeatures(state.features || {});
  $("bookTitle").textContent = state.book.title;
  $("bookTitle").title = `${state.book.title} by ${state.book.authors}`;
  const task = pdfjsLib.getDocument({
    url: `/pdf/${BOOK}`,
    cMapUrl: "/vendor/pdfjs/cmaps/",
    cMapPacked: true,
    standardFontDataUrl: "/vendor/pdfjs/standard_fonts/",
    wasmUrl: "/vendor/pdfjs/wasm/",
    iccUrl: "/vendor/pdfjs/iccs/",
  });
  task.onProgress = ({ loaded, total }) => {
    if (total) $("loading").textContent = `Opening the book… ${Math.round((100 * loaded) / total)}%`;
  };
  const doc = await task.promise;
  labels = await doc.getPageLabels();          // before the first page shows, so it shows its printed number
  pdf = doc;
  viewer.setDocument(pdf);
  linkService.setDocument(pdf, null);
  if (labels) viewer.setPageLabels(labels);
  buildOutline(await pdf.getOutline());
}

// What the host can do: the phone app has no model for asking or checking answers yet.
let features = { ask: true, check: true };

function applyFeatures(f) {
  features = { ...features, ...f };
  $("askBtn").hidden = !features.ask;
  document.querySelector('[role="tab"][data-tab="ask"]').hidden = !features.ask;
  if (!features.ask && currentTab === "ask") openPanel("quiz");
}

eventBus.on("pagesinit", () => {
  viewer.currentScaleValue = pref("zoom", "auto");
  if (state.last_page != null) viewer.currentPageNumber = state.last_page + 1;
  $("loading").hidden = true;
  updatePageUI();
  $("viewerContainer").focus();
});

function updatePageUI() {
  if (!pdf) return;
  const n = viewer.currentPageNumber;
  const total = viewer.pagesCount;
  const label = labels?.[n - 1];
  if (document.activeElement !== $("pageInput")) $("pageInput").value = label ?? n;
  $("pageCount").textContent = label && label !== String(n) ? `(${n} of ${total})` : `of ${total}`;
  $("prevBtn").disabled = n <= 1;
  $("nextBtn").disabled = n >= total;
  $("readMark").hidden = !readPages.has(n - 1);
}

let positionTimer;
eventBus.on("pagechanging", () => {
  updatePageUI();
  clearTimeout(positionTimer);
  positionTimer = setTimeout(() => api("position", { page: viewer.currentPageNumber - 1 }).catch(() => {}), 1500);
});

$("prevBtn").addEventListener("click", () => viewer.previousPage());
$("nextBtn").addEventListener("click", () => viewer.nextPage());
$("pageInput").addEventListener("keydown", (e) => {
  if (e.key !== "Enter") return;
  const v = e.target.value.trim();
  if (labels?.includes(v)) viewer.currentPageLabel = v;
  else if (/^\d+$/.test(v)) viewer.currentPageNumber = Math.min(Math.max(1, +v), viewer.pagesCount);
  e.target.blur();
  $("viewerContainer").focus();
});
$("pageInput").addEventListener("blur", updatePageUI);

// Reading: a page is read once it has been the current page for READ_AFTER_S seconds in all.
setInterval(() => {
  if (!pdf || document.hidden) return;
  const idx = viewer.currentPageNumber - 1;
  if (readPages.has(idx)) return;
  const t = (dwell.get(idx) || 0) + 1;
  dwell.set(idx, t);
  if (t < READ_AFTER_S) return;
  readPages.add(idx);
  updatePageUI();
  api("pages", { pages: [idx] }).then(setState).catch(() => readPages.delete(idx));
}, 1000);

function setState(s) {
  state = { ...state, ...s };
  readPages.clear();
  for (const p of s.pages_read) readPages.add(p);
  const f = s.furthest;
  $("progressText").textContent = s.units_read
    ? `Read ${s.units_read} of ${s.units} sections${f ? `, up to §${f.section} ${f.title}` : ""}`
    : "Nothing read yet. A page counts as read after you've spent a little while on it.";
  updatePageUI();
}

// -- zoom ----------------------------------------------------------------------

function setZoom(value) {
  viewer.currentScaleValue = value;
  savePref("zoom", value);
}

function zoom(dir) {
  const cur = viewer.currentScale;
  const next = dir > 0 ? ZOOM_STEPS.find((s) => s > cur + 0.01) : [...ZOOM_STEPS].reverse().find((s) => s < cur - 0.01);
  if (next) setZoom(String(next));
}

// Preset zooms follow the window: re-apply after the layout changes (panel, outline, resize).
function refit() {
  requestAnimationFrame(() => {
    if (pdf && PRESETS.has(viewer.currentScaleValue)) viewer.currentScaleValue = viewer.currentScaleValue;
  });
}

eventBus.on("scalechanging", ({ scale, presetValue }) => {
  const sel = $("zoomSelect");
  const custom = sel.querySelector('[value="custom"]');
  const value = presetValue && [...sel.options].some((o) => o.value === presetValue) ? presetValue
    : [...sel.options].find((o) => o.value !== "custom" && !PRESETS.has(o.value) && Math.abs(+o.value - scale) < 0.001)?.value;
  if (value) {
    sel.value = value;
    custom.hidden = true;
  } else {
    custom.textContent = `${Math.round(scale * 100)}%`;
    custom.hidden = false;
    sel.value = "custom";
  }
});
$("zoomSelect").addEventListener("change", (e) => e.target.value !== "custom" && setZoom(e.target.value));
$("zoomIn").addEventListener("click", () => zoom(1));
$("zoomOut").addEventListener("click", () => zoom(-1));
window.addEventListener("resize", refit);
$("viewerContainer").addEventListener("wheel", (e) => {
  if (!e.ctrlKey) return;
  e.preventDefault();
  zoom(e.deltaY < 0 ? 1 : -1);
}, { passive: false });

// Pinch to zoom on touch screens, around the point between the fingers.
let pinchDist = 0;
const touchDist = ([a, b]) => Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
$("viewerContainer").addEventListener("touchstart", (e) => {
  if (e.touches.length === 2) pinchDist = touchDist(e.touches);
}, { passive: true });
$("viewerContainer").addEventListener("touchmove", (e) => {
  if (e.touches.length !== 2 || !pinchDist) return;
  e.preventDefault();
  const d = touchDist(e.touches);
  if (Math.abs(d / pinchDist - 1) < 0.04) return;
  const [a, b] = e.touches;
  viewer.updateScale({ scaleFactor: d / pinchDist, origin: [(a.clientX + b.clientX) / 2, (a.clientY + b.clientY) / 2],
    drawingDelay: 300 });
  pinchDist = d;
}, { passive: false });
$("viewerContainer").addEventListener("touchend", (e) => {
  if (e.touches.length >= 2 || !pinchDist) return;
  pinchDist = 0;
  savePref("zoom", String(viewer.currentScale));
});

// -- find ----------------------------------------------------------------------

let findTimer;
function find(type, previous = false) {
  const query = $("findInput").value;
  if (!query) {
    eventBus.dispatch("findbarclose", { source: window });
    $("findCount").textContent = "";
    return;
  }
  eventBus.dispatch("find", {
    source: window, type, query, caseSensitive: false, entireWord: false,
    highlightAll: true, findPrevious: previous, matchDiacritics: false,
  });
}

function showFindCount(m) {
  $("findCount").textContent = $("findInput").value && m?.total ? `${m.current}/${m.total}` : "";
}

$("findInput").addEventListener("input", () => {
  clearTimeout(findTimer);
  findTimer = setTimeout(() => find(""), 250);
});
$("findInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    find("again", e.shiftKey);
  } else if (e.key === "Escape") {
    $("findInput").value = "";
    find("");
    $("viewerContainer").focus();
  }
});
eventBus.on("updatefindmatchescount", ({ matchesCount }) => showFindCount(matchesCount));
eventBus.on("updatefindcontrolstate", ({ state: s, matchesCount }) => {
  if (s === pdfjsViewer.FindState.NOT_FOUND && $("findInput").value) $("findCount").textContent = "none";
  else showFindCount(matchesCount);
});

// -- contents ------------------------------------------------------------------

function buildOutline(items) {
  const nav = $("outline");
  if (!items?.length) {
    nav.replaceChildren(el("p", "hint", "This PDF has no table of contents."));
    return;
  }
  const list = (entries) => {
    const ul = el("ul");
    for (const it of entries) {
      const li = el("li");
      const row = el("div", "item");
      const twisty = el("button", "twisty", "▸");
      twisty.type = "button";
      const link = button(it.title, () => {
        if (it.dest) linkService.goToDestination(it.dest);
        else if (it.url) window.open(it.url, "_blank", "noopener");
        if (matchMedia("(max-width: 900px)").matches) toggleOutline(false);
      }, "link");
      row.append(twisty, link);
      li.append(row);
      if (it.items?.length) {
        li.classList.add("collapsed");
        li.append(list(it.items));
        twisty.setAttribute("aria-label", `Show sections of ${it.title}`);
        twisty.addEventListener("click", () => {
          twisty.textContent = li.classList.toggle("collapsed") ? "▸" : "▾";
        });
      } else {
        twisty.classList.add("leaf");
        twisty.tabIndex = -1;
      }
      ul.append(li);
    }
    return ul;
  };
  nav.replaceChildren(list(items));
}

function toggleOutline(on = $("outline").hidden) {
  $("outline").hidden = !on;
  $("outlineBtn").setAttribute("aria-pressed", String(on));
  savePref("outline", on ? "1" : "");
  refit();
}
$("outlineBtn").addEventListener("click", () => toggleOutline());

// -- panel ---------------------------------------------------------------------

let currentTab = null;

function openPanel(tab) {
  $("panel").hidden = false;
  currentTab = tab;
  for (const b of document.querySelectorAll('[role="tab"]')) b.setAttribute("aria-selected", String(b.dataset.tab === tab));
  $("askTab").hidden = tab !== "ask";
  $("quizTab").hidden = tab !== "quiz";
  $("askBtn").setAttribute("aria-pressed", String(tab === "ask"));
  $("quizBtn").setAttribute("aria-pressed", String(tab === "quiz"));
  savePref("panel", tab);
  if (tab === "ask") $("askInput").focus();
  if (tab === "quiz" && !quizItem) nextQuestion();
  refit();
}

function closePanel() {
  $("panel").hidden = true;
  currentTab = null;
  $("askBtn").setAttribute("aria-pressed", "false");
  $("quizBtn").setAttribute("aria-pressed", "false");
  savePref("panel", "");
  refit();
  $("viewerContainer").focus();
}

$("askBtn").addEventListener("click", () => (currentTab === "ask" ? closePanel() : openPanel("ask")));
$("quizBtn").addEventListener("click", () => (currentTab === "quiz" ? closePanel() : openPanel("quiz")));
$("closePanel").addEventListener("click", closePanel);
for (const b of document.querySelectorAll('[role="tab"]')) b.addEventListener("click", () => openPanel(b.dataset.tab));

// -- asking --------------------------------------------------------------------

let selection = "";

function setSelection(text) {
  selection = text;
  $("selChip").hidden = !text;
  $("selChip").querySelector("span").textContent = text ? `“${text}”` : "";
}
$("selChip").querySelector("button").addEventListener("click", () => setSelection(""));

function scrollThread() {
  const t = $("thread");
  t.scrollTop = t.scrollHeight;
}

async function ask(question, mode, quote, showQuestion = true) {
  const thread = $("thread");
  thread.querySelector(".hint")?.remove();
  if (showQuestion) {
    const q = el("div", "q");
    if (quote) q.append(el("blockquote", null, quote));
    q.append(document.createTextNode(question));
    thread.append(q);
  }
  const a = el("div", "a");
  const status = statusEl(mode === "preview" ? "Reading ahead in the book…" : "Thinking…");
  a.append(status);
  thread.append(a);
  scrollThread();
  const full = quote ? `About this passage: "${quote}"\n\n${question}` : question;
  try {
    await stream("ask", { question: full, mode, page: viewer.currentPageNumber - 1 }, (ev) => {
      if (ev.type === "status") status.textContent = ev.text;
      else if (ev.type === "progress") status.textContent = `Writing the answer… ${Math.round(ev.tokens * 0.75)} words`;
      else if (ev.type === "error") a.replaceChildren(errorEl(ev.text));
      else if (ev.type === "answer") renderAnswer(a, ev, { question, quote });
    });
    setState(await api("state"));
  } catch (err) {
    a.replaceChildren(errorEl(err.message));
  }
  scrollThread();
}

function renderAnswer(a, ev, ctx) {
  a.replaceChildren();
  if (ev.kind === "forward" || ev.kind === "fallback") {
    const note = el("div", "note");
    note.append(el("div", null, ev.kind === "forward"
      ? "This needs material you haven't read yet:"
      : "I couldn't answer that without material you haven't read yet:"));
    const ul = el("ul");
    const seen = new Set();
    for (const p of ev.pointers) {
      const key = `${p.concept}|${p.section}`;
      if (seen.has(key)) continue;
      seen.add(key);
      const li = el("li");
      li.append(el("strong", null, p.concept), document.createTextNode(" "));
      if (p.section) li.append(citeButton(p, `§${p.section} ${p.title}, p. ${p.page}`));
      ul.append(li);
    }
    note.append(ul);
    const actions = el("div", "actions");
    const followUp = (mode) => () => {
      for (const b of actions.querySelectorAll("button")) b.disabled = true;
      ask(ctx.question, mode, ctx.quote, false);
    };
    if (ev.kind === "forward") actions.append(button("Answer with what I've read", followUp("within")));
    actions.append(button("Explain anyway (spoilers)", followUp("preview")));
    note.append(actions);
    a.append(note);
    return;
  }
  if (ev.mode === "preview") a.append(el("div", "note spoiler", "This answer uses material from later in the book."));
  a.append(mdEl(ev.text, new Map(ev.citations.map((c) => [c.unit, c]))));
  const secs = ev.stats?.total_s;
  a.append(el("div", "meta", `${ev.kind === "qa" ? "From the book's study questions" : "Based on the pages you've read"}${secs ? ` · ${secs}s` : ""}`));
}

$("askForm").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("askInput").value.trim();
  if (!q && !selection) return;
  ask(q || "Explain this.", "ask", selection);
  $("askInput").value = "";
  setSelection("");
});
$("askInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("askForm").requestSubmit();
  }
});
$("thread").addEventListener("click", (e) => {
  const c = e.target.closest(".cite[data-page]");
  if (c) goToPage(+c.dataset.page);
});

// Selecting text on a page offers "Ask about this".
function pageSelection() {
  const s = window.getSelection();
  if (!s || s.isCollapsed || !s.rangeCount || !$("viewer").contains(s.anchorNode)) return null;
  const text = s.toString().replace(/\s+/g, " ").trim();
  if (text.length < 3) return null;
  return { text: text.slice(0, 600), rect: s.getRangeAt(0).getBoundingClientRect() };
}

let selTimer;
document.addEventListener("selectionchange", () => {
  clearTimeout(selTimer);
  selTimer = setTimeout(() => {
    const sel = features.ask ? pageSelection() : null;
    const b = $("askSel");
    b.hidden = !sel;
    if (!sel) return;
    const x = Math.min(window.innerWidth - b.offsetWidth - 8, Math.max(8, sel.rect.right - b.offsetWidth));
    const y = sel.rect.bottom + 44 > window.innerHeight ? sel.rect.top - 40 : sel.rect.bottom + 8;
    b.style.left = `${x}px`;
    b.style.top = `${y}px`;
  }, 200);
});
$("viewerContainer").addEventListener("scroll", () => { $("askSel").hidden = true; }, { passive: true });
$("askSel").addEventListener("mousedown", (e) => e.preventDefault());
$("askSel").addEventListener("click", () => {
  const sel = pageSelection();
  $("askSel").hidden = true;
  if (!sel) return;
  setSelection(sel.text);
  window.getSelection().removeAllRanges();
  openPanel("ask");
});

// -- quiz ----------------------------------------------------------------------

let quizItem = null;

async function nextQuestion() {
  const card = $("quizCard");
  card.replaceChildren(statusEl("Picking a question…"));
  let it;
  try {
    it = await api(`quiz?scope=${$("quizScope").value}&page=${pdf ? viewer.currentPageNumber - 1 : ""}`);
  } catch (err) {
    card.replaceChildren(errorEl(err.message));
    return;
  }
  quizItem = it;
  if (it.empty) {
    $("quizCount").textContent = "";
    const box = el("div", "hint");
    if (!it.any_read) {
      box.textContent = "Quiz questions come from the pages you've read. Read a few pages first: "
        + "a page counts as read after about 20 seconds on it.";
    } else {
      box.textContent = "You haven't read anything in this chapter yet. ";
      box.append(button("Use everything I've read", () => {
        $("quizScope").value = "all";
        nextQuestion();
      }, "btn"));
    }
    card.replaceChildren(box);
    return;
  }
  $("quizCount").textContent = `${it.unasked} new of ${it.pool}`;
  const cites = new Map(it.citations.map((c) => [c.unit, c]));
  const quiz = el("div", "quiz");
  const from = el("div", "from", "From ");
  from.append(citeButton(it.source, `§${it.source.section} ${it.source.title}, p. ${it.source.page}`));
  const answer = el("textarea");
  answer.placeholder = features.check ? "Your answer (or just think it through, then show the answer)"
    : "Think it through or jot it down, then show the answer";
  const result = el("div");
  const actions = el("div", "actions");
  const done = () => { for (const b of actions.querySelectorAll("button")) b.disabled = true; answer.readOnly = true; };
  const reference = () => {
    const ref = el("div", "reference");
    ref.append(el("h4", null, "The book's answer"), mdEl(it.answer, cites, "body"));
    ref.addEventListener("click", (e) => {
      const c = e.target.closest(".cite[data-page]");
      if (c) goToPage(+c.dataset.page);
    });
    return ref;
  };
  const next = () => el("div", "actions");
  const check = async () => {
    const text = answer.value.trim();
    if (!text) return reveal();
    done();
    result.replaceChildren(statusEl("Checking your answer…"));
    try {
      await stream("quiz/check", { qa_id: it.qa_id, answer: text }, (ev) => {
        if (ev.type === "status") result.replaceChildren(statusEl(ev.text));
        else if (ev.type === "error") result.replaceChildren(errorEl(ev.text));
        else if (ev.type === "grade") {
          const label = { correct: "Correct", partly: "Partly right", incorrect: "Not quite" }[ev.verdict];
          const row = next();
          row.append(button("Next question", nextQuestion, "primary"));
          result.replaceChildren(el("span", `verdict ${ev.verdict}`, label), mdEl(ev.feedback), reference(), row);
        }
      });
    } catch (err) {
      result.replaceChildren(errorEl(err.message));
    }
  };
  const reveal = () => {
    done();
    const row = next();
    const grade = (r) => () => {
      api("quiz/result", { qa_id: it.qa_id, result: r }).catch(() => {});
      nextQuestion();
    };
    row.append(el("span", "muted", "Did you get it?"), button("Yes", grade("correct"), "btn"),
      button("Partly", grade("partly"), "btn"), button("No", grade("incorrect"), "btn"));
    result.replaceChildren(reference(), row);
  };
  answer.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) (features.check ? check : reveal)();
  });
  if (features.check) actions.append(button("Check my answer", check, "primary"));
  actions.append(button("Show answer", reveal, features.check ? "btn" : "primary"),
    button("Skip", () => {
      api("quiz/result", { qa_id: it.qa_id, result: "skipped" }).catch(() => {});
      nextQuestion();
    }, "btn"));
  quiz.append(from, mdEl(it.question, cites, "question"), answer, actions, result);
  card.replaceChildren(quiz);
  answer.focus();
}
$("quizScope").addEventListener("change", nextQuestion);

// -- reading history menu ------------------------------------------------------

function toggleMenu(on = $("progressMenu").hidden) {
  $("progressMenu").hidden = !on;
  $("progressBtn").setAttribute("aria-expanded", String(on));
}
$("progressBtn").addEventListener("click", (e) => {
  e.stopPropagation();
  toggleMenu();
});
document.addEventListener("click", (e) => {
  if (!$("progressMenu").hidden && !e.target.closest("#progressMenu")) toggleMenu(false);
});
$("markBtn").addEventListener("click", async () => {
  toggleMenu(false);
  const n = viewer.currentPageNumber;
  if (!confirm(`Count everything up to page ${labels?.[n - 1] ?? n} as read? Answers and quizzes can then use all of it.`)) return;
  setState(await api("mark", { through: n - 1 }));
  quizItem = null;
});
$("resetBtn").addEventListener("click", async () => {
  toggleMenu(false);
  if (!confirm("Forget which pages you've read, your questions and your quiz results?")) return;
  setState(await api("reset", {}));
  dwell.clear();
  quizItem = null;
  if (currentTab === "quiz") nextQuestion();
});

// -- keyboard ------------------------------------------------------------------

document.addEventListener("keydown", (e) => {
  const mod = e.ctrlKey || e.metaKey;
  if (mod && e.key === "f") {
    e.preventDefault();
    $("findInput").focus();
    $("findInput").select();
    return;
  }
  if (mod && (e.key === "=" || e.key === "+")) { e.preventDefault(); zoom(1); return; }
  if (mod && e.key === "-") { e.preventDefault(); zoom(-1); return; }
  if (mod && e.key === "0") { e.preventDefault(); setZoom("auto"); return; }
  if (mod || e.altKey || e.target.closest?.("input, textarea, select")) return;
  if (e.key === "ArrowLeft") { e.preventDefault(); viewer.previousPage(); }
  else if (e.key === "ArrowRight") { e.preventDefault(); viewer.nextPage(); }
  else if (e.key === "Escape" && !$("panel").hidden) closePanel();
});

// The phone's back button: close what is open before leaving the book.
window.autobookBack = () => {
  if (!$("progressMenu").hidden) toggleMenu(false);
  else if (!$("panel").hidden) closePanel();
  else if (!$("outline").hidden) toggleOutline(false);
  else return false;
  return true;
};

// -- go ------------------------------------------------------------------------

if (pref("outline", "")) toggleOutline(true);
const tab = pref("panel", "");
if (tab) openPanel(tab);
start();
