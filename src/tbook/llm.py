"""LM Studio client (OpenAI-compatible) with a persistent response cache.

Every digest call goes through `chat`; identical requests are answered from
build/llm_cache.sqlite, which makes long stages resumable after a crash.
"""

import base64
import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from openai import OpenAI, APIConnectionError, APITimeoutError, BadRequestError, InternalServerError

from . import config

_client = None
_cache_lock = threading.Lock()
_cache_con: sqlite3.Connection | None = None


@dataclass
class Usage:
    calls: int = 0
    cached: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, prompt: int, completion: int, seconds: float) -> None:
        with self.lock:
            self.calls += 1
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.seconds += seconds

    def summary(self) -> dict:
        return {
            "calls": self.calls,
            "cached": self.cached,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "llm_seconds": round(self.seconds, 1),
        }


usage = Usage()


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=config.LMSTUDIO_URL, api_key="lm-studio", timeout=900)
    return _client


def _cache() -> sqlite3.Connection:
    global _cache_con
    if _cache_con is None:
        config.LLM_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _cache_con = sqlite3.connect(config.LLM_CACHE_PATH, check_same_thread=False, timeout=60)
        _cache_con.execute("PRAGMA journal_mode=WAL")
        _cache_con.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, model TEXT, tag TEXT, "
            "response TEXT, created_at REAL)"
        )
    return _cache_con


def _cache_get(key: str) -> str | None:
    with _cache_lock:
        row = _cache().execute("SELECT response FROM cache WHERE key=?", (key,)).fetchone()
    return row[0] if row else None


def _cache_put(key: str, model: str, tag: str, response: str) -> None:
    with _cache_lock:
        _cache().execute(
            "INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?)", (key, model, tag, response, time.time())
        )
        _cache().commit()


def cache_key(model: str, messages: list, schema: dict | None, temperature: float, max_tokens: int) -> str:
    blob = json.dumps(
        {"m": model, "msg": messages, "s": schema, "t": temperature, "n": max_tokens},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def image_part(png: Path | bytes) -> dict:
    data = png if isinstance(png, bytes) else Path(png).read_bytes()
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()}}


class LLMError(RuntimeError):
    pass


def chat(
    messages: list,
    model: str = config.DIGEST_MODEL,
    *,
    schema: dict | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    use_cache: bool = True,
    tag: str = "",
    reasoning: str | None = None,
):
    """Return the reply text, or the parsed object when `schema` is given.

    `reasoning="none"` turns off a reasoning model's thinking (Gemma 4 in LM Studio
    thinks by default and can spend the whole token budget doing so).
    """
    key = cache_key(model, messages, schema, temperature, max_tokens)
    if reasoning is not None:
        key = hashlib.sha256((key + ":reasoning=" + reasoning).encode()).hexdigest()
    if use_cache:
        hit = _cache_get(key)
        if hit is not None:
            with usage.lock:
                usage.cached += 1
            return json.loads(hit) if schema else hit

    kwargs = {}
    if reasoning is not None:
        kwargs["extra_body"] = {"reasoning_effort": reasoning}
    if schema is not None:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "result", "strict": True, "schema": schema},
        }
    last_err = None
    for attempt in range(6):
        t0 = time.time()
        try:
            resp = client().chat.completions.create(
                model=model, messages=messages, temperature=temperature if attempt < 2 else 0.3,
                max_tokens=max_tokens, **kwargs,
            )
        except (APIConnectionError, APITimeoutError, InternalServerError) as e:
            last_err = e
            time.sleep(min(60, 5 * 2**attempt))
            continue
        except BadRequestError as e:
            if "Context size has been exceeded" not in str(e):
                raise
            last_err = e                  # the shared KV cache was full: wait for other slots to finish
            time.sleep(min(60, 5 * 2**attempt))
            continue
        text = resp.choices[0].message.content or ""
        u = resp.usage
        usage.add(u.prompt_tokens if u else 0, u.completion_tokens if u else 0, time.time() - t0)
        if not text.strip() and resp.choices[0].finish_reason == "length":
            last_err = LLMError("token budget used up before an answer (reasoning ran long)")
            max_tokens = int(max_tokens * 1.5)
            continue
        if schema is not None:
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as e:
                last_err = e
                if resp.choices[0].finish_reason == "length":
                    max_tokens = int(max_tokens * 2)          # truncated JSON: give it room
                continue
            if use_cache:
                _cache_put(key, model, tag, text)
            return obj
        if use_cache:
            _cache_put(key, model, tag, text)
        return text
    raise LLMError(f"chat failed after retries: {last_err}")


def chat_stream(messages: list, model: str, *, temperature: float = 0.2, max_tokens: int = 900):
    """Yield text deltas; the final item is a dict of timing stats."""
    t0 = time.time()
    first = None
    n = 0
    stream = client().chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, stream=True,
        stream_options={"include_usage": True},
    )
    completion_tokens = None
    for ev in stream:
        if ev.usage is not None:
            completion_tokens = ev.usage.completion_tokens
        if not ev.choices:
            continue
        delta = ev.choices[0].delta.content
        if delta:
            if first is None:
                first = time.time()
            n += 1
            yield delta
    end = time.time()
    toks = completion_tokens or n
    gen_time = max(1e-6, end - (first or end))
    yield {"ttft_s": round((first or end) - t0, 2), "tokens": toks, "tok_s": round(toks / gen_time, 1)}


EMBED_PREFIX = {"document": "search_document: ", "query": "search_query: ", "clustering": "clustering: "}


def embed(texts: list[str], kind: str = "document", batch: int = 64) -> np.ndarray:
    """Unit-normalized float32 embeddings from the shared embedder."""
    prefix = EMBED_PREFIX[kind]
    out = []
    for i in range(0, len(texts), batch):
        part = [prefix + (t or " ")[:6000] for t in texts[i:i + batch]]
        for attempt in range(6):
            try:
                resp = client().embeddings.create(model=config.EMBED_MODEL, input=part)
                break
            except (APIConnectionError, APITimeoutError, InternalServerError):
                time.sleep(min(60, 5 * 2**attempt))
        else:
            raise LLMError("embedding failed after retries")
        out.extend(d.embedding for d in resp.data)
    arr = np.asarray(out, dtype=np.float32)
    if len(arr):
        arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr.reshape(len(texts), -1) if len(texts) else np.zeros((0, config.EMBED_DIM), np.float32)


def load_prompt(name: str) -> str:
    return (config.PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")
