"""Load/unload models through the `lms` CLI so each phase gets a known setup."""

import json
import subprocess
import time

from . import config


def _lms(*args: str, check: bool = True) -> str:
    proc = subprocess.run(
        [str(config.LMS_EXE), *args], capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"lms {' '.join(args)} failed: {proc.stderr or proc.stdout}")
    return proc.stdout


def loaded() -> list[dict]:
    for attempt in range(4):
        try:
            return json.loads(_lms("ps", "--json") or "[]")
        except json.JSONDecodeError:
            return []
        except RuntimeError:              # `lms ps` occasionally exits non-zero with no output
            if attempt == 3:
                raise
            time.sleep(10 * (attempt + 1))


def ensure_loaded(model: str, ctx: int, parallel: int = 1, *, exclusive: bool = True) -> None:
    """Load `model` with this context/parallelism; unload other LLMs if `exclusive`.

    The embedder is never unloaded: it is small and every phase uses it.
    """
    current = loaded()
    for m in current:
        key = m.get("modelKey") or m.get("identifier")
        if key == model and m.get("contextLength", ctx) >= ctx and m.get("parallel", parallel) == parallel:
            if exclusive:
                _unload_others(current, keep=model)
            return
    if exclusive:
        _unload_others(current, keep=None)
    _lms("load", model, "-c", str(ctx), "--parallel", str(parallel), "--gpu", "max", "-y",
         "--identifier", model)


def ensure_embedder() -> None:
    if not any((m.get("modelKey") or m.get("identifier")) == config.EMBED_MODEL for m in loaded()):
        _lms("load", config.EMBED_MODEL, "-y", "--identifier", config.EMBED_MODEL)


def _unload_others(current: list[dict], keep: str | None) -> None:
    for m in current:
        ident = m.get("identifier") or m.get("modelKey")
        if ident in (keep, config.EMBED_MODEL) or m.get("modelKey") in (keep, config.EMBED_MODEL):
            continue
        _lms("unload", ident, check=False)
