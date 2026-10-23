#!/usr/bin/env bash
# Resume after the 2026-10-22 run: MR's vlm_ml pass (skipped by a transient `lms ps`
# failure), finalize both books (MR's repair overflowed the shared KV cache), then
# digest and the leak gate. MinerU/Marker output is already stored.
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
TBOOK=(.venv/Scripts/python.exe -m tbook.cli)
stamp() { echo "== $(date +%T) $*"; }

stamp "mr vlm_ml"
"${TBOOK[@]}" convert mr --methods vlm_ml --pages "0-17,154-615"
stamp "finalize"
"${TBOOK[@]}" finalize mr --pages "0-615"
"${TBOOK[@]}" finalize fob --pages "0-542"
stamp "digest"
"${TBOOK[@]}" digest mr
"${TBOOK[@]}" digest fob
stamp "leak gate"
"${TBOOK[@]}" eval mr --rebuild
"${TBOOK[@]}" eval fob --rebuild
stamp "all done"
