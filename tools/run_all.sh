#!/usr/bin/env bash
# Everything after `tbook pages`: extraction, digest and the leak gate, both books.
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
TBOOK=(.venv/Scripts/python.exe -m tbook.cli)
stamp() { echo "== $(date +%T) $*"; }

bash tools/run_full_extraction.sh
stamp "digest"
"${TBOOK[@]}" digest mr
"${TBOOK[@]}" digest fob
stamp "leak gate"
"${TBOOK[@]}" eval mr --rebuild
"${TBOOK[@]}" eval fob --rebuild
stamp "all done"
