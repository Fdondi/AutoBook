#!/usr/bin/env bash
# Full-book extraction for both books: four transcriptions per page, then
# consensus + checks + repair. Resumable: LLM calls are cached, and convert
# only stores finished batches. Back matter (bibliography, index) is skipped.
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
TBOOK=(.venv/Scripts/python.exe -m tbook.cli)     # not `uv run`: a re-sync would kill a long run
MR_PAGES="0-17,154-615"
FOB_PAGES="0-542"
stamp() { echo "== $(date +%T) $*"; }

stamp "glyph-hinted VLM transcriptions"
"${TBOOK[@]}" convert mr --methods vlm_ml_fast,vlm_ml --pages "$MR_PAGES"
"${TBOOK[@]}" convert fob --methods vlm_ml_fast,vlm_ml --pages "$FOB_PAGES"
stamp "MinerU and Marker"
"${TBOOK[@]}" convert mr --methods mineru,marker --pages "$MR_PAGES"
"${TBOOK[@]}" convert fob --methods mineru,marker --pages "$FOB_PAGES"
stamp "finalize (consensus, checks, repair)"
"${TBOOK[@]}" finalize mr --pages "0-615"
"${TBOOK[@]}" finalize fob --pages "$FOB_PAGES"
stamp "done"
