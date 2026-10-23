#!/usr/bin/env bash
# Creates one isolated venv per PDF converter (their torch/model deps conflict).
# Usage: bash tools/setup_converters.sh [marker|mineru|docling ...]
set -u
cd "$(dirname "$0")/.."
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
targets=("$@")
[ ${#targets[@]} -eq 0 ] && targets=(marker mineru docling)

for name in "${targets[@]}"; do
  venv="tools/venvs/$name"
  py="$venv/Scripts/python.exe"
  log="logs/setup_$name.log"
  echo "=== $name ($(date)) ===" | tee "$log"
  [ -x "$py" ] || uv venv "$venv" --python 3.12 >>"$log" 2>&1
  case "$name" in
    marker)  pkgs=(marker-pdf) ;;
    mineru)  pkgs=("mineru[core]") ;;
    docling) pkgs=(docling) ;;
    *) echo "unknown converter $name" | tee -a "$log"; continue ;;
  esac
  uv pip install --python "$py" "${pkgs[@]}" torch torchvision \
    --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match >>"$log" 2>&1
  "$py" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" >>"$log" 2>&1
  echo "=== $name done: exit $? ===" | tee -a "$log"
done
