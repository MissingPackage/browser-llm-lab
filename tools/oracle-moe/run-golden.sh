#!/usr/bin/env bash
# Run canonica dei golden logits GLM (goal engine-fase-c2, fase 1). Dalla root del repo.
set -euo pipefail
HERE=$(dirname "$(realpath "$0")")
MODEL=${MODEL:-$HOME/.cache/blab-models/GLM-4.7-Flash-Q4_0.gguf}
GGUF_SHA=${GGUF_SHA:-d0bbdfcde6e323ebf90a8b9e95da57100e972be1ec6f0bfa0fad0feaa426557e}
COMMIT=$(git -C "$HOME/Projects/llama.cpp-oracle" rev-parse HEAD)
DATE=$(date -u +%Y-%m-%d)
OUTDIR="results/engine/golden/glm47flash"
mkdir -p "$OUTDIR"
OUT="$OUTDIR/golden-glm47flash-q4_0-$DATE.json"
CORPUS_HASH=$(cat "$HERE"/corpus/*.txt | sha256sum | cut -d' ' -f1)

ARGS=()
for f in "$HERE"/corpus/*.txt; do ARGS+=(--prompt "$f"); done

"$HERE/golden" --model "$MODEL" "${ARGS[@]}" \
  --threads "${THREADS:-16}" --n-predict "${NPREDICT:-128}" --top-k 32 \
  --out "$OUT" \
  --corpus-hash "$CORPUS_HASH" --gguf-sha256 "$GGUF_SHA" --commit "$COMMIT"
echo "exit=0 out=$OUT"
