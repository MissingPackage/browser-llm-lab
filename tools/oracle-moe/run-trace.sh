#!/usr/bin/env bash
# Run canonica della traccia di routing (spec engine-fase-c1). Dalla root del repo.
set -euo pipefail
HERE=$(dirname "$(realpath "$0")")
MODEL=${MODEL:-$HOME/.cache/blab-models/GLM-4.7-Flash-Q4_0.gguf}
GGUF_SHA=${GGUF_SHA:-d0bbdfcde6e323ebf90a8b9e95da57100e972be1ec6f0bfa0fad0feaa426557e}
COMMIT=$(git -C "$HOME/Projects/llama.cpp-oracle" rev-parse HEAD)
DATE=$(date -u +%Y-%m-%d)
OUT="results/engine/moe-oracle/trace-$DATE"
CORPUS_HASH=$(cat "$HERE"/corpus/*.txt | sha256sum | cut -d' ' -f1)

ARGS=()
for f in "$HERE"/corpus/*.txt; do ARGS+=(--prompt "$f"); done

RC=0
"$HERE/trace" --model "$MODEL" "${ARGS[@]}" \
  --threads "${THREADS:-16}" --n-predict "${NPREDICT:-640}" \
  --out-prefix "$OUT" \
  --corpus-hash "$CORPUS_HASH" --gguf-sha256 "$GGUF_SHA" --commit "$COMMIT" || RC=$?
[ -f "$OUT.jsonl" ] && gzip -f "$OUT.jsonl"
echo "exit=$RC out=$OUT.jsonl.gz summary=$OUT-summary.json"
exit $RC
