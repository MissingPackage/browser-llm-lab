#!/usr/bin/env bash
# Run canonica dei golden logits Qwen3.5-4B (goal engine-fase-q1, fase 4).
# Pattern di run-golden.sh (GLM) con provenance piena: SHA GGUF, commit
# dell'oracle tree, hash del corpus. Dalla root del repo.
set -euo pipefail
HERE=$(dirname "$(realpath "$0")")
MODEL=${MODEL:-$HOME/.cache/blab-models/q35/Qwen3.5-4B-Q4_0.gguf}
GGUF_SHA=${GGUF_SHA:-298fcb5fe7a77ccc79745ae24751560c5ac56874caff4bb39b1f2055bd72b8bb}
COMMIT=$(git -C "$HOME/Projects/llama.cpp-oracle" rev-parse HEAD)
DATE=$(date -u +%Y-%m-%d)
OUTDIR="results/engine/golden/q35"
mkdir -p "$OUTDIR"
OUT="$OUTDIR/golden-q35-4b-full-$DATE.json"
CORPUS_HASH=$(cat "$HERE"/corpus/*.txt | sha256sum | cut -d' ' -f1)

ARGS=()
for f in "$HERE"/corpus/*.txt; do ARGS+=(--prompt "$f"); done

"$HERE/golden" --model "$MODEL" "${ARGS[@]}" \
  --threads "${THREADS:-16}" --n-predict "${NPREDICT:-128}" --top-k 32 \
  --out "$OUT" \
  --corpus-hash "$CORPUS_HASH" --gguf-sha256 "$GGUF_SHA" --commit "$COMMIT"
echo "exit=0 out=$OUT"
