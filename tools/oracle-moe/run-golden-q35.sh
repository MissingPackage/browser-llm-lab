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
# tag del modello nel nome file (it.11: il 9B non deve sovrascrivere il 4B)
TAG=$(basename "$MODEL" | sed -E 's/^Qwen3\.[56]-//; s/-.*$//' | tr 'A-Z' 'a-z')
# CORPUS_DIR: smoke (corpus-q35) o full (corpus) — it.15
CORPUS_DIR=${CORPUS_DIR:-corpus}
SUFFIX=full; if [ "$CORPUS_DIR" != "corpus" ]; then SUFFIX=smoke; fi
OUT="$OUTDIR/golden-q35-${TAG}-${SUFFIX}-$DATE.json"
CORPUS_HASH=$(cat "$HERE"/$CORPUS_DIR/*.txt | sha256sum | cut -d' ' -f1)

ARGS=()
for f in "$HERE"/$CORPUS_DIR/*.txt; do ARGS+=(--prompt "$f"); done

"$HERE/golden" --model "$MODEL" "${ARGS[@]}" \
  --threads "${THREADS:-16}" --n-predict "${NPREDICT:-128}" --top-k 32 \
  --out "$OUT" \
  --corpus-hash "$CORPUS_HASH" --gguf-sha256 "$GGUF_SHA" --commit "$COMMIT"
echo "exit=0 out=$OUT"
