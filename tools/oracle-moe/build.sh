#!/usr/bin/env bash
# Build del tool di traccia contro la build CPU-only dell'oracolo (zero patch upstream).
set -euo pipefail
LLAMA=${LLAMA:-$HOME/Projects/llama.cpp-oracle}
HERE=$(dirname "$(realpath "$0")")
g++ -O2 -std=c++17 "$HERE/trace.cpp" \
  -I "$LLAMA/include" -I "$LLAMA/ggml/include" \
  -L "$LLAMA/build/bin" -lllama -lggml -lggml-base \
  -Wl,-rpath,"$LLAMA/build/bin" \
  -o "$HERE/trace"
echo "built: $HERE/trace (oracolo: $LLAMA)"
