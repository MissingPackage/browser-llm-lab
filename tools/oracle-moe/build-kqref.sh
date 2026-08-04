#!/usr/bin/env bash
# Build del generatore di vettori di riferimento K-quant (fase 4c slice A).
set -euo pipefail
LLAMA=${LLAMA:-$HOME/Projects/llama.cpp-oracle}
HERE=$(dirname "$(realpath "$0")")
gcc -O2 -std=c11 "$HERE/kqref.c" \
  -I "$LLAMA/ggml/include" -I "$LLAMA/ggml/src" \
  -L "$LLAMA/build/bin" -lggml-base \
  -Wl,-rpath,"$LLAMA/build/bin" \
  -o "$HERE/kqref"
echo "built: $HERE/kqref (oracolo: $LLAMA)"
