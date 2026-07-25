#!/usr/bin/env bash
# Lancia Chrome branded per i bench manuali: profilo dedicato + flag minimo.
# NON usare chrome://flags nel profilo quotidiano (enable-vulkan corrompe il
# compositing su NVIDIA/Wayland; force-enable-webgpu-interop crasha all'avvio).
exec google-chrome \
  --user-data-dir="$HOME/.local/share/blab-bench" \
  --ignore-gpu-blocklist \
  "${1:-http://localhost:5173}"
