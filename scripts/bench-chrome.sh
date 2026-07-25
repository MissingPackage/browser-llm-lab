#!/usr/bin/env bash
# Lancia Chrome branded per i bench manuali: profilo dedicato + flag minimo.
# NON usare chrome://flags nel profilo quotidiano (enable-vulkan corrompe il
# compositing su NVIDIA/Wayland; force-enable-webgpu-interop crasha all'avvio).
# --disable-gpu-sandbox: su Fedora/NVIDIA il sandbox del processo GPU nega gli
# ICD Vulkan a Dawn ("Found no drivers"); solo il GPU process resta unsandboxed,
# renderer e browser mantengono il sandbox pieno. Ritestare a ogni major di Chrome.
exec google-chrome \
  --user-data-dir="$HOME/.local/share/blab-bench" \
  --ignore-gpu-blocklist \
  --disable-gpu-sandbox \
  "${1:-http://localhost:5173}"
