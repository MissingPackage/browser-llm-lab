# Docket — engine-fase-c2 (decisioni PI pendenti)

1. **PLAN-CHECK** (iteration 0, 2026-07-31): approvazione di PHASES.md prima
   dell'iterazione 1. Il contratto GOAL.md è già approvato in chat (con
   emendamento non-regressione); qui si approva la decomposizione: 7 fasi,
   gruppo parallelo A (floor+golden ‖ spec), fasi 3-6 gated dal ruling di
   spec, gate tok/s hard in fase 6 (decode ≥13.4 / prefill ≥56.6 = oracolo
   CPU C1) + non-regressione Qwen.

Costanti di gate (fase 1 le formalizza nel report):
- Floor tok/s da `results/engine/moe-oracle/llama-bench-glm47flash-q4_0-2026-07-30.json`:
  decode 13.43 ±0.10 tok/s (n_gen 64), prefill 56.58 ±3.74 tok/s (n_prompt 512),
  llama.cpp 5f55650 CPU-only, i9-14900HX 16 thread.
