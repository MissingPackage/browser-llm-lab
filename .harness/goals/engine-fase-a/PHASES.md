# PHASES — engine-fase-a

Decomposizione del contratto (GOAL.md) in fasi loop-runnable. Cambiabile dopo
l'iterazione 0 solo via docket. Le soglie marcate "da spec" vengono fissate in fase 1
e da lì in poi sono vincolanti per i verifier.

| # | phase | done-when (mechanical) | authority delta | owns | status |
|---|-------|------------------------|-----------------|------|--------|
| 1 | **Spec fase A** — IR/piano statico, formato interno pesi (da GGUF q4), contratto tap, telemetria, soglie conformance (top-1/KL, overhead telemetria), struttura `src/engine/` | `docs/superpowers/specs/*engine-fase-a-design.md` esiste con le sezioni contratto grep-abili ("Piano statico", "Formato pesi", "Tap", "Telemetria", "Soglie di conformance", "Struttura"); entry di richiesta ruling appesa a `docket.md` | none | `docs/superpowers/specs/`, `.harness/goals/engine-fase-a/` | **done** (2026-07-29; ruling pendente, docket 2) |
| 2 | **Misure di contorno** — tool banda OPFS + run; M2 (contatore call-site `flushCommands` sul bundle WebLLM) | tool in `.harness/tools/opfs-bench.mjs` (o pagina dedicata) + ≥1 JSON in `results/opfs-bench/`; `docs/engine/estimates.md` contiene i conteggi per call-site di M2 | none | `.harness/tools/`, `results/opfs-bench/`, `docs/engine/estimates.md` | **done** (2026-07-29) |
| 3 | **GGUF load + layout interno + oracolo** — parser GGUF (subset Qwen2.5-0.5B q4), conversione al layout interno, golden logits generati da llama.cpp locale, corpus fisso committato | `npm test` verde con unit del parser su fixture committata; `scripts/gen-golden*` eseguito → golden file in `results/engine/golden/`; corpus in repo | none | `src/engine/`, `tests/`, `scripts/`, `results/engine/golden/` | **done** (2026-07-29, su branch; avviata pre-ruling — razionale nel docket 2) |
| 4 | **Forward WebGPU v0 + parità** — kernel minimi (GEMV/GEMM dequant-fusa q4, RMSNorm, RoPE, attention, sampling), dispatch naive, correttezza prima della velocità | `scripts/conformance-engine.mjs` exit 0 sulla 4090 headed (top-1 ≥99% su ≥512 token vs golden — soglia finale da spec); `npm test` verde | none | `src/engine/**`, `tests/`, `scripts/` | **done** (2026-07-29: conformance 98.05%, mismatch set = cpuref-f64) |
| 5 | **Piano statico L1-L3 + telemetria + tap** — bind group al load, un submit/token, fusione ≤100 dispatch/token, telemetria nativa, tap hidden-states | run profiler sulla pagina bench motore: in finestra decode `createBindGroup=0`, `submit/token=1`, `dispatch/token≤100`; JSON bench con sezione telemetry; test overhead-spenta <1% verde; test strutturale tap verde; conformance ancora exit 0 | none | `src/engine/**`, pagina bench/prof del motore | **done** (2026-07-29: bindGroup=0 ✓, submit=1 ✓, dispatch=123 ≤130 [ruling docket 4] ✓; telemetria liv.1 ✓ overhead 0.24%, liv.2 opt-in per known-issue; tap ✓) |
| 6 | **First-light + chiusura** — bake-off contro WebLLM stesso giorno stesso device | `results/engine/` contiene JSON motore + JSON baseline WebLLM (stesso giorno, q4f32_1, protocollo warmup+3 repliche) con `decodeToksPerSec.mean` motore > baseline; entry "candidatura merge engine/fase-a" appesa al docket; HANDOFF aggiornato | none | `results/engine/`, `HANDOFF.md`, docket | **done** (2026-07-29: FIRST LIGHT 122.0-123.6 vs WebLLM 116.5-117.8 same-day; gate doppio PASS [ruling docket 3]; merge autorizzato [docket 5]) |

Note di spine:
- Parallel-group A (fasi 1+2): owns disgiunti, nessuna dipendenza reciproca.
- La sequenza 3→4→5 separa deliberatamente correttezza (parità con dispatch naive) da
  velocità (L1-L3): così il valore delle leve è misurato SUL NOSTRO motore come delta
  4→5, non solo stimato da WebLLM.
- Fase 4 è la più grossa (kernel WGSL): se supera 4 iterazioni, split via docket, non
  in silenzio.
- Il merge su main NON è una fase: è docket-born by design (GOAL.md, must-docket).
