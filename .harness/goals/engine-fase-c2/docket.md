# Docket — engine-fase-c2 (decisioni PI pendenti)

1. ~~**PLAN-CHECK**~~ RISOLTO (2026-07-31, ruling PI in chat: "Approvo,
   parti"). Decomposizione approvata senza modifiche; iterazione 1 = fasi
   1+2 in parallelo. Testo originale:

   **PLAN-CHECK** (iteration 0, 2026-07-31): approvazione di PHASES.md prima
   dell'iterazione 1. Il contratto GOAL.md è già approvato in chat (con
   emendamento non-regressione); qui si approva la decomposizione: 7 fasi,
   gruppo parallelo A (floor+golden ‖ spec), fasi 3-6 gated dal ruling di
   spec, gate tok/s hard in fase 6 (decode ≥13.4 / prefill ≥56.6 = oracolo
   CPU C1) + non-regressione Qwen.

2. **RULING RICHIESTO — spec C2** (2026-07-31, fase 2): approvazione di
   `docs/superpowers/specs/2026-07-31-engine-fase-c2-design.md`. Decisioni
   proposte da ratificare (dettaglio in spec §10):
   (a) il GGUF "Q4_0" è a quant MISTA (FINDING dal dump: ffn_down_exps
       Q4_1, shexp Q5_K/Q6_K, output Q6_K, proiettori MLA Q8_0) ⇒ si
       implementano i kernel dequant Q4_1/Q5_K/Q6_K; NIENTE repack (la
       requant romperebbe la conformance con l'oracolo sullo stesso file);
   (b) MLA in formulazione ABSORBED nel motore (cache 576/token/layer =
       54 KB/token f16, quella di direction §3); la naive solo in cpuref
       f64 — supera l'[ASSUMED naive-first] del contratto (la naive ha una
       cache 18× e non sta nel budget);
   (c) residenza minima: il GGUF vive in OPFS (copiato al primo load) e i
       miss expert leggono da lì — DEVIAZIONE dichiarata dalla lettera del
       contratto ("staging RAM", "OPFS-backed experts" a must-docket):
       15.6 GB di ArrayBuffer in un tab su host 31 GB non sono credibili;
       niente policy/tier/prefetch su OPFS (quello resta C3);
   (d) policy cache VRAM = LRU pura (a budget 87% il simulatore C1 dà
       96.4% decode hit; tuned/prefetch sono C3);
   (e) soglie: conformance argmax ≥99% vs cpuref-f64 E ≥97% vs golden
       (gate doppio, landmine oracolo-q8); routing set-match ≥99% vs
       traccia C1; bench = protocollo B2;
   (f) timebox: 4 iterazioni fase 4 (MLA) + 4 fase 5 (MoE+residenza);
       sforamento ⇒ docket con analisi.

Costanti di gate (fase 1 le formalizza nel report):
- Floor tok/s da `results/engine/moe-oracle/llama-bench-glm47flash-q4_0-2026-07-30.json`:
  decode 13.43 ±0.10 tok/s (n_gen 64), prefill 56.58 ±3.74 tok/s (n_prompt 512),
  llama.cpp 5f55650 CPU-only, i9-14900HX 16 thread.
