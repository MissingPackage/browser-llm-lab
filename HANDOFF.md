# HANDOFF — browser-llm-lab   (updated 2026-07-30, session 12)

## 1. Next decidable

**Goal `engine-fase-b1`, FASE 4: rollback KV — crop(P) con length pointer** — su
branch `engine/fase-b1`. Riancorarsi da: `.harness/goals/engine-fase-b1/{GOAL,PHASES,
docket,journal}.md` + spec `docs/superpowers/specs/2026-07-29-engine-fase-b1-design.md`
§Crop (contratto: `kvLen` interno, `crop(toLen)` zero-GPU, `pos===kvLen` hard su
forwardToken, `reset()`≡`crop(0)`; verificare che TUTTI i call-site interni usino il
pointer — conformance/bench/diag ripartono da pos 0). Done-when: unit CPU-side sulla
semantica crop verdi; report JSON in `results/engine/` da run meccanico "genera N,
crop a P, rigenera" vs run fresco: sequenze IDENTICHE, exit 0 (`scripts/kv-rollback.mjs`
da creare). Poi fase 5 (OPFS) → 6 (bench+chiusura).

## 2. State delta (sessione 12, 2026-07-30)

- **FASE 3 DONE in 1 iterazione**: forward multi-token M≤8 conforme. Nuovi:
  `src/engine/prefillplan.ts` (chunking puro + 13 unit CI in
  `tests/engine-chunking.test.ts`), 6 kernel chunk in `kernels/wgsl.ts`,
  `prefillChunked()` in `gpuforward.ts` (zero readback, submit/64, lm_head solo
  ultima posizione, tap per chunk). Conformance gira col percorso M>1: GATE DOPPIO
  PASS 98.05% golden / 100.00% cpuref, anche con `telemetryGpu` attivo. 135/135 unit.
- **Bug WGSL/Tint trovato e fixato** (root-cause della prima conformance a 82%):
  `var` array dichiarata nel body di un loop NON viene ri-azzerata a ogni iterazione
  su Chrome/Tint ⇒ azzeramento esplicito nel GEMM chunk (vedi Landmine).
- **Sim rimossa** (spec §Struttura): `prefillBatched` + modalità `prefillsim` +
  `scripts/prefill-sim.mjs`. Nuovo scaffolding di fase (rimozione a fase 6):
  `scripts/prefill-diag.mjs` (parità seq-vs-chunked per lunghezza),
  `scripts/kernel-diag.mjs` (GEMM chunk vs dequant CPU, guardia sul bug Tint).
- Report conformance esteso: campi `telemetryGpu` e `prefill{path,mMax,submitTokens}`
  (nota verifier fase 2 recepita).

## 3. Open threads

- **Branch `engine/fase-b1` NON merged** (merge a goal chiuso, ruling permanente).
- Goal B1: fasi 4-5-6 a cascata, nessun ruling pendente.
- **B2 da ri-inquadrare al goal-brief** (docket goal B1, item 2): GPU busy ≈2.2 su
  8.1 ms/token ⇒ ~73% del decode è sync/encode, non dispatch.
- Scaffolding temporaneo da rimuovere a fase 6: knob `tsqDiag`, modalità
  `prefilldiag`/`kerneldiag` + script relativi.
- Goal harness stale mai chiusi: `fase-1b-matrice` (11 docket item),
  `fase-2-deep-dive` (5) — igiene da /weekly-maintenance.
- Sweep fase 1b (wllama/transformersjs su S22) fuori goal; GLM-5 uscito → ledger §H a v2.

## 4. Landmines

- **WGSL/Tint: mai fidarsi dell'azzeramento implicito di una `var` array dichiarata
  nel body di un loop** — su Chrome/Tint NON viene ri-azzerata a ogni iterazione
  (spec WGSL dice il contrario): azzerare esplicitamente. Colpiva solo kernel con
  >64 blocchi/riga (unico loop multi-iterazione); trovato in fase 3, guardia:
  `scripts/kernel-diag.mjs`.
- **mai mapAsync su un buffer referenziato da un submit non ancora emesso**: Dawn
  droppa l'INTERO command buffer in silenzio (tsq-diag-2026-07-29.md).
- Chrome headless Linux/NVIDIA → SwiftShader: driver Playwright con HEADED=1; Chrome
  da shell sandboxata → SwiftShader anche headed (bench col sandbox disabilitato).
- Vite: server di sessioni vecchie su :5173+ servono CODICE STALE — porta dedicata
  (`npx vite --port 5199 --strictPort`), kill a fine sessione (`pkill -f "[v]ite --port"`).
- Device senza `requiredLimits` espliciti nasce a 128 MiB binding → garbage silenzioso;
  grid > 65535/dim ⇒ submit no-op muto (uncapturederror fatale già nel motore).
- Chrome branded Linux/NVIDIA NON espone shader-f16 → f32-first; timestamp GPU
  quantizzati ~100 µs.
- I tok/s sotto patch del profiler non sostituiscono i bench (S22 −40%); telemetria
  liv.2 opt-in `telemetryGpu`, zero-overhead da spenta.
- llama.cpp SOLO oracolo (mai vendored); oracolo CPU quantizza le attivazioni (q8):
  noise floor 98.05% ⇒ gate doppio (cpuref-f64 ≥99% E golden ≥97%).
- `erasableSyntaxOnly` in tsconfig; contratto B1 (da fase 4): `pos === kvLen` hard
  su forwardToken — chi usa pos libere si rompe by design (crop/reset).

## 5. Docket (decisioni PI pendenti)

4-5. Ereditati: promozione skill `bottleneck-brainstorm`; #10-fase-1b qualityScore
     (goal evals futuro); #8 sorveglianza wllama (v3.1 ha WebGPU — rilevante per il
     benchmark pubblico).
16. Headline del benchmark pubblico (ledger §E): serve prima del goal benchmark.
17. RISOLTO (2026-07-30): ruling spec B1 (a-g in blocco) — dettaglio nel docket del
    goal, item 3; rimandi in ideas-ledger §I.
