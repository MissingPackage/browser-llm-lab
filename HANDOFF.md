# HANDOFF — browser-llm-lab   (updated 2026-07-30, session 11)

## 1. Next decidable

**Goal `engine-fase-b1`, FASE 3: forward multi-token M≤8 + parità** — in sessione
nuova, su branch `engine/fase-b1` (esiste su origin). Riancorarsi da:
`.harness/goals/engine-fase-b1/{GOAL,PHASES,docket,journal}.md` + spec APPROVATA
`docs/superpowers/specs/2026-07-29-engine-fase-b1-design.md` (§Forward multi-token =
il piano; §Soglie = i gate) + `docs/engine/ideas-ledger.md` §I (rimandi con trigger).
Done-when fase 3: `npm test` verde con unit chunking CPU-side; `conformance-engine.mjs`
exit 0 (gate doppio) col percorso prefill M>1 attivo. Le fasi 4 (crop) → 5 (OPFS) →
6 (bench+chiusura) seguono a cascata. Fase 3 è la più grossa: >4 iterazioni ⇒ split
via docket, non in silenzio.

## 2. State delta (sessione 11, 2026-07-29→30)

- **Goal `engine-fase-b1` aperto** (split PI: B1 = memoria/latenza di entrata; B2 =
  floor dispatch, goal futuro). Contratto + PHASES (6 fasi) approvati; tag
  `goal-engine-fase-b1-start` su main; lavoro su branch `engine/fase-b1`.
- **Soglia prefill fissata per simulazione** (criterio Pareto, ruling PI): **3×** vs
  baseline seq same-day. Sim committata (`results/engine/prefill-sim-4090-*.json` +
  `scripts/prefill-sim.mjs`): floor dei trucchi (no-readback + skip lm_head, zero
  kernel nuovi) = 1.53×; analitico M=8 = 5-10×; knee submit-granularità ≈ 64 token.
- **Fase 2 DONE — known-issue telemetria liv.2 RISOLTO**: era un bug NOSTRO di fase A
  (mapAsync prima del submit ⇒ Dawn droppa l'intero command buffer: corruzione + zeri).
  Fix `armTsq` post-submit in `gpuforward.ts`; matrice A/B pulita, gpuMs reali
  ~2.2 ms/token, conformance GATE DOPPIO PASS con liv.2 attivo. Nota:
  `docs/engine/tsq-diag-2026-07-29.md`.
- **Fase 1 DONE — spec B1 scritta e APPROVATA** (ruling PI 2026-07-30, decisioni a-g
  in blocco: chiave token-id, lookup v1 esatto, no logits nel checkpoint, LRU 512 MB,
  contratto hard `pos===kvLen`, soglie).
- **Registro rimandi creato**: `ideas-ledger.md` **§I** — 8 rimandi espliciti con fase
  di ripresa e trigger di riattivazione (richiesta PI: le fasi future devono ritrovarli).
- Verifier gate iterazione 2: PASS 8/8. `npm test` 122/122, tsc pulito.

## 3. Open threads

- **Branch `engine/fase-b1` NON merged** (merge a goal chiuso, ruling permanente).
- Goal B1 fasi 3-6: tutte ready/a cascata, nessun ruling pendente.
- **B2 da ri-inquadrare al goal-brief** (docket goal B1, item 2): GPU busy ≈ 2.2 su
  8.1 ms/token ⇒ ~73% del decode è sync/encode, non dispatch — la leva potrebbe non
  essere il dispatch count.
- Nota verifier per fase 3+: aggiungere campo `telemetryGpu` allo schema del report
  di conformance (gate auto-evidente dal JSON).
- La sim `prefillBatched` e i knob `tsqDiag` sono temporanei: si rimuovono quando il
  piano prefill vero li sostituisce (detto in spec §Struttura).
- Goal harness stale mai chiusi formalmente: `fase-1b-matrice` (11 docket item),
  `fase-2-deep-dive` (5) — igiene da /weekly-maintenance.
- Sweep fase 1b (wllama/transformersjs su S22) fuori goal; GLM-5 uscito → ledger §H a v2.

## 4. Landmines

- **mai mapAsync su un buffer referenziato da un submit non ancora emesso**: Dawn
  droppa l'INTERO command buffer in silenzio (salvo uncapturederror) — root-cause del
  falso "bug browser" di fase A (tsq-diag-2026-07-29.md).
- Chrome headless Linux/NVIDIA → SwiftShader: driver Playwright con HEADED=1; Chrome
  da shell sandboxata → SwiftShader anche headed (bench col sandbox disabilitato).
- Vite: server di sessioni vecchie vivi su :5173-:5177 servono CODICE STALE — per i
  run usare porta dedicata (`npx vite --port 5199 --strictPort`) e ucciderla a fine
  sessione (pattern pkill: `"[v]ite --port"`, altrimenti si auto-matcha).
- Device senza `requiredLimits` espliciti nasce a 128 MiB binding → garbage silenzioso;
  grid > 65535/dim ⇒ submit no-op muto (uncapturederror fatale già nel motore).
- Chrome branded Linux/NVIDIA NON espone shader-f16 → f32-first; timestamp GPU
  quantizzati ~100 µs.
- I tok/s sotto patch del profiler non sostituiscono i bench (S22 −40%); telemetria
  del motore zero-overhead da spenta (liv.2 opt-in `telemetryGpu`, ora funzionante).
- llama.cpp SOLO oracolo (mai vendored); oracolo CPU quantizza le attivazioni (q8):
  noise floor 98.05% ⇒ gate doppio (cpuref-f64 ≥99% E golden ≥97%).
- `erasableSyntaxOnly` in tsconfig; contratto B1: `pos === kvLen` hard su forwardToken
  (chi usa pos libere si rompe by design — usare crop/reset).

## 5. Docket (decisioni PI pendenti)

4-5. Ereditati: promozione skill `bottleneck-brainstorm`; #10-fase-1b qualityScore
     (goal evals futuro); #8 sorveglianza wllama (v3.1 ha WebGPU — rilevante per il
     benchmark pubblico).
16. Headline del benchmark pubblico (ledger §E): serve prima del goal benchmark.
17. RISOLTO (2026-07-30): ruling spec B1 (a-g in blocco) — dettaglio nel docket del
    goal, item 3; rimandi in ideas-ledger §I.
