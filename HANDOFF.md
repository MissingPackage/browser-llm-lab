# HANDOFF — browser-llm-lab   (updated 2026-07-30, session 12)

## 1. Next decidable

**GOAL ATTIVO: `engine-fase-b2` v2** (re-scope opzione (a), ruling PI 2026-07-30:
"il miglior motore in assoluto", niente fretta). Contratto v2: wall di decode al
ctx bench verso il floor GPU — leve: (1) attention decode split-context (il
kernel a 14 wg vale ~4 ms/token a ctx 570), (2) loop multi-step K (~1.5 ms di
sync), gate provvisorio decode ≥240 tok/s da fissare in spec. Fase 1 DONE
(attribuzione: quota fuori-GPU reale 20%, non 73%). Fasi 2-3 DONE (ruling spec
2026-07-30: soglia 230, token_embd su GPU, dispatch ≤100 archiviato). Kernel
split INTEGRATO (decode 248.3 a K=1, conformance identica a B1). Fase 4 DONE
(it.4): decodeBatch K≤8 con feedback token on-GPU (embedGatherQ4, token_embd
~68 MB su GPU), UNA mapAsync/batch, EOS via crop — token-identity PASS (K=8/5/1
identici su 256 vs oracolo), conformance invariata, informale K=8 ~3.46 ms/tok
(~289 tok/s). **Prossimo: fase 5** = telemetria liv.2 + profiler nel nuovo loop
(gpuMs reale non-null, submit/token ~1/K, dispatch ≤160), poi fase 6
(bench+chiusura, gate decode ≥230 + non-regressione). Nessun PI-gate pendente.
Riancorarsi da:
`.harness/goals/engine-fase-b2/{GOAL,PHASES,docket,journal}.md`,
results/engine/decode-attrib-4090-*2026-07-30*.json, tsq-diag §Conseguenze
(CORRETTA). Branch: engine/fase-b2. Altri PI-gated (non bloccanti): igiene goal
stale (fase-1b-matrice, fase-2-deep-dive) da /weekly-maintenance. Benchmark
pubblico: FUORI dai goal engine (contributo separato, repo/paper/sito propri).

## 2. State delta (sessione 12, 2026-07-30 — goal B1 completo, iterazioni 3-6)

- **Fase 3**: forward multi-token M≤8 (prefillplan puro, kernel chunk, prefillChunked
  zero-readback, submit/64, lm_head solo ultima posizione). Bug Tint #1 trovato:
  var array in loop non ri-azzerata.
- **Fase 4**: length pointer `kvLen` (kvlen.ts puro, contratto hard pos===kvLen,
  crop zero-GPU, reset≡crop(0)); rollback meccanico PASS.
- **Fase 5**: prefix-cache OPFS (kvstore.ts: envelope BKV1 + chiave token-id + LRU
  puri; I/O SyncAccessHandle budget 512 MB); readKv/writeKv; restore in worker
  nuovo token-identico.
- **Fase 6**: kernel chunk FUSI stile fase A (4 righe/wg, M srotolato in scalari
  generati; 7 dispatch/layer) ⇒ **prefill 700.5 ms vs 2410.9 seq same-day = 3.44×**
  (gate 3×, soglia 804 ms); decode **122.4 ±1.1** (≥120, = fase A); profiler
  finestra decode 0 createBindGroup / 1 submit / 123.6 dispatch per token;
  restore 106.5 ms < re-prefill 699 ms; conformance GATE DOPPIO PASS (98.05/100.00)
  anche con telemetryGpu. Bug WGSL #2 (vedi Landmine): `var stride` ridichiarata ⇒
  pipeline invalida ⇒ submit droppati in silenzio con readback STALE plausibili.
- **Cleanup**: sim prefillBatched (fase 3) e knob tsqDiag + tsq-diag.mjs (fase 6)
  RIMOSSI da spec. TENUTI come harness di parità permanenti: `prefill-diag.mjs`
  (cold-start golden + confronto KV riga-per-riga), `kernel-diag.mjs` (GEMM chunk
  vs dequant CPU), gate `kv-rollback.mjs` e `prefix-cache.mjs`.
- Report estesi: conformance {telemetryGpu, prefill}; bench schemaVersion 3 con
  {prefillMs, seqBaselineMs, speedupVsSeq, gatePass}; engine-prof a finestra decode.

## 3. Open threads

- **Merge su main**: fatto a goal chiuso dopo verifier PASS (ruling permanente) —
  se questa riga è presente e il merge non risulta in `git log main`, il merge è
  stato interrotto: rifarlo (fast-forward di engine/fase-b1).
- **B2 plan-check** (PI-gated): PHASES.md da approvare (docket B2 item 1);
  contratto già approvato in chat 2026-07-30.
- Goal harness stale mai chiusi: `fase-1b-matrice` (11 docket item),
  `fase-2-deep-dive` (5) — igiene da /weekly-maintenance.
- Sweep fase 1b (wllama/transformersjs su S22) fuori goal; GLM-5 uscito → ledger §H a v2.
- Rimandi di fase registrati in ideas-ledger §I (longest-prefix, chiave testuale,
  logits nel checkpoint, scoring eviction, fusioni, parametro M, telemetria liv.3).

## 4. Landmines

- **Pipeline WebGPU invalida = submit droppati IN SILENZIO con readback STALE
  plausibili**: un errore di compilazione WGSL (es. ridichiarazione di variabile
  nello stesso scope — kernel generati che emettono lo stesso snippet due volte!)
  invalida la pipeline; ogni submit che la contiene viene droppato da Dawn; le
  mapAsync successive risolvono coi dati del run precedente ⇒ "parità perfetta"
  fasulla. Difese nel repo: error scope validation/oom come CONTRATTO di
  prefillChunked; diag a COLD-START (mai fidarsi di un confronto dopo un run che
  ha già scritto gli stessi buffer); gli uncapturederror arrivano come pageerror
  ASINCRONI — mai troncarli via `tail`/filtri.
- **WGSL/Tint: mai fidarsi dell'azzeramento implicito di una `var` array dichiarata
  nel body di un loop** — non viene ri-azzerata a ogni iterazione: azzerare
  esplicitamente (guardia: `scripts/kernel-diag.mjs`).
- **mai mapAsync su un buffer referenziato da un submit non ancora emesso**: Dawn
  droppa l'INTERO command buffer in silenzio (tsq-diag-2026-07-29.md).
- Chrome headless Linux/NVIDIA → SwiftShader: driver Playwright con HEADED=1; Chrome
  da shell sandboxata → SwiftShader anche headed (bench col sandbox disabilitato).
- Vite: server di sessioni vecchie su :5173+ servono CODICE STALE — porta dedicata
  (`npx vite --port 5199 --strictPort`), kill a fine sessione (`pkill -f "[v]ite --port"`).
- Device senza `requiredLimits` espliciti nasce a 128 MiB binding → garbage silenzioso;
  grid > 65535/dim ⇒ submit no-op muto.
- Chrome branded Linux/NVIDIA NON espone shader-f16 → f32-first; timestamp GPU
  quantizzati ~100 µs.
- llama.cpp SOLO oracolo (mai vendored); oracolo CPU quantizza le attivazioni (q8):
  noise floor 98.05% ⇒ gate doppio (cpuref-f64 ≥99% E golden ≥97%).
- `erasableSyntaxOnly`; contratto `pos === kvLen` hard su forwardToken/prefillChunked
  (pos libere ⇒ throw; usare crop/reset).

## 5. Docket (decisioni PI pendenti)

4-5. Ereditati: promozione skill `bottleneck-brainstorm`; #10-fase-1b qualityScore
     (goal evals futuro); #8 sorveglianza wllama (v3.1 ha WebGPU — rilevante per il
     benchmark pubblico).
16. Headline del benchmark pubblico (ledger §E): serve prima del goal benchmark.
    Nota 2026-07-30: il benchmark è un CONTRIBUTO SEPARATO dall'engine (ruling PI:
    repo/paper/sito propri, confronto imparziale che include il nostro engine;
    split repo rimandato alla pubblicazione) — ci si torna su iniziativa PI.
18. ~~B2 goal-brief da approvare~~ RISOLTO (2026-07-30, ruling PI in chat:
    contratto B2 approvato col gate a quota fuori-GPU + guard-rail gpuBusy;
    goal `engine-fase-b2` aperto). Resta il plan-check (docket B2 item 1).
