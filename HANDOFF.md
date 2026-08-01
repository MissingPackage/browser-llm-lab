# HANDOFF — browser-llm-lab   (updated 2026-08-01, session 17 — goal engine-fase-c2 CHIUSO)

## 1. Next decidable

**Goal C3 — setup del contratto (decisione PI, non del loop).** C2 è CHIUSO
(deroga item 6a: correttezza + residenza dimostrate, prestazione sotto il
floor con attribuzione). C3 ha già tutto per partire:
- **Gate d'ingresso**: decode ≥13.43 / prefill ≥56.58 tok/s (floor oracolo
  CPU C1 — la deroga lo ha spostato da C2 a C3; il confronto con la funzione
  obiettivo ~30 tok/s resta la stella polare, gap 6.5×).
- **Baseline misurata**: decode 4.64 tok/s = 215.5 ms/token, scomposti in
  docket C2 item 8 (56.1 stallo residenza + 158.9 struttura).
- **Leve dimensionate, in ordine valore/rischio** (docket C2 item 8):
  (1) repack all'import (−42.5 ms/token), (2) eliminazione/batching dei 46
  sync router (il collo: senza, tetto ~7 tok/s), (3) prefill batched M>1
  (capacità nuova — il gate prefill è irraggiungibile senza), (4) prefetch
  LOOKA (residuo sul dev-box; decisivo su mobile/ctx lunghi).
- Perimetro C3 da contratto C1/direction: slab+tier+AUTOPIN+PILOT-real,
  modello di banda, instant-on come gate intermedio, WP banda fredda browser.
Quando vuoi: `/goal-brief` per il contratto C3.
Riancorarsi da: docket C2 item 8 (input C3 completo), direction.md §7
(fase C aggiornata), `.harness/goals/engine-fase-c2/{GOAL,journal}.md`
(emendamento 3, checklist chiusura it.12).

## 2. State delta (sessione 17 — chiusura C2)

- **it.11**: harness `glmbench` nuovo (B2 sul forward di produzione, decode
  greedy reale, telemetria per-token); gate GLM falliti con attribuzione;
  debug non-regressione Qwen (codice scagionato byte-identico, firma
  throttling host). Verifier PASS alla terza passata.
- **Post-it.11**: PI quiesce l'host (kill zen: 5g10h CPU accumulata,
  rendering loop WebRender; brief indagine CPU in
  `~/cpu-investigation/brief.md`, fuori repo). Ri-misure: **Qwen PASS tutti
  i gate (K=8 321.9 — sopra il baseline 287.5, anch'esso contaminato;
  prefill 600.2; K=1 244.0)**; GLM quiescente 4.64/5.22 — gate ancora FAIL,
  attribuzione ripulita (pack 42.5, struttura 158.9).
- **it.12 (fase 7)**: ruling PI — item 6 = DEROGA (floor → C3), item 5 =
  campione 2/8 ratificato, coda item 7 = baseline Qwen quiescente adottato
  senza clausola termica formale ("nessuno fa i benchmark con altri processi
  aperti"). Chiusura: docket item 8 (INPUT C3), GOAL emendamento 3,
  direction §7 + ledger riga paging aggiornati, checklist DONE WHEN 7/7
  (punto 5 in deroga), digest.
- Stato test: suite 220 passed + 2 skipped, `tsc --noEmit` pulito, ktest
  30/30 (da it.10). Tutto su main, push a chiusura verificata.

## 3. Cosa c'è nel motore adesso (eredità C2)

- GLM-4.7-Flash end-to-end: reader deepseek2 (quant mista validata hard),
  MLA absorbed (cache 576/token/layer), MoE router bit-fedele a
  build_moe_ffn, residenza minima OPFS→VRAM (ExpertCache LRU due
  size-class, import SHA-streaming 17.2 GB), output head Q6_K, harness
  glmroute/glmconf/glmbench + runner.
- Correttezza: argmax ≡ cpuref-f64 100% (256/256, campione ratificato);
  top-1 vs golden 98.83% full-corpus; identità naive↔absorbed in suite.
- Qwen intatto: conformance bit-identica, bench sopra il baseline nuovo.

## 4. Open threads

- **Goal C3**: aspetta il contratto (PI). Tutti gli input in docket C2 item 8.
- **Indagine CPU host**: sessione separata col brief in
  `~/cpu-investigation/`; esito rilevante per l'igiene dei bench futuri
  (la norma operativa: bench a macchina scarica).
- Goal harness stale mai chiusi: `fase-1b-matrice` (11 docket),
  `fase-2-deep-dive` (5) — igiene da /weekly-maintenance.
- Duplicato results/opfs-bench-*.json in due posizioni (segnalato, non toccato).
- Nota minore: `scripts/engine-bench.mjs` e `conformance-engine.mjs` hanno
  BASE_URL default :5173, convenzione repo è :5199 (si sovrascrive da env).

## 5. Landmines

- **Bench su questa macchina: farli a macchina SCARICA** (norma PI
  2026-08-01). Con carico utente pesante la GPU resta thermally-capped
  (osservato: 1425-1620 MHz su 3105, −8% sui path a carico sostenuto,
  path latency-bound intatti — quella firma è il discriminatore).
  Diagnostica rapida: `nvidia-smi --query-gpu=temperature.gpu,clocks.sm,
  power.draw,utilization.gpu`.
- Oracolo CPU quantizza le attivazioni q8: MAI gate diretti engine-vs-oracolo
  su selezioni near-tie; il confronto giusto è vs cpuref-f64; golden ≥97%
  assorbe il drift. maxAbsDeltaLogit è metrica di SCALA: mai gate.
- VRAM 16.4 GB: slab 12 GiB + head OK a ctx corto (KV 54 KB/token); ctx 6k
  ⇒ KV 361 MB ⇒ OOM. Budget slab ctx-aware in C3.
- `/tmp` è tmpfs 16 GB: profilo Chrome con OPFS 17 GB in
  `~/.cache/blab-glmroute-profile` (E2E_PROFILE); import skip su size-match.
- mlaAttnDecode: ctx>4k richiede maxComputeWorkgroupStorageSize 32 KB
  negoziato (fail-fast in createGlmModel).
- Runner in background: NIENTE pipe su tail/grep (maschera exit code);
  output diretto su file.
- Mini-modello sintetico ktest: NON estendere in profondità (overflow f32).
- Storiche: error scope su ogni submit; vite :5199; f32-first Chrome/Linux;
  `pos === kvLen` hard; var WGSL nei loop da azzerare; mai mapAsync su
  buffer di submit non emesso.

## 6. Docket (user decisions pending)

- **Nessun item C2 aperto** (1-8 tutti risolti o informativi; item 8 =
  input C3, non richiede decisione).
- Prossima decisione PI: contratto goal C3 (quando vuoi).
- fase-1b-matrice (11 item) e fase-2-deep-dive (5 item): stale, da triage
  weekly-maintenance.
