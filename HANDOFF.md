# HANDOFF — browser-llm-lab   (updated 2026-08-01, session 18 — contratti C3a/C3b approvati)

## 1. Next decidable

**Fase 3 in corso — repack all'import.** Spec approvata (docket item 6), fasi
3-6 sbloccate. Della fase 3 è FATTA la **negoziazione dei limiti WebGPU**
(`src/engine/gpulimits.ts`, applicato ai 4 worker GLM); **manca il repack**:
secondo file OPFS `*.slabs.bin` da 15.68 GB con header (magic + versione di
layout + SHA del sorgente), invalidazione su mismatch, scrittura su temporaneo
+ rename, e `ExpertCache.ensure` che salta `packExpertSlab` quando lo slab
arriva già impacchettato. Obiettivo: pack CPU da 41.4 a <1.0 ms/token.

**Decisione PI aperta e non bloccante: docket item 8** — pagare ~0.67 GiB di
qualità (5% del parco expert, quant asimmetrica sui più freddi) per ottenere la
residenza totale ed eliminare il drain della coda. È una scelta sulla funzione
obiettivo (capienza vs velocità), non di implementazione.

**Contesto — le due assunzioni del contratto smentite dalla misura.**
1. Le tre leve originali **non raggiungono il gate 13.43** (it.1) ⇒ emendamento 1,
   quarta leva (granularità dispatch) nel perimetro.
2. Il costo dei 46 sync **non è latenza dei submit** (che vale 0.6 ms/token) ma
   il fatto che `mapAsync` è una **barriera**: ogni layer drena la coda (it.3).
   Confermato aritmeticamente: `gpuBusy`/wall 36.4% ≈ utilizzo GPU 34.6%, e
   `gpu_idle` attivo in 34/40 campioni mentre power-cap e thermal stanno a 1/40.
   ⇒ emendamento 2, prefetch LOOKA nel perimetro (sovrappone ~54 ms/token di
   lavoro CPU al lavoro GPU).
Il pattern che risolve è provato da ORT/ggml-webgpu/MLC: binding **fisso**,
expert come **offset aritmetico** nello shader. Ma vale solo a residenza
totale ⇒ item 8.

Dettaglio della fase 1: le tre leve originali **non raggiungono il gate 13.43**.
- Wall decode 215.0 ms/token = `gpuBusy` **78.2** + stallo residenza **53.8** +
  sync/CPU **83.0** (63.6% fuori GPU). Ma i 46 readback veri costano solo
  **7.6 ms/token** (probe indipendente): il resto è latenza di submit e bolle.
- Leve 1+2 al 100% ⇒ **10.18 tok/s**; il gate vuole ≤74.46 ms/token e `gpuBusy`
  da solo ne occupa 78.2.
- **Però** la GPU è sotto-clockata dalle bolle che la leva 2 rimuove (1746 MHz
  medi su 3105, utilizzo 34.6%) ⇒ limite ottimistico **15.63 tok/s, PASS**.
  **Forbice 10.2-15.6 col gate dentro**: la leva 2 ha un payoff di secondo
  ordine mai dimensionato.
- Cross-check: 2.22 GB pesi/token su 576 GB/s ⇒ floor memory-bound 3.85
  ms/token, `gpuBusy` è **20× sopra** (1816 dispatch a 43 µs l'uno). Quarta leva
  disponibile (granularità dispatch), fuori contratto e ultima per direction §2.
- Ruling PI: quarta leva **dentro** C3a (GOAL emendamento 1, fase 4b), gate
  invariato. La fase 4 chiude con una ri-misura obbligatoria di clock e
  `gpuBusy` che dimensiona la 4b.
Aperto anche: **item 2** (clausola di fallback) — rimandato dal PI a fine fase 4,
quando la ri-misura avrà sciolto la forbice. La fase 4 ha in done-when l'obbligo
di ripresentarlo.
- **Perimetro C3a (struttura)**: repack all'import, eliminazione/batching dei
  46 sync router, prefill batched M>1. **Gate di chiusura**: decode ≥13.43 /
  prefill ≥56.58 tok/s (floor oracolo CPU C1, ereditato dalla deroga C2).
- **Doppio livello sui numeri** (ruling PI 2026-08-01, docket C3a item 1): il
  floor è gate d'ingresso, NON obiettivo. Ogni bench riporta anche il gap
  dalla soglia UX — **decode 30 tok/s (60 in thinking) e TTFT ≤4 s**. Oggi:
  4.64 tok/s (gap 6.5×) e 88 s di TTFT. L'assenza di quei numeri dal report
  è un FAIL di checklist.
- **Baseline misurata**: 215.5 ms/token = 56.1 stallo residenza + 158.9
  struttura (docket C2 item 8). Soglie confermate dal PI: sync/token ≤2,
  pack <1.0 ms/token nel path caldo.
- **C3b (paging) è CHARTERED e non avviato**: slab ctx-aware, tier.h,
  AUTOPIN, PILOT-real, modello di banda, WP banda fredda browser, instant-on
  come TTFT a freddo. Parte a C3a chiusa, dopo aver fissato i numeri di
  non-regressione che C3a lascia (docket C3b item 1).
Riancorarsi da: `.harness/goals/engine-fase-c3a/{GOAL,docket}.md`, docket C2
item 8 (input C3 completo), direction.md §2 (funzione obiettivo a due
termini) e §7 (fase C splittata).

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

- **Goal C3a**: fasi 1-2 DONE (it.1-2), fasi 3-6 gated dal ruling di spec
  (docket item 6).
- **[VERIFY] sciolto** (it.3): `scripts/webgpu-limits.mjs` +
  `results/engine/webgpu-limits-4090laptop-2026-08-02.json`. Il device prendeva
  i default di spec su 3 limiti — ora negoziati (`src/engine/gpulimits.ts`).
  `maxStorageBufferBindingSize` 2 GiB−4 è un tetto duro NVIDIA (workaround Dawn
  per un bug su `OpArrayLength`): lì il cap del codice era corretto.
- **`subgroup-matrix` è disponibile** su questo adapter ⇒ direction §8 rischio 1
  corretto; nuove righe in ideas-ledger §B. Vale per il ceiling, NON per i
  confronti pubblici (la vediamo solo con `--enable-unsafe-webgpu`).
- **Record/replay del grafo comandi precluso** finché esiste un readback per
  layer (graph capture richiede nessun kernel su CPU) — alza la posta di item 8.
- **Goal C3b**: chartered, parte a C3a chiusa.
- **TTFT misurato per la prima volta**: 88.06 s a ctx 461 — 22× il budget UX di
  4 s. Il prefill sequenziale (nessun percorso M>1) è la causa diretta.
- **Verifier di protocollo non lanciato** in it.1: la policy di sessione vieta
  subagent non richiesti dal PI. Se vuoi il gate indipendente a ogni ciclo,
  serve un'autorizzazione esplicita.
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
- **C3a item 1 RISOLTO** (2026-08-01): split C3→C3a/C3b + formulazione del
  gate prefill (velocità misurata sull'UX, TTFT 4 s) + soglie confermate.
  Prossimo item atteso: PLAN-CHECK a iteration 0 di C3a.
- C3b: nessuna decisione pendente finché C3a non chiude.
- fase-1b-matrice (11 item) e fase-2-deep-dive (5 item): stale, da triage
  weekly-maintenance.
