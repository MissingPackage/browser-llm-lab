# Digests — engine-fase-c3c

## it.0 (2026-08-08) — goal aperto

- Contratto riletto (4 modifiche split coerenti), PHASES 9 fasi sequenziali:
  WP banda fredda → spec → slab ctx-aware → prefetch in-forward → policy
  tier+AUTOPIN vs LRU → modello di banda → instant-on → gate floor+nonreg →
  chiusura.
- Primo target: fase 1 — tool tools/opfs-cold/** per la banda OPFS fredda
  IN Chrome (i numeri OS di C1 non si estrapolano).
- PI-gated noti: docket item 1 (budget instant-on, si decide con la spec in
  fase 2); nessuna fase docket-born.

## it.1 (2026-08-08) — fase 1 DONE: banda fredda browser = parità OS

- Tool `tools/opfs-cold/opfs-cold-bench.mjs`: OPFS cold in Chrome, eviction
  fadvise da fuori sui backing file, freddezza PROVATA nel JSON (fincore
  0 B post-drop + delta warm 8× sugli stessi offset).
- Numeri (2 run): random expert-size COLD 1.79-1.94 GB/s, p50 2.9-3.1
  ms/expert (OS C1: 1.63 / 3.74); seq 3.43-3.73; streaming expert-size
  p50 1.3-1.4 ms. Tassa browser sul freddo: ZERO (ratio 1.07-1.19).
- Implicazione instant-on: non-routed+caldi 1.53 GB ≈ 0.44 s di banda
  disco — il TTFT a freddo non muore di I/O.
- Verifier: sostanza PASS; sanature applicate (digest, range ratio).
- Next: fase 2 — spec C3c coi numeri del WP; lì la PROPOSTA per il ruling
  docket item 1 (budget instant-on).

## it.2 (2026-08-08) — fase 2 DONE: spec depositata, proposta item 1 al PI

- Spec `2026-08-08-engine-fase-c3c-design.md`: budget slab ctx-aware
  (formula con numeri verificati dal codice), prefetch in-forward (K=4,
  predictor al confine ESCLUSO), tier.h+AUTOPIN (pin ≤12.5% HARD), modello
  di banda ±15%, instant-on con freddezza riproducibile (protocollo WP).
- PROPOSTA per il ruling item 1 (docket 1-bis): (a) raccomandata 1.25×
  AUTO-ancorato alla config della run instant-on; senza overlap l'aritmetica
  dice 1.28-1.61× ⇒ il target morde sull'overlap I/O. Blocca solo fase 7.
- Scoperta: landmine "KV 54 KB/token" era stale 2× (vero 108 288 B/token,
  f32; ctx 6k = 665 MB non 361). HANDOFF corretto, docket item 4.
- Next: fase 3 — slab ctx-aware (formula in codice, ctx 6k senza OOM,
  non-regressione b12, rimisura syncLogits).

## it.3 (2026-08-08) — fase 3 DONE: slab ctx-aware, ctx 6k senza OOM

- Formula in produzione: budget = ceiling misurato − nonExpert − KV(ctx) −
  work(ctx) − riserva; glmbench --budget-gib auto + --ctx-max; 6 unit test.
- ctx 6144: NO OOM a budget calcolato 12.146 GiB (prima: OOM garantito);
  ctx 525 auto in sessione viva = regime tetto (16.55 tok/s, TTFT 12.53,
  strutturale 1.875 PASS) senza config a mano; b12 migliorato (+11%/−11%).
- Due OOM istruttivi: attnPartialsM ×16 mancante dal work (253 MiB @6k) e
  riserva tarata 256→512 MiB sui punti OOM osservati. syncLogits rimisurato
  (0.09-0.11 ms): landmine iv chiusa.
- Next: fase 4 — prefetch in-forward (tap hidden L → router L+1) + recall
  in-engine vs oracolo 92% @K=8.

## it.4 (2026-08-08) — fase 4 DONE: prefetch in-forward, oracolo replicato

- Tap in-forward (+1 GEMV router L+1, stessa mapAsync, zero sync extra);
  fetch dei predetti nella finestra d'attesa del router; predizioni
  strutturalmente confinate al token (stato dentro il forward).
- Recall in-engine full-corpus: **91.92% @K=8 vs 92.0 oracolo (−0.08pp)**,
  77.04% @K=4 vs 77.5 — il potere predittivo di C1 arriva intero.
- Identità blindata: firma 14b ai conteggi ESATTI con prefetch acceso,
  cpuref 256/256, ktest 69/69, suite 344+7, tsc pulito.
- Next: fase 5 — tier.h+AUTOPIN vs LRU a 1472/736 slot (il confronto vero
  sui miss al momento d'uso; ceiling Belady +9-19pp).

## it.5 (2026-08-08) — fase 5 (1/2): la policy è in codice

- tier.h+AUTOPIN+REPIN dentro ExpertCache dietro policy:"tier" (lru default
  intatto): eusage persistibile additivo, cap pin 12.5% HARD con throw,
  LFRU con isteresi, pin mai evinti. 5 unit test, suite 349+7, tsc ok.
- glmroute: --policy e --park-frac (1472/736 slot esatti per classe).
- it.6 = batch notturno 4 run full-corpus + analisi (delta vs LRU e vs
  ceiling Belady) + chiusura riga 5.

## it.6 (2026-08-09) — fase 5 DONE: +9.43pp @1472, +25.04pp @736

- Use-hit (depurato dal prefetch): tier 93.75%/87.37% vs LRU 84.33%/62.34%
  — a 1472 colma il 94% del gap Belady; a 736 SUPERA Belady (il prefetch
  anticipa i fetch: fuori dal perimetro del bound demand-fetch, spiegato).
- Pin sempre al cap 12.5% esatto (assert mai scattato); firma 14b esatta
  in tutte e 4 le run; costo dichiarato +31-40% byte letti totali.
- Regola di metodo registrata: full-corpus solo per firma/nonreg/
  riferimenti; sim o subset di prompt interi per esplorare.
- Next: fase 6 — modello di banda (bench brevi ai 3 budget, ±15%).

## it.7 (2026-08-09) — fase 6 DONE: modello di banda a ±1%; STOP BY DESIGN

- bandmodel.ts: BASE indipendente (c3b) + costo/fetch parametrico in banda
  + gradino gpuBusy con evidenza misurata (48→84 ms/token). 3 punti bench
  nuovi (5.41/2.84/1.93 tok/s), b12 FUORI dal fit = predizione +0.9%.
- Il ±15% del done-when è un TEST permanente in npm test (10 nuovi, 359+7).
- Proiezioni fredde per fase 7: 5.02/2.24/1.27 tok/s; coldTtftMs pronto.
- STOP BY DESIGN: fase 7 bloccata dal ruling item 1 (proposta 1-bis sul
  tavolo: raccomandato 1.25× auto-ancorato). Il loop riparte al ruling.

## it.8 (2026-08-09) — fase 7 DONE: instant-on 1.247 ≤ 1.25

- Ruling (a) recepito; overlap costruito (prefetch nel prefill chunked:
  tap batched + unione top-4, cold −6.0 s).
- Verdetto al protocollo v2 PRE-dichiarato (mediane 3+3, eviction provata
  a ogni sessione fredda): 24 727 / 19 824 ms = 1.247 PASS — margine
  sottile (0.3%) dichiarato; v1 singola era 1.2505 (rumore).
- Coda fase 6 verificata: coldTtftMs −5.8% (±15%). Gap UX 6.18× → fase D.
- Next: fase 8 — gate floor + non-regressione a perimetro pieno (batch
  lungo, albero congelato), poi fase 9 chiusura.

## it.9 (2026-08-09) — fase 8 DONE: floor 15.641 ≥ 13.43 PASS in produzione

- Il gate C1 passa SECCO alla config migliore (budget CALCOLATO 12.737 GiB,
  sessione utente viva) — niente clausola. Strutturale 2.000 PASS.
- Nonreg piena: sync b12 e Qwen in banda, optimistic b12 13.172/14.74
  (nuovo riferimento, item 5 risolto), golden 98.828% AL PIN, cpuref
  256/256 + 512/512, firma routing esatta, ktest 69/69, suite 359+7.
- Next: fase 9 — chiusura del goal (checklist DONE WHEN, direction §7,
  ledger, HANDOFF, input hero-demo M4 a docket, tag).
