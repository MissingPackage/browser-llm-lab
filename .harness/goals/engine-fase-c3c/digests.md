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
