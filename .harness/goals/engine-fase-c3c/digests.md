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
