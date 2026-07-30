# Journal — engine-fase-b1

2026-07-29 — Iterazione 2 (cont.), fase 1 DONE: spec B1 scritta
(specs/2026-07-29-engine-fase-b1-design.md, 5 sezioni contratto verificate via grep).
Decisioni proposte al ruling (docket 3): chiave su token-id, lookup v1 esatto, niente
logits nel checkpoint (1 forward al restore), LRU semplice, contratto hard pos===kvLen,
budget 512 MB. Soglia prefill 3× già PI-ruled (Pareto, it.1); resto soglie =
non-regressione fase A. Con fase 2 già chiusa, TUTTO il residuo (fasi 3-6) è
PI-gated dal ruling spec ⇒ stop-by-design da working protocol.

2026-07-29 — Iterazione 2, fase 2 DONE (diagnosi telemetria liv.2, dentro il timebox).
Matrice A/B (5 varianti, detector = 64 token greedy + logits finali): H1 timestampWrites
refutata, H3 ring refutata, H2 resolve-in-encoder confermata ma incompleta; il buffer
etichettato ha dato il colpevole: "tsq-staging used in submit while mapped". ROOT CAUSE
= bug NOSTRO fase A (mapAsync chiamata prima del submit che riempie lo staging ⇒ Dawn
droppa l'intero command buffer: token saltato = corruzione, copy mai eseguita = zeri;
"own" curava solo la matematica; il microbench attendeva il submit ⇒ funzionava).
FIX: flushTsq encoda soltanto, armTsq mappa dopo il submit. VERIFICA: matrice post-fix
tutta pulita (idsMatch=✔, maxΔ=0, gpuMs 2.17-2.26 ms/token REALI), conformance completa
con telemetryGpu:true GATE DOPPIO PASS (98.05% golden / 100.00% cpuref, = fase A),
npm test 122/122. Landmine HANDOFF aggiornata (decade "timestamp azzerati", resta la
lezione mapAsync-post-submit). Nota: docs/engine/tsq-diag-2026-07-29.md. Osservazione
fuori scope → docket 2: GPU busy 2.2 ms/token vs 8.1 wall ⇒ ~73% del decode è sync/
encode, dato che ri-inquadra il goal-brief di B2.

2026-07-29 — Iterazione 1 (parziale, pre-spec): plan-check APPROVATO dal PI ("sul
resto tutto ok"). Soglia prefill decisa per simulazione col criterio di Pareto
(mandato PI in chat). Sim su 4090 (prefill-sim-4090-*.json, 2 run concordanti):
seq-await 5.16 ms/token; batched senza readback 3.80 (submit/64); + skip lm_head
per posizioni non finali 3.38 ⇒ il floor "trucchi senza kernel nuovi" = 1.53×.
Residuo overhead-bound: ~28 µs/dispatch × 120 dispatch ⇒ analitico M=8 ≈ 5-10×.
SOGLIA FISSATA: 3× (≤1/3 della baseline seq same-day) — 2× sopra il floor dei
trucchi, metà della predizione conservativa. GOAL.md aggiornato (ruling registrato).
Metodo: prefillBatched() sim nel motore (embedding+pos pre-caricati, copy in-encoder,
submit a granularità variabile, decodeMatch=true su tutte le varianti vs run seq).
NOTA per fase 3: il knee submit-granularità è a 64 (all-in-one peggiora: 4.22).

2026-07-29 — Goal aperto. Contratto approvato dal PI in chat (goal-brief, "vai con 2
goal": approvazione in blocco, [ASSUMED] compresi). SPLIT della fase B di direction:
B1 = memoria/latenza di entrata (questo goal), B2 = floor dispatch ≤100 (goal futuro,
annotato in HANDOFF). Soglie numeriche marcate "da confermare in spec" vanno fissate
nella spec B1 prima del codice.

2026-07-29 — Verifier gate iterazione 2: PASS (8/8 punti, zero violazioni). Nota a
costo zero dal verifier per la fase 3+: aggiungere il campo `telemetryGpu` allo schema
del report di conformance, così il gate "liv.2 non perturba" diventa auto-evidente
dal JSON.

2026-07-30 — Ruling PI: spec B1 APPROVATA (docket 3, decisioni a-g in blocco).
Richiesta PI aggiuntiva recepita: i rimandi deliberati vanno documentati dove le fasi
successive li ritrovino → creato registro ideas-ledger.md §I ("Rimandi espliciti di
fase": 8 righe con fase di ripresa + trigger di riattivazione — longest-prefix,
chiave testuale/tokenizer, logits nel checkpoint, scoring eviction, re-inquadramento
B2, fusioni annotate, parametro M per spec-dec, telemetria liv.3). Fase 3 ready.
