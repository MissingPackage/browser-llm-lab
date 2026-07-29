# Journal — engine-fase-b1

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
