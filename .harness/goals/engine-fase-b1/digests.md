# Digests — engine-fase-b1

(un digest per ciclo, dal working protocol)

## Iterazione 2 (2026-07-29, fasi 1+2)
Fase 2 DONE dentro il timebox: il known-issue telemetria liv.2 era un bug NOSTRO
(mapAsync prima del submit ⇒ Dawn droppava l'intero command buffer: corruzione + zeri).
Fix (armTsq post-submit) verificato: matrice A/B pulita, gpuMs reali ~2.2 ms/token,
conformance GATE DOPPIO PASS con liv.2 attivo, 122/122 test. Landmine HANDOFF chiusa.
Fase 1 DONE: spec B1 scritta (multi-token M≤8, crop con kvLen, prefix-cache OPFS stile
ds4 ristretta a token-id, LRU 512 MB, soglie con prefill 3× Pareto) → ruling richiesto
(docket 3). Scoperta fuori scope per B2 nel docket 2: ~73% del decode è fuori GPU.
Next: fasi 3-6 blocked sul ruling spec — stop-by-design.

## Iterazione 0 (2026-07-29, goal-setup)
Contratto approvato in blocco (split B1/B2 deciso dal PI). PHASES.md scritto: 6 fasi —
parallel-group 1+2 (spec B1 ∥ diagnosi telemetria liv.2 timeboxed), poi spine
sequenziale 3 (multi-token M≤8 + parità) → 4 (rollback KV) → 5 (prefix-cache OPFS) →
6 (bench + chiusura). Primo target: fase 1 (spec) + fase 2 (diagnosi). Docket-born:
plan-check (STOP: PHASES.md attende approvazione PI prima dell'iterazione 1). B2
(floor dispatch ≤100) resta fuori goal per contratto.

## Iterazione 3 (2026-07-30) — fase 3 DONE
Prefill multi-token M≤8 implementato e conforme in 1 iterazione: piano prefill
compilato a parte (GEMM chunk a M colonne, attention causale intra-chunk, zero
readback, submit/64, lm_head solo ultima posizione), unit chunking 13/13, npm test
135/135. Prima conformance FAIL 82% → root-cause con bisezione a 4 sensori: bug
Chrome/Tint (var array nel body di un loop WGSL non ri-azzerata per iterazione,
colpiva solo il down-proj a 152 blocchi/riga) → fix azzeramento esplicito →
GATE DOPPIO PASS 98.05% golden / 100.00% cpuref, anche con telemetria liv.2 attiva.
Sim prefillBatched rimossa (da spec). Landmine nuova in HANDOFF. Next: fase 4 (crop).
