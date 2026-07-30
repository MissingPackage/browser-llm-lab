# Journal — engine-fase-b2

2026-07-30 — Iterazione 1, FASE 1 DONE con FINDING CHE REFUTA LA PREMESSA DEL
CONTRATTO. Costruita modalità worker `attrib` (finestra decode pura post-prefill,
wall per-token, delta encode liv.1 / gpuBusy liv.2 per replica, probe sync-floor
sullo stesso device, predizione analitica K∈{2,4,8}; rotta ?attrib=1 in page.ts,
orchestratore scripts/decode-attrib.mjs con PREFIX/GEN per il contesto corto).
Protocollo repo: warmup scartato + 3 repliche HEADED, crop() per riavvolgere la
finestra fra le repliche.

RISULTATI (2 JSON committati in results/engine/):
- ctx bench (~470-670, decode-attrib-4090-2026-07-30T14-03-15): wall 8.09 ms/tok
  = encode 0.05 + gpuBusy 6.46 + sync 1.59 ⇒ quota fuori-GPU 20.2%.
  predictionByK: K=2 137, K=4 145, K=8 149 tok/s (batched; pipelined +1%).
- ctx corto (32-96, decode-attrib-4090-ctx32-…T14-06-46): wall 3.99 = 0.04 +
  gpuBusy 2.43 + sync 1.53 ⇒ quota 39.1%. Il gpuBusy riproduce il 2.17-2.26
  della tsq-diag B1 (stessa finestra prefill 32/decode 64) ⇒ misura liv.2 SANA.
- Probe sync-floor (GPU ~vuota): mapRoundTrip 0.4-0.9 ms mean (p50 0.08-0.24),
  ordine compatibile con la sync residua; sync per-token ~1.5-1.6 ms FISSA nei
  due regimi (readback argmax mapAsync + event loop).

DIAGNOSI: il "GPU busy 2.2 vs wall 8.1 ⇒ 73% fuori GPU" (docket B1 item 2, base
del ri-inquadramento del contratto B2) confrontava numeri a CONTESTI DIVERSI
(2.2 @ ctx~64 della tsq-diag vs wall @ ctx~570 del bench). Il GPU busy scala con
kvLen: pendenza ≈ (6.46−2.43)/(570−64) ≈ 8 µs/posizione/token = attention decode
(attnFusedWgsl: 14 workgroup = 1/head, loop sequenziale sul contesto — GPU
sottoutilizzata; a ctx 1024 proietta ~+7.7 ms/token). Componenti al ctx bench:
attention ~4.0 · GEMV+floor dispatch ~2.4 · sync ~1.6 · encode ~0.05.

CONSEGUENZE SUL CONTRATTO (⇒ docket item 2, RULING RICHIESTO):
- Il gate "quota fuori-GPU da ~73% a ≤50%" è GIÀ soddisfatto oggi (20%): vacuo.
- La de-sync pura (multi-step K) vale al massimo ~1.22× (149 tok/s a K=8):
  il 240 tok/s provvisorio NON è raggiungibile senza ridurre il GPU busy.
- La leva primaria al contesto bench è il kernel attention (split sul contesto,
  stile flash-decoding); sync e floor GEMV/dispatch sono secondarie ma reali.
- Nessuna soglia proposta col Pareto (il done-when di fase 1 la prevedeva): fissare
  una soglia dentro un inquadramento refutato sarebbe teatro — la proposta è il
  re-scope, PI-gated by design (anti-ratcheting: il contratto non si riscrive da
  soli). Opzioni nel docket item 2.

Doc stale corrette (ruling 2026-07-29): tsq-diag §Conseguenze (claim 73%
barrato + correzione), ideas-ledger §I riga floor-dispatch. Deviazione owns
fase 1 dichiarata: toccati engine.worker.ts e page.ts (plumbing della modalità
diag, autorizzato dal GOAL §may-do; owns di PHASES citava solo scripts/pagina).
Unit 156/156 invariati, tsc pulito (i cambi sono modalità diag additive).
Next: STOP by design — fase 2 (spec) è gated dal ruling re-scope.
