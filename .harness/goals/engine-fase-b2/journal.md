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

2026-07-30 — Iterazione 2, FASE 2 DONE (ruling spec pendente, docket 3). Dopo il
re-scope v2 (ruling PI: opzione (a), "il miglior motore in assoluto"): scritti i
kernel attnSplitPartWgsl/attnSplitReduceWgsl (split del contesto in blocchi da 64,
1 pos/thread, griglia FISSA (14,16) — piano statico invariato, partizioni vuote
escono su begin>=n; pass 2 log-sum-exp esatto; owner-rule rope/append invariato)
e il microbench in isolamento (worker mode attnbench, pattern kernel-diag: input
sintetici deterministici, riferimento CPU f64, 480 op/submit serializzate dagli
hazard RW, error-scope da landmine B1).

RISULTATI (attn-bench-4090-2026-07-30T14-59-18-433Z.json):
- split ~28-38 µs/layer PIATTO da ctx 64 a 1024; fuso 29→219 µs lineare.
  ctx 576: 144.3 vs 31.8 µs = 4.53×; ctx 1024: 7.18×; ctx 64: 0.77× (accettato).
- Parità: maxΔ vs CPU f64 IDENTICO fuso/split a ogni ctx (5.7e-8 @ 64, ~3e-3 @
  lunghi = accumulo f32 uguale per entrambi) ⇒ split numericamente equivalente.
- Proiezione ctx 576: gpuBusy 6.46−2.70=3.76 ms/tok ⇒ K=1 185, K=4 238, K=8 249
  tok/s. Plateau a leva singola: sync-only 149, kernel-only 185.

Spec v2 scritta (2026-07-30-engine-fase-b2-design.md, sezioni: Attention split,
Decode loop multi-step, Streaming, Tap e telemetria, Metodologia gpuBusy, Soglie,
Rischi). Soglia proposta col Pareto: decode ≥230 (sopra entrambi i plateau,
margine 8% sotto predizione; alternativa 240 al PI). Dispatch ≤100: proposta di
archiviazione motivata (floor GEMV = lavoro reale; +24 dispatch dello split e il
wall scende comunque), bound sanity 160. Decisione grossa in spec: token_embd su
GPU (~68 MB) per il feedback on-GPU (embedGatherQ4), deviazione dichiarata dalla
fase A. Next: STOP by design — fasi 3-6 gated dal ruling spec (docket 3).

2026-07-30 — Iterazione 3, FASE 3 DONE (1 iterazione su timebox 4). Ruling spec
incassato a inizio ciclo (docket 3: soglia 230, token_embd su GPU, dispatch
archiviato). Integrazione del kernel split nel piano fuso: attnF sostituito da
attnPart (griglia (14,16)) + attnReduce (14) in gpuforward.ts, buffer attnPartials
(14×16×66 f32), modulo puro attnsplit.ts (CHUNK/sMax/owner/range + lseReduce di
riferimento) usato per il sizing e testato (5 unit nuove: geometria partizioni +
proprietà "lseReduce ≡ softmax monolitica" anche con score estremi). attnFusedWgsl
resta nel sorgente (microbench/debug), fuori dal piano di produzione.

GATE (tutti PASS):
- npm test 161/161 (156+5); tsc pulito.
- Conformance col kernel split: GATE DOPPIO PASS 98.05% golden / 100.00% cpuref
  (512/512) — IDENTICA alla B1 (conformance-4090-2026-07-30T18-44-19).
- prefill-diag cold-start: argmaxMatch=true su tutte le L, chunk≡seq, badRows
  vuote; kernel-diag: maxErr ~1e-7.
- Profiler finestra decode: createBindGroup=0, submit/token 1.036,
  dispatch/token 147 (≤160 da spec; engine-prof-4090 JSON).

SORPRESA POSITIVA (bench informale, bench-4090-2026-07-30T18-45-56):
decode 248.3 ±3.9 tok/s GIÀ A K=1 — la proiezione additiva (185 kernel-only) era
PESSIMISTA: il gate 230 è superato prima del multi-step. Lettura: parte della
"sync" dell'attribuzione era coda GPU dietro i kernel lenti, non costo fisso.
Overhead telemetria 0.65%. Prefill 701 ms invariato (≤810 ✓); la baseline seq
same-day è scesa a 1907 ms (anche lei usa lo split) ⇒ il gatePass RELATIVO del
bench era diventato fuorviante: ancorato all'ASSOLUTO di spec (PREFILL_GATE_MS
810, deviazione dichiarata: campo report, non percorso di calcolo).
Nota fase 4: il margine sul gate non cambia il contratto — il multi-step resta
in scope (la sync residua ~0.9 ms/token si amortizza comunque, e serve al
contratto streaming/spec-dec §I); la predizione K=8 va rifatta coi numeri nuovi.
Next: fase 4 (decode loop multi-step K + token-identity).
