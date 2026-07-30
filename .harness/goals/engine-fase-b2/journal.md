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

2026-07-30 — Iterazione 4, FASE 4 DONE. Decode loop multi-step da spec:
(1) kernel embedGatherQ4 (dequant riga token_embd on-GPU, id letto da amaxOut —
stessa aritmetica ESATTA di dequantQ4_0Row, quindi x identica al percorso CPU);
token_embd repackato su GPU (~68 MB, ruling decisione c) SENZA togliere il
percorso CPU di forwardToken (oracolo intatto). (2) decodeBatch(prev, posStart,
k≤8): K forward in un submit, seed del feedback via writeBuffer(amaxOut), P
per-step da dbSlots (pattern pcSlots), copy amaxOut→idsBatch per step, UNA
mapAsync per batch su stagingIds; error scope validation/oom come CONTRATTO
(landmine B1); tsq per-step (timestampWrites per pass, al più un flush/batch);
kvLen advance(k). (3) modulo puro decodebatch.ts (planDecodeBatch + trimAtEos
per EOS mid-batch via crop) con 5 unit. (4) worker mode idcheck + script
token-identity.mjs (exit 0 solo su pass).

GATE (tutti PASS):
- npm test 166/166 (161+5); tsc pulito.
- Token-identity (token-identity-4090-2026-07-30T19-00-43): K=8, K=5 (non
  divisore) e K=1-degenere TUTTI match=true su 256 token vs oracolo per-token.
- Conformance INVARIATA post-integrazione: GATE DOPPIO PASS 98.05/100.00
  (conformance-4090-2026-07-30T19-00-59).
- Tempi informali (nota nel JSON, non sono il bench): per-token 3.97 ms/tok,
  K=8 3.46 (~289 tok/s), K=5 3.53 — il multi-step amortizza ~0.5 ms/token
  di sync residua, coerente con l'attesa post-fase 3.
Next: fase 5 (telemetria liv.2 + profiler nel nuovo loop: run tsq con gpuMs
reale, profiler con submit/token 1/K, dispatch target di spec ≤160 già ok).

2026-07-30 — Iterazione 5, FASE 5 DONE. Telemetria e profiler sul loop
multi-step: (1) runBench portato sul percorso decodeBatch (k=8 default di spec,
k=1 = per-token per baseline; finestra di rate post-primo-batch, dichiarata nel
report); (2) idcheck con telemetryGpu: liv.2 ATTIVO sul loop multi-step
(timestampWrites per step) con identità INVARIATA e gpuMs REALE; (3) engine-prof
K-aware (schemaVersion 2: submit/forward atteso 1/k, dispatch atteso
dispatchesPerToken+1 per l'embedGather, bound spec ≤160); (4) ledger §I: riga
floor-dispatch aggiornata ad ARCHIVIATO-su-desktop con trigger mobile (ruling f).

GATE (tutti PASS):
- npm test 166/166; tsc pulito.
- Liv.2 sul loop nuovo (token-identity-4090-2026-07-30T19-13-14, tsq attivo):
  K=8/K=5/K=1 match=true, gpuMs/token 2.862 REALE (non-null) — il liv.2 non
  perturba il multi-step e misura davvero.
- Conformance con telemetryGpu:true: GATE DOPPIO PASS 98.05/100.00 invariata
  (conformance-4090-2026-07-30T19-13-34).
- Profiler sul bench multi-step (engine-prof-4090-2026-07-30T19-13-54, exit 0):
  bindGroup 0, submit/forward 0.125 = 1/8 ESATTO, mapAsync/forward 0.125,
  dispatch/forward 148 = 147+1 (embedGather), ≤160 ✓.
Dato per fase 6: gpuBusy col kernel split ≈ 2.86 ms/token (dall'idcheck tsq, da
rimisurare nel bench col protocollo §Metodologia gpuBusy).
Next: fase 6 (bench canonico k=8 + baseline k=1 same-day + gpuBusy liv.2,
non-regressione rollback/prefix-cache, checklist DONE WHEN, chiusura+merge).

2026-07-30 — Iterazione 6, FASE 6 DONE e GOAL CHIUSO. Bench canonico: runBench
esteso con baseline K=1 same-day (runOnce(label, kRun)) e repliche liv.2
DEDICATE per gpuBusy (secondo engine con tsq, senza taps — caveat verifier it.5
— stessa finestra post-primo-batch; headline sempre dalle repliche OFF); gates
auto-evidenti nel report (decode ≥230, overhead ≤2, prefill ≤810 assoluto).

CHECKLIST DONE WHEN DEL CONTRATTO v2 (7/7):
1. [x] Spec B2 v2 scritta e approvata (ruling PI 2026-07-30, docket 3: soglia
   230, token_embd su GPU, dispatch archiviato).
2. [x] Attribuzione decode wall: 2 JSON (ctx pieno + ctx32), it.1 — quota reale
   20%, gpuBusy scala con kvLen; verifier PASS.
3. [x] Decode più veloce: bench-4090-2026-07-30T19-24-35 — headline K=8 OFF
   287.5 ±2.3 tok/s ≥230 (2.35x vs B1 122.4), baseline K=1 same-day 238.3,
   msPerToken ~3.48, quota fuori-GPU 13.2% (gpuBusy 3.06, repliche liv.2
   dedicate), warmup+3 repliche HEADED, schemaVersion 3.
4. [x] Token-identity multi-step: K=8/K=5/K=1 IDENTICI su 256 token vs oracolo
   (it.4; ri-verificato it.5 con tsq ATTIVO).
5. [x] Parità INVARIATA: conformance GATE DOPPIO PASS 98.05/100.00 (512/512)
   col kernel split (it.3), post multi-step (it.4) e con telemetryGpu:true
   (it.5) — identica alla B1.
6. [x] Non-regressione prefill e persistenza: prefill 697.8 ms ≤810 (assoluto
   di spec); kv-rollback-4090-2026-07-30T19-24-58 PASS exit 0;
   prefix-cache-4090-2026-07-30T19-25-06 PASS (restore worker nuovo 173.3 ms <
   re-prefill 700 ms, continuazione identica).
7. [x] Telemetria viva: gpuMs 2.86-3.06 REALE nel loop nuovo (it.5 + bench);
   overhead da spenta -0.002% (≤2); profiler exit 0 (0 bindGroup, submit 1/8
   esatto, 148 dispatch/forward ≤160); dispatch ≤100 ARCHIVIATO nel ledger §I
   con trigger mobile (ruling f).

RISULTATO DEL GOAL: decode al ctx bench 122.4 → 287.5 tok/s (2.35x), wall
8.17 → 3.48 ms/token, quota fuori-GPU 73%(apparente)/20%(reale) → 13.2%,
parità e persistenza intatte. Le due leve hanno reso: kernel split ~2x,
multi-step +16% (238→287.5 same-day). Chiusura: merge su main + push (ruling
permanente 2026-07-29) DOPO verifier gate.
