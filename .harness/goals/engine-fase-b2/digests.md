# Digests — engine-fase-b2

2026-07-30 — it.0 (setup). Goal aperto: floor di decode via de-sync (quota
fuori-GPU ~73% → ≤ soglia da spec, provv. ≤50%; guard-rail gpuBusy ≤ +5% vs B1;
dispatch ≤100 retrocesso a soglia-da-spec). 5 fasi sequenziali: attribuzione →
spec → loop multi-step K/submit → telemetria+profiler → bench+chiusura. Primo
target: fase 1 (scomposizione 8.1 ms/token + predizione K∈{2,4,8}).
Docket-born: plan-check (item 1) — STOP prima dell'iterazione 1.

2026-07-30 — it.1 (fase 1 DONE, finding maggiore). Attribuzione misurata: wall
8.09 ms/tok @ ctx bench = gpuBusy 6.46 + sync 1.59 + encode 0.05 ⇒ quota
fuori-GPU 20%, NON 73% (il dato del docket B1 confrontava contesti diversi:
gpuBusy@ctx64=2.43 riproduce il "2.2" della tsq-diag). GPU busy scala con kvLen
(attention decode, 14 wg). De-sync multi-step da sola: max 149 tok/s (1.22×).
Premessa del contratto refutata ⇒ RULING re-scope richiesto (docket item 2,
raccomandata: leva attention split + K multi-step, gate ≥240 tok/s). STOP by
design: fase 2 gated dal ruling. 2 JSON committati; doc stale corrette.

2026-07-30 — re-scope v2 (ruling PI: opzione (a), "il miglior motore in
assoluto"). GOAL.md riscritto: leva primaria attention split-context, secondaria
multi-step K; gate provv. ≥240 tok/s da fissare in spec; guard gpuBusy rimosso
(ridurlo È l'obiettivo), guard = parità+non-regressione+headline. PHASES v2:
fase 2 = spec + microbench attn-split isolato (ready), fase 3 kernel (timebox 4
it., clausola split pre-negoziata), 4 multi-step, 5 telemetria, 6 chiusura.
Loop riparte: iterazione 2 = fase 2.

2026-07-30 — it.2 (fase 2 DONE, ruling spec pendente). Kernel attention split
scritti e misurati in ISOLAMENTO: ~32 µs/layer piatto da ctx 64 a 1024 vs
29→219 µs del fuso (4.53× @ ctx576, 7.18× @ ctx1024), parità identica al fuso
vs CPU f64. Proiezione: gpuBusy 6.46→3.76 ms/tok ⇒ 185 tok/s @ K=1, 249 @ K=8.
Spec v2 scritta con soglie Pareto (proposta ≥230, alternativa ambiziosa ≥240 —
scelta al PI) e archiviazione motivata del dispatch ≤100. Verifier: sostanza
PASS (matematica split verificata a mano, diff additivo, 156/156), FAIL
iniziale su digest/HANDOFF stale → corretto in questo commit. STOP by design:
fasi 3-6 gated dal ruling spec (docket item 3, decisioni a-g).
