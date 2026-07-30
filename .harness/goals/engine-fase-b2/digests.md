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

2026-07-30 — it.3 (fase 3 DONE in 1 iterazione). Ruling spec incassato (230 /
token_embd GPU / dispatch archiviato). Kernel split integrato nel piano fuso:
conformance IDENTICA a B1 (98.05/100.00), prefill-diag/kernel-diag PASS,
profiler 0 bindGroup / 1.04 submit / 147 dispatch per token. SORPRESA: decode
248.3 tok/s GIÀ a K=1 (proiezione kernel-only diceva 185; parte della "sync"
era coda GPU) — gate 230 superato prima del multi-step. Prefill 701 ms ≤810;
gate bench ancorato all'assoluto di spec. Next: fase 4 (multi-step + identity).

2026-07-30 — it.4 (fase 4 DONE). Multi-step operativo: embedGatherQ4 (token_embd
su GPU, dequant esatta), decodeBatch K≤8 con feedback on-GPU e UNA mapAsync per
batch, EOS via crop (trimAtEos puro). Token-identity PASS: K=8, K=5 e K=1
degenere IDENTICI su 256 token vs oracolo per-token; conformance invariata
(98.05/100.00); 166/166 unit. Informale: 3.97→3.46 ms/token con K=8 (~289
tok/s). Next: fase 5 (telemetria liv.2 + profiler nel nuovo loop).

2026-07-30 — it.5 (fase 5 DONE). Liv.2 sul loop multi-step: gpuMs 2.862 REALE
con identità K=8/5/1 invariata (tsq non perturba); conformance tsq PASS;
engine-prof v2 K-aware exit 0 (0 bindGroup, submit/forward 0.125 = 1/8 esatto,
148 dispatch/forward ≤160); bench portato su decodeBatch k=8 default; ledger §I:
dispatch ≤100 archiviato-desktop con trigger mobile. Next: fase 6 = bench
canonico (k=8 + baseline k=1 + gpuBusy da metodologia spec), non-regressione
rollback/prefix-cache, checklist 7/7, chiusura + merge.
