# Digests — engine-fase-b2

2026-07-30 — it.0 (setup). Goal aperto: floor di decode via de-sync (quota
fuori-GPU ~73% → ≤ soglia da spec, provv. ≤50%; guard-rail gpuBusy ≤ +5% vs B1;
dispatch ≤100 retrocesso a soglia-da-spec). 5 fasi sequenziali: attribuzione →
spec → loop multi-step K/submit → telemetria+profiler → bench+chiusura. Primo
target: fase 1 (scomposizione 8.1 ms/token + predizione K∈{2,4,8}).
Docket-born: plan-check (item 1) — STOP prima dell'iterazione 1.
