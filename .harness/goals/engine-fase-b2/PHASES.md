# PHASES — engine-fase-b2

Decomposizione del contratto (GOAL.md) in fasi loop-runnable. Cambiabile dopo
l'iterazione 0 solo via docket. Le soglie marcate "da spec" (quota fuori-GPU
[provvisoria ≤50%], target dispatch/token) vengono fissate in fase 2 sui dati di
fase 1 e da lì in poi sono vincolanti per i verifier. Il guard-rail anti-gaming
(gpuBusy ≤ +5% vs B1 ~2.2 ms/token) è di contratto, NON rinegoziabile in spec.

| # | phase | done-when (mechanical) | authority delta | owns | status |
|---|-------|------------------------|-----------------|------|--------|
| 1 | **Attribuzione decode wall** — scomposizione degli 8.1 ms/token B1 in gpuBusy / sync readback / encode CPU / altro (4090 HEADED, protocollo repo) + predizione analitica del guadagno multi-step per K ∈ {2,4,8} | JSON committato in `results/engine/` con la scomposizione (somma componenti ≈ wall dichiarata nel JSON) e `predictionByK`; journal aggiornato con la soglia quota fuori-GPU proposta (criterio di Pareto, come la 3× di B1) | none | `scripts/`, pagina diag dedicata, `results/engine/` | ready (dopo plan-check) |
| 2 | **Spec fase B2** — architettura decode loop multi-step (K/submit, feedback token on-GPU, cadenza readback, fallback pipelining se l'embedding gather non legge da buffer GPU), contratto streaming a raffiche di K, interazione tap+telemetria liv.2, metodologia misura gpuBusy nel bench (repliche liv.2 vs dedicate, prerequisito timestamp-query), soglia quota fuori-GPU (da fase 1) e target dispatch/token (rientro ≤100 O archiviazione ledger §I) | `docs/superpowers/specs/*engine-fase-b2-design.md` esiste con sezioni grep-abili ("Decode loop multi-step", "Streaming", "Tap e telemetria", "Metodologia gpuBusy", "Soglie", "Rischi"); entry di richiesta ruling appesa a `docket.md` | none | `docs/superpowers/specs/`, `.harness/goals/engine-fase-b2/` | blocked (dati fase 1) |
| 3 | **Decode loop multi-step + parità** — implementazione da spec (kernel/piano per K forward/submit, token feedback on-GPU o fallback, argmax on-GPU in catena), tap preservati, error scope da landmine B1 | `npm test` verde con unit CPU-side sul piano del loop multi-step; report JSON token-identity committato: run K>1 vs per-token dallo stesso prefisso, greedy, ≥256 token IDENTICI, exit 0; `scripts/conformance-engine.mjs` exit 0 (gate doppio) attraverso il nuovo loop | none | `src/engine/**`, `tests/`, `scripts/` | blocked (ruling spec) |
| 4 | **Telemetria liv.2 + profiler nel nuovo loop** — tsq ring compatibile con K forward/submit (mapAsync SOLO post-submit, landmine tsq-diag), finestra decode del profiler aggiornata | run liv.2 con gpuMs reale non-null (JSON committato); conformance exit 0 con `telemetryGpu:true`; profiler JSON con submit/token e dispatch/token del nuovo loop; target dispatch di spec rispettato O archiviazione motivata scritta nel ledger §I | none | `src/engine/**` (percorso telemetria), `scripts/`, `results/engine/`, `docs/engine/ideas-ledger.md` (§I) | blocked (fase 3) |
| 5 | **Bench + non-regressione + chiusura** — quota fuori-GPU vs soglia, guard-rail, prefill/persistenza invariati | bench JSON in `results/engine/` (schemaVersion 3, baseline same-day): quota fuori-GPU ≤ soglia di spec E gpuBusy ms/token ≤ +5% vs B1 E prefillMs mean ≤ 810 E overhead telemetria spenta ≤ 2%, con decodeToksPerSec.mean/msPerToken headline; `kv-rollback` e `prefix-cache` re-run exit 0 (JSON committati); checklist DONE WHEN 7/7 nel journal; HANDOFF aggiornato; merge+push a goal chiuso (may-do, ruling permanente) DOPO verifier gate PASS | none | `results/engine/`, `HANDOFF.md`, docket, journal | blocked |

Note di spine:
- 1→2 è sequenziale by design: la spec fissa le soglie SUI DATI dell'attribuzione
  (stesso metodo della soglia prefill B1: misura/simulazione → Pareto → soglia in
  spec → vincolante). Niente parallel-group in questo goal: ogni fase consuma
  l'output della precedente.
- Fase 3 è la più grossa (feedback on-GPU tocca embedding gather + catena sampling
  + struttura del submit): se supera 4 iterazioni, split via docket (candidato
  naturale: prima pipelining del readback, poi feedback on-GPU).
- La token-identity di fase 3 usa il percorso per-token B1 come oracolo interno:
  va tenuto funzionante (flag/percorso legacy) almeno fino a fase 5 — rimuoverlo
  prima è vietato dal done-when di fase 3.
- Il readback batched cambia la cadenza di consegna dei token (raffiche di K): il
  contratto di streaming è deciso in spec (fase 2), non improvvisato in fase 3.
- Fusioni cross-layer/megakernel: SOLO se la spec le arruola per il target
  dispatch; altrimenti restano rimandi §I (il contratto B1 le teneva must-docket,
  qui entrano solo via spec approvata).
- Benchmark pubblico: fuori dal goal by contract (contributo separato, ruling PI
  2026-07-30) — nessuna fase lo tocca.
