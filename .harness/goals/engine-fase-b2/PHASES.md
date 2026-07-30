# PHASES — engine-fase-b2 (v2, re-scope opzione (a), ruling PI 2026-07-30)

Decomposizione del contratto GOAL.md v2. Ridisegnata dopo il ruling re-scope
(docket item 2 RISOLTO) come previsto dal protocollo: cambiabile solo via docket.
La soglia decode (provv. ≥240 tok/s) diventa vincolante quando la fase 2 la fissa
in spec col Pareto su attribuzione + microbench. Fase 1 del v1 resta DONE.

| # | phase | done-when (mechanical) | authority delta | owns | status |
|---|-------|------------------------|-----------------|------|--------|
| 1 | **Attribuzione decode wall** (v1) | 2 JSON committati + journal | none | scripts/, results/engine/ | **done** (2026-07-30 it.1: quota fuori-GPU 20%, gpuBusy scala con kvLen; verifier PASS) |
| 2 | **Spec B2 v2 + microbench attn-split** — design kernel attention split-context (partizioni kvLen, softmax parziale m/l/acc, riduzione, fattore split, scratch), loop multi-step (K/submit, feedback on-GPU, streaming a raffiche), tap+telemetria, metodologia gpuBusy, soglie (decode, dispatch) da Pareto su attrib+microbench; microbench del kernel split IN ISOLAMENTO (pattern kernel-diag: pesi/KV reali, confronto vs riferimento CPU + tempo vs attnFusedWgsl a ctx bench) | `docs/superpowers/specs/*engine-fase-b2-design.md` esiste con sezioni grep-abili ("Attention split", "Decode loop multi-step", "Streaming", "Tap e telemetria", "Metodologia gpuBusy", "Soglie", "Rischi"); JSON microbench committato in `results/engine/` (correttezza + ms a ctx ~570 vs kernel attuale); entry richiesta ruling appesa a `docket.md` | none | `docs/superpowers/specs/`, `src/engine/kernels/` (SOLO kernel nuovo per microbench), `scripts/`, `results/engine/`, `.harness/goals/engine-fase-b2/` | **done** (2026-07-30 it.2: spec scritta, microbench 4.53×@ctx576/7.18×@ctx1024 con parità identica al fuso, JSON committato; ruling pendente, docket 3) |
| 3 | **Kernel attention split nel decode + parità** — integrazione nel piano fuso (sostituisce attnF nel percorso decode), error-scope da landmine B1, kernel-diag esteso | `npm test` verde (unit CPU-side sul piano split); `scripts/conformance-engine.mjs` exit 0 (gate doppio) col kernel split attivo; kernel-diag/prefill-diag PASS a cold-start; profiler: createBindGroup=0, submit/token=1 invariati | none | `src/engine/**`, `tests/`, `scripts/` | blocked (ruling spec) — TIMEBOX 4 iterazioni: allo sforo, proposta di split B2/B3 via docket (clausola pre-negoziata, decide il PI) |
| 4 | **Decode loop multi-step K + token-identity** — K forward/submit, feedback token on-GPU (o fallback da spec), argmax in catena, readback 1/K; percorso per-token tenuto vivo come oracolo interno fino a fase 6 | unit CPU-side sul piano del loop verdi; report JSON token-identity: K>1 vs per-token, greedy, ≥256 token IDENTICI, exit 0; conformance exit 0 attraverso il nuovo loop | none | `src/engine/**`, `tests/`, `scripts/` | blocked (fase 3) |
| 5 | **Telemetria liv.2 + profiler nel nuovo loop** — tsq ring compatibile con K/submit (mapAsync SOLO post-submit), finestra decode del profiler aggiornata | run liv.2 con gpuMs reale non-null (JSON); conformance exit 0 con telemetryGpu:true; profiler JSON con submit/token e dispatch/token; target dispatch di spec rispettato O archiviazione motivata nel ledger §I | none | `src/engine/**` (percorso telemetria), `scripts/`, `results/engine/`, `docs/engine/ideas-ledger.md` (§I) | blocked (fase 4) |
| 6 | **Bench + non-regressione + chiusura** | bench JSON (schemaVersion 3, baseline same-day): decodeToksPerSec.mean ≥ soglia di spec E prefillMs mean ≤ 810 E overhead telemetria spenta ≤2%, con quota fuori-GPU/gpuBusy riportati; kv-rollback + prefix-cache re-run exit 0 (JSON); checklist DONE WHEN 7/7 nel journal; HANDOFF aggiornato; merge+push a goal chiuso (may-do, ruling permanente) DOPO verifier gate PASS | none | `results/engine/`, `HANDOFF.md`, docket, journal | blocked |

Note di spine:
- Il microbench attn-split sta in FASE 2 (non 3) by design: la soglia di spec si
  fissa su numeri misurati del kernel isolato (metodo B1: simulazione→Pareto→
  soglia vincolante), e il rischio-kernel si vede PRIMA di integrare.
- Ordine 3→4: prima il kernel (la leva grossa, ~4 ms), poi il loop multi-step
  (~1.5 ms) — ogni fase tiene conformance exit 0 come invariante; il multi-step
  si costruisce sopra un attention già conforme.
- Fase 3 ha il timebox esplicito (4 iterazioni) con la clausola di split
  pre-negoziata dal ruling: la proposta va a docket, la decide il PI. "Non
  abbiamo fretta" (ruling): il timebox protegge dal ratholing, non dalla
  lentezza — iterazioni che AVANZANO con evidenza non fanno scattare la clausola.
- Il guard-rail v1 sul gpuBusy non esiste più: i guard sono parità (conformance),
  non-regressione (prefill/rollback/prefix-cache) e headline con baseline
  same-day nello stesso JSON.
- Benchmark pubblico: fuori dal goal by contract — nessuna fase lo tocca.
