# PHASES — engine-fase-b1

Decomposizione del contratto (GOAL.md) in fasi loop-runnable. Cambiabile dopo
l'iterazione 0 solo via docket. Le soglie marcate "da spec" (prefill 2×, decode ≥120)
vengono fissate in fase 1 e da lì in poi sono vincolanti per i verifier.

| # | phase | done-when (mechanical) | authority delta | owns | status |
|---|-------|------------------------|-----------------|------|--------|
| 1 | **Spec fase B1** — layout forward multi-token (M≤8), contratto crop/length-pointer, formato on-disk prefix-cache OPFS (chiave hash token-id + sha256 GGUF + versione layout, mismatch ⇒ throw), quota/eviction, soglie (prefill ≤1/3 baseline [ruling Pareto], decode ≥120) | `docs/superpowers/specs/*engine-fase-b1-design.md` esiste con sezioni contratto grep-abili ("Forward multi-token", "Crop", "Formato prefix-cache", "Quota/eviction", "Soglie"); entry di richiesta ruling appesa a `docket.md` | none | `docs/superpowers/specs/`, `.harness/goals/engine-fase-b1/` | **done** (2026-07-29; ruling pendente, docket 3) |
| 2 | **Diagnosi telemetria liv.2** (TIMEBOXED 1 iterazione, da contratto) — timestamp azzerati + corruzione compute su Chrome/Linux/NVIDIA; A/B a una variabile partendo dal microbench che funziona standalone | nota scritta in `docs/engine/` con esito (root-cause O "riprodotto e documentato come bug browser" con repro minimo); journal aggiornato; se fix: conformance ancora exit 0 con `telemetryGpu:true` | none | `docs/engine/` (nota diag), pagina diag dedicata, `src/engine/gpuforward.ts` SOLO percorso telemetria | **done** (2026-07-29: ROOT-CAUSE trovata e FIXATA — mapAsync pre-submit, bug nostro; matrice A/B 5 varianti; conformance PASS con tsq attivo; docs/engine/tsq-diag-2026-07-29.md) |
| 3 | **Forward multi-token M≤8 + parità** — kernel/piano per chunk di prefill (GEMM small-batch o loop M sul piano fuso, decisione in spec), tap preservati | `npm test` verde con unit CPU-side sul piano di chunking; `scripts/conformance-engine.mjs` exit 0 (gate doppio fase A) col percorso prefill M>1 attivo | none | `src/engine/**`, `tests/`, `scripts/` | **done** (2026-07-30, it.3: conformance PASS con M>1 attivo, 135/135 unit; root-cause+fix bug Tint decl-in-loop) |
| 4 | **Rollback KV** — crop(P) con length pointer, semantica da spec | unit CPU-side sulla semantica crop verdi; report JSON committato in `results/engine/` da run meccanico "genera N, crop a P, rigenera" vs run fresco: sequenze token IDENTICHE, exit 0 | none | `src/engine/**`, `tests/`, `scripts/`, `results/engine/` | **done** (2026-07-30, it.4: 3 check identici, conformance invariante PASS) |
| 5 | **Prefix-cache OPFS** — save/restore stato KV, restore in worker NUOVO (sessione fredda), continuazione token-identica | unit sull'indice prefix-cache verdi; report JSON committato: continuazione identica al run ininterrotto + tempo restore < tempo re-prefill (entrambi nel JSON) | none | `src/engine/**`, `tests/`, `scripts/`, `results/engine/` | **done** (2026-07-30, it.5: restore 104 ms vs re-prefill 2977 ms, continuazione identica in worker nuovo) |
| 6 | **Bench + non-regressione + chiusura** — prefill vs baseline fase A, decode invariato, contatori invariati | bench JSON in `results/engine/`: prefillMs mean ≤ soglia di spec (default 1/2 di ~2.44s) E decodeToksPerSec.mean ≥ 120; JSON profiler: createBindGroup=0, submit/token=1, dispatch/token ≤130; checklist DONE WHEN 8/8 nel journal; HANDOFF aggiornato; merge+push a goal chiuso (may-do, ruling permanente) DOPO verifier gate PASS | none | `results/engine/`, `HANDOFF.md`, docket, journal | ready (ATTENZIONE: prefill chunked oggi ~3.1 s/468 tok, sopra baseline — servono fast-kernel GEMM chunk per la soglia 3×) |

Note di spine:
- Parallel-group A (fasi 1+2): owns disgiunti (spec vs docs/engine+percorso telemetria);
  la diagnosi è indipendente dalle decisioni di spec ed è timeboxed by contract — a
  timebox scaduto si chiude con la nota "documentato", non si estende.
- La sequenza 3→4→5 mette la parità multi-token PRIMA della persistenza: crop e
  prefix-cache si testano contro un prefill veloce già conforme, e ogni passo tiene
  la conformance exit 0 come invariante (convenzione fase A: parità invariata a ogni
  passo).
- Fase 3 è la più grossa (GEMM small-batch): se supera 4 iterazioni, split via docket.
- Il lavoro sul floor dispatch ≤100 NON è in nessuna fase: è B2 (must-docket by
  contract — le occasioni di fusione viste in fase 3 si annotano, non si implementano).
