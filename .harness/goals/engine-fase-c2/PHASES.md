# PHASES — engine-fase-c2

Decomposizione iteration-0 (2026-07-31). Sequenziale salvo il gruppo A (owns
disgiunti). Le fasi 3-6 sono gated dal ruling di spec (fase 2), come da
convenzione C1. Nessuna fase richiede authority oltre il grant di GOAL.md.

| # | phase | done-when (mechanical) | authority delta | owns | status |
|---|-------|------------------------|-----------------|------|--------|
| 1 | Floor & golden: pin del floor tok/s (llama-bench C1) in un doc di gate; golden logits GLM via oracolo (pattern gen-golden, commit 5f55650) sul corpus C1 [provvisorio: la spec può estenderlo] | script run exit 0; `results/engine/golden/glm47flash/*.json` esistono con SHA-256 GGUF + commit oracolo nel payload; floor registrato in `docket.md` come costante di gate | none | scripts/gen-golden*, tools/oracle-moe/ (riuso read-only), results/engine/golden/glm47flash/ | done (it.1) |
| 2 | Spec C2 (spec-first): subset GGUF deepseek2, formulazione MLA (naive→absorbed come decisione), kernel MoE + replica router C1, residenza minima RAM→VRAM LRU predisposta al prefetch, soglie conformance (logits KL/top-k; routing ≥99% target), corpus, sanity-gate | `docs/superpowers/specs/2026-07-31-engine-fase-c2-design.md` esiste e copre i 6 punti del DONE WHEN §1; entry RULING RICHIESTO nel docket | none | docs/superpowers/specs/ | done (it.1, ruling 2026-07-31) |
| 3 | Reader GGUF deepseek2: estensione gguf.ts/shape.ts/quant.ts al subset GLM con validazione hard (exit su chiave/tensore mancante o layout inatteso) | `npm test` verde con suite parsing GLM (SHA d0bbdfc… pinnato nel test); `tsc --noEmit` pulito; load headless del GGUF exit 0 | none | src/engine/{gguf,shape,quant}.ts, tests/ | done (it.2) |
| 4 | Kernel MLA (kv_lora 512 + rope 64, f32-first) + primo layer denso: conformance per-layer vs cpuref-f64 e golden | unit ktest verdi sui kernel nuovi; conformance layer-level entro soglie di spec (gate doppio cpuref/golden, lezione landmine) | none | src/engine/kernels/, src/engine/{attnsplit,cpuref,gpuforward}.ts, tests/ | done (it.3-5; gate golden al full-model, v. journal it.5) |
| 5 | MoE: router (sigmoid+bias+top-4+norm, replica C1) + GEMV dequant-fusa per-expert + shared expert + residenza minima (staging RAM, cache VRAM LRU ~2.573 slot, telemetria hit/upload) | conformance routing vs traccia C1 ≥ soglia spec su corpus C1; unit verdi su router/cache; contatore upload RAM→VRAM presente nel report telemetria | none | src/engine/ (moe/residenza nuovi file), tests/ | tecnica DONE it.6-9; gate routing → RULING docket item 4 |
| 6 | E2E: forward GLM completo nel piano statico, conformance logits full-model, bench con i GATE (decode ≥13.4, prefill ≥56.6 tok/s) + non-regressione Qwen (conformance A + bench first-light ≥ ultimi valori committati) | run conformance exit 0 + JSON in results/engine/ entro soglie; bench JSON con entrambi i gate tok/s PASS; bench+conformance Qwen ri-eseguiti ≥ valori 2026-07-30 | none | src/engine/, results/engine/ | blocked (fasi 3-5) |
| 7 | Chiusura: docket input C3 (residenza minima osservata vs simulatore, costo upload/expert), ledger/direction ove stale, HANDOFF refresh, merge su main | checklist DONE WHEN 7/7 nel journal; verifier gate finale PASS; merge+push eseguiti (ruling permanente) | none | docs/, .harness/, HANDOFF.md | blocked (fase 6) |

Note:
- Fase 1 non tocca src/engine e riusa infra C1: loop-runnabile mentre il
  ruling di spec pende. Se la spec cambia il corpus, i golden extra si
  rigenerano in fase 6 (costo: minuti).
- Il rischio grosso del goal è la fase 4 (MLA): nessuna clausola di fallback
  pre-negoziata qui — se il timebox di spec sfora, la via va a docket (il
  contratto non autorizza scorciatoie sul path attention).
- Emendamento non-regressione (contratto v1): i gate di fase 6 sono HARD;
  regressione non assorbibile ⇒ docket, non merge.
