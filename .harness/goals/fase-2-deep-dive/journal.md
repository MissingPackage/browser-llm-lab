# Journal — fase-2-deep-dive

## 2026-07-27 — Iterazione 0 (setup)

Scaffold del goal spine: GOAL.md (contratto approvato in chat dopo goal-brief), PHASES.md
(7 fasi sequenziali), docket.md (voce #1 plan-check con pre-autorizzazione condizionale),
digests.md. Tag `goal-fase-2-start` su `main` per il check diff-clean di
`src/adapters/webllm.ts` a fine goal. HANDOFF.md §1 puntato a questo goal, fase 1.
Prossimo: product-loop, iterazione 1 = skill `bottleneck-brainstorm` v1.

## 2026-07-27 — Iterazione 1 (fase 1: skill bottleneck-brainstorm)

Ciclo TDD writing-skills completo. **RED**: 2 subagent senza skill su scenari reali
(buffer-2GB, KV/S22) — baseline NON fallisce in capacità (prior-art, scoring, citazioni già
presenti con lo spec su disco); fallisce in FORMA (varianza di struttura tra run). Skill
quindi scopata come contratto di output (ricetta), non disciplina. **GREEN**: 1 subagent
CON skill su scenario nuovo (dequant) — contratto rispettato in tutti e 5 i punti, artefatto
verbatim in `baseline/run-C-green-dequant.md`. Nessun loophole → niente REFACTOR.
Commit `d79bc32` su feat/fase-2-deep-dive. Done-when fase 1: tutti i check verdi (grep
frontmatter/sezioni, commit sul branch).

**Findings load-bearing emersi dai run** (per fasi future, salvati in `baseline/`):
- Run A: WebLLM richiede hardcoded 1 GiB di `maxStorageBufferBindingSize` (fallback 128 MiB,
  bundle index.js:4067-4082) — il soffitto per-binding reale è 1 GiB, non ~2 GiB del probe.
  Confermato dal run LLVMPIPE (134217728 = 1<<27). → fase 3.
- Run GREEN: il bench usa `q4f32_1` ovunque; l'S22 espone `shader-f16` mai usato. Candidato
  esperimento quasi a costo zero (swap modelId). → docket #2, decisione PI.
- Run B: KV cache dimensionata sul max contesto configurato, page_size hardcoded 16,
  ipotesi DVFS per la varianza TTFT S22 (si lega a docket #12 ereditato). → fase 5.

**Verifier gate**: FAIL alla prima passata (giusto) — journal/digest/HANDOFF/PHASES fermi
all'iterazione 0 e GREEN senza artefatto; sanato in questa stessa iterazione (questo entry,
run-C salvato, PHASES riga 1 → done, HANDOFF §1 → fase 2). Re-check focalizzato: vedi sotto.

**Invocazioni skill bottleneck-brainstorm** (registro dogfooding richiesto dal done-when
delle fasi 2-5): [fase 2: 2026-07-27, iterazione 2, subagent con skill → sezione
"Bottleneck & vie d'uscita" di compute-shader-dispatch.md, contratto rispettato, nessun
raffinamento skill necessario] [fase 3: —] [fase 4: pre-run GREEN 2026-07-27, da rifare
in-fase] [fase 5: —]

## 2026-07-27 — Iterazione 2 (fase 2: doc compute-shader-dispatch.md)

Doc completo in `docs/deep-dive/compute-shader-dispatch.md`. Metodo: estrazione meccanica
dal bundle via subagent (6 punti, righe citate e verificate), sezioni "Cosa fa" + "Perché
i numeri sono quelli" scritte nel main loop (3 citazioni spot-check a mano: flushCommands
4411-4418, encoder condiviso 4644-4649, get_fmap 7440-7442, sync per-token 11126-11135 —
tutte confermate), sezione bottleneck via dogfood skill (primo uso ufficiale in-fase).

Fatti architetturali chiave stabiliti: kernel WGSL nel wasm TVM (get_fmap/get_shader/
update_prebuild); encoder condiviso con UN submit per molti kernel e UNA sync per token
(readback id campionato); nessun timestamp-query richiesto → tempo GPU per-kernel
invisibile; load warm (1.5-1.9s 4090, 6.1s S22) = candidato compile-pipeline, fetch pesi
e compile shader oggi sequenziali per struttura (12549-12564), non per dipendenza.

Done-when fase 2: 3 heading letterali ✓, ≥1 citazione results (10) ✓, bundle con versione
(0.2.84) ✓, invocazione skill in journal ✓ (sopra), nessun raffinamento skill da
committare. 6 marker [VERIFY] nel doc — ammessi in-fase, sweep a fase 7.

Docket delta: #3 nuovo (candidati esperimento dal dogfood: multi-step decode, overlap
fetch/compile — scelta slot PI; timestamp-query è già infrastruttura di fase 6 per spec).
