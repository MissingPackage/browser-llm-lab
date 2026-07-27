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
raffinamento skill necessario] [fase 3: 2026-07-27, iterazione 3, subagent con skill →
sezione bottleneck di buffer-limit-2gb.md partendo dal materiale baseline run-A
ri-verificato, contratto rispettato, nessun raffinamento] [fase 4: pre-run GREEN
2026-07-27, da rifare in-fase] [fase 5: —]

## 2026-07-27 — Iterazione 3 (fase 3: doc buffer-limit-2gb.md)

Doc completo in `docs/deep-dive/buffer-limit-2gb.md`. Le citazioni chiave del materiale
baseline run-A ri-verificate a mano prima dell'uso (blocchi 4050-4066 e 4067-4082:
richiesta hardcoded 1 GiB per ENTRAMBI i limiti, fallback 256/128 MiB; 7086-7106:
upload per-tensore con `yield device.sync()` per ogni tensore — dettaglio NUOVO non
presente in run-A, emerso dalla ri-verifica).

Tesi del doc: il "muro dei 2 GB" è a tre piani (adapter ~2 GiB / richiesta WebLLM 1 GiB /
fallback 128-256 MiB) e il runtime vive al piano di mezzo; lo spezzettamento pesi è il
default (un buffer per tensore), quindi il cap limita il singolo tensore più grande, non
la taglia modello; nessun run committato ha mai toccato il muro — il bottleneck osservato
oggi è l'upload serializzato con sync per-tensore dentro la finestra di load warm.

Done-when fase 3: 3 heading ✓, citazioni results (8) ✓, bundle con versione (0.2.84, 2
occorrenze) ✓, invocazione skill in journal ✓. Dogfood: contratto rispettato, spec riga
104 citata correttamente (verificata).

Docket delta: #4 nuovo — quarto candidato esperimento (diradare sync per-tensore
nell'upload pesi; attacca il load warm misurato, criterio binario loadMs prima/dopo).

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
