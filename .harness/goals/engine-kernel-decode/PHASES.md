# PHASES — engine-kernel-decode (decomposizione it.0, 2026-08-13)

Sequenziale: le fasi 1 e 2 toccano lo STESSO file (`src/engine/kernels/wgsl.ts`),
quindi nessun `parallel-group` — la proprietà esclusiva dei path non c'è.

**Metrica obiettivo del goal**: tok/s del 4B a contesto ≥ 6000. Oggi **9,95**
(100,52 ms/token, misura it.59). Bersaglio 30 (33,3 ms/token). Ogni riga
dichiara quanto muove QUESTA metrica, o si dichiara gate.

**Punto di partenza, scomposto** (it.59 + revisione): a ctx 6333 il token è
100,5 ms = **65,8 di scansione KV** (10,4 µs/posizione, 6,3 GB/s) + **27,1 di
corpo** (133 G pesi/s) + **7,6 di coda** (lm_head + argmax + readback).

Gate PERMANENTI a ogni fase (impliciti in ogni done-when): ktest tutti PASS,
`npm test` verde, `npx tsc --noEmit` pulito, argmax del golden 4B IDENTICO al
pin, GLM non-regredito.

| # | phase | done-when (mechanical) | veicolo | authority delta | owns | status |
|---|-------|------------------------|---------|-----------------|------|--------|
| 0 | **Sonde e varianti** — decidere SE le leve esistono, prima di riscrivere | JSON committato in `results/engine/` con, per OGNI variante isolata, G pesi/s e GB/s misurati su forma reale, zero run di modello: **(a)** sonda `subgroups`/`subgroupAdd` — presente/assente sul device, dichiarato; **(b)** attenzione: forma attuale · un workgroup per gruppo GQA (KV letta 1 volta invece di 4) · letture vec4 · softmax in streaming senza `scores[ctxMax]` · split del contesto su più workgroup; **(c)** GEMV q4_0: forma attuale · vec4 · `subgroupAdd` · 2-4 righe per workgroup · `dot4I8Packed` se esposto. Predizioni PRE-REGISTRATE prima di eseguire e graduate dopo. **REGOLA DI STOP: se nessuna variante supera la forma attuale di ≥ 1,5x, la fase chiude il goal e il piano si riscrive** | `research-campaign` (prediction-gated: pre-registra → esegui → grade indipendente → memo) | none | `src/microbench/**`, `scripts/**`, `results/engine/**`, `docs/deep-dive/**` | ready |
| 1 | **Attenzione riscritta** — è il 65% del token a contesto vero | (a) `attnDecodeWgsl` sostituito da un kernel con **softmax in streaming** (niente `scores[ctxMax]`), **KV letta una volta per gruppo GQA** e **letture vec4**; (b) ktest dell'attenzione PASS contro il cpuref f64 con la tolleranza di oggi — **NON bit-identico per costruzione**: cambia l'ordine delle somme, dichiarato PRIMA di iniziare; (c) argmax del golden 4B IDENTICO al pin (gate secco); (d) ms/token a ctx 388 **e** a ctx 6333 PRIMA e DOPO nello **stesso** JSON, con la pendenza µs/posizione ricalcolata (riferimento da battere: 10,4 µs/pos, 6,3 GB/s); (e) `maxComputeWorkgroupStorageSize` richiesto dal motore **< 16384 B a qualunque ctxMax**, e il commento di `gpulimits.ts` che dichiara il path Qwen "indipendente dal contesto" corretto o reso vero (docket item 2) | `sdd-conductor` (build multi-task: kernel + limits + ktest + fixture) → poi `second-opinion` sul kernel numerico | none | `src/engine/kernels/**`, `src/engine/gpulimits.ts`, `src/engine/q35gpumodel.ts`, `tests/**`, `results/engine/**` | ready |
| 2 | **GEMV quantizzati** — il corpo, 27,1 ms a 133 G pesi/s | (a) i GEMV quantizzati adottano la variante vincente di fase 0 (vec4 / subgroup / N righe per workgroup); (b) G pesi/s del corpo PRIMA e DOPO nello stesso JSON — riferimento 133, esterno sullo stesso GGUF/host 500 (llama.cpp Vulkan); (c) **copertura della convenzione dichiarata**: quanti dei GEMV la adottano su quanti esistono, con la worklist e le eccezioni motivate — una convenzione applicata al 30% è il difetto, non il progresso; (d) GLM non-regredito: b12 optimistic entro ±5% di 13.172 / 31,26 / 14,74, misura fresca su host dichiarato; (e) ktest tutti PASS, argmax golden identico | `sdd-conductor` + `pattern-coverage` come gate pre-merge | none | `src/engine/kernels/**`, `src/engine/**`, `tests/**`, `results/engine/**` | ready |
| 3 | **CHECKPOINT — 30 tok/s o esclusione** | `node scripts/q35-bench-run.mjs --prompt-idx 0 --n-decode 64 --vram-gib 8 --declared quiescent` → JSON committato con `decodeContext.startPositions ≥ 6000` e `decode.tokS ≥ 30,0`; OPPURE JSON `kind: "q35-decode-kernel-checkpoint"`, `decision: "excluded-by-numbers"`, con banda efficace per segmento, tetto residuo e il conto che mostra perché la soglia non si raggiunge. In entrambi i rami: ratchet di correttezza intatti e non-reg GLM fresca | diretto | none | `results/engine/**`, `docs/engine/**` | ready |
| 4 | **Chiusura** | checklist del DONE WHEN del contratto voce per voce con evidenza; HANDOFF §1 aggiornato; triage del docket; il goal successivo (TEMPO AL PRIMO TOKEN) ereditata la lista di ciò che è stato escluso coi numeri | diretto | none | `docs/**`, `HANDOFF.md`, `.harness/goals/**` | ready |

**Legame con l'obiettivo, riga per riga** (ruling C9):

- riga 0 — **GATE**, non muove la metrica: decide se le leve esistono prima di
  spendere iterazioni a riscrivere. È l'unica riga che può chiudere il goal.
- riga 1 — attacca i **65,8 ms** di KV. Se la scansione passa da 6,3 GB/s a 60
  (10x, un decimo del picco misurato), la KV scende a ~6,6 ms e il token a
  ~41 ms = **24 tok/s** [proiezione, non promessa].
- riga 2 — attacca i **27,1 ms** di corpo. A 300 G pesi/s (2,3x, sotto i 500 di
  llama.cpp) il corpo scende a ~12 ms e il token, con la riga 1, a ~26 ms =
  **38 tok/s** [proiezione].
- riga 3 — **GATE**: misura il vero e decide.
- riga 4 — **GATE** di chiusura.

Tre righe di lavoro, tre gate: il piano non fotografa lo stato.

**Taglie stimate** (1-4 iterazioni; a 3x di sforamento la riga si ferma e si
ri-pianifica, ruling C9): 0 → 1-2 · 1 → **2-4** (la riscrittura dell'attenzione
tocca il kernel più delicato del motore e cambia i risultati numerici per
costruzione) · 2 → 2-4 · 3 → 1 · 4 → 1.

**Rischio più alto: la riga 1.** È l'unico kernel di cui il gate non può essere
la bit-identità — la somma in streaming cambia l'ordine delle operazioni. Il
paracadute è a due strati: il ktest contro il riferimento CPU in f64 con la
tolleranza dichiarata PRIMA, e l'argmax del golden identico al pin, che è secco.
Per questo la riga passa da `second-opinion` dopo la build: un solo revisore su
un kernel numerico è fiducia mal riposta.
