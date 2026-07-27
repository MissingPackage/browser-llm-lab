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
ri-verificato, contratto rispettato, nessun raffinamento] [fase 4: 2026-07-27,
iterazione 5, subagent con skill sui fatti del dump WGSL reale → sezione bottleneck di
dequant-kernels.md; il pre-run GREEN è stato aggiornato e in parte SMENTITO dai fatti
nuovi (swap q4f16 declassato); contratto rispettato] [fase 5: 2026-07-27, iterazione 6,
subagent con skill su run-B ri-verificato + kernel 008/009/028 del dump; l'agente ha
verificato da fonte primaria HF i config (context 32768, chunk 2048, 24 layer, GQA 14:2)
superando i miei [VERIFY]; contratto rispettato]

## 2026-07-27 — Iterazione 6 (fase 5: doc kv-cache-layout.md)

Doc completo in `docs/deep-dive/kv-cache-layout.md`. Pezzo forte: il layout di pagina
letto direttamente dall'aritmetica di indicizzazione del kernel 009 —
`[pagina][K|V][kv-head][pos∈16][dim∈64]`, 4096 f32 = 16 KiB/pagina, V a offset +2048.
Attention decode (028) = FlashAttention-style online softmax, workgroup (16,7,2) = 14
Q-head GQA. Allocazione: create_tir_paged_kv_cache dimensionata su context_window_size
32768 → 768 MiB di KV f32 al tetto = ~2.9× i pesi q4; working set reale dei run: ~11 MB.

Fatti nuovi verificati dal dogfood (fonte primaria HF): prefill_chunk_size=2048 → il
prompt bench (469 tok) gira come UN mega-dispatch, il chunking non è mai stato
esercitato da nessun run; sliding_window=-1 (attention piena). Il dogfood ha anche
scartato con argomenti solidi l'adattamento ds4-swap alla KV (incoerente con attention
piena) e l'offload stile FlexGen (su memoria unificata non libera nulla).

Done-when fase 5: 3 heading ✓, citazioni results (4) ✓, bundle versione ✓, skill in
journal ✓. Docket delta: #6 (warm-up pre-ramp = QUINTO candidato per 2 slot, attacca il
#12 ereditato).

## 2026-07-27 — Iterazione 5 (fase 4 parte 2: doc dequant-kernels.md)

Doc completo in `docs/deep-dive/dequant-kernels.md`, primo doc con fonte primaria (i 34
WGSL del dump, non il bundle). Struttura stabilita: schema q4 (8 nibble/u32, offset −7,
scale f32 per 32, FMA fusa — mai un buffer dequant in VRAM); DUE forme di kernel generate
da TVM (prefill: GEMM tiled 8×8 con dequant in shared memory; decode: GEMV 64×1 con
dequant nei registri e riduzione in workgroup).

**Finding principale del goal finora** (sezione "Perché i numeri"): il roofline di banda
pesi dà tetti ~2000 (4090) e ~180 (S22) tok/s contro 101-116 e 6.99 misurati — **entrambi
i device al 4-6% del tetto**. Il rapporto relativo (~14×) segue la banda (~11×), ma in
assoluto il cap NON è la banda: candidati occupancy GEMV vs launch overhead (~34
dispatch/token). Ridimensiona sia il "memory-bound da manuale" sia il valore atteso dello
swap q4f16 (docket #2 → secondario, vedi docket #5). Il micro-bench di fase 6 è il
discriminatore progettato apposta.

Done-when fase 4: 3 heading ✓, citazioni results (4) ✓, bundle con versione ✓, skill in
journal ✓ (sopra). Dogfood: contratto ok, idea fuori dagli schemi con prior art solido
(megakernel, TensorRT-LLM 14.6% launch overhead su Qwen2.5-1.5B → engine-notes).

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

## 2026-07-27 — Iterazione 4 (fase 4, parte 1: dump WGSL reale)

Infrastruttura di dump costruita e funzionante: `tools/wgsl-dump.mjs` (Playwright, patch
di `GPUDevice.prototype.createShaderModule` iniettata NEL worker via `page.workers()` —
zero modifiche alla SPA). **34 kernel WGSL reali** catturati dal run live 4090
(Qwen2.5-0.5B q4f32_1, Chrome branded headed) in `wgsl-dump/`: famiglia
`fused_dequantize*_NT_matmul*` (fase 4), `batch_prefill_ragged_kv` +
`batch_decode_paged_kv` + `tir_kv_cache_transpose_append` (fase 5), sampling/argsort.

Debug onesto, due tentativi falliti prima del successo:
1. run #1: timeout 900s senza osservazione (script senza logging) + colpiva un dev server
   preesistente su :5173 NON nostro (pid 487440, lasciato vivo — probabilmente di
   Cristiano); il mio vite era su :5177.
2. run #2 (con logging): fetch ok, poi **hang eterno in warm-up** — causa: Chrome headless
   cade su SwiftShader (`maxComputeInvocationsPerWorkgroup=256` = spia software rasterizer)
   e il mio script non aveva la guardia isReal del driver e2e. Fix: HEADED=1 + guardia
   isReal portata nello script.
3. run #3 headed su GPU reale: STATUS done, 34 shader, exit 0.

Conferme dal vivo sul kernel 014 (decode matmul embed/lm_head): 8 pesi nibble per u32,
dequant `(w>>k & 15) − 7` × scale per gruppo di 32 (threadIdx.x>>2), FMA fusa, riduzione
workgroup via red_buf0, guard blockIdx.z del workaround 65535 presente in testa.

Fase 4 done-when NON ancora soddisfatto (doc mancante): il doc dequant-kernels.md si
scrive nell'iterazione 5 con questo materiale primario + run-C. PHASES riga 4 resta
ready con nota "dump fatto".

## 2026-07-27 — Iterazione 7 (fase 6 parte 1: motore micro-bench + run reale 4090)

Costruito il micro-bench matmul: `src/microbench/` (mbSchema v1 con `skipped[]`, stats
pure unit-testate, kernels.ts con GEMV parametrici modellati sulla forma dei kernel TVM
dumpati — q4f32 con nibble/offset−7/scale∈32, f32, f16 —, runner con timestamp-query +
fallback CPU, worker + page) + entry `microbench.html` nel build vite. Driver headed in
`tools/microbench-run.mjs`. Test 102/102 (12 nuovi), `tsc --noEmit` pulito, build ok.

Due bug REALI trovati dal primo run e fixati (entrambi lezioni da doc):
1. **Quantizzazione timestamp Chrome (~100µs)**: singolo dispatch piccolo → delta 0 →
   "tempo non positivo". Fix metodologico: batch di 16 dispatch per campione (serializzati
   dal WAW hazard su y), delta/16.
2. **Device senza requiredLimits** → binding default 128 MiB → cella f32 8192² garbage
   silenziosa. Lo STESSO errore concettuale del cap hardcoded di WebLLM documentato in
   buffer-limit-2gb.md, riprodotto in casa. Fix: requiredLimits al max dell'adapter +
   error scope validation/oom + checksum → celle fallite = `skipped[]` esplicito.

Run valido committato: `results/microbench/microbench-4090-linux-2026-07-27T04-28-42-421Z.json`
(10 celle, 0 skip, tutto timestamp-query). Segnale:
- f32 L2-resident (≤64MB): 480-1000 GB/s (banda cache); f32 oltre L2 (256MB, 1GB):
  **432-436 GB/s** = banda VRAM misurata (~75% del datasheet ~576).
- q4 a 16384² (160MB > L2): 86 GB/s "effettivi" MA — CORREZIONE del verifier, il mio
  primo conto (172G, 1.6×) divideva i GB/s per 0.5 B/peso dimenticando scale+I/O in
  bytesRead — pesi/s reali: q4 137.4G vs f32 108.9G = **1.26×** oltre-L2. A 8192² il
  rapporto apparente è 3.6× ma è sleale (q4 40MB ancora L2-resident, f32 256MB già
  VRAM-bound). Confronto onesto = solo la riga 16384². Analisi per il doc.
- f16 assente: Chrome branded non espone shader-f16 su questo stack Linux/NVIDIA
  (skip loggato dal runner, non silenzioso) — dettaglio da riportare nel doc.

Fase 6 done-when PARZIALE: route+test+build+run reale ✓; manca `micro-bench-matmul.md`
(iterazione 8). PHASES riga 6 resta ready con nota.

## 2026-07-27 — Iterazione 8 (fase 6 parte 2: doc micro-bench-matmul.md)

Doc completo in `docs/deep-dive/micro-bench-matmul.md`: metodologia (incl. le 2 lezioni
del primo run come parte del design), tabella numeri 4090 con regime L2/VRAM per cella,
3 letture (banda VRAM reale ~435 GB/s = roofline onesto −25%; curva piegata sul confine
L2; q4 1.26× f32 in pesi/s oltre-L2 con smontaggio della metrica GB/s ingannevole),
stima 75-85% del budget/token = orchestrazione (chiude la domanda di dequant-kernels.md
per ordini di grandezza; conferma kernel-per-kernel instradata a engine-notes), slot
M4/S22 "pending" espliciti con istruzioni per i run manuali.

Done-when fase 6 COMPLETO: build ✓ (entry microbench nel dist), test 102/102 ✓, JSON
schema-versionato da run reale non-SwiftShader in results/microbench/ ✓ (iterazione 7),
doc con metodologia+numeri+slot pending ✓ (grep "pending": 2). Fase 6 done.

## 2026-07-27 — Iterazione 9 (fase 7: engine-design-notes + closure sweep)

**Closure sweep [VERIFY]**: 25 marker nei 5 doc → 0. Risoluzioni: 4 chiusi con misure
del micro-bench di fase 6 (quota orchestrazione); 2 chiusi con fonti esterne (datasheet
4090 laptop 576 GB/s — VideoCardz/TechSpot; Exynos 2200 51.2 GB/s — nanoreview/
Notebookcheck); 1 verificato in-repo (tutti i run sono 0.5B); i restanti riformulati da
asserzioni-con-marker a questioni aperte esplicite o buchi dichiarati, consolidati nella
sezione "Questioni aperte" di engine-design-notes.md.

**engine-design-notes.md scritto** (doc personale): 5 fatti che vincolano il design,
backlog delle 12 idee instradate engine-notes dai 4 doc, 6 questioni aperte, sintesi
"la forma del motore in tre frasi" (minimizzare sync, misurarsi da solo, negoziare i
limiti). Contiene i filename di tutti gli altri 5 doc (grep ok).

**Done-when fase 7, tutto verde meccanicamente**: engine-notes con 5 filename ✓;
grep [VERIFY] docs/deep-dive/*.md = 0 ✓; npm test 102/102 ✓; tsc --noEmit pulito ✓;
npm run build exit 0 ✓; git diff goal-fase-2-start -- src/adapters/webllm.ts = vuoto ✓;
experiments/ = 0 entry (≤2) ✓.

## 2026-07-27 (sera) — Post-chiusura: dati PI (M4 + S22 f16) e correzione doc

Il PI ha eseguito il docket #2 (S22 q4f16_1) e il primo run M4 Pro; dati committati su
main (340ab91). Esiti nel docket #7-#8. Correzione a buffer-limit-2gb.md sul branch:
la tesi "binding cap 2³¹−4 = vincolo dell'API" è falsificata dal probe M4 (2³²−4 su
Metal) → riformulata come scelta di implementazione per-backend. Nessun altro doc
richiede correzioni da questi dati (la tesi orchestrazione ne esce anzi rafforzata:
M4 ≈ 4090 in decode con metà banda).
