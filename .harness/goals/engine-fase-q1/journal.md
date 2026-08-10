# Journal — engine-fase-q1

## it.0 (2026-08-10) — goal-setup

- Contratto v1 chartered e APERTO (PI: "partiamo con la spec e poi
  direttamente il loop"); tag `goal-engine-fase-q1-start` su 8bbd56e.
- Emendamenti PI assorbiti nel contratto: terza taglia densa = 4B
  mobile-target (2B = eventuale futuro, fuori goal).
- PHASES.md: 9 fasi sequenziali (spec → reader/tokenizer → kernel DeltaNet
  kernel-level → 4B e2e → 9B+WP gap → leve bounded → 35B parametrizzato →
  tier/recall/bandmodel → chiusura), authority delta none ovunque,
  nessun parallel-group (GPU singola, albero congelato nelle run).
- Docket item 1: plan-check pre-autorizzato; punto di revisione naturale =
  prima della fase 3.
- Next: it.1 = fase 1 (spec + ratifica).

## it.1 (2026-08-10) — fase 1 DONE (verifier PASS)

- Spec depositata: `docs/superpowers/specs/2026-08-10-engine-fase-q1-design.md`
  (9 sezioni); ratifica formale del contratto in §0 (emendamento 4B incluso).
- SHA GGUF pinnate e cross-verificate sui pointer LFS (verifier): 4B Q4_0
  298fcb5f… (2.58 GB), 9B Q4_0 17670346… (5.38 GB), 35B UD-Q4_K_S
  a8138f18… (20.9 GB). Disco ~29 GB ≤ 60 authority.
- FINDING carico-portante: il Q4_0 del 35B-A3B non esiste da fonte pulita
  (unsloth/Qwen/hub verificati) → deviazione quant DICHIARATA (spec §2,
  docket item 3): MoE = UD-Q4_K_S, prezzo = dequant Q4_K expert; byte
  identici motore/oracolo preservati. Auto-quant scartata (>100 GB, must-
  docket).
- Config 4B verificata da HF (32 layer, 8 full, hidden 2560, 16/4 head,
  tie embeddings TRUE); [VERIFY] residui (9B/35B dettagli) si chiudono in
  fase 2 dal header GGUF.
- [VERIFY] llamaweb CHIUSO (spec §7): subgroups shipped Chrome 134;
  subgroup-matrix in standardizzazione, NESSUN origin trial Chrome 146-148
  (ago 2026), solo Dawn sperimentale → fase 6 = spike empirico sul flag.
- Verifier: PASS su tutti i punti del done-when, SHA cross-check incluse;
  drift none, violazioni none.
- Next: it.2 = fase 2 (reader GGUF + tokenizer; download pinnati, enum
  type del file UD, verifica supporto oracolo b10333).

## it.2 (2026-08-10) — fase 2 IN CORSO, checkpoint strumentazione (c59c76c)

- Download dei 3 GGUF pinnati in background (~29 GB, hf CLI, sequenziale
  4B→9B→35B in ~/.cache/blab-models/q35/).
- Strumentazione committata E validata: q35-manifest.json (SHA spec §1);
  q35-verify-sha.mjs (gate secco byte+hash, FAIL pulito su assenti
  VERIFICATO); q35-header-dump.py (VALIDATO su GLM: 844 tensori,
  deepseek2, 47 block — chiuderà i [VERIFY] spec §3 + inventario UD);
  corpus-tok/09-12 (emoji/CJK/ZWJ, whitespace, numeri/code, special
  adversariali; 01-08 GLM riusati per il regime).
- Al risveglio (notifica download): verify-sha (exit 0), header-dump sui
  3 file, verifica supporto qwen3_5 dell'oracolo b10333 (llama-tokenize),
  poi reader+tokenizer (grosso di fase 2, it.3).
- Oracolo localizzato: ~/.cache/llamacpp-vulkan/llama-b10333/ (include
  llama-tokenize, llama-quantize, llama-perplexity).

## it.2 seguito (2026-08-10) — download DONE, misure a valle DONE

- SHA: `node scripts/q35-verify-sha.mjs` → 3× PASS, exit 0 (done-when
  punto 1 di fase 2 CHIUSO).
- Header dump committato (`results/engine/q35-header-dump-2026-08-10.json`):
  [VERIFY] spec §3 CHIUSI (tabella aggiornata), mix UD enumerato (expert
  Q4_K 117 + Q6_K 3), densi nel set type già supportato, KV 35B
  40 960 B/token. Docket item 5.
- Oracolo b10333: supporto famiglia PROVATO (llama-bench 4B pp16 36.1 /
  tg8 17.2 CPU; llama-tokenize ok). Docket item 4 RISOLTO, no upgrade.
  Nota: llama-cli -no-cnv si impianta (probabile stdin) — llama-bench e
  llama-tokenize sono gli strumenti del protocollo, nessun impatto.
- Fixture tokenizer: `tests/fixtures/q35-tok-oracle.json` via
  `scripts/q35-tok-oracle-gen.mjs` — 12 file, 27 714 token, protocollo
  `--ids --no-bos --no-parse-special`, cross-check 4B==35B PASS in
  generazione (vocab di famiglia identico).
- RESTANO per chiudere fase 2 (it.3): reader TS `qwen35`/`qwen35moe`
  (load test sui 3), tokenizer TS con id==fixture in npm test, tsc.

## it.3 (2026-08-10) — fase 2 DONE (verifier PASS)

- Inventario nomi-tensori completo dei 3 file (q35-header-dump esteso ai
  nomi): qkv FUSA sui layer DeltaNet [d,(2·nK+nV)·hd], attn_q dei full =
  q+gate fusi [d,2·nHead·256] (attn_output_gate), QK-norm per-head, pattern
  UD sui down_exps (Q4_1 su 4 layer del 4B, Q6_K su 3 del 35B).
- `src/engine/q35shape.ts`: shape DERIVATA dai metadata + validazione hard
  dell'inventario completo (dims calcolate, allow-list tipi chiusa, throw su
  tensori extra) — la parametrizzazione di spec §3, primo pezzo. gguf.ts:
  aggiunta SOLO additiva Q4_K (superblocco 144 B).
- `src/engine/q35tokenizer.ts`: PRIMO tokenizer in-engine — BPE byte-level
  GPT-2, regex pre `qwen35` verbatim da llama.cpp (verifier: riscontrata
  IDENTICA nel binario b10333 via strings), partizionamento special fedele
  (USER_DEFINED sempre, CONTROL solo con parseSpecial — llama-vocab.cpp:3171).
- DUE correzioni di protocollo scoperte dal gate secco (docket item 6):
  `--no-escape` obbligatorio (corpus 11: \\ processato) e semantica
  USER_DEFINED (corpus 12: <think> = token singolo). Fixture v2 rigenerato,
  cross-check 4B==35B PASS. Il gate secco ha pagato: 2 bug di fedeltà presi
  PRIMA di toccare i kernel.
- Gate: SHA 3/3 exit 0; shape test 4/4 (426/427/733 tensori); tokenizer
  12 file id-identici + roundtrip decode; suite 364 PASS (+5) | 7 skip;
  tsc pulito. Verifier PASS con spot-check oracolo indipendente.
- Next: it.4 = fase 3 (kernel DeltaNet WGSL kernel-level vs cpuref-f64;
  prima il cpuref TS della catena linear-attn, spec §4).

## it.4 (2026-08-10) — fase 3 prima metà: cpuref-f64 DeltaNet (verifier PASS)

- Semantica presa dalla FONTE pinnata alla build dell'oracolo (llama.cpp
  b10333): qwen35.cpp (build_layer_attn_linear), delta-net-base.cpp (path
  AUTOREGRESSIVO = la ricorrenza; il chunked è l'ottimizzazione della stessa
  matematica), ops.cpp (ssm_conv = dot k4 su concat(stato,x); l2_norm con
  eps a FLOOR: 1/max(√Σx²,eps)), ggml-impl.h (softplus x>20?x:log1p(eˣ)).
- Ordine ESATTO della ricorrenza fissato nel cpuref: decay → lettura sk →
  delta β(v−sk) → update S → output S·q. Broadcast k-head→v-head = h mod nK
  (ggml_repeat TILA, non raggruppa); q/√hd; gated norm = RMS(o)·silu(z).
- `src/engine/q35cpuref.ts`: catena f64 completa con tap intermedi esposti
  per il ktest (qkvPreConv, convOut, q/kNorm, oCore, gated); core esportato
  puro (deltaNetStepCore).
- Test (7 PASS): 5 identità algebriche in aritmetica esatta (fit β=1 ⇒
  o(q=k)=v/√hd; ortogonalità non disturba; β=0 stato invariato; g→−∞ wipe;
  discriminante dell'ORDINE decay-prima-della-lettura) + softplus al gomito
  + campione sintetico T=12 pinnato su fixture (helper condiviso, generatore
  esplicito scripts/q35-deltanet-fixture-gen.mjs — niente auto-pin).
- Suite 371 PASS (+7) / 7 skip, tsc pulito, GLM intatto.
- RESTA per chiudere fase 3 (it.5): kernel WGSL DeltaNet + ktest vs cpuref
  sul campione (+ pesi reali 4B layer-scale), ktest 69/69 invariati.

## it.5 (2026-08-10) — fase 3 DONE: kernel WGSL DeltaNet, ktest 75/75

- `src/engine/kernels/deltanet.ts`: 3 kernel — conv causale k4 + SiLU +
  shift stato (un thread/canale, colonna propria); gates β/g (softplus
  gomito 20); CORE = un workgroup per v-head, thread = colonna j dello
  stato, decay FUSO nelle due passate (mai sweep separata), l2norm q/k in
  workgroup reduction, gated norm RMS·silu(z) in coda. Stato f32 (obbligo
  config), accumulatori tutti inizializzati (landmine Tint).
- Campione spostato in `src/engine/q35sample.ts` (il worker non importa da
  tests/; helper re-esporta per i test node).
- ktest 6 nuovi, TUTTI PASS al primo run reale: conv-256/conv-8192 (shift
  ESATTO), gates, core-hd16 e core-hd128 (dims reali, out+stato), catena
  T=12 col campione pinnato e stato PERSISTENTE su GPU (maxAbs 5.4e-7,
  maxRel 1.1e-3 su near-zero). Proiezioni via gemv-f32 esistente.
- Totale ktest 75/75 (69 preesistenti PASS = GLM intatto). Suite node 371
  PASS / 7 skip, tsc pulito. FASE 3 done-when coperto per intero.
- Il rischio dominante del goal (DeltaNet) è DIMEZZATO: numerica ricorrente
  provata kernel-level a dims reali; resta l'e2e coi pesi veri (fase 4).
- Next: it.6 = fase 4 (path 4B end-to-end: GQA variante + mrope + ibrido
  3:1 + ffn denso; argmax==cpuref; golden 4B a soglia ratchet).

## it.6 (2026-08-10) — fase 4 slice 1: cpuref 4B e2e == ORACOLO (verifier PASS)

- Semantica full-attention dalla fonte b10333: mrope TEXT-ONLY COLLASSA a
  NEOX su 64 dims (sezioni [11,11,10,0] con posizioni uguali ⇒ le theta
  t/h/w/e scalano in lockstep; ggml_mrope_cache_init + rotate_pairs offset
  32), q+gate fusi a stride 512 per head (gate: niente norm/rope, sigmoid a
  valle), QK-norm per head PRIMA del rope, GQA h/(nHead/nKvHead).
- `src/engine/q35cpurefmodel.ts`: forward teacher-forced streaming
  per-layer (pesi f32 un layer alla volta ~300 MB, attivazioni f64, head
  tied Q6_K dequantizzata una volta ~2.5 GB); i layer linear RIUSANO
  Q35DeltaNetRef (ktestata 75/75).
- Oracolo: il binario golden di tools/oracle-moe è GENERICO — funziona sul
  4B senza rebuild (tree llama.cpp-oracle @5f55650 ha già qwen35).
  Golden smoke committato (results/engine/golden/q35/, prompt committato,
  SHA modello, chat template dell'oracolo INCLUSO nei token del JSON:
  teacher-forced, zero logica template nel cpuref; il 4B apre con <think>).
- GATE E2E: argmax cpuref == oracolo su TUTTE le 6 posizioni generate
  (prefill 34 + 5 tf, 32 layer, pesi reali) AL PRIMO RUN NUMERICO — 130 s
  (test gated Q35_E2E=1: fuori dalla suite permanente, la raddoppierebbe).
- Fix: readFileSync cap 2 GiB → readLargeFile a chunk nel test.
- Igiene notata (non bloccante): golden.cpp scrive "arch":"deepseek2"
  HARDCODED nel JSON anche per qwen35 — cosmetico, i token/logit sono veri;
  da sistemare a igiene fuori run (registrare a docket).
- La COMPRENSIONE del modello è provata: da qui ogni divergenza GPU è un
  bug di kernel (pattern GLM). RESTANO per fase 4: forward WGSL 4B
  (engine path), argmax==cpuref sul campione ratificato via GPU, golden
  full-corpus a soglia ratchet, riferimenti full-resident con hostState.
- Next: it.7 = forward GPU 4B (orchestratore: kernel esistenti + deltanet).

## it.7 (2026-08-10) — fase 4 slice 2: assembly GPU layer reali (verifier gate)

- Kernel: ropeNeoxWgsl esteso con `ropeDims` opzionale (default headDim =
  testo storico INVARIATO; qwen35 usa 64 su 256, il corpo era già corretto
  per il parziale) + `sigmoidMulWgsl` (output gate attn).
- cpuref: estratto `attnLayerRef(l, hidden)` da forward (fonte unica per i
  fixture); guardia e2e RIESEGUITA post-refactor: PASS identico (130 s).
- Fixture attn con pesi REALI (public/models/q35-attn, 46 MB, NON in git,
  rigenerabile: scripts/q35-attn-fixture-gen.mjs, SHA GGUF pinnata dentro):
  byte raw quantizzati di blk.0 (linear) e blk.3 (full) + input seeded T=3
  + attesi dal cpuref.
- ktest +4, TUTTI PASS al primo run: rope-partial (canali ≥64 invariati
  bit-a-bit), sigmoid-mul, e i due ASSEMBLY con pesi reali —
  q35-attn-linear-real-blk0 L2rel 7.7e-7 (gemv q4_0/q8_0 + gates + conv +
  core + gemv Q5_K, stato persistente), q35-attn-full-real-blk3 L2rel
  2.1e-7 (gemv + deinterleave stridedCopy + QK-norm batch + rope parziale +
  kvAppend + attnDecode 16/4/256 + sigmoidMul + wo). Totale 79/79.
- Ogni PEZZO del forward 4B è ora provato su GPU con pesi veri: l'it.8
  è solo orchestrazione (loop 32 layer + embed + head) + gate argmax.
- Suite node 371/8 skip, tsc pulito, GLM intatto (69 ktest verdi dentro i 79).

## it.8 (2026-08-10) — fase 4 slice 3: FORWARD GPU FULL-MODEL, argmax == ORACOLO

- `src/engine/q35gpumodel.ts`: orchestratore correttezza-prima del denso
  qwen35 — piano di step PRECOSTRUITO (pipeline cache + bind group statici
  su scratch condivisi), 562 dispatch/token, un submit/token, full
  residency (tier 16 GB by design), head Q6_K con gemv a griglia, embed
  row Q6_K dequant CPU-side, reader astratto (Range o byte in RAM).
- Setup ktest (riproducibile): symlink `public/models/Qwen3.5-4B-Q4_0.gguf`
  → ~/.cache/blab-models/q35/…; `cp results/engine/golden/q35/golden-q35-
  4b-smoke-2026-08-10.json public/models/q35/golden-smoke.json`.
- DUE pareti trovate e risolte nel run: (i) fetch/ArrayBuffer 2.6 GB
  sfonda i cap Chromium (arrayBuffer 2 GiB E allocazione singola) → loader
  a fetch RANGE per-tensore (vite risponde 206, verificato) — la stessa
  postura streaming del cpuref; (ii) head Q6_K 527 MB > maxBufferSize
  negoziato dal device ktest (256 MB) → extraBindings esteso (l'errore
  URLA via uncapturederror: il punto-unico gpudevice ha fatto il suo
  lavoro, niente no-op silenzioso).
- GATE: **argmax GPU == oracolo 6/6** sul golden smoke (39 posizioni
  teacher-forced in 1.9 s, load 10 s). ktest TOTALE 80/80.
- FASE 4: restano golden full-corpus a soglia RATCHET (run-golden-q35.sh
  con provenance) + riferimenti decode/prefill/TTFT full-resident con
  hostState → it.9.
- Suite node 371/8, tsc pulito, GLM intatto.

## it.9 (2026-08-10) — fase 4: golden full-corpus + SOGLIA RATCHET 4B

- `run-golden-q35.sh` (provenance piena) → golden full-corpus COMMITTATO:
  8 prompt, prefill 27 312, 1024 posizioni greedy top-32 (SHA 298fcb5f,
  oracle 5f55650a, corpus-hash 2882e4ab).
- Infrastruttura conformance: q35conf worker/page/html + runner playwright
  (pattern glmconf); q35gpumodel: resetState() per prompt + step(read=false)
  per il prefill sequenziale senza readback.
- RUN GPU full-corpus (albero congelato, 29 min): **top-1 = 1012/1024 =
  98.828125%** — SOGLIA 4B FISSATA a docket item 8 (ratchet
  pre-autorizzato dal contratto). Per-prompt 125-128/128; load 11.3 s.
  Coincidenza notevole e verificabile: stessa cifra del PIN GLM (stesso
  corpus, near-tie q4 comparabili; i miss cadono su prompt diversi).
- RESTA per chiudere fase 4 (it.10): riferimenti decode/prefill/TTFT
  full-resident con hostState dichiarato (orchestratore correttezza-prima,
  DICHIARATO come frame di partenza: il numero onesto pre-ottimizzazioni).

## it.10 (2026-08-10) — FASE 4 COMPLETA: riferimenti full-resident 4B

- Modalità bench nel worker q35conf (?bench=idx,n): prefill sequenziale
  read=false con sync onSubmittedWorkDone, poi decode GREEDY (argmax →
  token successivo) con readback, p50; runner q35-bench-run.mjs con
  hostState PRIMA/DOPO (scripts/lib/hoststate.mjs).
- RIFERIMENTI (q35-bench-4b-fullresident-2026-08-10.json, host
  user-session-light dichiarato, GPU idle 31°C/210MHz al via):
  **decode 22.93 tok/s (p50 43.6 ms) · prefill seq 26.0 tok/s · TTFT
  25.8 s (load 10.9 s + prefill 387 tok in 14.9 s + first 49 ms)** su
  prompt 05-math-en (correzione verifier: la prima stesura trascinava il
  load di it.9 e arrotondava i token). DICHIARATO
  nel JSON: correttezza-prima (562 dispatch/token, zero fusioni, readback
  per token) = FRAME DI PARTENZA, non un numero competitivo.
- Lettura onesta del frame: il 4B denso full-resident fa GIÀ 23 tok/s
  SENZA alcuna ottimizzazione (GLM paginato: 15.6 dopo tre fasi di
  lavoro) — il regime full-resident è un altro sport; la soglia UX 30
  è a 1.3× di distanza con tutte le leve ancora in canna.
- PHASES fase 4 → done (it.6-10). FASI 1-4 COMPLETE su 9.
- Next: it.11 = fase 5 — 9B (golden MODEL=9B, conformance riusando
  l'infra parametrica) + WP decomposizione gap full-residency vs
  llama.cpp Vulkan (stesso GGUF/protocollo p512/n64).

## it.11 (2026-08-10) — fase 5 prima metà: 9B CONFORME, soglia ratchet

- Infra parametrizzata sul modello (worker/page/runner con --model, tabella
  URL+SHA pinnata; run-golden-q35.sh con TAG nel filename — preso PRIMA
  che il 9B sovrascrivesse il golden 4B, run killata e rilanciata).
- SORPRESA DAL FILE (fail veloce, poi fix): il 9B ha embd Q4_0 e head
  Q6_K — INVERSO del 4B. q35gpumodel ora gestisce entrambi gli
  orientamenti (embd Q6_K|Q4_0, head Q6_K|Q4_0); il cpuref li gestiva già.
- Golden 9B committato (1024 pos, provenance piena). Conformance GPU
  (42 min, albero congelato): **top-1 = 1000/1024 = 97.65625%** → SOGLIA
  9B FISSATA (docket item 9) CON ANALISI NEAR-TIE: tutti i 24 miss sono
  near-tie oracolo (23/24 = top-2, margine mediano 0.066 logit, zero >1)
  — firma benigna f32, near-tie mai gateati (C2).
- Riferimenti full-resident 9B (host dichiarato): **decode 14.55 tok/s ·
  prefill seq 15.4 · TTFT 40.3 s** (load bench 15.1 s; correzione
  verifier: il 15.6 era della run di conformance) — scala ~coerente col
  rapporto parametri attivi vs 4B (22.9→14.55 ≈ 0.63× per 2.1× parametri:
  la banda pesi domina, come atteso su gemv-bound).
- RESTA per fase 5 (it.12): WP decomposizione gap — llama.cpp Vulkan
  stesso GGUF 9B p512/n64 vs questi riferimenti, scomposizione
  kernel/dispatch/safety-check nel doc di studio, leve per ROI.

## it.12 (2026-08-10) — FASE 5 COMPLETA: WP decomposizione gap

- Nativo Vulkan STESSO GGUF (llama-bench -ngl 99 p512/n64 r3, GPU 31°C
  idle): 4B 118.7/4755, 9B 67.2/2622. Browser: 22.93/26.0 e 14.55/15.4.
- GAP DECODE: **5.18× (4B) / 4.62× (9B)**; scomposto CON MISURA:
  readback+sync = decode−prefillSeq = 5.1/3.6 ms/token (12%, stessi 562
  dispatch, read=false salta solo copy+map); compute residuo 4.6×/4.4×
  (dispatch stimato 15-29%, kernel non tuned, check WebGPU 14-42% prior).
- PREFILL 183×/171× = ASSENZA di batching (prefill sequenziale by
  design), non gap: leva n.1, pattern GLM già in casa.
- Leve per ROI (doc §4): prefill batched → decode multi-step no-readback
  (pattern GLM B2, −5.1 ms misurati) → fusione dispatch → tuning+
  dot4I8Packed (fase 6 contratto) → spike subgroup-matrix. Docket item 10:
  priorità informata per il PI, nessun cambio contratto.
- Per il WRITEUP: gap a parità di regime 4.6-5.2× decode; prefill solo
  dopo il batching. Artefatti: native-vs-browser-q35 JSON (aritmetica
  auto-verificata) + doc studio. PHASES fase 5 → done (it.11-12).
- Next: it.13 = fase 6 (leve bounded da contratto: dot4I8Packed e tuning
  dietro flag SE in cima al ROI — il WP dice che il ROI più alto è nei
  pattern GLM (fase 7-8/D), quindi fase 6 = misura leve contratto +
  esclusioni motivate coi numeri + spike subgroup-matrix).

## it.13 (2026-08-10) — FASE 6 COMPLETA: 2 esclusioni motivate + probe

- dot4I8Packed e tuning tile: ESCLUSE PER ORA coi numeri del WP (ramo
  previsto dal done-when): il collo del decode è readback+dispatch+fusione
  (non l'ALU); il +41% del tuning è sul matmul batched che ancora non
  esiste (leva n.1). Doc §5, docket item 11.
- SPIKE subgroup-matrix ESEGUITO (probe empirico committato, 3 config di
  flag): SORPRESA — `chromium-experimental-subgroup-matrix` è ESPOSTA E
  CONCESSA al device sul nostro Chrome stable coi flag runner standard
  (adapter nvidia/lovelace); MA config INT8-ONLY ({u8,i8}→{u32,i32},
  16×16×32|16×8×32), niente f16/f32, shader-f16 assente dall'adapter;
  compilazione naive rifiutata (packing u32 richiesto) ⇒ la misura di
  throughput richiede la stessa infra int8-packed di dot4I8Packed: le due
  leve CONVERGONO, si rivalutano dopo il prefill batched. Spec §7
  aggiornata (per l'utente finale senza flag resta non raggiungibile:
  probe ≠ deployment).
- Fix di metodo nel probe (2 iterazioni): launchPersistentContext + origin
  http (WebGPU assente su about:blank con launch semplice); adapter
  FRESCO per ogni requestDevice (Chrome consuma l'adapter).
- PHASES fase 6 → done. FASI 1-6 su 9.
- Next: it.14 = fase 7 — MoE 35B-A3B parametrizzato (il pezzo grosso
  rimasto: slotTable/classi parametrici con GLM invariato, dequant Q4_K
  expert, forward 35B con paging C3c).

## it.14 (2026-08-10) — fase 7 slice 1: Q4_K + cpuref MoE (verifier gate)

- SEMANTICA MoE dalla FONTE b10333 (qwen35moe.cpp + build_moe_ffn), NON
  da GLM: router F32 → softmax 256 (niente bias) → top-8 per probs →
  norm sum-1 con clamp ESATTO 6.103515625e-5 → NESSUNO scale
  (expert_weights_scale non letto ⇒ 0 ⇒ ramo skip) → pesatura DOPO il
  down; shared expert = SwiGLU · sigmoid(gate SCALARE ffn_gate_inp_shexp).
  (GLM: sigmoid+bias+norm+scale 1.8 — tutto diverso: letto, non assunto.)
- `dequantQ4_K` (quant.ts, 144 B/superblocco = Q5_K senza qh) ancorato a
  ORACOLO ESTERNO: gguf-py dequantize sugli stessi byte REALI del 35B
  (blk.0.ffn_gate_exps, fixture con base64) — 1024 valori identici <1e-7.
- `gemvQ4KWgsl` (wgsl.ts, 36 word/sb) + testGemvC2 esteso: ktest
  gemv-q4_K-2048x512 e 512x2048 (dims expert reali) PASS al primo run —
  ktest TOTALE 82/82.
- `q35MoeFfnRefF64` (q35cpurefmodel.ts) + 3 property test: pesi sum-1
  ordinati; topK=nExpert ≡ mixture densa (ricalcolo indipendente <1e-12);
  shared gate scalare isolato. Suite 375 (+4).
- RESTA fase 7 (it.15+): estensione Q35CpuRefModel a qwen35moe con reader
  LAZY (20.9 GB non stanno in RAM insieme al resto: range da fd),
  cpuref e2e vs golden smoke 35B, poi forward GPU con paging C3c.
