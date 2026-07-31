# Journal — engine-fase-c2

## it.1 — fasi 1+2 in parallelo (2026-07-31)

**Fase 2 (spec) DONE-pending-ruling**: dump header GGUF (metadata deepseek2
completi + inventario 844 tensori) ⇒ TRE fatti che la spec recepisce:
1. **quant MISTA nel file "Q4_0"** (ffn_down_exps Q4_1 su tutti i 2944
   expert; shexp Q5_K/Q6_K; output Q6_K; proiettori MLA Q8_0) ⇒ servono
   kernel dequant nuovi, repack escluso (romperebbe la conformance);
2. metadati MLA confermano la formulazione absorbed come design point
   (cache 576/token/layer = 54 KB/token f16, direction §3);
3. gating: sigmoid (func=2) + exp_probs_b per la selezione,
   weights_norm=true, weights_scale=1.8 per il mixing.
Spec: `docs/superpowers/specs/2026-07-31-engine-fase-c2-design.md` (10
sezioni); RULING RICHIESTO a docket item 2 con 6 decisioni (a)-(f), incluse
2 deviazioni dichiarate dal contratto: MLA absorbed (supera l'ASSUMED
naive-first) e OPFS come sorgente dei byte expert (il contratto diceva
staging RAM: 15.6 GB di ArrayBuffer in un tab su host 31 GB non reggono).
Fasi 3-6 restano gated dal ruling.

**Fase 1 (floor+golden)**: floor registrato nel docket allo scaffold
(decode 13.43 / prefill 56.58 tok/s, llama-bench C1). Tool
`tools/oracle-moe/golden.cpp` (nuovi file, zero modifiche ai file C1;
owns esteso rispetto alla riga PHASES — deviazione minore registrata qui):
prefill chunked 512 + decode greedy 128/prompt, argmax + top-32 logit f32
per posizione, envelope sha/commit/threads/corpusHash. Run canonica
lanciata sul corpus C1 (8 prompt); esito in coda a questa entry.

**Fase 1 esito (run 2026-07-31, exit 0, ~10 min)**: golden scritto —
`results/engine/golden/glm47flash/golden-glm47flash-q4_0-2026-07-31.json`
(647 KB): 26.154 prefill (IDENTICO alla traccia C1: stesso corpus/template,
sanity incrociata), 1.024 posizioni golden (8×128, zero EOS anticipati),
argmax+top-32 f32 ordinati, envelope sha/commit/corpusHash verificati.
Cross-check indipendente: i 128 token greedy per prompt coincidono 8/8 coi
primi 128 decode della traccia C1 (stesso oracolo, tool diverso ⇒ il golden
e la traccia sono mutuamente coerenti). Checkout oracolo pulito, commit
5f55650a. Done-when fase 1: run exit 0 ✓, JSON con envelope ✓, floor nel
docket ✓ (registrato allo scaffold). FASE 1 DONE (pending verifier).

**Verifier it.1: FAIL → correzioni applicate → ri-verifica.** Il verifier
(PASS su golden, vincoli, floor, envelope, cross-check traccia) ha refutato
il claim "ffn_down_exps Q4_1 su tutti i 2944 expert": il mio dump teneva
solo il PRIMO esemplare per suffisso (blk.1). Layout vero (riverificato
per-layer sul file): Q4_1 solo su blk.1-4 (256 expert, 5.505.024 B), Q4_0
su blk.5-46 (2.688 expert, 5.308.416 B); media pesata 5.325.512 = ESATTO
l'expertBytesQ4 di residency-sim C1 (quadratura indipendente). Corretti:
spec §1 (tabella + nota) e §5 (slab a DUE size-class), docket item 2(a).
La decisione (a) resta valida (Q4_1 serve comunque: 256 expert + down
denso). Lezione strumento: mai riassumere un inventario tensori campionando
il primo elemento per nome — enumerare i tipi per layer.

## it.2 — fase 3: reader GGUF deepseek2 (2026-07-31)

Implementato secondo spec §2 (ruling (a)-(f) risolto a inizio iterazione):
- `gguf.ts`: GGML_TYPE + {Q4_1:3, Q5_K:13, Q6_K:14}, tensorByteSize esatti
  (20 B/32; 176 B/256; 210 B/256) con throw su ne[0] non multiplo.
- `quant.ts`: dequant CPU di riferimento per Q4_1, Q5_K, Q6_K portate
  riga-per-riga da ggml-quants.c dell'oracolo (5f55650): scaleMinK4 (6-bit
  packed), qh bit alto Q5_K, d IN CODA nel blocco Q6_K, scales int8.
- `shape.ts`: Glm47Shape + GLM47_FLASH (metadati completi dal dump),
  GLM47_FLASH_SHA256 pinnato, validateGlm47Flash (844 tensori attesi,
  down_exps Q4_1 solo blk.1-4 via GLM47_DOWN_EXPS_Q4_1_LAST, throw su ogni
  mismatch — postura ds4 invariata).
- `tests/engine-gguf-glm.test.ts` (18 test): fixture sintetica 844 tensori +
  7 mutazioni negative (tipo/dims/mancante/conteggio/gating/arch), byte per
  expert delle due size-class asseriti (5.308.416/5.505.024), dequant vs
  fixture da byte REALI del GGUF con atteso calcolato da gguf-py 0.17.1
  (oracolo indipendente, `scripts/gen-quant-fixtures.py`, match f32 a
  ULP-zero), load headless del file reale (header 64 MiB + validazione +
  bounds + taglia file 17.216.676.192).
- FIX in corsa: la mia costante taglia-file usava il model_size di
  llama-bench (sola sezione dati); corretta al byte-size su disco già
  annotato dal verifier C1.

**Evidenza**: `npx tsc --noEmit` pulito; `npm test` 21 file / 199 test PASS
(nessuna regressione sul pregresso; il real-file test ESEGUITO, non
skippato). Done-when fase 3: suite parsing GLM verde con SHA pinnato ✓,
tsc pulito ✓, load headless exit 0 ✓ (dentro vitest). Pending verifier.

Verifier it.2: **PASS** (loop-verifier, 2026-07-31 — porting dequant confrontato
riga-per-riga col C dell'oracolo; fixture riverificata con parser e dequant
scritti ex novo dal verifier; shape vs file reale con parser indipendente;
zero regressioni; suite 199 verde). Osservazioni recepite: (1) deviazione
owns non annotata — `scripts/gen-quant-fixtures.py` sta in scripts/ (pattern
gen-golden di fase 1), fuori dagli owns della riga PHASES; registrata QUI,
nessun conflitto di ownership; (2) il goal lavora su main mentre GOAL.md
§AUTHORITY dice branch engine/fase-c2 con merge a goal chiuso (C1 usò il
branch) → divergenza di merge-policy, NON la decido io: docket item 3.

## it.3 — fase 4, primo slice: GEMV dequant-fuse dei formati GLM (2026-07-31)

Slice scelto (il più grande verificabile in un'iterazione): i kernel WGSL
GEMV per i tipi quant nuovi — prerequisito di MLA (proiettori) e MoE (expert
e shexp). Implementato:
- `quant.ts`: repackQ4_1 (qs 4 u32/blocco; scales = 1 u32/blocco con d|m,
  unpack2x16float-ready) e repackKQuant (superblocco GGUF grezzo in u32 LE,
  stride allineato: Q5_K 44 word esatte, Q6_K 53 con 2 B pad).
- `kernels/wgsl.ts`: gemvQuantWgsl esteso a kind "q4_1" (contributo blocco
  = d·Σq·x + m·Σx); gemvQ5KWgsl (scaleMinK4 6-bit replicata in WGSL, qh bit
  alto, accumulo d·sc·dot − dmin·min·Σx per gruppo da 32); gemvQ6KWgsl
  (nibble+2bit, scales int8 sign-extended, d f16 in coda al superblocco).
  Correttezza-prima come da spec §9 rischio 1 (niente tiling).
- ktest: testGemvC2 (kernel vs dequant CPU di it.2, dati seeded, fixScalesAt
  per i f16 nei nuovi offset), 5 casi a taglie reali GLM; page.ts espone
  window.__report; runner nuovo `scripts/ktest-run.mjs` (pattern kernel-diag,
  Chrome headed, vite :5199 dedicata + kill).

**Evidenza**: run ktest su 4090 (Chrome headed) status "done", 16/16 PASS —
i 5 nuovi: q4_1 1536x2048 maxRel 5.1e-7, q4_1 10240x256 9.3e-6, q5_K
2048x1536 2.2e-4, q6_K 1536x2048 1.4e-5, q6_K 2048x1024 1.5e-5 (tolleranza
2e-4 rel / 1e-3 abs come i gemv esistenti); zero regressioni sugli 11 kernel
preesistenti; `npm test` 199/199; `tsc --noEmit` pulito. Fase 4 prosegue
(it.4: cpuref MLA naive + kernel absorbed). Pending verifier.

Verifier it.3: **PASS** (ri-run ktest indipendente 16/16 su 4090; matematica
WGSL confrontata col C dell'oracolo incluso il ramo is>=4 dei 6-bit; catena
di fiducia gguf-py → dequant CPU → kernel GPU confermata). WATCH ITEM per
fase 6 (dal verifier): gemv-q5_K passa per tolleranza ASSOLUTA (maxAbs
7.6e-4 ≤ 1e-3) con maxRel 2.25e-4 appena sopra la relTol 2e-4 — se la
conformance per-layer stringesse le soglie, è il primo kernel da guardare.

## it.4 — fase 4, slice 2: kernel MLA absorbed (2026-07-31)

Semantica verificata NEL SORGENTE dell'oracolo prima di scrivere WGSL
(deepseek2.cpp, llama-model.cpp a 5f55650) — tre fatti che avrebbero rotto
la conformance se sbagliati:
1. **RoPE = tipo NORM** (coppie consecutive), NON NEOX: deepseek2 sta nel
   gruppo "normal RoPE" di llama-model.cpp; applicato alle sole 64 dim rope
   di q (offset 192 per head) e di kv_cmpr_pe (offset 512), PRIMA della
   kv_a_norm (che tocca solo le prime 512 componenti);
2. **kq_scale = 1/sqrt(256)** (n_embd_head_k_mla, la head dim
   MHA-equivalente) — NON 1/sqrt(576); mscale=1 (niente yarn nel GGUF);
3. layout Qcur/Kcur = [abs/ckv 512 | rope 64] (rope in coda), V = c_kv
   NORMATA (512): lo score è un dot unico a 576.
Kernel nuovi (wgsl.ts): ropeMlaNormWgsl (parametrico nVec/stride/offset —
serve sia q sia k_pe), gemvQ8HeadsWgsl (GEMV Q8_0 con x per-head: copre
wk_b [192,512,20] assorbimento e wv_b [512,256,20] uscita),
mlaAttnDecodeWgsl (MQA su cache 576, softmax stile attnDecode esistente,
out in spazio c_kv [20×512]). Riferimenti JS f64 inline nel ktest
(ropeNormRef, per-head matmul, softmax f64).

**Evidenza**: ktest su 4090 status "done", **21/21 PASS** (nuovi:
rope-mla-norm-q, rope-mla-norm-kpe maxRel<5e-4; gemv-q8-heads-absorb-kb e
-vout-vb entro 2e-4; mla-attn-decode maxAbs 3.7e-8); zero regressioni sui
16 preesistenti; `npm test` 21 file/199 PASS; `tsc --noEmit` pulito.
Restano per la fase 4 (timebox: 2 it.): assemblaggio layer GLM in
gpuforward + conformance layer-level con pesi reali (gate doppio). Pending
verifier.

Verifier it.4: **PASS** (ri-run ktest 21/21; RoPE NORM riscontrato in
llama-model.cpp:2510-2549 e ggml ops.cpp con rotate_pairs consecutivi;
kq_scale 1/sqrt(256) in deepseek2.cpp:154/168-172; ordine rope→norm e
layout concat [512|64] alle righe 279-316; riferimento f64 indipendente;
layout per-head di gemvQ8Heads coerente col GGUF row-major). Nota:
l'assunzione "niente yarn ⇒ mscale=1" è da sorgente+metadata, verrà
riverificata empiricamente coi golden alla conformance con pesi reali.

## it.5 — fase 4, slice 3: layer GLM assemblato + conformance layer-level con pesi reali (2026-07-31)

Slice finale della fase 4. Implementato:
- `cpuref.ts`: `GlmDenseLayerRefF64` — layer denso blk.0 in f64 con MLA
  in formulazione NAIVE: wk_b ad accesso TRASPOSTO (k_nope per posizione),
  V decompressa PER POSIZIONE via wv_b (l'absorbed la applica all'aggregato):
  riferimento deliberatamente diverso dal motore. [dicitura corretta
  post-verifier it.5]
- `glmforward.ts` (file NUOVO — deviazione owns come it.2, stessi criteri:
  pattern gpuforward, nessun conflitto): `createGlmLayer0` assembla il layer
  su GPU — 22 dispatch/token, MLA absorbed coi kernel di it.3/4 + ffn denso
  (gate/up Q4_0, down Q4_1), error scope da contratto, cache 576/token.
- `kernels/wgsl.ts`: `stridedCopyWgsl` (assemblaggio q576 = [q_ckv|q_rope]
  per head — copyBufferToBuffer non ha stride); ktest esatto (tol 0).
- `scripts/gen-glm-layer-fixture.py`: fixture 52 MB (byte GREZZI dei 13
  tensori blk.0 + 16 righe embd dei primi 16 token del corpus golden p0) in
  public/models/ (gitignored, rigenerabile). Il ktest la fetch-a e su di essa
  costruisce ENTRAMBI i riferimenti (stessi byte, due formulazioni).
- `tests/engine-cpuref-glm.test.ts`: identità algebrica naive↔absorbed in
  f64 su pesi sintetici (implementazione absorbed indipendente nel test,
  rms/rope/matvec propri): maxAbs < 1e-8 su 2 posizioni. Un errore di
  trasposizione o d'ordine rope/norm in uno dei due la rompe.

**Evidenza**: ktest su 4090 status "done", **23/23 PASS** — nuovo
`glm-layer0-conformance`: pesi reali blk.0, 16 posizioni decode replay,
**L2rel 2.35e-7, max|Δ| 8.8e-8** (gate harness: L2rel ≤ 1e-3, max|Δ| ≤ 5e-2 —
ampiamente dentro; l'accordo è al rounding f32), maxRel 1.7e-3 solo su
componenti ≈0; `strided-copy-qrope` esatto; zero regressioni sui 21
preesistenti. `npm test` 200/200 (nuovo test identità incluso); `tsc
--noEmit` pulito.

**Lettura del "gate doppio cpuref/golden" a livello layer** (registrata, il
PI può obiettare): i golden di fase 1 contengono SOLO logits full-model
(argmax + top-32 per posizione) — non esistono hidden per-layer nell'oracolo.
Il gate (i) vs cpuref-f64 è quindi l'unico applicabile a livello layer ed è
PASS; il gate (ii) vs golden si applica by construction al full-model e
matura in fase 6 (soglie §7: argmax ≥99% cpuref / top-1 ≥97% golden). Il
cpuref f64 full-model necessario al gate (i) di fase 6 richiede il MoE ref
(fase 5). Fase 4 chiusa a it.5 (timebox 4 it., usate 3). Pending verifier.

Verifier it.5: **PASS** (loop-verifier — ri-run ktest indipendente 23/23 con
numeri identici al claim; tsc+suite 200/200 fresh; kq_scale/rope NORM/ordine
rope→kv_a_norm/layout [512|64]/v_mla riscontrati in deepseek2.cpp:172,279-316
e llama-model.cpp:2526; dualità naive/absorbed verificata algebricamente
(qᵀ(Wᵀc)=(Wq)ᵀc); fixture ricalcolata a mano vs shape.ts e golden; 22
dispatch contati a mano). Osservazioni recepite: (1) dicitura wv_b nel
journal corretta (sopra); (2) sha fixture = costante-vs-costante, pin
ancorato a it.1/2 (non ri-hashato il 17 GB); (3) WATCH ITEM fase 6: il gate
layer-level (L2rel ≤ 1e-3) è ~4 ordini più largo del risultato (2.35e-7) —
quando le soglie §7 saranno fissate sul full-model, stringerlo a canary di
non-regressione (si collega al watch-item q5_K di it.3).

**FASE 4 CHIUSA** (it.3-5, timebox 4 it.: usate 3). Next: fase 5 (MoE +
residenza minima, timebox 4 it.).

## it.6 — fase 5, slice 1: router MoE + GEMV per-expert sugli slab (2026-07-31)

Semantica router verificata riga-per-riga in `build_moe_ffn`
(llama-graph.cpp, oracolo 5f55650), come richiesto dalla spec §4:
- probs = **sigmoid**(logits) (gating func=2, r.1864); logits = gate_inp·cur
  (F32, nessun gate_inp_b nel path deepseek2);
- bias `exp_probs_b` sommato SOLO per la selezione (r.1883: selection_probs);
- top-4 = argsort_top_k su selection_probs (r.1926), pareggi → indice minore
  (argsort stabile, stesso tie-break della replica C1 trace.cpp);
- pesi di mixing = get_rows su **probs SENZA bias** (r.1940), normalizzati a
  somma 1 con denominatore clampato a **6.103515625e-5** (min f16 normale,
  r.1958), poi **×1.8** (w_scale, r.1967);
- pesatura applicata DOPO il down (r.2122, weight_before_ffn=false);
- shared expert: build_ffn sullo STESSO input post-ffn_norm, sommato a
  moe_out (deepseek2.cpp r.403-412); niente swiglu_clamp per deepseek2
  (caricato solo da deepseek4.cpp); n_expert_groups=1 ⇒ niente gruppi;
  ffn_gate_up_exps merged ASSENTE nel nostro GGUF (path separate gate/up).

Implementato:
- `moe.ts` (file NUOVO, owns di fase 5): `routerSelect` (replica CPU della
  catena sigmoid→bias→top4→norm×1.8 — la selezione DEVE stare su CPU: decide
  quali slab bindare) + layout slab a due size-class ESATTE
  (`SLAB_DOWN_Q4_0` 5.308.416 B / `SLAB_DOWN_Q4_1` 5.505.024 B, offset dei 6
  segmenti [qs|scales]×{gate,up,down} tutti multipli di 256 =
  minStorageBufferOffsetAlignment) + `packExpertSlab` (repack CPU
  all'upload: parte del costo di miss che la telemetria misurerà).
- `kernels/wgsl.ts`: `gemvF32Wgsl` (router: ffn_gate_inp è F32) e opzione
  `scaledAccum` su `gemvQuantWgsl` (y[r] += accScale[0]·dot: il down
  per-expert accumula il contributo pesato direttamente su moe_out — 1
  dispatch risparmiato per expert, l'ordine delle somme differisce
  dall'oracolo solo al rounding f32).
- `cpuref.ts`: `glmMoeFfnRefF64` — blocco MoE-FFN f64 con router e selezione
  INDIPENDENTI (sort-based vs scan-based di moe.ts, pattern naive/absorbed).
- ktest nuovi (5): `gemv-f32-router`, `gemv-q4_0-accum`, `gemv-q4_1-accum`,
  `moe-ffn-block-downq4_0`, `moe-ffn-block-downq4_1` — il blocco completo a
  dims reali GLM: router GEMV su GPU → selezione CPU → shexp Q5_K/Q6_K → 4
  catene expert dagli slab impacchettati con **bind group a offset per-slot**
  (lo stesso meccanismo della residenza) → confronto vs ref f64.
- vitest `engine-moe-router.test.ts` (8): tie-break, bias solo-selezione,
  Σpesi=1.8, clamp, concordanza con selezione indipendente su 200 input
  casuali, taglie/allineamento/contenuto slab, validazione hard.

**Evidenza**: ktest su 4090 status "done", **28/28 PASS** (nuovi:
gemv-f32-router maxRel 1.6e-6; accum q4_0 maxRel 3.9e-4; accum q4_1 maxAbs
5.7e-6 — maxRel 3.7e-3 solo su componenti ≈0, pass via absTol [dicitura
corretta post-verifier]; moe-ffn-block-downq4_0 maxRel 6.0e-5, -downq4_1
maxRel 2.3e-4 — il maxAbs alto dei due blocchi è artefatto delle scale
K-quant sintetiche, valori ~1e7); zero regressioni sui 23 preesistenti. `npm test` **208/208** (8
nuovi); `tsc --noEmit` pulito. Restano per la fase 5 (timebox 4 it.):
slice 2 = residenza minima (OPFS → staging → cache VRAM LRU per classe,
telemetria) e slice 3 = forward multi-layer + conformance routing ≥99% vs
traccia C1. Nota di design registrata: la selezione su CPU implica un punto
di sincronizzazione GPU→CPU per layer MoE nel decode (readback dei 64
logits) — il costo va misurato in fase 6, mitigazioni eventuali a docket
(non sono C2). Pending verifier.

Verifier it.6: **PASS** (loop-verifier — ri-run indipendente: tsc pulito,
suite 208/208, ktest 4090 28/28 con numeri coincidenti; tutte le righe
dell'oracolo riscontrate — r.1864/1883/1926/1940/1958/1967/2121-22,
deepseek2.cpp r.403-412; routerSelect e glmMoeFfnRefF64 verificati
semanticamente e davvero indipendenti; aritmetica slab ricalcolata ex novo;
bind a offset reali nel ktest; zero drift, zero violazioni, nessun push
pre-verifica). Osservazioni recepite: (1) dicitura tolleranze accum corretta
(sopra); (2) riferimenti riga nel commento di moe.ts allineati all'oracolo;
(3) WATCH ITEM slice 3: il check selezione GPU↔ref nel ktest confronta i
top-4 come INSIEME — per la conformance routing vs traccia C1 la spec §7
richiede il set-match (l'ordine è escluso per scelta di spec: "la
composizione dell'insieme decide la residenza"), ma i PESI di mixing vanno
comunque confrontati per (posizione, layer); (4) digest it.5 e it.6 appesi a
digests.md (il gap di it.5 era consolidato).

## it.7 — fase 5, slice 2: residenza minima (OPFS + cache VRAM LRU + telemetria) (2026-07-31)

Implementato secondo spec §5 (il MINIMO che fa girare 17 GB in 16 GiB;
niente policy/tier/prefetch su OPFS — quelli sono C3; il path di load resta
random-access, compatibile col prefetch di C3 come da ruling C1 item 4):
- `expertstore.ts` (NUOVO): `Sha256Stream` (FIPS 180-4 incrementale:
  crypto.subtle vuole l'intero buffer — impraticabile su 17.2 GB; il costo
  una-tantum dell'import va in telemetria); `ExpertOpfsStore` (import
  streaming URL→OPFS con hash in corsa e VALIDAZIONE HARD: mismatch ⇒
  truncate(0)+throw; read sincrona di range via SyncAccessHandle, solo
  worker dedicato — stessa posture di kvstore.ts); `GgufExpertIndex`
  (expert e = fetta CONTIGUA di tensorByteSize/64 del tensore 3D: funzione
  pura, testata in node sul file reale).
- `residency.ts` (NUOVO): `ExpertCache` — slot (classe, buffer, offset),
  buffer ≤ maxStorageBufferBindingSize negoziato, riparto del budget tra le
  classi in proporzione al parco (256/2688), cap al parco; LRU PURA per
  classe via Map insertion-order (touch O(1)); `ensure()` con readRaw SOLO
  su miss → packExpertSlab → writeBuffer allo slot vittima; argomento
  `pinned` (i top-4 del token corrente non sono mai vittime — invariante
  che il forward di slice 3 userà); telemetria liv.1 (hit/miss/eviction/
  byte/read-pack-upload ms) con performance.now GATED da opts.timing
  (zero-overhead spenta, contratto); `slotBindRanges` = i 6 sotto-range
  nell'ordine dei binding dei kernel GEMV di it.6.

**Evidenza**: ktest su 4090 status "done", **29/29 PASS** — nuovo
`residency-opfs-roundtrip`: import 37.6 MB con SHA streaming verificato vs
crypto.subtle (212 ms), rifiuto+truncate su SHA sbagliato, 8 miss / 1 hit /
2 eviction con contatori ESATTI, readback GPU byte-esatto degli slab
(inclusi slot riusati post-eviction e classe q4_1 separata); zero
regressioni sui 28. `npm test` **219/219** (11 nuovi in
engine-residency.test.ts: Sha256Stream vs webcrypto sulle taglie limite del
padding, indice sul GGUF reale con identità aritmetica vs tensorByteSize
per tutti i 46 layer MoE, LRU/pinned/classi/riparto con device mock);
`tsc --noEmit` pulito.

**WATCH ITEM fase 6 (load-bearing, numeri dal roundtrip)**: il costo di
miss misurato è ~9.7 ms (read warm 1.9 + **pack 5.3** + upload 2.5): il
repack CPU DOMINA. A 4×46=184 ensure/token e hit 96.4% (sim C1) ⇒ ~6-7
miss/token ⇒ **~60-70 ms/token di solo stallo residenza**, contro un budget
di 74 ms/token al floor 13.43 tok/s. Rischio concreto sul gate decode di
fase 6. Leve possibili (decisione a valle della misura E2E, eventualmente
docket): repack più veloce (blocchi 36 B = 9 word), kernel GEMV che legge
il layout GGUF grezzo (i K-quant già indicizzano byte nelle word), o
riordino [qs|scales] in scrittura OPFS (repack pagato all'import, una
volta). Pending verifier.

Verifier it.7: **PASS** (loop-verifier — ri-run fresco: tsc pulito, suite
219/219, ktest 29/29 con contatori identici; Sha256Stream validata
INDIPENDENTEMENTE su vettori NIST ("" / "abc" / 448-bit al limite del
padding) + 9 taglie in chunk irregolari vs node:crypto; semantica
LRU/pinned/riparto riscontrata nel codice; aritmetica expert-range
ricalcolata; scope solo nel grant, nessun push pre-verifica; nota: `pinned`
è invariante di correttezza intra-token, NON il pin di C3). Osservazioni
recepite: (1) watch item CONFERMATO e leggermente peggiorato dal run fresco
(10,35 ms/miss, pack 5,85 dominante ⇒ ~66-68 ms/token di stallo proiettato
su budget 74,5): margine <10 ms/token sul gate decode di fase 6; la leva
indicata dal verifier col miglior rapporto rischio/beneficio è "repack
pagato all'import" (riordino [qs|scales] in scrittura OPFS: azzera il
termine dominante senza toccare i kernel) — la decisione matura alla misura
E2E di fase 6; (2) minore: bytesRead/bytesUploaded incrementano anche a
timing:false (due addizioni, zero-overhead materialmente rispettato).

## it.8 — fase 5, slice 3a: forward GLM multi-layer sul path di produzione (2026-07-31)

Slice 3 spezzata in due (timebox fase 5: 4 it., questa è la 3ª): it.8 =
assemblaggio del forward multi-layer VERIFICATO kernel-level; it.9 = replay
del corpus reale con import OPFS 17 GB + gate routing ≥99% (spec §7).

Implementato:
- `glmmodel.ts` (NUOVO): `createGlmModel(device, GlmWeightSource, opts)` —
  l'orchestratore che unisce fase 4 (attention MLA absorbed, pattern
  glmforward), slice 1 (router/moe.ts) e slice 2 (residency.ts). Per token:
  per ogni layer MoE, [submit fino a ffn_norm+router GEMV f32 → copy logits]
  → [sync mapAsync] → routerSelect su CPU → ensure×4 con `pinned` →
  [encode shexp + 4 catene expert da slotBindRanges + residuo] che prosegue
  nel submit del layer successivo. Pipeline WGSL condivise tra i layer
  (shape identiche), bind group fissi per-layer al load, bind group expert
  CACHED per-slot (stabili: la chiave (buffer, offset) sopravvive
  all'eviction perché il binding non cambia, cambia il contenuto);
  `GlmWeightSource` astrae la sorgente byte (OPFS in produzione via
  ExpertOpfsStore+GgufExpertIndex, mock sintetico nei test); error scope per
  submit; routing log per (posizione, layer) — l'output che it.9 confronta
  con la traccia C1. 16 dispatch/layer attn + 6 denso + 23 per layer MoE ⇒
  proiezione full-model 47 layer: 16×47+6+23×46 = **1.816 dispatch/token**
  (la stima di spec §6 ~800-1000 era ottimista; è MISURA, non gate — il
  contratto non eredita il ≤100 di fase B).
- `cpuref.ts` (refactor senza cambi numerici + estensione): attention MLA
  naive f64 estratta in `GlmMlaAttnRefF64` (identica su tutti i 47 layer),
  `GlmDenseLayerRefF64` delega; NUOVO `GlmMoeLayerRefF64` = attention +
  rms(ffn_norm) + `glmMoeFfnRefF64` + residui, con routing esposto.

**Evidenza**: ktest su 4090 status "done", **30/30 PASS** — nuovo
`glm-model-2layer`: mini-modello blk.0 denso + blk.1 MoE (classe down
Q4_1), pesi sintetici, 6 posizioni decode attraverso il path di PRODUZIONE
con cache stretta (6 slot ⇒ 13 hit / 11 miss / 5 eviction con re-fetch):
**L2rel 2.36e-7** sull'hidden post-2-layer (stesso ordine del layer-level
con pesi reali di it.5 ⇒ l'assemblaggio non aggiunge errore), top-4
set-match 6/6 vs selezione sort-based indipendente, pesi di mixing
wMaxRel 1.84e-7, 61 dispatch/token esatti come da formula; zero regressioni
sui 29. `npm test` 219/219 (cpuref refactor trasparente: il test identità
naive/absorbed passa invariato); `tsc --noEmit` pulito.

Restano per fase 5 (it.9, ultima del timebox): loader OPFS reale
(GlmWeightSource su ExpertOpfsStore + import one-shot 17.2 GB con SHA
streaming ~2-3 min), harness replay traccia C1 (formato in
tools/oracle-moe/trace.cpp) e gate set-match ≥99% su (posizione decode,
layer) + confronto pesi mixing (watch it.6). Nota dimensionale registrata:
il replay full-corpus (26k prefill + 1k decode × 46 sync/token) al passo
decode-only è nell'ordine delle decine di minuti/ore — la scelta del
sottoinsieme/strategia (prefill chunked prima? subset di spec?) si decide
in it.9 dentro il perimetro di spec §7. Pending verifier.

Verifier it.8: **PASS** (loop-verifier — ri-run fresco tsc/suite 219/ktest
30-30 con numeri identici; percorso attention confrontato riga-per-riga con
glmforward di fase 4; semantica MoE (shexp sovrascrive, expert accumulano
post-down, residuo) riscontrata nel WGSL; necessità del sync per layer
argomentata dal codice; cache bind group per-slot corretta anche dopo
eviction — conferma empirica: 5 eviction e L2rel invariato; refactor cpuref
testualmente identico; riferimento davvero indipendente — cpuref non importa
mai moe/glmmodel; aritmetica dispatch verificata; scope nel grant, nessun
push pre-verifica; split 3a/3b giudicato non-drift, motivato dal timebox).
Osservazioni recepite: (1) ampiezze sintetiche ~6e7 su 2 layer — se il
mini-modello sintetico crescesse di layer, rischio overflow f32 che
maschererebbe errori: il pattern NON va esteso in profondità, il full-model
si valida coi pesi reali (it.9/fase 6); (2) il costo per-sync del router
non è ancora misurato singolarmente — va nel report E2E di fase 6 col
watch item residenza; (3) nota harness: pkill nel comando ktest matcha la
shell wrapper (exit 144 spurio innocuo a run già concluso).

## it.9 — fase 5, slice 3b: replay reale, gate routing NON passato, discriminatore, docket (2026-07-31)

Costruito l'harness di replay completo e portato il motore sui PESI REALI
end-to-end per la prima volta:
- `glmsource.ts` (NUOVO): GlmWeightSource di produzione — GGUF in OPFS
  (import one-shot 17.2 GB via importFromUrl con SHA-256 streaming
  VERIFICATO sull'intero file; skip su size-match ai run successivi),
  header parse+validazione hard da OPFS, non-expert per nome, expert per
  range con buffer riusati.
- `glmroute/` (worker+page) + `glmroute.html` + `scripts/glm-route-run.mjs`:
  replay teacher-forced della traccia C1 (31.274 righe, token inclusi) su
  createGlmModel a 47 layer; set-match per (posizione, layer), contatori
  per fase/layer, sample, telemetria residenza nel report JSON. Profilo
  Chrome SU DISCO (~/.cache/blab-glmroute-profile): l'OPFS da 17 GB non può
  stare nel tmpfs di /tmp (16 GB — catch scoperto in corsa).

**Esito gate (spec §7)**: SOTTO SOGLIA. Prompt 4 completo (1.004 pos):
prefill 90.80%, decode **85.85%** vs ≥99% (report
routing-smoke-p4-2026-07-31.json). Replay 3,7 pos/s; hit residenza 95,9% a
12 GiB slab (2.216+203 slot); import OPFS con SHA pass al primo run.

**Debug del router come da spec ("ci si ferma e si debugga")** — eseguito
il discriminatore e il router è SCAGIONATO:
1. Firma dei mismatch: TUTTI swap di un solo expert (4°/5°), ~2% già al
   layer 1 (dove non c'è alcun expert a monte), degrado liscio con
   profondità (98%→80%) e con la lunghezza del contesto — drift, non bug.
2. `tests/analysis-route-cpuref.test.ts` (NUOVO, gated ROUTE_ANALYSIS):
   full-model **cpuref f64 esatto** (naive, layer-streaming, zero GPU, zero
   codice condiviso) sulle prime 16 pos del prompt 4: **96.20% = IDENTICO
   al motore GPU sullo stesso subset, con gli stessi 28/28 mismatch alle
   stesse (pos, layer)** (routing-cpuref-analysis-2026-07-31.json). Il
   motore riproduce l'aritmetica esatta; il disaccordo è dell'ORACOLO
   (attivazioni q8, landmine C1) sui near-tie del top-4. La soglia 99% era
   tarata sull'autotest C1 (hidden dell'oracolo, recall 0.999) che non
   copre il replay engine-vs-oracolo.

**Conseguenze**: (i) la taratura del gate è decisione di spec ⇒ RULING
RICHIESTO a docket item 4 con 3 opzioni (raccomandata: routing informativo,
gate su logits full-model fase 6 — il doppio gate esistente); (ii) timebox
fase 5 (4 it.) esaurito ⇒ lo stesso item vale da docket-con-analisi per
ruling (f); (iii) fase 6 resta bloccata dal ruling. Evidenza di build:
`npm test` 219/219 + 1 skipped (l'analisi è env-gated), `tsc --noEmit`
pulito, nessuna regressione (ktest non toccato in questa it.).
Corroborazione in coda: replay completo prompt 7 in run (i numeri nel
report smoke-p7 quando esce; stessa metodologia). Pending verifier.

Verifier it.9: **PASS** (loop-verifier — percentuali RICALCOLATE dai campi
grezzi dei 3 report; subset coincidente verificato; insiemi dei 28 mismatch
estratti e confrontati: uguali, e sotto i cap di campionamento ⇒ confronto
esaustivo; indipendenza del discriminatore verificata sul sorgente — zero
import da glmmodel/moe/residency, selezione sort-based propria; teacher
forcing riscontrato; spec §7 INVARIATA nel diff e docket item 4 che
presenta opzioni SENZA deciderle; 4 ipotesi di bug comune valutate
esplicitamente ed escluse dalla firma — lettura traccia, embedding, moduli
condivisi, file diverso). Osservazioni recepite: (1) dicitura "zero codice
condiviso" corretta in "zero codice di calcolo condiviso" (addendum docket);
(2) corroborazione p7 arrivata: decode 94.11% — varianza tra prompt coerente
col meccanismo near-tie; (3) replay full-corpus restante: si esegue solo a
valle del ruling (se (a): per il report informativo; se (b): con la traccia
nuova).

**FASE 5: lavoro tecnico COMPLETO (it.6-9, 4 verifier PASS), gate routing
in attesa di ruling PI (docket item 4). STOP BY DESIGN del loop**: fase 6
è gated dal ruling; nessun altro passo decidibile senza il PI.

## Ruling docket item 4 (2026-07-31, PI in chat): OPZIONE (a) ADOTTATA

"ok andiamo con A. Se in fase 6 dovessimo avere dei dubbi e per qualche
motivo ci servisse di nuovo un oracolo, lo faremmo rigirare."

Applicato: spec §7 emendata (routing = misura informativa con analisi e
contingenza oracolo-f32 on-demand; gate correttezza MoE = doppio gate
logits full-model); GOAL.md emendamento 2 (commento in testa, il DONE WHEN
rimanda alla soglia di spec che ora è l'emendamento); docket item 4
RISOLTO; PHASES: fase 5 **DONE** (it.6-9, timebox sanato dal ruling),
fase 6 **READY**. Lanciato il replay FULL-CORPUS (8 prompt, 31.274 pos,
~2.5h) in background per il report informativo di routing — è anche il
primo stress-test lungo della residenza (hit-rate a regime su tutto il
corpus = input C3). Il loop riparte su fase 6: E2E (output head + final
norm), conformance logits doppio gate, bench coi gate tok/s (decode ≥13.43,
prefill ≥56.58) + non-regressione Qwen. Nota operativa: niente bench
finché il replay occupa la GPU (contesa falserebbe i tempi); l'it.10
parte da implementazione+ktest (correttezza).

## Report informativo routing FULL-CORPUS (2026-08-01, post-ruling item 4a)

`results/engine/routing-conformance-glm47flash-2026-07-31.json` (replay
completo 8 prompt, 31.274 pos, 199 min — il fix maxComputeWorkgroupStorageSize
32KB era necessario: ctxMax 6688):
- Set-match top-4 (MISURA, non gate — ruling a): decode **88.51%**
  (208.460/235.520), prefill 87.05% (1.05M/1.20M); per-layer decode
  82.0-98.7% — tutto coerente col meccanismo near-tie discriminato in it.9
  (p4 85.8% e p7 94.1% stanno nel range).
- **Residenza a regime su tutto il corpus (input C3)**: hit-rate **97.56%**
  a 12 GiB slab (140.153 miss, 701 GiB riletti da OPFS); costo miss medio
  10.8 ms di cui **pack 6.1 ms** (858 s totali) vs read 3.1 + upload 1.6 —
  conferma definitiva del watch item: il repack CPU è il termine da
  eliminare (leva: repack all'import). Hit-rate sopra il 96.4% del
  simulatore C1 (budget effettivo ~82% del parco).
- Nota harness: il campo `gate` del report riflette la soglia pre-ruling
  (exit done-gate-fail): cosmetico, il ruling governa; da allineare alla
  prossima passata sull'harness.
