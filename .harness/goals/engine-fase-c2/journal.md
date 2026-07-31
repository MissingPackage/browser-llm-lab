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
