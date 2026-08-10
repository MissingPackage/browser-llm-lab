# HANDOFF — browser-llm-lab   (updated 2026-08-10, sessione 27 — GOAL q1: IL 35B GIRA SU GPU CON PAGING (5/5 vs oracolo, residenza 70.3%); in coda: golden full 35B CPU → conf GPU ~12h per la soglia ratchet)

## 1. Next decidable

**AVVIO GOAL `engine-fase-q1` — GENERALIZZAZIONE.** La direzione post-fase-C
NON è più una decisione aperta: il PI l'ha data (docket c3c item 9, "vai") e
poi raffinata (item 10). Sequenza congelata: **q1 (generalizzazione) →
release (split repo + paper Zenodo + blog)**. I due prerequisiti immediati
sono FATTI: (a) baseline nativa llama.cpp b10333 Vulkan stesso hardware —
decode 66.6 / prefill 1230 tok/s vs browser 15.6/35.7 = **gap 4.3× / 34×**,
limite INFERIORE (`results/engine/native-baseline-llamacpp-vulkan-2026-08-09.json`);
(b) recon famiglia Qwen 3.5/3.6 (`docs/engine/study/2026-08-09-qwen35-family-recon.md`).
**GOAL `engine-fase-q1` APERTO** (tag `goal-engine-fase-q1-start`,
contratto v1 approvato dal PI, taglie: 4B mobile-target + 9B + 35B-A3B; il
2B = eventuale futuro). Spine in `.harness/goals/engine-fase-q1/PHASES.md`:
9 fasi sequenziali, plan-check pre-autorizzato (docket q1 item 1, revisione
naturale prima della fase 3). **Fasi 1-2 DONE (it.1-3, verifier PASS)**: spec con ratifica
(`docs/superpowers/specs/2026-08-10-engine-fase-q1-design.md`); GGUF in
casa e SHA-verificati (~29 GB, q35-verify-sha exit 0); reader parametrico
(`q35shape.ts`: shape dai metadata, inventario 426/427/733); primo
tokenizer in-engine (`q35tokenizer.ts`: BPE byte-level, id==oracolo sul
corpus 12 file, protocollo v2 `--no-escape`, semantica USER_DEFINED
llama.cpp:3171); oracolo b10333 supporta la famiglia senza upgrade. **Fase 3 DONE
(it.4-5)**: cpuref-f64 DeltaNet dalla fonte b10333 + 3 kernel WGSL
(`kernels/deltanet.ts`) — ktest 75/75 (69 GLM intatti + 6 nuovi, catena
T=12 stato-persistente maxAbs 5.4e-7, core a dims reali hd128). Suite 371
PASS, tsc pulito. **Fase 4 slice 1 DONE (it.6)**: cpuref-f64
4B e2e == ORACOLO al primo run (argmax su tutte le posizioni generate del
golden smoke committato; mrope text-only = NEOX-64 provato dalla fonte;
`q35cpurefmodel.ts`, test gated Q35_E2E=1). **Slice 2 DONE (it.7)**: assembly GPU
dei layer attn con pesi REALI == cpuref a L2rel ~1e-7 (ktest 79/79; fixture
q35-attn rigenerabile, rope parziale + sigmoidMul nuovi). **Slice 3 DONE (it.8): IL 4B GIRA
SU GPU** — orchestratore `q35gpumodel.ts` (562 dispatch/token, piano
precostruito, full residency), argmax GPU == ORACOLO 6/6 sul golden smoke,
ktest 80/80; pareti Chromium risolte (Range per-tensore; maxBufferSize per
la head 527 MB). **it.9 DONE**: golden full-corpus
committato (8 prompt, 1024 pos, provenance piena) e **SOGLIA 4B FISSATA a
RATCHET: top-1 ≥ 1012/1024 = 98.828125% AL PIN** (docket q1 item 8; run
GPU 29 min; conformance infra: q35conf + run-golden-q35.sh). **it.10 DONE — FASE 4 COMPLETA**:
riferimenti full-resident 4B committati (decode 22.93 tok/s p50 43.6 ms,
prefill seq 26.0, TTFT 25.8 s; hostState user-session-light; frame
correttezza-prima dichiarato — zero fusioni, readback per token). **it.11 DONE**: 9B CONFORME —
ratchet **1000/1024 = 97.656%** con analisi near-tie al pin (24 miss tutti
near-tie, 23/24 top-2, mediana 0.066 logit; docket item 9); embd/head
invertiti sul 9B gestiti (Q4_0/Q6_K); riferimenti 9B: decode 14.55, prefill
15.4, TTFT 40.3 s. **it.12 DONE — FASE 5 COMPLETA**: WP gap
fatto — decode 5.18×/4.62× vs nativo Vulkan stesso GGUF, readback 5.1/3.6
ms/token MISURATO (decode−prefillSeq), prefill 183×/171× = assenza di
batching; leve per ROI in docket item 10 e doc studio
(2026-08-10-q35-gap-decomposition.md); per il writeup: 4.6-5.2× a parità
di regime. **it.13 DONE — FASE 6 COMPLETA**:
dot4I8Packed/tuning escluse coi numeri del WP (rami previsti dal
done-when); probe subgroup-matrix committato — feature ESPOSTA sul nostro
harness ma INT8-ONLY, converge con dot4I8Packed post-prefill-batched;
spec §7 corretta. **Slice 1 DONE (it.14)**: dequantQ4_K
ancorato a gguf-py su byte reali + gemvQ4K ktestato a dims expert (82/82)
+ `q35MoeFfnRefF64` dalla fonte (softmax→top8→norm clamp esatto→no scale;
shared con gate sigmoid scalare — TUTTO diverso da GLM). **Slice 2 DONE (it.15)**: cpuref
35B-A3B == ORACOLO al primo run (130 s; reader lazy da fd; golden smoke
35B committato; fix head non-tied). Verifier: FAIL amministrativo journal
sanato; elevata a docket item 7 l'arch falsa nei metadata dei golden q35.
**Slice 3a DONE (it.16)**: DECISIONE — residency GLM intatto, paging q35
= strato nuovo parametrico; blocco MoE GPU reale == cpuref L2rel 2-3e-7
(entrambe le classi down, arena offset-binding; slot in byte REPACKED:
Q6_K 210→212). **it.17 DONE nella sostanza** (verifier: 5/5 ricalcolato, root-cause
corroborato al byte; FAIL puntuale sanato — la run full era abortita per
golden mancante): 35B su GPU con paging (arena a CHUNK ≤2 GiB, LRU
on-miss, residenza 70.3% al budget 12 GiB, firma routing, zero OOM;
esecuzione segmentata correttezza-prima 1 sync/layer). **IN CODA**: (1)
golden full 35B su CPU in generazione (task bo9rnqy88, ~30-60 min; arch
fix golden.cpp fatto — item 7 mezzo sanato, restano i campi dei golden
4B/9B da rigenerare prima del paper); (2) al termine: conf GPU full 35B
(~11-12 h, ALBERO CONGELATO, scripts/q35-conf-run.mjs --model 35b
--golden-kind full --timeout-min 900) → SOGLIA RATCHET 35B a docket coi
numeri. Nel frattempo il loop può fare solo lavoro documentale o
attendere (GPU e albero vincolati). Dopo: fase 8 (tier+recall+bandmodel)
e fase 9 (chiusura). Perimetro: path testo Qwen
3.5/3.6, fedeltà bit-verificata metodo GLM, tier mobile+8/12/16 emulati, WP
decomposizione gap kernel-vs-paging, leve kernel bounded. Riancorarsi da:
`.harness/goals/engine-fase-q1/{GOAL,PHASES,journal,docket}.md`.
Rischio tecnico dominante già nominato: **kernel DeltaNet WGSL** (ibrida 3:1
linear-attention : GQA, mamba f32). Fuori sequenza, non aperti: hero-demo M4
(docket c3c item 8, PI-gated per hardware), fase D — moltiplicatori/spec-dec
MTP (DOPO q1, è per-modello), igiene fuori-goal (§3). Riancorarsi da:
`.harness/goals/engine-fase-c3c/{GOAL,journal,docket}.md` (item 9/10),
`docs/publishing/{split-plan,paper-contract-draft}.md`, direction §7, ledger §A.

## 2bis. State delta (sessione 26, 2026-08-09 — post-fase-C, fuori goal)

- **Baseline nativa** (429bfd2): llama.cpp b10333 Vulkan, stessa GPU/driver/
  GGUF, p512/n64 — 66.6 / 1230 tok/s (best `-ncmoe 8`; 26.7/338 con tutti gli
  expert su CPU). Chiude il [VERIFY] del writeup. Chiave architetturale:
  llama.cpp fa **compute-at-data** sugli spillover (computa su CPU), noi li
  trasferiamo ⇒ direzione WASM-SIMD REGISTRATA, non promessa.
- **Recon Qwen** (429bfd2): 3.5 + 3.6 Apache 2.0, GGUF maturi (oracolo
  llama.cpp preservato); tutta la famiglia è IBRIDA 3:1 (KV solo su 1/4 dei
  layer ⇒ ctx lungo quasi gratis), MTP nativa ovunque (fase D apparecchiata);
  35B-A3B = 256 expert top-8 da ~1.7 MB Q4 (granularità 3× più fine del nostro
  paging). 3.8 = solo API, niente pesi open.
- **Publishing**: split in 3 repo congelato ALLA PUBBLICAZIONE (20f3d11,
  `docs/publishing/split-plan.md`); paper = companion del rilascio su Zenodo
  preprint, si charterizza a valle di q1 coi numeri finali (e28c0a8,
  `docs/publishing/paper-contract-draft.md`). Il lab resta il workshop.
- Nessun goal aperto in questa sessione; nessun codice engine toccato.

## 2. State delta (sessione 25, 2026-08-08/09 — goal C3c intero)

- Fasi 1-9 in 10 iterazioni, ogni iterazione verificata da loop-verifier
  e committata su main (10 commit, 5c2de37..58f5cf4).
- Meccanismi nuovi in `src/engine/`: `slabBudgetCtxAware` (residency.ts,
  riserva tarata 512 MiB), prefetch in-forward decode+prefill chunked
  (glmmodel.ts, tap router L+1, recall 91.92% @K=8), policy tier+AUTOPIN
  (residency.ts, pin ≤12.5% HARD), `bandmodel.ts` (±1% sui 3 punti, test
  permanente). Tutto dietro flag: default = comportamento storico.
- Tool nuovi: `tools/opfs-cold/` (banda fredda browser, freddezza provata),
  `scripts/glm-instanton-run.mjs` (instant-on, protocollo mediane 3+3).
- Riferimenti nuovi (docket c3c item 2/5-bis): b12 optimistic
  13.172/31.26/14.74; hit-rate policy 1472/736 (routing-policy-*.json);
  instant-on 1.247 ≤ 1.25× (ruling item 1a).
- Docs: direction §7 fase C CHIUSA, §8.3 banda browser; ledger §A riga
  paging CHIUSA; spec `2026-08-08-engine-fase-c3c-design.md`.
- Storia completa: `.harness/goals/engine-fase-c3c/journal.md` (it.0-10);
  lo storico delle sessioni 17-24 vive nei journal dei goal c3a/c3b.

## 3. Open threads

- **Fuori-goal, non bloccanti**: ratifiche c3a item 14b / item 2 formale /
  19-21 prese d'atto; `npm run test:conformance` punta all'harness
  benchmark; BASE_URL :5173 in 2 script; goal fase-1b/fase-2 in STANDBY
  deliberato (benchmark, ripresa più avanti — NON stale, PI 2026-08-09); campi power hostState scambiati (c3c item 7);
  selezioni prefill non in eusage (c3c item 6); profilo bench ~6.4 GiB in
  `~/.cache/blab-opfs-cold-profile` (cancellabile).
- C3b/C3a/C2: chiusi, nessun item aperto che richieda azione.

## 4. Landmines

- Bench su questa macchina: gate a macchina scarica o host state DICHIARATO
  nel JSON; costo per-fetch varia ~4× fra sessioni host (upload 0.65-3.7
  ms) — mai confrontare bande ±5% fra host state diversi.
- Margine instant-on 0.3%: qualunque ritocco al prefill invalida il
  riferimento — rimisurare col protocollo v2 (mediane 3+3), non a sessione
  singola.
- Prefill in scarsità collassa (30→4 tok/s a 1472/736 slot): su device
  piccoli il TTFT è prefill-bound, primo numero da guardare in fase D.
- Run GPU lunghe SOLO ad albero congelato (HMR vite uccide la run) e 60 s
  fra run; niente pipe sull'output dei runner; `/tmp` è tmpfs (profili
  Chrome in ~/.cache); KV = 108 288 B/token (f32, NON il vecchio 54 KB).
- Full-corpus SOLO per firma/nonreg/riferimenti nuovi; sim su traccia o
  subset di PROMPT INTERI per esplorare (mai --cap: perde il decode).
- Oracolo CPU quantizza q8: mai gate su near-tie; golden ≥98.828% è un PIN.
- Storiche: error scope su ogni submit; vite :5199; f32-first; `pos ===
  kvLen` hard; var WGSL nei loop da azzerare; mai mapAsync su submit non
  emesso; lock OPFS transitorio possibile al primo accesso (retry).

## 5. Docket (user decisions pending)

- **Via all'apertura di q1** — è il §1: contratto su disco e approvato,
  manca solo il "partiamo" del PI per lanciare /goal.
- Timing del blog (prima del paper o insieme) — aperto, non bloccante
  (docket c3c item 10).
- Freshness di HANDOFF: hook "messaggio nomina HANDOFF ⇒ HANDOFF.md in
  index" = tampone approvato dal PI; il meccanismo vero (garanzia di
  refresh a fine goal, non a ogni commit) è da disegnare insieme —
  parcheggiato (PI 2026-08-09).
- c3c item 8: input hero-demo M4 (PI-gated per hardware).
- Ratifiche c3a pendenti (14b, 2 formale, 19-21) — prese d'atto, non urgenti.
