# HANDOFF — browser-llm-lab   (updated 2026-08-09, sessione 26 — fase C COMPLETA, direzione post-C DECISA; next = avvio goal q1 generalizzazione)

## 1. Next decidable

**AVVIO GOAL `engine-fase-q1` — GENERALIZZAZIONE.** La direzione post-fase-C
NON è più una decisione aperta: il PI l'ha data (docket c3c item 9, "vai") e
poi raffinata (item 10). Sequenza congelata: **q1 (generalizzazione) →
release (split repo + paper Zenodo + blog)**. I due prerequisiti immediati
sono FATTI: (a) baseline nativa llama.cpp b10333 Vulkan stesso hardware —
decode 66.6 / prefill 1230 tok/s vs browser 15.6/35.7 = **gap 4.3× / 34×**,
limite INFERIORE (`results/engine/native-baseline-llamacpp-vulkan-2026-08-09.json`);
(b) recon famiglia Qwen 3.5/3.6 (`docs/engine/study/2026-08-09-qwen35-family-recon.md`).
**Passo successivo = (c) goal-brief q1**: il contratto è approvato in sostanza
dal PI in chat ("mi torna") ma è **ASSUMED, non è su disco** — va scritto,
ratificato in spec (pattern c3a item 3 / c3b item 4) e poi aperto con /goal.
Perimetro previsto dal ruling: famiglia Qwen 3.6/3.8, 2-3 taglie, 2-3 baseline
hardware consumer (8-12 GB = scarsità vera), prefill/TTFT DENTRO il goal.
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

- **Ratifica del contratto q1** (generalizzazione) — è il §1: il PI ha
  approvato in sostanza in chat, va scritto su disco e ratificato in spec.
- Timing del blog (prima del paper o insieme) — aperto, non bloccante
  (docket c3c item 10).
- Freshness di HANDOFF: hook "messaggio nomina HANDOFF ⇒ HANDOFF.md in
  index" = tampone approvato dal PI; il meccanismo vero (garanzia di
  refresh a fine goal, non a ogni commit) è da disegnare insieme —
  parcheggiato (PI 2026-08-09).
- c3c item 8: input hero-demo M4 (PI-gated per hardware).
- Ratifiche c3a pendenti (14b, 2 formale, 19-21) — prese d'atto, non urgenti.
