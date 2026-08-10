# HANDOFF — browser-llm-lab   (updated 2026-08-10, sessione 27 — goal engine-fase-d in corso: core unificato + gate a invarianti, GLM bit-identico; next = it.7 migrazione q35gpumodel (chiude la fase 1))

## 1. Next decidable

**AL LAVORO: goal `engine-fase-d`** (chartered 2026-08-10, tag
`goal-engine-fase-d-start`, PHASES 9 fasi). Nasce dal ruling PI
2026-08-10 (direction §7-ter) e dalla riapertura di q1 (docket q1 item
14): una famiglia nuova non è importata finché non ha le stesse
ottimizzazioni delle esistenti; il codice si UNIFORMA.

**Fatto (it.1-3, ogni iterazione verificata)**: `moe.ts` e `residency.ts`
sono una meccanica sola — tabella di geometria dei quant, UN router con
due configurazioni (GLM sigmoid+bias+1.8 / Qwen softmax), UN builder di
slab, e il MOTORE della cache (stati di classe, ripartizione del budget,
arena, repin, stats, path caldo `ensure`) guidato da `MoeModelConfig`.
GLM è una configurazione e resta **BIT-IDENTICO** (ktest 84/84:
2layer L2rel 2.07e-7, arena-vs-slotrange BIT-A-BIT, layer0 2.35e-7).
Suite 387. Nato il GATE STRUTTURALE (`tests/engine-one-mechanism.test.ts`)
che vieta un secondo router/layout/arena.

**Gate anti-duplicazione (it.4-6)**: il verifier ha bocciato TRE versioni
di un gate a scansione del sorgente, l'ultima con 5 evasioni eseguite —
di cui una (un router Qwen legittimo: softmax puro, niente clamp, niente
nomi di tensori, niente VRAM) NON catturabile da nessuna impronta
testuale, perché la differenza fra duplicazione e seconda famiglia è
SEMANTICA. Esito: **l'invariante vive nel SISTEMA DI TIPI** — marchio di
conio (`unique symbol`) su `SlotRef`, che solo `residency.ts` può CONIARE
(11 sonde ostili del verifier rifiutate da `tsc`). Ferma la CONTRAFFAZIONE,
non l'indifferenza: q35gpumodel oggi ignora `SlotRef` e tsc è verde — il
marchio diventa portante con it.7. E
`tests/types/slotref-brand.ts` (`@ts-expect-error`) va rosso se il
marchio sparisce. `tests/engine-one-mechanism.test.ts` resta come
**RATCHET** su impronte note (scansione di tutto `src/`, ancorata,
estensioni incluse) con la pretesa ridimensionata per iscritto: non è una
prova, ed è sbagliato usarlo come tale. Docket item 4 CHIUSO.

**Al lavoro: it.7** = migrazione di `q35gpumodel` alla meccanica unica.
È questa — non il gate — che elimina DAVVERO la duplicazione: le tre voci
DEBITO NOTO dell'allowlist spariscono lì e la fase 1 chiude. Poi le fasi 2-5
(slab pre-impacchettati, decode multi-step, prefill batched, policy MoE).
**Regola del goal**: bench pieni SOLO alle fasi 6 e 8 — durante lo
sviluppo micro-bench e ktest.

## 2. State delta (sessione 27, 2026-08-10 — goal q1 intero, it.0-21)

- 21 iterazioni in ~14 h, verifier gate su ognuna (2 FAIL sanati: journal
  it.15, run-morta it.17). Storia completa nel journal del goal.
- Moduli nuovi in `src/engine/`: q35shape (shape dai metadata),
  q35tokenizer (PRIMO tokenizer in-engine, BPE byte-level), q35cpuref +
  q35cpurefmodel (cpuref-f64 famiglia, reader lazy), q35sample,
  kernels/deltanet (conv+gates+core WGSL), q35gpumodel (orchestratore
  denso+MoE con arena a chunk e LRU), q35conf/ (conformance+bench+tap).
  gguf/quant/wgsl estesi SOLO additivi (Q4_K, axpy, sigmoidMul, ropeDims).
  GLM (glmmodel/residency/moe/bandmodel) INTATTO.
- Strumenti: run-golden-q35.sh (provenance, CORPUS_DIR, arch fix),
  q35-conf-run/q35-bench-run (--model/--arena-gib), q35-looka-run,
  q35-bandmodel-fit, q35-tier-mobile-gen, webgpu-subgroup-matrix-probe,
  q35-verify-sha, fixture generators.
- Soglie ratchet nel docket q1 (item 8/9/12); leve e convergenza int8
  (item 10/11); non-reg host-gated (item 13).
- Docs: direction §7-bis (generalizzazione coi numeri), ledger +4 righe,
  spec q1 con §7 aggiornata dal probe, gap-decomposition §5.

## 3. Open threads (fuori goal, non bloccanti)

- **Rerun non-reg a boot pulito** (docket q1 item 13): glm-bench
  optimistic autobudget (>=13.43, banda vs 15.641) + glm-conf full +
  Qwen2.5. È il gate d'apertura della prossima sessione.
- Golden q35 4B/9B: campo arch "deepseek2" stale nei metadata (fix del
  tool fatto; rigenerare i CAMPI prima del paper — docket q1 item 7).
- BASE_URL :5173 in engine-bench.mjs e conformance (thread c3a).
- Ratifiche c3a pendenti (14b, 2, 19-21); campi power hostState scambiati
  (c3c item 7); profilo bench ~6.4 GiB in ~/.cache cancellabile.
- Goal fase-1b/fase-2 = STANDBY deliberato benchmark (non stale).
- Disco: +50 GB questa sessione (29 GGUF q35 + profili/golden).

## 4. Landmines

- **Non-reg NUMERICA solo a host comparabile DICHIARATO** (lezione it.21):
  page cache OPFS e VRAM baseline muovono stallResidenza 4.3→29 ms/token
  e P(dirty) 0.81→0.98; mai confrontare bande fra host diversi (c3c).
- MoE q35: prefill bench con `await` (mai fire-and-forget: mapAsync del
  router); arena SOLO a chunk ≤2 GiB (adapter cap 4 GB, il monolitico
  fallisce SILENZIOSO); slot in byte REPACKED (Q6_K 210→212).
- fetch/ArrayBuffer >2 GiB: sempre Range/pread (3 pareti trovate).
- recall@4 lookahead ha denominatore 8 (tetto 50%): mai confrontarlo
  con @8 senza dichiararlo.
- Storiche: run GPU ad albero congelato + 60 s; niente pipe sui runner;
  near-tie mai gateati; f32-first; var WGSL azzerate; full-corpus solo
  per riferimenti; KV GLM 108 288 B/token; q35: 40 960 (35B) / 65 536
  (densi).

## 5. Docket (user decisions pending)

- **Prossimo passo post-q1**: release (split+paper+blog, sequenza c3c
  item 10) vs fase D (moltiplicatori) — la decisione di §1.
- Rerun non-reg a boot pulito (q1 item 13) — operativo, primo atto.
- Hero-demo M4 (c3c item 8) — PI-gated per hardware.
- Timing del blog (prima del paper o insieme) — aperto, non bloccante.
