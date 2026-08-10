# HANDOFF — browser-llm-lab   (updated 2026-08-10, sessione 27 — GOAL engine-fase-q1 CHIUSO: famiglia Qwen 3.5/3.6 conforme; next = decisione PI + rerun non-reg a boot pulito)

## 1. Next decidable

**DECISIONE PI: prossimo passo dopo q1.** Il goal `engine-fase-q1`
(generalizzazione) è **CHIUSO** (it.21, checklist DONE WHEN 11/11 nel
journal, tag `goal-engine-fase-q1-done`): il motore esegue il path testo
della famiglia Qwen 3.5/3.6 — **4B 98.828% · 9B 97.656% · 35B-A3B 98.926%
top-1 vs oracolo** (ratchet full-corpus, near-tie analizzati), il 35B col
paging VERO (121k eviction, zero OOM). Gap nativo **4.6-5.2× decode a
parità di regime** (scomposto con misura); tier 8/12/16 GB misurati col
collasso attribuito; recall lookahead 82.67%@8 spiegato. Numeri =
CORRETTEZZA-PRIMA dichiarata (i moltiplicatori sono fase D).
**PRIMA DI TUTTO alla riapertura: rerun non-reg GLM a boot pulito**
(docket q1 item 13 — HOST-GATED in questa sessione: page cache + VRAM
baseline; correttezza GLM piena e verde, 84/84). Poi la sequenza congelata
(docket c3c item 10): **release = split repo + paper Zenodo + blog** —
oppure fase D (moltiplicatori sul modello target: prefill batched = leva
1, decode no-readback −5.1 ms misurati, poi spec-dec MTP). Riancorarsi:
`.harness/goals/engine-fase-q1/{GOAL,PHASES,journal,docket}.md`,
direction §7-bis (i numeri), `docs/publishing/{split-plan,
paper-contract-draft}.md`.

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
