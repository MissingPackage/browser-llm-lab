# HANDOFF — browser-llm-lab   (updated 2026-08-01, session 16 — goal engine-fase-c2 in fase 6)

## 1. Next decidable

**it.11 del goal `engine-fase-c2` — fase 6 slice 2: BENCH coi gate hard**
(ultimo ostacolo sostanziale; dopo resta solo la fase 7 di chiusura).
Protocollo B2 (mediane su run ripetute, finestre e ctx dichiarati) sul
forward GLM di produzione (`glmmodel.ts` + `glmsource.ts`, OPFS già
importato): **GATE decode ≥13.43 tok/s e prefill ≥56.58 tok/s** (floor =
oracolo CPU C1, `results/engine/moe-oracle/llama-bench-*.json`) +
**non-regressione Qwen** (conformance fase A verde; first-light K=8
≥287.5−2σ, K=1 ≥238.3−2σ, prefill ≤697.8+2σ, metodologia same-day B2) +
report telemetria (tok/s, 1.816 dispatch/token, hit-rate, stallo
ms/token, occupazione) = input C3. Scelte da fare NEL bench (parametri
dichiarati nel report, non policy): budget slab 11 vs 12 GiB (a ctx corto
il KV si riduce ⇒ si può tornare a ~12; con head a 12 GiB la VRAM sforava
— OOM osservato); prefill decode-only vs chunked. Rischi quantificati:
stallo residenza (pack 6.1 ms/miss DOMINANTE su read 3.1 + upload 1.6;
leva pronta: repack all'IMPORT, journal it.7) e 46 sync router/token
(strutturali: la selezione CPU decide i bind). Gate fallito ⇒ misura
onesta nel report + docket (deroga = decisione PI, non-regressione
permanente); niente ottimizzazioni fuori scope C2.
Riancorarsi da: `.harness/goals/engine-fase-c2/{GOAL,PHASES,docket,journal}.md`,
spec §6-§8, `src/engine/{glmmodel,glmsource,residency,moe}.ts`, harness
`scripts/glm-{route,conf}-run.mjs` (pattern runner), bench Qwen esistente
(`engine.worker.ts` + protocollo B2 in docs/superpowers/specs C1/B2).

## 2. State delta (questa sessione, 16)

- **Fase 5 CHIUSA** (it.6-9, 4 verifier PASS): kernel MoE (router replica
  build_moe_ffn riga-per-riga), residenza minima (OPFS 17.2 GB SHA-streaming
  + ExpertCache LRU due size-class), forward 47 layer di produzione
  (`glmmodel.ts`), harness replay. Gate routing sotto soglia (85.8-94.1%) →
  discriminatore cpuref-f64 = motore al 100% (stessi mismatch) → **ruling
  PI item 4 = opzione (a)**: routing informativo, spec §7 emendata.
- Report informativo routing full-corpus: decode 88.51%; **hit residenza
  97.56% @12 GiB** su 31k posizioni (input C3).
- **Fase 6 slice 1 CHIUSA** (it.10, verifier PASS): output head +
  conformance logits full-model — **gate (i) motore≡cpuref-f64 256/256,
  gate (ii) top-1 vs golden 1012/1024 = 98.83%**; su p7 E p4 la divergenza
  dal golden è lo stesso token per motore e cpuref (firma q8). mscale=1
  confermato. Nuovi: `glmsource.ts`, `glmconf/*`, `glmroute/*`,
  `GlmMlaAttnAbsorbedRefF64` (+identità in suite), analisi env-gated
  `tests/analysis-{route,conf}-cpuref.test.ts`.
- Stato test: ktest 30/30 (4090), suite 220+2 skipped, tsc pulito. Tutto
  pushato su origin/main fino a `940d954`.

## 3. Open threads

- Goal C2: fase 6 slice 2 (bench) → fase 7 (chiusura: input C3 a docket,
  ledger/direction, merge — ruling permanente). Timebox: nessuno attivo
  sulle slice di fase 6.
- Alla CHIUSURA goal il PI ratifica: campione gate (i) 2/8 prompt (docket
  item 5) + eventuali derive bench.
- Goal harness stale mai chiusi: `fase-1b-matrice` (11 docket),
  `fase-2-deep-dive` (5) — igiene da /weekly-maintenance.
- Duplicato results/opfs-bench-*.json in due posizioni (segnalato, non toccato).

## 4. Landmines

- Oracolo CPU quantizza le attivazioni q8: MAI gate diretti engine-vs-oracolo
  su selezioni near-tie (routing 99% è morto così, it.9); il confronto giusto
  è vs cpuref-f64; golden ≥97% assorbe il drift. maxAbsDeltaLogit è metrica
  di SCALA (58 con KL 2e-7 osservato): mai promuoverla a gate.
- VRAM 16.4 GB: slab 12 GiB + head (~270 MB) + KV ctx 6k = OOM (bind group
  "invalid due to previous error" a cascata) — visto e riprodotto; budget 11
  GiB con head, o ctx corto.
- `/tmp` è tmpfs 16 GB: il profilo Chrome con OPFS 17 GB sta in
  `~/.cache/blab-glmroute-profile` (E2E_PROFILE); l'import si salta su
  size-match.
- mlaAttnDecode tiene scores[ctxMax] in workgroup memory: ctx>4k richiede
  maxComputeWorkgroupStorageSize 32 KB negoziato (fail-fast in createGlmModel).
- Runner in background: NIENTE pipe su tail/grep (bufferizza e maschera gli
  exit code — successo due volte in questa sessione); output diretto su file.
- Il mini-modello sintetico (ktest 2-layer) NON si estende in profondità:
  ampiezze ~6e7 già a 2 layer, rischio overflow f32.
- Storiche sempre valide: error scope su ogni submit; vite porta 5199
  dedicata; f32-first Chrome/Linux; `pos === kvLen` hard; var WGSL nei loop
  da azzerare; mai mapAsync su buffer di submit non emesso.

## 5. Docket (user decisions pending)

- engine-fase-c2 item 5: ratifica del campione gate (i) (2/8 prompt,
  256/256) — alla chiusura del goal.
- (Ratificati in sessione: item 4 = opzione (a), routing informativo.)
- fase-1b-matrice (11 item) e fase-2-deep-dive (5 item): stale, da triage
  weekly-maintenance.
