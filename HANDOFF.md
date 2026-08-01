# HANDOFF — browser-llm-lab   (updated 2026-08-01, session 17 — goal engine-fase-c2 bloccato su ruling)

## 1. Next decidable

**RULING PI su docket engine-fase-c2 item 6 e 7 — il loop è in STOP BY
DESIGN, nessun passo tecnico decidibile senza di te.**
Il bench di fase 6 (it.11) ha misurato e i gate hard NON passano:
- **GLM: decode 3.30 tok/s vs gate 13.43, prefill 4.41 vs 56.58** (config
  migliore slab 12 GiB; a 11 GiB 2.96/3.87). Attribuzione misurata: dei
  302.7 ms/token, 112.4 sono stallo residenza (pack 71.6 dominante) e
  190.3 sono struttura (1.816 dispatch + **47 sync CPU↔GPU/token**); i
  token a zero miss costano 136.3 ms ⇒ **anche con residenza perfetta si
  starebbe a 5.3-7.3 tok/s**. Il gate prefill non è un divario omogeneo ma
  una CAPACITÀ MANCANTE: il floor è il pp512 *batched* di llama.cpp, il
  motore fa 461 forward sequenziali (nessun percorso M>1). Le leve esistono
  e sono dimensionate, ma stanno **fuori dallo scope C2** (sono C3):
  serve la tua decisione (item 6: deroga con misura onesta / estensione
  scope / FAIL formale).
- **Qwen: K=8 263.5 vs soglia 282.9 e prefill 747.9 ms vs 726.3 sotto
  soglia**, MA conformance identica (98.05/100.00), K=1 e seq prefill
  MIGLIORI del baseline, codice del percorso Qwen byte-identico al commit
  di baseline. Firma da throttling dell'host (browser utente `zen` all'86%
  di CPU da 6 giorni, package 99 °C, GPU 1425-1620 MHz sotto carico su
  3105 max), non da regressione di codice (item 7: accettare l'attribuzione
  con o senza ri-misura a macchina quiescita / trattarla da regressione).
Dopo il ruling resta solo la fase 7 (chiusura: input C3 a docket,
ledger/direction, merge).
Riancorarsi da: `.harness/goals/engine-fase-c2/{PHASES,docket,journal}.md`
(journal it.11 = misure e debug per esteso), report
`results/engine/bench-glm-4090-b1{1,2}-2026-08-01.json`,
`results/engine/bench-4090-2026-08-01T13-51-55-018Z.json` (Qwen).

## 2. State delta (sessione 17)

- **Fase 6 slice 2 (bench) ESEGUITA**: harness nuovo `glmbench`
  (`src/engine/glmbench/`, `glmbench.html`, `scripts/glm-bench-run.mjs`) —
  protocollo B2 sul forward di produzione, decode greedy reale, telemetria
  per-token che permette l'attribuzione senza strumentare il motore.
- Parametri dichiarati decisi nel bench: **slab 12 GiB** (a ctx corto il KV
  è 28 MB ⇒ nessun OOM; domina 11 GiB su tutto) e **prefill decode-only**
  (non esiste path batch M>1).
- Non-regressione Qwen ri-misurata same-day (2 run + conformance) con
  debug completo dell'attribuzione (v. §1 e journal it.11).
- Stato test: suite 220 passed + 2 skipped, `tsc --noEmit` pulito.

## 3. State delta (sessione 16)

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

## 4. Open threads

- Goal C2: fase 6 MISURATA ma gate non passati → fase 7 (chiusura) BLOCCATA
  dai ruling item 6/7. Nessun timebox attivo.
- Alla CHIUSURA goal il PI ratifica: campione gate (i) 2/8 prompt (docket
  item 5) + le derive del bench (item 6/7).
- Ri-misura Qwen a macchina quiescita: richiede la chiusura del browser
  utente `zen` (azione umana) — schedulata dal PI se opta per item 7 (a).
- Goal harness stale mai chiusi: `fase-1b-matrice` (11 docket),
  `fase-2-deep-dive` (5) — igiene da /weekly-maintenance.
- Duplicato results/opfs-bench-*.json in due posizioni (segnalato, non toccato).

## 5. Landmines

- **Bench su questa macchina: la GPU è thermally/power-capped dalla CPU.**
  Con carico utente pesante (osservato: `zen` all'86%, package 99 °C) la GPU
  resta a 76-83 °C e 1425-1620 MHz su 3105 max: i path a CARICO SOSTENUTO
  perdono ~8%, quelli latency-bound no. Ogni bench di non-regressione va
  letto con clock/temperatura a fianco (`nvidia-smi --query-gpu=
  temperature.gpu,clocks.sm,power.draw,utilization.gpu`).
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

## 6. Docket (user decisions pending)

- **engine-fase-c2 item 6: gate tok/s GLM falliti** (decode 3.30/13.43,
  prefill 4.41/56.58) con attribuzione — deroga / estensione scope / FAIL.
- **engine-fase-c2 item 7: non-regressione Qwen K=8+prefill sotto soglia**,
  attribuita all'host — accettare / ri-misurare a macchina quiescita /
  trattare da regressione.
- engine-fase-c2 item 5: ratifica del campione gate (i) (2/8 prompt,
  256/256) — alla chiusura del goal.
- (Ratificati in sessione: item 4 = opzione (a), routing informativo.)
- fase-1b-matrice (11 item) e fase-2-deep-dive (5 item): stale, da triage
  weekly-maintenance.
