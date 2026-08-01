# HANDOFF — browser-llm-lab   (updated 2026-08-01, session 17 — goal engine-fase-c2 bloccato su ruling)

## 1. Next decidable

**RULING PI su docket engine-fase-c2 item 6 (unico aperto) — il loop è in
STOP BY DESIGN.** L'item 7 (Qwen) è RISOLTO: host quiescito dal PI, ri-misura
PASS su tutti e tre i gate con margine (K=8 **321.9** vs soglia 282.9 — sopra
anche il baseline 287.5, che era ANCH'ESSO su host degradato; prefill 600.2 ms
vs ≤726.3; K=1 244.0 vs 226.5; gpuBusy 2.22 vs 3.06 ms/token).
Resta l'**item 6 — gate GLM falliti anche a macchina quiescente** (numeri
puliti nel ri-bench b12-quiesced, che SOSTITUISCONO i 3.30/4.41 contaminati):
- **decode 4.64 tok/s vs gate 13.43** (FAIL 2.9×); attribuzione pulita:
  215.5 ms/token = stallo residenza 56.1 (pack 42.5) + struttura **158.9**
  (1.816 dispatch + **47 sync CPU↔GPU/token**). Proiezione a residenza
  perfetta: **6.3-7.0 tok/s, ancora sotto il floor** ⇒ fallimento
  strutturale del piano per-token, non della residenza.
- **prefill 5.22 vs 56.58**: CAPACITÀ MANCANTE, non divario — il floor è il
  pp512 *batched* di llama.cpp, il motore fa forward sequenziali (nessun
  percorso M>1).
Le leve sono dimensionate ma fuori scope C2 (sono C3): item 6 = deroga con
misura onesta [raccomandata] / estensione scope / FAIL formale. Alla
chiusura goal il PI sceglie anche il baseline Qwen permanente (287.5
conservativo vs 321.9 con condizioni termiche dichiarate — coda item 7).
Dopo il ruling resta solo la fase 7 (chiusura: input C3 a docket,
ledger/direction, merge).
Riancorarsi da: `.harness/goals/engine-fase-c2/{PHASES,docket,journal}.md`
(journal it.11 + "Post-it.11" = misure e debug per esteso), report
`results/engine/bench-glm-4090-b12-quiesced-2026-08-01.json` (GLM pulito),
`results/engine/bench-4090-2026-08-01T16-34-04-484Z.json` (Qwen pulito).

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
- **Post-it.11**: PI quiesce l'host (kill zen+VSCode; brief indagine CPU in
  `~/cpu-investigation/brief.md`, fuori repo) e ordina la ri-misura =
  ruling item 7(a). Qwen PASS tutti i gate; ri-bench GLM b12 quiescente:
  4.64/5.22, gate ancora FAIL con attribuzione ripulita. Item 7 RISOLTO,
  item 6 aggiornato coi numeri puliti.
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

- Goal C2: fase 6 MISURATA (anche da quiescente) ma gate GLM non passati →
  fase 7 (chiusura) BLOCCATA dal ruling item 6. Nessun timebox attivo.
- Alla CHIUSURA goal il PI ratifica: campione gate (i) 2/8 prompt (docket
  item 5) + baseline Qwen permanente (coda item 7).
- Indagine CPU host in corso in sessione separata (brief
  `~/cpu-investigation/brief.md`): l'esito può cambiare le condizioni
  termiche "tipiche" della macchina — rilevante per la scelta del baseline.
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

- **engine-fase-c2 item 6: gate tok/s GLM falliti anche da quiescente**
  (decode 4.64/13.43, prefill 5.22/56.58 = capacità mancante) con
  attribuzione pulita — deroga / estensione scope / FAIL.
- engine-fase-c2 item 5: ratifica del campione gate (i) (2/8 prompt,
  256/256) — alla chiusura del goal.
- Coda item 7 (risolto): scelta del baseline Qwen permanente alla chiusura
  — 287.5 (conservativo, host tipico) vs 321.9 (quiescente, condizioni
  termiche dichiarate obbligatorie nel protocollo).
- (Ratificati in sessione: item 4 = opzione (a), routing informativo.)
- fase-1b-matrice (11 item) e fase-2-deep-dive (5 item): stale, da triage
  weekly-maintenance.
