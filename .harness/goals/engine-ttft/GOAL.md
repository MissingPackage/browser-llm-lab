GOAL: Il Qwen3.5-4B in browser mostra il primo token entro 4 secondi su un
prompt da >= 6000 token a modello caldo, dove oggi il solo prefill ne costa
~192 (proiezione da 32,91 tok/s misurati) — o l'irraggiungibilità è dimostrata
coi numeri e col conto che la mostra.

<!-- CONTRATTO v1 (chartered 2026-08-13, PI in chat: "procediamo col goal in
     loop ... sulla scelta preliminare sono d'accordo con l'opzione A").
     Goal precedente: engine-kernel-decode, CHIUSO (47,93 tok/s a ctx 6333
     contro una soglia di 30, 4,82x). Questo goal eredita dal suo docket gli
     item 4 (clausola di portabilità e2), 5 (`--out` nei runner) e 6 (freeze
     dei due call-site GLM nel ktest).

     NASCE DA UNA CORREZIONE. Il numero "TTFT 22,7 s su un prompt da 6k" che
     HANDOFF portava è sbagliato in due punti, entrambi verificati sull'artefatto
     `results/engine/q35-bench-4b-tier8-fase-d-it43.json`:
       (1) il prompt è prompt-idx 4 = 388 token, non 6k. I 6333 sono il
           prompt-idx 0, usato per il DECODE del goal precedente. I due numeri
           vengono da run su prompt diversi e sono stati appaiati per errore.
       (2) `ttftMs = loadMs + prefillMs + firstMs` (q35conf.worker.ts:246):
           dei 22,695 s, 10,890 sono il LOAD del modello e 11,760 il prefill di
           387 token. Il primo token vero costa 46 ms.
     Il TTFT su un prompt da 6k NON È MAI STATO MISURATO. -->

RULING PRELIMINARE — OPZIONE A (PI, 2026-08-13):

  L'obiettivo dei 4 secondi vale sul TTFT **A MODELLO CALDO**: `prefill.ms +
  decode.firstMs`. Il LOAD del modello (10,89 s misurati) ha una soglia sua,
  separata, e non appartiene a questo goal — in una chat si paga una volta per
  sessione, il prefill a ogni turno.

  VINCOLO DI MISURA CHE SOPRAVVIVE AL RULING: ogni artefatto pubblica sempre i
  tre termini SCOMPOSTI (`loadMs`, `prefill.ms`, `decode.firstMs`) accanto al
  `ttftMs` aggregato, così che un ruling diverso sia ricalcolabile senza rifare
  le run.

DONE WHEN (all measurable):

- BASELINE RIMISURATA PRIMA DI QUALUNQUE RISCRITTURA, su prompt-idx 0
  (6333 token): JSON in `results/engine/` con `prompt.tokens >= 6000`,
  `hostState.declared = "quiescent"`, e i tre termini `loadMs` / `prefill.ms` /
  `decode.firstMs` presenti e distinti.
  MOTIVO: tutti gli artefatti di TTFT esistenti sono `*fase-d-it43*` del
  2026-08-11, che la landmine di HANDOFF marca come "percorso vecchio", e
  nessuno è sul prompt da 6k. Questo goal parte da un numero che oggi non esiste.

- TTFT A CALDO sul prompt-idx 0: `prefill.ms + decode.firstMs <= 4000 ms`,
  misurato da `scripts/q35-bench-run.mjs` con host dichiarato, JSON committato;
  OPPURE un JSON `kind: "q35-ttft-kernel-checkpoint"` con
  `decision: "excluded-by-numbers"` che porta banda efficace per segmento,
  tetto residuo e il conto che mostra perché la soglia non è raggiungibile.

- PREFILL: `prefill.tokS >= 1600` sul prompt-idx 0 (6333 token in <= 3,96 s).
  È la stessa clausola vista dal lato del rate, e serve perché è confrontabile
  col riferimento da battere: **32,91 tok/s**
  (`q35-bench-4b-tier8-fase-d-it43.json`, 387 token in 11.760 ms), da
  riconfermare sul codice di oggi dalla baseline.

- IL PREFILL TORNA PIÙ VELOCE DEL DECODE: `prefill.tokS > decode.tokS` sullo
  stesso JSON. Oggi è FALSO — 32,91 contro 47,93 dopo il goal sui kernel — ed è
  la firma del difetto: il chunk a M=8 rende 1,27x invece di ~M.

- FASE 0 SUPERATA PRIMA DI OGNI RISCRITTURA: micro-bench isolato delle varianti
  candidate, zero run di modello, JSON committato con G pesi/s e GB/s per
  variante. **REGOLA DI STOP**: se nessuna variante supera la forma attuale di
  >= 1,5x, la fase si chiude e il piano si riscrive invece di procedere.
  (È il meccanismo che ha chiuso il goal precedente: si eredita testuale.)

- LEVA 1 (GEMM multi-riga con riuso vero dei pesi) misurata: G pesi/s PRIMA e
  DOPO nello stesso JSON, a M = 1, 8, 16, 32. Il done-when non è il tempo ma il
  RIUSO: **byte di peso letti per token prefillato scende di >= 4x a M=8**
  (il massimo teorico a M=8 è 8x; chiedere il pieno sarebbe chiedere una cache
  perfetta).

- LEVA 2 (attenzione a chunk del prefill): `attnDecodeWgsl` con `batch: true`
  oggi instrada al LEGACY (`wgsl.ts:530`), che ha i tre difetti già chiusi sul
  decode — ridondanza GQA 4x, letture scalari, `scores: array<f32, ctxMax>`.
  Misura: ms per chunk a ctx 388 e a ctx >= 6000, PRIMA e DOPO, stesso JSON.
  Il golden `tests/fixtures/attn-decode-legacy.golden.json` inchioda il testo
  attuale: va aggiornato deliberatamente, non per sbaglio.

- PORTABILITÀ — clausola (e2) ereditata dal goal chiuso (suo docket item 4), in
  due pezzi perché il done-when originale ne nominava uno solo:
  (e2a) `engineNeeds(...).maxComputeWorkgroupStorageSize <= 16384 B` a QUALUNQUE
        ctxMax. Oggi vale 30.848 B per `rmsPairGemmSiluChunkFast`
        (K = dModel 896, mMax = PREFILL_M 8), il kernel FUSO DEL PREFILL.
  (e2b) il modulo dei limiti CONTA anche il ramo `batch` legacy dell'attenzione,
        che chiede `4·ctxMax + 256` B e OGGI NON È DICHIARATO
        (`gpulimits.ts:180-188`, commento esplicito in loco). A ctxMax 6400 sono
        25.856 B richiesti contro 1.536 dichiarati: sul percorso di prodotto la
        sotto-dichiarazione è già viva, non teorica. Il pareggio col termine
        fuso è a ctxMax ~7.392.
  Verificabile da `tests/gpulimits.test.ts`, che oggi asserisce il contrario
  (`WEBGPU_GUARANTEED = 16384` nominato come debito dichiarato).

- CORRETTEZZA, gate secchi a ogni merge: `node .harness/tools/engine-ktest.mjs`
  tutti PASS; argmax del golden 4B IDENTICO al pin; `npx vitest run` verde;
  `npx tsc --noEmit` pulito.

- NON-REGRESSIONE, banda ±5%: decode 4B >= 45,5 tok/s a ctx 6333 (riferimento
  47,93, `decode-kernel-checkpoint-4090-2026-08-13T04-30-46-419Z.json`) e GLM
  b12 optimistic entro ±5% di 13.172 / 31,26 / 14,74, misurata fresca su host
  dichiarato.

- Ogni numero pubblicato porta il proprio contesto: nessun tok/s senza
  `decodeContext`, nessun `ttftMs` senza i tre termini scomposti.

EVIDENCE OF DONE:
- `node scripts/q35-bench-run.mjs --prompt-idx 0 --n-decode 64 --vram-gib 8
   --declared quiescent` → JSON con `prompt.tokens`, `loadMs`,
   `prefill.{ms,tokS}`, `decode.firstMs`, `ttftMs`, `decodeContext`.
  **ATTENZIONE**: `--prompt-idx` ha default 4 = 388 token. È il flag che ha
  prodotto il numero sbagliato in HANDOFF. Leggere i flag PRIMA di spendere GPU
  (landmine di HANDOFF, terza recidiva registrata in it.9 del goal precedente).
- `node .harness/tools/engine-ktest.mjs` con `npx vite` acceso → tutti PASS
- `npx vitest run` verde · `npx tsc --noEmit` pulito
- micro-bench di fase 0: un JSON per variante in `results/engine/`
- portabilità: `npx vitest run tests/gpulimits.test.ts` con l'asserzione
  riscritta da debito dichiarato a garanzia
- non-reg GLM: runner GLM con host dichiarato

AUTHORITY GRANTED:
- may do autonomously: scrivere e riscrivere kernel WGSL, il piano di dispatch e
  il piano di prefill (`prefillplan.ts`, `PREFILL_M`); aggiornare il golden
  `attn-decode-legacy.golden.json` quando la riscrittura lo richiede,
  dichiarandolo nel journal; creare runner e fixture nuovi; eseguire bench GPU e
  ktest; committare e pushare su origin/main a fine task; mergiare a goal chiuso
  e verificato; aprire e chiudere item di docket che sono lavoro mio; escludere
  una leva coi numeri e dichiararlo; usare i workflow dell'harness
  (sdd-conductor, second-opinion, research-campaign, pattern-coverage) e i
  subagent.
- must docket (never do): cambiare la definizione di `ttftMs` o la soglia dei
  4 s (è il ruling preliminare qui sopra); cambiare i gate di correttezza o le
  bande di non-regressione; riattivare lo spec-dec MTP senza che la leva 1
  esista e sia misurata; cancellare o riscrivere codice committato da più di
  30 giorni fuori dal brief; pubblicare numeri fuori dal tag di release;
  spendere denaro; toccare i goal in standby (fase-1b, fase-2); toccare il 35B e
  le famiglie K-quant/Q8_0 (è il goal accanto, non questo).

CONSTRAINTS:
- WebGPU reale, non CUDA: `shader-f16` NON è disponibile su questo stack
  Chrome/Linux — ogni piano che lo assume è morto qui (vivo su M4/S22).
  `subgroupAdd` è stato confermato utilizzabile nella fase 0 del goal precedente
  e la riduzione di sottogruppo è già in produzione sul decode.
- Il GEMV veloce nuovo è **q4_0-only per costruzione** (`wgsl.ts:216-249`
  rifiuta q8_0/q4_1/bias/batch/scaledAccum): la leva 1 va progettata sapendo che
  la forma attuale NON copre il `batch`, che è esattamente ciò che serve al
  prefill.
- Nessuna attribuzione AI in commit, PR o merge.
- Le metriche misurate non peggiorano mai: banda ±5% su tok/s e TTFT, gate di
  correttezza secchi.
- I bench costosi si eseguono su codice finale, non su stati intermedi.
- Ogni misura dichiara host e contesto.
- Full-corpus solo per firma/non-reg/riferimenti nuovi; l'esplorazione va su
  simulazione o su prompt interi scelti, mai su `--cap`.
- Il codice dello spec-dec MTP resta in albero e gated finché la leva 1 non
  esiste: non si rimuove, non si attiva a intuito.

FUORI SCOPE (registrati, non aperti):
- Il LOAD del modello (10,89 s misurati): per il ruling A ha una soglia sua ed è
  un goal di famiglia residency/IO — parente del reperto GLM "il delta è tutto
  in `read` da disco" (docket engine-kernel-decode item 8).
- Il 35B e le famiglie K-quant/Q8_0: goal accanto a questo, con la sua fase 0.
- GLM: resta residency-bound; nessuna di queste leve lo porta a 30.
- I benchmark comparativi fra stack: un goal chiuso, l'altro in pausa dichiarata.

WORKING PROTOCOL: skills loop-iteration + done; verifier gate per ciclo; digest
ogni ciclo; stop-by-design quando il resto è docket-gated.

CONTEXT ANCHORS:
- `HANDOFF.md` — mappa e landmine
- `.harness/goals/engine-kernel-decode/{journal.md,docket.md}` — la fase 0 che ha
  funzionato, le leve escluse coi numeri (fusione GQA: misurata più lenta), e gli
  item 4/5/6/7 che questo goal eredita
- `src/engine/kernels/wgsl.ts` — `attnDecodeWgsl` (514, lo switch a 530),
  `attnDecodeLegacyWgsl` (552), `rmsPairGemmSiluChunkFastWgsl`
- `src/engine/gpulimits.ts:160-200` — la sotto-dichiarazione (e2b), commentata
  in loco
- `src/engine/prefillplan.ts` — `PREFILL_M = 8`
- `src/engine/q35conf/q35conf.worker.ts:246` — la definizione di `ttftMs`
- `docs/deep-dive/headroom-2026-08-12.md` — memo della revisione + ADDENDUM it.59
