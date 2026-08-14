STATUS: CLOSED (2026-08-14) — ruling del PI in chat: «Ok chiudere il goal.
Abbiamo fatto un ottimo lavoro. I prossimi sono i k quant e il 0.5B (non so
ancora in che ordine). Per il 0.5B migreremo il possibile.»

ESITO: TTFT a caldo 87.618 -> 32.127 ms = 2,727x. La barra meccanica del
contratto (< 21.905 ms) NON e' stata raggiunta: manca 1,467x. Dieci clausole su
dodici soddisfatte; le due che cadono sono la stessa — la barra e la sua gemella
`prefill.tokS > 289`. La causa e' MISURATA e stava fuori dalla portata del goal:
il 37,9% del prefill e' `gemm:deltanet-out`, che cade sul fallback legacy perche'
`ssm_out` e' Q5_K mentre tutte le leve di questo goal sono q4_0-only PER
CONTRATTO (v. i vincoli qui sotto).

Consuntivo voce per voce: docs/engine/ttft-consuntivo-2026-08-14.md
Checkpoint: results/engine/q35-ttft-kernel-checkpoint-4b-2026-08-14.json

GOAL: Il Qwen3.5-4B in browser porta il tempo al primo token a modello caldo,
su un prompt da >= 6000 token, il piu' in basso che questa macchina consenta —
esaurendo le leve identificate dalla fase di sonde e dimostrando coi numeri, per
segmento, il tetto che resta.

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

RULING (PI, 2026-08-13, dopo la riga 1) — TRE MODIFICHE AL CONTRATTO:

  1. **Il bersaglio dei 4 secondi e' tolto.** «Va bene se non arriviamo a 4
     secondi su questa macchina. Scendiamo il piu' possibile.» La funzione
     obiettivo diventa la discesa massima dimostrata, non una soglia; la sua
     forma meccanica e' nei done-when qui sotto (quarto della baseline +
     esaurimento delle leve + contabilita' del tetto residuo).
  2. **Non sfruttare tutti i 49.152 B concessi va bene.** «Il modello 4B
     evidentemente non li sfrutta, ma modelli piu' grossi lo farebbero.» La
     macchina di negoziazione del docket item 6 resta valida e va costruita
     pensando alle famiglie piu' grandi: qui non e' una leva, e P6 lo ha
     misurato (piatto entro ±5% da 16.384 a 49.152).
  3. **`packed_4x8_integer_dot_product` e' autorizzato** (docket item 11):
     entra come leva di questo goal, non del prossimo. E' compute-side, cioe'
     attacca il collo che la riga 1 ha identificato.

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

- TTFT A CALDO sul prompt-idx 0: `prefill.ms + decode.firstMs` **< 21.905 ms**,
  cioe' meno di un quarto della baseline misurata (87.618 ms), da
  `scripts/q35-bench-run.mjs` con host dichiarato e JSON committato.
  **Perche' questa soglia e non i 4.000**: il ruling del 2026-08-13 (v. sotto)
  toglie il bersaglio dei 4 s, che la riga 1 ha proiettato irraggiungibile su
  questa macchina (pavimento modellato 8.665 ms, senza 24 layer su 32, norm,
  RoPE e dispatch). Ma «il piu' in basso possibile» non e' graduabile da un
  verificatore: il quarto della baseline e' la barra meccanica che dice «il goal
  ha prodotto un salto d'ordine di grandezza», sta comodamente sopra il
  pavimento proiettato, e fallirla vorrebbe dire che qualcosa e' andato storto.

- ESAURIMENTO DELLE LEVE, ed e' la clausola che sostituisce il bersaglio: ogni
  leva nominata dalla riga 1 e' **in produzione e misurata PRIMA/DOPO nello
  stesso JSON**, oppure **esclusa coi numeri e dichiarata**. Alla data del
  ruling sono: moltiplicatore multi-riga `splitk` · attenzione del prefill in
  streaming · `packed_4x8_integer_dot_product` (docket item 11, autorizzato dal
  PI). Una leva lasciata cadere in silenzio e' un done-when mancato.

- CONTABILITA' DEL TETTO RESIDUO: un JSON
  `kind: "q35-ttft-kernel-checkpoint"` con banda efficace **per segmento**,
  workgroup in volo per dispatch, e il conto che mostra dove finisce il tempo
  che resta. E' l'artefatto che rende «il piu' in basso possibile» una
  affermazione verificabile invece che una resa.

- PREFILL: `prefill.tokS > 289` sul prompt-idx 0 — la stessa clausola vista dal
  lato del rate (6332 token in < 21,9 s). Riferimento da battere, misurato in
  it.1 sul codice di oggi: **72,30 tok/s**.

- IL PREFILL RESTA PIÙ VELOCE DEL DECODE: `prefill.tokS > decode.tokS` sullo
  stesso JSON. **Gia' vero alla baseline** (72,30 contro 47,79, it.1) — resta
  come guardia: se una riscrittura lo fa tornare falso, il prefill e' ricaduto
  nel path del decode. Il 32,91 che questa clausola citava veniva da codice
  vecchio su un prompt da 388 token: corretto in it.1.

- FASE 0 SUPERATA PRIMA DI OGNI RISCRITTURA: micro-bench isolato delle varianti
  candidate, zero run di modello, JSON committato con G pesi/s e GB/s per
  variante. **REGOLA DI STOP**: se nessuna variante supera la forma attuale di
  >= 1,5x, la fase si chiude e il piano si riscrive invece di procedere.
  (È il meccanismo che ha chiuso il goal precedente: si eredita testuale.)

- LEVA 1 (GEMM multi-riga con riuso vero dei pesi) misurata: G pesi/s PRIMA e
  DOPO nello stesso JSON, a M = 1, 8, 16, 32. Il done-when non è il tempo ma il
  RIUSO: **byte di peso letti per token prefillato scende di >= 5,5x a M >= 16,
  misurato sull'INVENTARIO PER-LAYER INTERO del 4B** (ruling PI 2026-08-13,
  it.14).

  PERCHE' 5,5 E NON 8. Il done-when precedente chiedeva >= 8x ed era
  ARITMETICAMENTE IRRAGGIUNGIBILE, non per un difetto del kernel ma per la
  COPERTURA: la forma nuova e' q4_0-only per costruzione, e nel 4B 24 tensori
  `ssm_out` in Q5_K piu' 4 `ffn_down` in Q4_1 sono l'11,54% dei byte e restano
  sul percorso vecchio, pagandosi M volte anche dopo. Con l'88,46% dei byte a
  16x e l'11,54% a 1x:

      1 / (0,1154 + 0,8846/M)   =>   M=8: 4,43x   M=16: 5,86x   M=32: 6,99x

  Il tetto a M infinito e' 8,67x, e il >= 8x richiederebbe M >= 92. L'errore era
  mio e nasceva dal banco, che misurava una shape sola dove la copertura e' 100%
  e il rapporto e' esattamente 16x. Misurato sull'inventario vero: **5,86x**, e
  5,5 e' la barra col suo margine.

  IL RESIDUO HA UNA CASA: l'11,54% non coperto e' materia del **goal sulle
  famiglie K-quant e Q8_0**, quello gia' identificato come «dare a K-quant e
  Q8_0 la stessa fase 0 che ha avuto la q4_0». Non e' debito silenzioso: e'
  scope di un altro goal, e va nominato nel consuntivo di questo.

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
