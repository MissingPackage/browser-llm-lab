GOAL: Il decode del Qwen3.5-4B in browser regge 30 token/s a CONTESTO
REALISTICO (>= 6000 posizioni), dove oggi ne fa 9,95 — o l'irraggiungibilità è
dimostrata coi numeri e col conto che la mostra.

<!-- CONTRATTO v1 (chartered 2026-08-13, PI in chat: "Ok su tutto ... partiamo").
     Nasce dalla revisione architetturale indipendente
     (docs/deep-dive/headroom-2026-08-12.md) e dalla misura it.59 che ne corregge
     il roofline: il termine KV mancava, e a contesto vero domina.
     Goal precedente: engine-fase-d, CHIUSO (blocco A raggiunto, spec-dec MTP
     escluso coi numeri). -->

DONE WHEN (all measurable):

- decode >= 30,0 tok/s sul 4B con prompt >= 6000 token, misurato da
  `scripts/q35-bench-run.mjs` con `hostState.declared = "quiescent"` e
  `decodeContext.startPositions >= 6000`, JSON committato in `results/engine/`;
  OPPURE un JSON `kind: "q35-decode-kernel-checkpoint"` con
  `decision: "excluded-by-numbers"` che porta banda efficace per segmento, tetto
  residuo, e il conto che mostra perché la soglia non è raggiungibile.

- FASE 0 SUPERATA PRIMA DI OGNI RISCRITTURA: micro-bench isolato delle varianti
  candidate, zero run di modello, JSON committato con G pesi/s e GB/s per
  variante. **REGOLA DI STOP**: se nessuna variante supera la forma attuale di
  >= 1,5x, la fase si chiude e il piano si riscrive invece di procedere.

- LEVA 1 (attenzione) — i tre difetti misurati SEPARATAMENTE in fase 0:
  (a) RIDONDANZA GQA: 4 workgroup leggono le stesse righe KV (`kvHead = h/4`).
      Variante: un workgroup tratta le 4 teste del gruppo, KV letta una volta.
      Quanto assorba già la L2 è ignoto: è la misura che decide.
  (b) LETTURE SCALARI: ciclo su 256 dimensioni una alla volta → variante vec4.
  (c) TETTO DI MEMORIA E PARALLELISMO: `scores: array<f32, ctxMax>` in memoria
      di gruppo ⇒ il motore chiede 4·ctxMax+256 B (25,9 KB a ctx 6400) contro i
      16 KB garantiti da WebGPU: sopra ~4000 posizioni non parte su un
      dispositivo conforme al minimo. Variante: softmax in streaming, che toglie
      il tetto E abilita lo split del contesto su più workgroup.
  Misura: ms/token a ctx 388 e a ctx >= 6000 PRIMA e DOPO nello stesso JSON, con
  la pendenza µs/posizione ricalcolata. Riferimento da battere: **10,4
  µs/posizione, 6,3 GB/s sulla scansione KV** (it.59).

- **PORTABILITÀ, indipendente dalla velocità**: `maxComputeWorkgroupStorageSize`
  richiesto dal motore torna sotto **16384 B a QUALUNQUE ctxMax** — il tetto di
  contesto non dipende più dalla memoria di gruppo.

- LEVA 2 (GEMV quantizzati) misurata: G pesi/s del corpo PRIMA e DOPO nello
  stesso JSON. Riferimento: **133 G pesi/s**; riferimento esterno sullo stesso
  GGUF/host/driver: llama.cpp Vulkan **118,68 tok/s ≈ 500 G pesi/s**.

- CORRETTEZZA, gate secchi a ogni merge: `node .harness/tools/engine-ktest.mjs`
  tutti PASS; argmax del golden 4B IDENTICO al pin; `npm test` verde;
  `npx tsc --noEmit` pulito.

- NON-REGRESSIONE GLM: b12 optimistic entro ±5% dei riferimenti
  13.172 / 31,26 / 14,74, misurata fresca su host dichiarato.

- Ogni numero pubblicato porta il proprio contesto: nessun tok/s senza
  `decodeContext` (regola nata da it.59).

EVIDENCE OF DONE:
- `node scripts/q35-bench-run.mjs --prompt-idx 0 --n-decode 64 --vram-gib 8
   --declared quiescent` (prompt 0 = 6333 token) → JSON con `decode.tokS` e
   `decodeContext`
- `node .harness/tools/engine-ktest.mjs` con `npx vite` acceso → tutti PASS
- `npx vitest run` verde · `npx tsc --noEmit` pulito
- micro-bench di fase 0: JSON per variante in `results/engine/`
- non-reg GLM: runner GLM con host dichiarato (leggere i flag PRIMA di
  spenderci GPU — ruling C8)

AUTHORITY GRANTED:
- may do autonomously: scrivere e riscrivere kernel WGSL e il piano di dispatch;
  creare runner e fixture nuovi; eseguire bench GPU e ktest; committare e
  pushare su origin/main a fine task; mergiare a goal chiuso e verificato;
  aprire e chiudere item di docket che sono lavoro mio; escludere una leva coi
  numeri e dichiararlo; usare i workflow dell'harness (sdd-conductor,
  second-opinion, research-campaign, pattern-coverage) e i subagent.
- must docket (never do): cambiare la funzione obiettivo o la soglia dei 30
  tok/s; cambiare i gate di correttezza o le bande di non-regressione;
  cancellare o riscrivere codice committato da più di 30 giorni fuori dal brief;
  pubblicare numeri fuori dal tag di release; spendere denaro; toccare i goal in
  standby (fase-1b, fase-2).

CONSTRAINTS:
- WebGPU reale, non CUDA: `shader-f16` NON è disponibile su questo stack
  Chrome/Linux — ogni piano che lo assume è morto qui (vivo su M4/S22).
  `subgroupAdd` risulta disponibile secondo la revisione: **da ri-confermare con
  una sonda in fase 0 prima di progettarci sopra**.
- Nessuna attribuzione AI in commit, PR o merge.
- Le metriche misurate non peggiorano mai: banda ±5% su tok/s e TTFT, gate di
  correttezza secchi.
- I bench costosi si eseguono su codice finale, non su stati intermedi.
- Ogni misura dichiara host e contesto.
- Il codice dello spec-dec MTP resta in albero e gated: non si rimuove, non si
  attiva finché la leva 3 (multi-riga) non esiste.

FUORI SCOPE (registrati, non aperti):
- Il TEMPO AL PRIMO TOKEN (22,7 s → 4 s) e il GEMM multi-riga che lo abilita:
  goal suo. Lì va anche l'attenzione a chunk del prefill (`attnDecodeWgsl` con
  `batch`), che ha gli stessi tre difetti del kernel di decode e che la stessa
  riscrittura sistemerebbe — stesso kernel, obiettivo diverso.
- La resurrezione dello spec-dec MTP: conseguenza del goal TTFT, non obiettivo.
- GLM: resta residency-bound, nessuna di queste leve lo porta a 30.

WORKING PROTOCOL: skills loop-iteration + done; verifier gate per ciclo; digest
ogni ciclo; stop-by-design quando il resto è docket-gated.

CONTEXT ANCHORS:
- `docs/deep-dive/headroom-2026-08-12.md` — memo della revisione + ADDENDUM
  it.59 (il termine KV che corregge il roofline): in quest'ordine
- `HANDOFF.md` — mappa e landmine
- `.harness/goals/engine-fase-d/{journal.md,docket.md}` — cosa è già escluso coi
  numeri: non ri-proporlo
- `src/engine/kernels/wgsl.ts` (`attnDecodeWgsl`, `gemvQuantWgsl`),
  `deltanet.ts`, `q35gpumodel.ts`
- `results/engine/native-vs-browser-q35-2026-08-10.json` — riferimento esterno

<!-- STATUS: CHIUSO il 2026-08-13 (it.10-11). Obiettivo RAGGIUNTO e superato:
     47,93 tok/s a contesto 6333 contro la soglia di 30, partendo da 9,95
     (4,82x). Checklist del DONE WHEN voce per voce nel journal it.11.
     Aperti e passati al PI: docket item 3 (il conductor installato tronca le
     patch a 16 KB — non e' codice di progetto), item 4 (il done-when (e) chiede
     piu' di quanto la riga 1 possa dare: il totale del motore resta a 30.848 B
     per il kernel FUSO DEL PREFILL, che appartiene al goal TTFT), item 5, 6, 7.
     Il goal successivo naturale e' il TEMPO AL PRIMO TOKEN, che eredita il
     multi-riga, l'attenzione a chunk del prefill e la clausola (e2). -->
