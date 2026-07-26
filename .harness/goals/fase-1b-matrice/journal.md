# Journal — fase-1b-matrice

## Iterazione 1 (2026-07-26)

- Branch `feat/fase-1b-matrice` creato da `main` (post plan-check approvato).
- Scritto e committato il piano di implementazione per la Fase 1 (adapter Transformers.js):
  `docs/superpowers/plans/2026-07-26-fase-1b-matrice-adapter-transformersjs.md` (commit
  `848cee3`). 4 task: (1) `TransformersJsAdapter` + `StackId` + dep `@huggingface/transformers`,
  (2) `BenchServer` multi-stack routing (registry `Record<StackId, factory>`, protocol esteso
  con `stack`), (3) UI stack selector con filtro modelli, (4) driver e2e `STACK` env + run reale
  4090 + commit risultato in `results/`.
- Ricerca API Transformers.js fatta via skill dedicato (`huggingface-skills:transformers-js`,
  non a memoria) prima di scrivere codice nel piano — pattern DI identico a `WebLLMAdapter`
  (engine/isCached/now iniettabili) per tenere i test unitari lontani da rete/GPU reali.
- **Nota corretta durante la scrittura**: la spec parla di "Transformers.js v3", ma npm registry
  (controllato 2026-07-26) mostra l'ultima pubblicata è `4.2.0` (v3 fermo a `3.8.1`). Stessa API
  (`pipeline`/`TextStreamer`), solo la label di versione è invecchiata — non uno scope change,
  loggato nel piano (Global Constraints) e qui.
- Non ancora eseguito: in attesa della scelta di esecuzione (subagent-driven vs inline) da
  Cristiano.

## Iterazione 2 (2026-07-26) — Fase 1 eseguita (subagent-driven)

Piano eseguito task-by-task con review avversaria per task + final review whole-branch (opus).

- **Task 1** (`fe4e657` + fix `33cad11`): `TransformersJsAdapter` + `StackId`, dep
  `@huggingface/transformers@4.2.0`. Review ha trovato un **bug reale nel piano**: cablava
  `callback_function` invece di `token_callback_function` sul `TextStreamer`. Confermato alla
  fonte (`streamers.js:161` fa scattare `callback_function` solo con testo non vuoto = confine di
  parola; `:96` fa scattare `token_callback_function` una volta per token). Avrebbe contato parole
  invece di token, corrompendo le metriche per cui l'adapter esiste. Corretto in codice e nel piano.
- **Task 2** (`2b81ff4`): routing per-stack — `BenchServer` con `adapters: Record<StackId, factory>`,
  `stack` nel protocollo + guard runtime. `handle()` preservato byte-identico a parte il dispatch.
  Review clean al primo giro.
- **Task 3** (`4b0470a`): selettore stack in UI con filtro modelli per `data-stack`. Verificato via
  Playwright headless sui 3 scenari. Review clean al primo giro.
- **Task 4** (`7b14369` + fix `216c31c`): driver `STACK` env + **run reale sulla 4090**. Il run ha
  dato la verifica dal vivo del fix del Task 1 (`completionTokens` 256/256/256 = `maxTokens`; col bug
  avremmo visto ~150-190) e ha **rivelato un secondo difetto**: `TextStreamer` installa `stdout_write`
  come `callback_function` di default (`streamers.js:65`), quindi 675 `console.log` per-parola
  *dentro la finestra cronometrata del decode* — contaminazione di misura in un bench comparativo,
  visto che WebLLM non stampa nulla. Corretto con no-op; 675 → 0.

**Final review whole-branch** (opus): 0 Critical, 5 Important. Ha verificato la simmetria tra i due
adapter su ogni asse (inizio del cronometro, definizione di "chunk", greedy, `repetition_penalty`
1.1 su entrambi, max tokens, prompt condiviso, cold/warm, formula del rate) — equivalenti; unica
asimmetria era `quant`. Ha ricalcolato a mano il datapoint committato: riproduce esattamente.
- #1 `quant` free-text non falsificabile per transformersjs (`DTYPE` hardcoded a `q4` ma `quant`
  arriva dal messaggio) → **fixed** `ae3d4d5`: `src/stacks.ts` con `STACK_FIXED_QUANT` + guard in
  `benchServer`, 2 test nuovi. Era raggiungibile solo con i flag documentati.
- #2 `HANDOFF.md` diceva il falso ("non ancora eseguito") → fixed.
- #3 `README:96` "non ancora in scope" contraddetto dal branch → **fixed** `98770c6`.
- #4 conformance test del done-when non soddisfatto → **docket #2** (decisione PI, gap simmetrico
  anche per WebLLM).
- #5 nessun run WebLLM in condizioni equivalenti → **fixed** `bc15375` (run reale, stessa sessione).

**Primo risultato cross-stack reale** (stessa GPU, stesso base model, stesso browser Chrome 151,
stessa sessione, schema v2, 3 repliche):

| stack | quant | decode tok/s | load | TTFT |
|---|---|---|---|---|
| webllm | q4f32_1 | **113.0 ±2.5** | 62.5 s cold | 408 ms |
| transformersjs | q4 | **55.5 ±2.1** | 157.6 s cold / 3.2 s warm | 444 ms |

→ **WebLLM ~2.04× più veloce di Transformers.js** sulla stessa 4090. Probe: `subgroups` presente,
`shader-f16` ancora assente (coerente coi landmine noti).

**⚠️ RITRATTAZIONE (stessa data, dopo i dati di Cristiano)**: il "2.04×" **non è affidabile**.
Cristiano ha fornito un run a 8 celle sul suo Chrome 150 branded
(`results/4090-linux-2026-07-26T19-54-55-278Z.json`) dove, con un solo probe e nella stessa
sessione, Transformers.js va da 31.7 a 49.3 tok/s (+55%) e WebLLM da 101.3 a 115.6 (+14%) al
progredire dei run. Esiste quindi un **effetto warm-up tra celle** che le repliche multiple non
catturano (sono consecutive dentro la stessa cella; la varianza intra-cella resta ±0.5–5 e il flag
`high-variance` non scatta mai). Ogni rapporto cross-stack dipende dall'ordine in cui le celle sono
state eseguite. Vedi **docket #5b**.
Ritrattata anche l'ipotesi che avevo formulato su un confronto parziale (4 celle su 8), secondo cui
il divario dipendeva dalle feature WebGPU `chromium-experimental-subgroup-matrix`/
`subgroup-size-control` assenti su Chrome 150: non può spiegare una variazione *interna* a un file
con un unico probe. A regime i due browser concordano (transformersjs ~48 vs 55.5, webllm ~111 vs
113.0).

Gate finale: `npm test` 47/47, `tsc --noEmit` pulito, `npm run build` ok.
