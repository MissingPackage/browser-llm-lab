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

## Iterazione 3 (2026-07-26) — conformance harness, chiusura Fase 1

Ruling PI su docket #2: conformance test per **tutti** gli stack **nel browser**, on-demand via
Playwright; `npm test` resta unit veloce e offline. Done-when di GOAL.md/PHASES.md emendato di
conseguenza (la formulazione "verifiable via `npm test`" non era realizzabile: WebLLM richiede
WebGPU e non gira in Node).

Costruito: `src/conformance/contract.ts` (8 check condivisi), `conformance.html` + `src/conformance/page.ts`
(entrypoint browser, escluso dal build di produzione), `scripts/conformance.mjs` (driver Playwright
col rifiuto del software-rasterizer), script `npm run test:conformance`. Il `TransformersJsAdapter`
ha ora `dtype`/`device` iniettabili (default invariati `q4`/`webgpu`, vincolati da `STACK_FIXED_QUANT`)
così il contratto esercita la **vera** `defaultEngineFactory` invece di un engine fake — è proprio il
punto cieco che stiamo chiudendo.

Modelli: transformersjs usa il fixture `Xenova/tiny-random-Phi3ForCausalLM` (~4 MB, ha un chat
template quindi esercita il percorso messages-array); webllm usa `Qwen2.5-0.5B-Instruct-q4f32_1-MLC`
— per MLC non esiste un fixture "tiny", va detto.

**Risultato reale**: 8/8 transformersjs + 8/8 webllm, exit 0, su 4090.

**Mutation test** (eseguito dal controller, non delegato): reintrodotto a mano il bug
`callback_function` al posto di `token_callback_function` → `[FAIL] one timestamp per generated
token — expected 16, got 10` (conteggio a parole, 62%, esattamente la banda prevista), exit 1,
webllm resta 8/8. Mutazione revertata, tree pulito, riverificato 8/8+8/8 exit 0. **Il contratto sa
fallire** e isola l'adapter giusto.
Verificato dal vivo anche il guard sul software rasterizer: primo lancio senza `HEADED=1` → finito su
SwiftShader → driver ha rifiutato con exit 2 invece di riportare un pass non attendibile.

---

## 2026-07-26 — sessione 3: ruling del PI applicati, protocollo di misura verificato

Cristiano ha risposto nel docket a 4 voci: #3 (push su `origin` approvato), #4 (OK, nessuna
azione), #5 (strict), #5b (cella di riscaldamento + media su 3-5 passate).

**#5 strict** — `"strict": true` in `tsconfig.json`. Il codebase era già conforme: `npx tsc
--noEmit --strict` dava zero errori *prima* di toccare qualsiasi cosa. Nessun `| null` da
sistemare, nessun `!` da difendere. Costo reale del ruling: una riga.

**#5b warm-up** — `needsWarmup()` in `src/benchServer.ts`: se `cacheState !== "warm"` si esegue
una generate a carico identico (PROMPT_512, 256 token) e la si scarta prima delle 3 repliche
misurate. `"unknown"` trattato come freddo.

Un test preesistente è caduto: "flags high-variance when decode rate spreads across replicates"
usava proprio il primo run lento come sorgente di spread, e il warm-up ora se lo mangia. Isolato
con `cacheState: "warm"`. **Che sia caduto è la prova che lo scarto funziona** — non è stato un
fastidio da silenziare.

**Verifica su GPU reale.** Serviva un driver multi-cella: `scripts/e2e-bench.mjs` fa una cella per
invocazione e l'effetto vive *tra* celle. Scritto `scripts/seq-bench.mjs` (sequenza via `SEQ`,
contatori `nvidia-smi` a ogni confine di cella). Due bug pagati nel driver: `playwright` non
risolve da `/tmp` (spostato in `scripts/`), e `#status` resta `"done"` dalla cella precedente →
la wait ritornava subito e il click successivo arrivava a bench in corso (ora azzerato prima del
click).

4 run sulla 4090, dati e log in `results/methodology/`:

| run | sequenza | decode tok/s | deriva |
|-----|----------|--------------|--------|
| alternato | tjs, webllm, tjs, webllm | tjs 56.5 → 48.6 / webllm 113.4 → 110.5 | tjs −13.9%, webllm −2.6% |
| mono ×3 | tjs ×3 | 49.9 → 45.6 → 45.1 | −9.5% |
| mono ×4 strumentato | tjs ×4 | 49.2 → 45.0 → 45.4 → 46.0 | −6.6% |
| warm-up **sempre** (esperimento) | tjs ×4 | 48.0 → 48.2 → 47.2 → 46.6 | −2.9% |

**Il +55% non è più riproducibile**: il ruling ha risolto il problema che mirava a risolvere.
Ma ha scoperto una deriva di segno opposto: la prima cella di *ogni sessione browser* fa 49-50,
le successive 45-46, su 3 sessioni indipendenti.

**Non è hardware, e l'ho misurato invece di ipotizzarlo**: temperatura 48→56 °C (throttling di una
4090 laptop è ~87 °C), memoria GPU piatta (2177→2197 MiB), clock stabile a 2265 MHz,
`clocks_throttle_reasons` = 0x0 durante le celle. L'ipotesi power/clock-state del docket è
confermata **solo per il primo run assoluto** (1455 MHz a freddo → 2265 dopo la prima cella); il
residuo è stato del processo, presumibilmente compilazione shader / cache di kernel di ONNX
Runtime Web.

Causa del residuo: il warm-up si applica solo alle celle cold, quindi ogni cella warm successiva
ripaga il primo-run lento. Controprova eseguita (warm-up forzato su tutte le celle): deriva a
−2.9% e TTFT stabilizzato (275/270/279/290 contro 347/403/459/429). **Esperimento revertato**: la
lettera del ruling dice "se il modello è cold", estenderlo è decisione del PI → docket #5b.

Rapporto cross-stack dalle misure pulite: **webllm/transformersjs ≈ 2.0-2.3×** (2.01 in posizione
1, 2.27 in posizione 2). Coerente col ~2.3× stimato, ancora ordine-dipendente.

Gate: `tsc --noEmit` pulito, `npm test` 50/50, `npm run build` ok. Commit `579c9e4`, pushato su
`origin/main` (`825e2b5..579c9e4`, 22 commit — `origin` era fermo a prima di tutta la Fase 1).

---

## 2026-07-27 — Fase 2: adapter wllama

Ruling applicati: warm-up esteso a tutte le celle ma **selezionabile** (`WarmupPolicy`), perché —
osservazione del PI — senza warm-up si misura ciò che un utente vero sperimenta al primo colpo,
che su una pagina pubblica è l'informazione più utile. Default `always` (benchmark/steady-state),
`never` per l'uso divulgativo, `cold-only` conservata perché è ciò che ha prodotto i dati in
`results/methodology/`. Nessuna UI: il motore è pronto, la pagina pubblica dovrà solo esporre il
controllo. Le due modalità **misurano cose diverse e non vanno confrontate fra loro** — vale anche
per come la UI le presenterà.

**Adapter wllama** (`@wllama/wllama` 3.5.1 — il nome `wllama` in GOAL.md non esiste su npm).
`modelId` = `owner/repo/file.gguf` col GGUF nominato per esteso: `loadModelFromHF({quant})` fa
fallback silenzioso Q4_K_M → Q8_0 → non quantizzato, e una cella etichettata Q4_K_M avrebbe potuto
contenere una misura Q8_0. Con il file esplicito il quant è nel modelId e non c'è fallback.

**Tre difetti trovati, tre strumenti diversi.** Vale la pena notare quale ha preso cosa:

1. *Il contratto di conformance* ha preso i chunk senza contenuto contati come token (17 timestamp
   per 16 token). Regola estratta in `chunkIsToken()` e ora coperta anche dai test unitari.
2. *Il ragionamento sul confronto cross-stack* ha preso il prompt cache: le repliche 2 e 3
   avrebbero saltato il prefill dei 512 token, con TTFT artificialmente basso, mentre gli altri due
   stack il prefill lo rifanno. Avremmo misurato il caching di uno stack invece della sua velocità.
   `cache_prompt: false`, verificato attivo dal log llama.cpp.
3. *Il run reale* ha preso `document is not defined`: l'adapter passava il conformance e falliva
   ogni cella di bench. Il conformance gira nel **main thread**, il bench nel **worker**, e
   `absoluteUrl()` di wllama usa `document.baseURI`. Il conformance non poteva vederlo per
   costruzione — un buco di copertura del harness, non un suo fallimento.

**Il difetto che resta (docket #8)**: il chunk di chiusura di ogni `generate()` viene consegnato
all'inizio della successiva, quindi ogni risposta perde la coda. Diagnosticato per conto nostro
strumentando i chunk, poi trovata la stessa causa radice in ngxson/wllama#263 (+ PR #264, entrambe
aperte). Due ipotesi falsificate lungo la strada e registrate nel docket per non rifarle:
non-determinismo multi-thread (identico con `n_threads: 1`) e penalità di ripetizione (identico con
penalità azzerate e seed fisso). Confermato anche sul percorso di produzione:
`completionTokens: 255` su `maxTokens: 256`. Ruling: documentare, nessun workaround nostro,
routine di sorveglianza ogni 3 giorni (`trig_018i6ZnQpZHF1tg6egjTsyST`).

Ho anche corretto una mia caratterizzazione imprecisa: avevo scritto "il testo è identico, è solo
il conteggio". Non è così — è la coda della risposta che si perde, e il decode rate esce calcolato
su n−1 token, sistematicamente.

**Prima misura a tre stack** (4090, Qwen2.5-0.5B, 3 repliche, warm-up scartato): webllm ~110 tok/s,
transformersjs ~46–48, wllama **25.97** (stdev 0.05, TTFT 8406 ms). wllama è CPU via WASM: il
divario dice quanto costa non avere accelerazione, non che la libreria sia lenta.

Gate: `npm test` 75/75, `tsc --noEmit` pulito, `npm run build` ok, conformance **8/8 + 8/8 + 7/8**,
run reale in `results/4090-linux-2026-07-26T22-51-39-379Z.json`. Mergiata in `main` (`277609e`) e
pushata.

## Iterazione (2026-07-27) — Fase 3 eseguita (inline, /loop /product-loop)

Branch `feat/fase-3-quality-schema-v3` creato da `main`. Prossimo decidibile per HANDOFF.md §1
era Fase 3 (PHASES.md riga 3), nessun gate a bloccarla.

- `src/quality.ts`: `computePerplexity` (`exp(-mean(logprobs))`, throw su array vuoto —
  stesso stile di `aggregateReplicates`), `evaluateExactMatch` (12 prompt, prompt senza risposta
  contano come sbagliati, non lanciano), `selectQualityMethod` (decide perplexity vs
  exact-match da `capabilities().logprobs`).
- `src/qualityPrompts.ts`: 12 prompt deterministici, 4 categorie (arithmetic/factual/format/json),
  matcher via regex o JSON.parse + shape-check, tolleranti a whitespace/punto finale.
- `src/schema.ts`: `SCHEMA_VERSION = 3`. `BenchCell.qualityScore?: QualityScore` — **opzionale**,
  perché il modulo non è ancora collegato a `benchServer.ts` (registrato in docket #10: wiring
  reale = 12 generate extra per cella o percorso logprobs, un costo/comportamento della pipeline
  che nessun docket ha ancora approvato esplicitamente — L1 docket+continue, non un blocco).
- **docket #7 eseguito nella stessa iterazione** (HANDOFF diceva esplicitamente di farlo qui,
  perché è dove lo schema si tocca comunque): `BenchCell.protocol { warmupPolicy, warmupApplied,
  replicateCount }` sostituisce la stringa `"protocol: warm-up run discarded…"` che abusava
  `anomalies`. `benchServer.ts` aggiornato per popolarlo. `WarmupPolicy` spostato da
  `benchServer.ts` a `schema.ts` (è forma dei dati, non comportamento) — elimina anche il
  cross-import inverso che `protocol.ts` aveva verso `benchServer.ts` per quel tipo.
  `GOAL.md` CONSTRAINTS emendato per includere `protocol` accanto a `qualityScore` nella clausola
  "no ad hoc field additions".
- Test aggiornati: `schema.test.ts`, `render.test.ts` (fixture con `protocol`),
  `benchServer.test.ts` (6 asserzioni riscritte da "anomalies contiene la stringa protocol" a
  "cell.protocol uguale a {...}" — più preciso, verifica tutti e tre i campi invece di un
  sottostringa). Nuovo `tests/quality.test.ts`, 12 test.

Gate: `npm test` 87/87 (75 + 12 nuovi), `tsc --noEmit` pulito, `npm run build` ok.

**Scope non incluso, registrato non deciso da solo**: wiring di `quality.ts` dentro la pipeline
di bench reale (docket #10) — `PHASES.md` riga 3 non lo richiede, e cambiare il costo/tempo di
ogni run reale sulle GPU fisiche è una decisione che eccede l'autorità di questa iterazione.
