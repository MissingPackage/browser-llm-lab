# HANDOFF — browser-llm-lab   (updated 2026-07-27, session 5)

## 1. Next decidable — nessun goal attivo, sweep manuale in corso da parte di Cristiano

**Goal `fase-1b-matrice` CHIUSO** (2026-07-27, ruling PI: "possiamo chiudere questo goal").
Storico in `.harness/goals/fase-1b-matrice/` (GOAL.md, PHASES.md, docket.md, journal.md). Tutte e
4 le fasi complete, mergiate e pushate su `main`.

**Prossimo passo è manuale, non autonomo**: Cristiano testa lui stesso M4 Pro e laptop (domani);
per l'S22 Ultra va ancora capito l'approccio (vedi risposta su come raggiungere il dev server da
telefono, registrata sotto in §3 e nella chat). Se lo sweep produce bug/fix da fare in codice,
quello sarà un **nuovo goal**, non una riapertura di `fase-1b-matrice`.

**Docket ereditati, vivi ma non bloccanti**:
- **#10**: `src/quality.ts` pronto e testato ma non collegato a `benchServer.ts` — nessun run
  reale porta un `qualityScore`. Decisione PI residua: se/quando collegarlo (costo: fino a 12
  `generate()` extra per cella, o un passaggio logprobs).
- **#8**: conformance wllama 7/8, routine cloud ogni 3 giorni che apre una PR da sola quando il
  fix upstream arriva. Nessuna azione ora.

**Nessun goal attivo per `/loop`.** Non inventare un nuovo goal per il device sweep senza che
Cristiano lo chieda esplicitamente.

## 2. State delta (session 5, 2026-07-27) — Fase 4

- **README**: nuova sezione "Fasce modello — il gap strutturale della fascia Large"
  (Qwen2.5-7B-Instruct / Llama-3.1-8B-Instruct: solo WebLLM può servirle — nessun repo ONNX
  web-runnable per Transformers.js, GGUF Q4_K_M sopra il tetto WASM 4GB per wllama). Nuova
  sezione "Fase 3 — modulo qualità + schema v3" che dichiara onestamente che `qualityScore` non
  è ancora popolato da run reali (docket #10).
- **Due righe README corrette perché rese stale dalla Fase 3** (trovate rileggendo il file, non
  nello scope dichiarato ma conseguenza diretta di quel lavoro): la label schema in "Quick
  start" diceva ancora v2; la nota di warm-up diceva che finiva in `anomalies`, non più vero da
  quando esiste `BenchCell.protocol`.
- Nessun codice toccato in questa iterazione — solo README + goal spine. Test 87/87 invariati,
  `tsc --noEmit` pulito, `npm run build` ok.
- **`loop-verifier` ha trovato un gap reale prima del merge**: i run reali esistenti per
  `transformersjs`/`wllama` in `results/` erano schema v2 (catturati prima del bump di Fase 3);
  `GOAL.md` DONE WHEN chiede esplicitamente "each a valid schema-v3 JSON". Corretto eseguendo
  l'e2e driver headed sulla 4090 reale (authority già concessa, nessun ruling nuovo):
  `results/4090-linux-2026-07-26T23-56-34-978Z.json` (transformersjs, schemaVersion 3, 57.8 tok/s)
  e `results/4090-linux-2026-07-26T23-57-10-621Z.json` (wllama, schemaVersion 3, 28.5 tok/s).
- **`GOAL.md` DONE WHEN risulta ora soddisfatto riga per riga, verificato non solo dichiarato**
  (STATUS NOTE aggiornata in `GOAL.md` con la correzione). Loop fermato **by design** — vedi §1.

## Session 4 (2026-07-27) — Fase 3, per riferimento

- **Fase 3 — modulo qualità + schema v3** (`src/quality.ts`, `src/qualityPrompts.ts`):
  `computePerplexity` (perplexity via `exp(-mean(logprobs))`), `evaluateExactMatch` (12 prompt
  deterministici, 4 categorie: arithmetic/factual/format/json), `selectQualityMethod` (sceglie il
  percorso da `capabilities().logprobs`). `SCHEMA_VERSION = 3`. `BenchCell.qualityScore`
  **opzionale** — modulo pronto e testato ma non collegato a `benchServer.ts` (nessuna cella reale
  calcola oggi un punteggio; vedi docket #10).
- **docket #7 implementato**: `BenchCell.protocol { warmupPolicy, warmupApplied,
  replicateCount }` sostituisce la nota testuale `"protocol: warm-up run discarded…"` che abusava
  `anomalies`. `WarmupPolicy` spostato da `benchServer.ts` a `schema.ts` (forma dei dati) —
  elimina anche il cross-import che `protocol.ts` aveva verso `benchServer.ts` per quel tipo.
  Constraint `GOAL.md` emendato per includerlo esplicitamente accanto a `qualityScore`.
- Test 87/87 (75 + 12 nuovi su `quality.ts`/`qualityPrompts.ts`). `tsc --noEmit` pulito,
  `npm run build` ok. `origin/main` allineato dopo il merge di `feat/fase-3-quality-schema-v3`.

### Session 3 (2026-07-26 → 27), per riferimento

- **Ruling #5 (strict)**: `"strict": true` in `tsconfig.json`; il codebase era già conforme.
- **Ruling #5b (warm-up)**: `WarmupPolicy` — `always` (default, steady-state), `never` (prima
  esperienza reale), `cold-only`. Selezionabile per run via `MainToWorker.bench.warmup`.
  **Le prime due misurano cose diverse e non vanno confrontate.**
- **4 run di metodologia sulla 4090** (`results/methodology/`): il +55% di docket #5b non è più
  riproducibile; resta ~3% di dipendenza dall'ordine col warm-up esteso. **Non è hardware** —
  temperatura, memoria GPU, clock e throttle reasons misurati e tutti negativi.
- **Fase 2 — adapter wllama** (`src/adapters/wllama.ts`, `@wllama/wllama` 3.5.1): terzo stack,
  CPU via WASM. Tre difetti trovati e corretti (chunk senza contenuto contati come token; prompt
  cache che falsava il TTFT delle repliche; `document.baseURI` assente nel Web Worker).
- **Prima misura a tre stack**: webllm ~110 tok/s, transformersjs ~46–48, wllama **25.97**.
- `scripts/seq-bench.mjs`: driver multi-cella (`SEQ`, `BENCH_URL`) con contatori `nvidia-smi` a
  ogni confine di cella — `e2e-bench.mjs` fa una cella per invocazione e l'effetto vive *tra* celle.

## 3. Open threads

- **Sweep manuale sui 3 device (in corso da Cristiano)**: M4 Pro e laptop domani via test diretto.
  **S22 Ultra — approccio da chiarire**: `npm run dev` gira solo su una macchina con Node (il
  telefono fa da client Chrome, non da host). Serve `npm run dev -- --host` (bind su tutte le
  interfacce) sulla macchina con GPU dedicata + navigare da Chrome sull'S22 verso
  `http://<IP-LAN-macchina>:5173` sulla stessa rete Wi-Fi. Nodo aperto: l'origine non è HTTPS, quindi
  non è un secure context per default — serve il flag Chrome
  `chrome://flags/#unsafely-treat-insecure-origin-as-secure` sul telefono con quell'URL aggiunto,
  altrimenti niente `crossOriginIsolated`/WebGPU. `wllama` (WASM) funziona comunque; `webllm` e
  `transformersjs` dipendono dal supporto WebGPU reale di Chrome su Adreno, da verificare dal probe
  box prima di fidarsi di un numero.
- **Nessuna fase residua** — tutte e 4 fatte. Vedi §1 per cosa resta (docket-gated/fuori scope).
- **docket #10 — decisione registrata, non un ruling bloccante**: `quality.ts` non è collegato a
  `benchServer.ts`. Nessun run reale porta un `qualityScore`. Va deciso se/quando collegarlo.
- **docket #8 — deroga attiva**: conformance wllama **7/8**. Difetto di wllama, non nostro
  (ogni risposta perde la coda, che arriva al giro dopo). Upstream ngxson/wllama#263 + PR #264,
  entrambe aperte; 3.5.1 è precedente alla PR. **Routine di sorveglianza attiva**
  `trig_018i6ZnQpZHF1tg6egjTsyST`, ogni 3 giorni: apre una PR quando il fix è rilasciato.
  Quando arriva → aggiornare la dipendenza, rieseguire il conformance **in locale** (serve GPU) e
  togliere la deroga da README, PHASES e docket.
- Rapporto cross-stack ancora **non dichiarabile** in un rapporto: ~3% di dipendenza dall'ordine.
- I run in `results/*.json` precedenti al 2026-07-26 sono ordine-dipendenti e nulla nel file lo dice.
- Branch `feat/fase-1a`, `fix/fase-1b-fixin1b`, `feat/fase-2-wllama`, `feat/fase-3-quality-schema-v3`
  merged, non cancellati.

## 4. Landmines

- **`shader-f16` è per-browser, non per-GPU**: assente su chromium-playwright (q4f16_1 crasha →
  usare q4f32_1), presente su Firefox.
- **Firefox silent-fallback a CPU**: FF 152 girava su llvmpipe dichiarando `webgpu:true`. Il probe
  ora rileva il software adapter (128 MiB + vendor vuoto).
- **Headless = SwiftShader**: mai accettare quei risultati come dati GPU. I driver e2e/conformance
  rifiutano da soli; `seq-bench.mjs` pure. Per la GPU reale serve
  `HEADED=1 CHANNEL=chrome CHROME_ARGS="--ignore-gpu-blocklist --disable-gpu-sandbox"`.
- **Chrome GPU sandbox vs Vulkan ICD (Fedora/NVIDIA)**: col sandbox attivo il GPU process non
  legge gli ICD ("Found no drivers") → niente adapter WebGPU. Ritestare a ogni update di Chrome.
- **Chrome flags & NVIDIA/Wayland**: `enable-vulkan` nel profilo corrompe il compositing,
  `force-enable-webgpu-interop` crasha all'avvio. NON rimetterli nel profilo quotidiano.
- **`#status` resta `"done"` tra una cella e l'altra**: qualunque driver multi-cella deve azzerarlo
  prima del click, o la wait ritorna subito e clicca a bench in corso (già pagato).
- **Il conformance gira nel main thread, il bench nel Web Worker.** Un adapter può passare 8/8 e
  fallire ogni cella di bench: è successo con wllama (`document.baseURI`, vedi
  `ensureWorkerDocumentShim`). Un adapter nuovo non è verificato finché non ha girato **anche**
  in un run reale.
- **`@wllama/wllama` 3.5.1**: niente `index.js` alla root né campo `exports` malgrado
  `main: "index.js"` → importare da `@wllama/wllama/esm/index.js`, o TypeScript ripiega sul
  sorgente `.ts` e `erasableSyntaxOnly` fallisce.
- **wllama e il quant**: usare sempre `file:` esplicito, mai `quant:` — quest'ultimo fa fallback
  silenzioso Q4_K_M → Q8_0 → non quantizzato e produce celle mislabellate.
- Playwright `waitForFunction(fn, arg, options)`: le options sono il TERZO argomento.
- `chromium.launch()` = profilo effimero → cold/warm richiede `launchPersistentContext`.
- tsconfig: `erasableSyntaxOnly` (niente parameter properties) e ora `strict`.
- `.superpowers/` è gitignorato: il ledger SDD non sopravvive a `git clean -fdx`.
- Il dev server su :5173 può essere già acceso da un'altra sessione — `npm run dev` ripiega su
  :5174 e i driver puntano a :5173 di default.

## 5. Docket (user decisions pending)

Vedi `.harness/goals/fase-1b-matrice/docket.md` per il testo integrale.

- **#8 — deciso, deroga attiva**: conformance wllama 7/8, documentato, sorveglianza schedulata.
  Nessuna azione finché la routine non segnala il fix (vedi §3).
- **#10 — registrato, non richiede ruling per procedere**: `qualityScore` non collegato alla
  pipeline di bench. Decisione PI residua (vedi §1).
- Chiusi in sessione 4: **#7** (`BenchCell.protocol` implementato, constraint emendato).
- Chiusi in sessione 3: **#3** (push approvato), **#4** (registrazione), **#5** (strict),
  **#5b** (warm-up selezionabile), **#9** (nome pacchetto `@wllama/wllama`).
- Nessun item aperto in attesa di ruling PI.
- Storico dei docket delle fasi precedenti: nel journal del goal.
