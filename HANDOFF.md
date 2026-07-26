# HANDOFF — browser-llm-lab   (updated 2026-07-27, session 4)

## 1. Next decidable

**Goal attivo**: `fase-1b-matrice` (`.harness/goals/fase-1b-matrice/` — GOAL.md, PHASES.md,
docket.md, journal.md). Design: `docs/superpowers/specs/2026-07-26-fase-1b-matrice-design.md`.
**Fasi 1, 2 e 3 complete e mergiate in `main`**, pushate su `origin`.

**Prossimo decidibile: Fase 4 — README + verifica finale whole-branch** (PHASES.md riga 4,
`docs/superpowers/specs/2026-07-26-fase-1b-matrice-design.md` §Ordine di implementazione, punto 4):
sezione "Fase 1b — matrice" nel README con il gap Large documentato + suite completa verde
(baseline + nuovi test) + `tsc`/`build` puliti + run reali `transformersjs` e `wllama` già
presenti in `results/` (fatto in Fase 1/2). **Nota**: `PHASES.md` riga 4 non menziona lo sweep
manuale sui 3 device (M4/S22) — quello resta fuori da queste fasi per costruzione (vedi GOAL.md
"must docket"), quindi Fase 4 è documentazione + verifica, non l'esecuzione dello sweep.

**Prima di dichiarare Fase 4 completa, decidere docket #10**: `quality.ts` è pronto e testato
(Fase 3) ma **non collegato** a `benchServer.ts` — nessun run reale porta oggi un `qualityScore`.
Se il README di Fase 4 deve riportare anche un confronto di qualità, il collegamento va fatto
prima; altrimenti Fase 4 riporta solo le metriche di velocità già esistenti.

## 2. State delta (session 4, 2026-07-27)

- **Fase 3 — modulo qualità + schema v3** (`src/quality.ts`, `src/qualityPrompts.ts`):
  `computePerplexity` (perplexity via `exp(-mean(logprobs))`), `evaluateExactMatch` (12 prompt
  deterministici, 4 categorie: arithmetic/factual/format/json), `selectQualityMethod` (sceglie il
  percorso da `capabilities().logprobs`). `SCHEMA_VERSION = 3`. `BenchCell.qualityScore`
  **opzionale** — modulo pronto e testato ma non collegato a `benchServer.ts` (nessuna cella reale
  calcola oggi un punteggio; vedi docket #10, decisione da confermare prima di Fase 4).
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

- **Fase 4** (README + verifica finale) da fare — vedi §1.
- **docket #10 — decisione registrata, non un ruling bloccante**: `quality.ts` non è collegato a
  `benchServer.ts`. Nessun run reale (Fase 1/2, né lo sweep manuale di Fase 4) porta un
  `qualityScore`. Va deciso se/quando collegarlo prima di considerare il goal completo.
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
  pipeline di bench. Va confermato/deciso prima di dichiarare il goal completo (vedi §1, §3).
- Chiusi in questa sessione: **#7** (`BenchCell.protocol` implementato, constraint emendato).
- Chiusi in sessione 3: **#3** (push approvato), **#4** (registrazione), **#5** (strict),
  **#5b** (warm-up selezionabile), **#9** (nome pacchetto `@wllama/wllama`).
- Nessun item aperto in attesa di ruling PI.
- Storico dei docket delle fasi precedenti: nel journal del goal.
