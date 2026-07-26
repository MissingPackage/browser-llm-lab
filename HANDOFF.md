# HANDOFF — browser-llm-lab   (updated 2026-07-27, session 3)

## 1. Next decidable

**Goal attivo**: `fase-1b-matrice` (`.harness/goals/fase-1b-matrice/` — GOAL.md, PHASES.md,
docket.md, journal.md). Design: `docs/superpowers/specs/2026-07-26-fase-1b-matrice-design.md`.
**Fasi 1 e 2 complete e mergiate in `main`**, pushate su `origin` (`277609e`).

**Prossimo decidibile: Fase 3 — modulo qualità + schema v3** (PHASES.md riga 3):
`src/quality.ts` + `src/qualityPrompts.ts` (perplexity se `capabilities().logprobs`, altrimenti
fallback a 12 prompt exact-match), `BenchCell.qualityScore`, `SCHEMA_VERSION=3`. Nessun gate lo
blocca. Due cose già decise vanno fatte **dentro** Fase 3, perché è lì che lo schema si tocca:
- **docket #7** (approvato): aggiungere `BenchCell.protocol` per registrare warm-up policy e
  numero di repliche. Oggi il warm-up è annotato in `anomalies`, che significa "cosa è andato
  storto" — e con due politiche di misura un file senza note è ambiguo.
- il constraint di `GOAL.md` "no ad hoc field additions beyond `qualityScore`" va emendato di
  conseguenza (il ruling #7 lo fa implicitamente: registrarlo esplicitamente in Fase 3).

## 2. State delta (session 3, 2026-07-26 → 27)

- **Ruling #5 (strict)**: `"strict": true` in `tsconfig.json`; il codebase era già conforme.
- **Ruling #5b (warm-up)**: `WarmupPolicy` in `src/benchServer.ts` — `always` (default,
  steady-state), `never` (prima esperienza reale), `cold-only`. Selezionabile per run via
  `MainToWorker.bench.warmup`. **Le prime due misurano cose diverse e non vanno confrontate.**
- **4 run di metodologia sulla 4090** (`results/methodology/`): il +55% di docket #5b non è più
  riproducibile; resta ~3% di dipendenza dall'ordine col warm-up esteso. **Non è hardware** —
  temperatura, memoria GPU, clock e throttle reasons misurati e tutti negativi.
- **Fase 2 — adapter wllama** (`src/adapters/wllama.ts`, `@wllama/wllama` 3.5.1): terzo stack,
  CPU via WASM. Tre difetti trovati e corretti (chunk senza contenuto contati come token; prompt
  cache che falsava il TTFT delle repliche; `document.baseURI` assente nel Web Worker).
- **Prima misura a tre stack**: webllm ~110 tok/s, transformersjs ~46–48, wllama **25.97**.
- `scripts/seq-bench.mjs`: driver multi-cella (`SEQ`, `BENCH_URL`) con contatori `nvidia-smi` a
  ogni confine di cella — `e2e-bench.mjs` fa una cella per invocazione e l'effetto vive *tra* celle.
- Test 75/75. `origin/main` allineato.

## 3. Open threads

- **Fase 3** (qualità + schema v3) e **Fase 4** (README + verifica finale) da fare.
- **docket #8 — deroga attiva**: conformance wllama **7/8**. Difetto di wllama, non nostro
  (ogni risposta perde la coda, che arriva al giro dopo). Upstream ngxson/wllama#263 + PR #264,
  entrambe aperte; 3.5.1 è precedente alla PR. **Routine di sorveglianza attiva**
  `trig_018i6ZnQpZHF1tg6egjTsyST`, ogni 3 giorni: apre una PR quando il fix è rilasciato.
  Quando arriva → aggiornare la dipendenza, rieseguire il conformance **in locale** (serve GPU) e
  togliere la deroga da README, PHASES e docket.
- Rapporto cross-stack ancora **non dichiarabile** in un rapporto: ~3% di dipendenza dall'ordine.
- I run in `results/*.json` precedenti al 2026-07-26 sono ordine-dipendenti e nulla nel file lo dice.
- Branch `feat/fase-1a`, `fix/fase-1b-fixin1b`, `feat/fase-2-wllama` merged, non cancellati.

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
- **#7 — deciso, da eseguire in Fase 3**: `BenchCell.protocol` per registrare warm-up policy e
  repliche. Serve anche emendare il constraint "no ad hoc field additions" di GOAL.md.
- Chiusi in questa sessione: **#3** (push approvato), **#4** (registrazione), **#5** (strict),
  **#5b** (warm-up selezionabile), **#9** (nome pacchetto `@wllama/wllama`).
- Nessun item aperto in attesa di ruling.
- Storico dei docket delle fasi precedenti: nel journal del goal.
