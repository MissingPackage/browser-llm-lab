# HANDOFF — browser-llm-lab   (updated 2026-07-26, session 3)

## 1. Next decidable

**Goal attivo**: `fase-1b-matrice` (`.harness/goals/fase-1b-matrice/` — GOAL.md, PHASES.md,
docket.md, journal.md). Design: `docs/superpowers/specs/2026-07-26-fase-1b-matrice-design.md`.
**Fase 1 completa e mergiata in `main`**; `main` è **pushato su `origin`** (`579c9e4`).

**Prossimo decidibile: Fase 2 — adapter wllama** (PHASES.md riga 2). Nessun gate lo blocca: il
push è approvato (docket #3), il merge a fine fase è approvato, il conformance harness accoglie il
terzo stack con una riga in `src/conformance/page.ts`.

**Ma prima, una risposta rapida che cambia i numeri di Fase 2** — docket #5b chiede se estendere
il warm-up a **tutte** le celle invece che alle sole cold. Misurato: porta la deriva ordine-
dipendente da −6.6% a −2.9% e stabilizza il TTFT. È una riga (`needsWarmup` → `return true`),
già isolata. Se la risposta arriva prima di Fase 2, i run wllama nascono con il protocollo giusto.

## 2. State delta (session 3, 2026-07-26)

- **Ruling #5 (strict)**: `"strict": true` in `tsconfig.json`. Il codebase era già conforme —
  zero errori da correggere.
- **Ruling #5b (warm-up)**: `needsWarmup()` in `src/benchServer.ts` — se `cacheState !== "warm"`,
  una generate a carico identico viene eseguita e scartata prima delle 3 repliche. La cella lo
  dichiara in `anomalies` come `protocol: warm-up run discarded`.
- **Nuovo `scripts/seq-bench.mjs`**: driver multi-cella (env `SEQ`) con contatori `nvidia-smi` a
  ogni confine di cella. Serviva perché `e2e-bench.mjs` fa una cella per invocazione, e l'effetto
  di docket #5b vive *tra* celle. Documentato in README.
- **4 run di verifica sulla 4090**, dati + log in `results/methodology/`: il **+55% non è più
  riproducibile**. Resta una deriva −7…−14% (prima cella di ogni sessione più veloce), **non
  hardware** — temperatura, memoria GPU, clock e throttle reasons misurati e tutti negativi.
- Test 50/50 (4 nuovi). Un test preesistente sull'high-variance è caduto perché il warm-up
  assorbe il primo run lento che usava come sorgente di spread — isolato con `cacheState: "warm"`.
- **Pushato `origin/main`** `825e2b5..579c9e4` (22 commit: `origin` era fermo a prima di Fase 1).

## 3. Open threads

- **Fase 2 (wllama)** non iniziata. Poi Fase 3 (qualità + schema v3), Fase 4 (README + verifica).
- Branch `feat/fase-1a` e `fix/fase-1b-fixin1b` merged, non cancellati.
- Rapporto cross-stack: **webllm/transformersjs ≈ 2.0–2.3×** sulla 4090 dalle misure pulite di
  oggi. Ancora ordine-dipendente (~3% anche col warm-up esteso) → **non dichiarabile** in un
  rapporto finché docket #5b non chiude.
- I run in `results/*.json` precedenti a oggi sono ordine-dipendenti e nulla nel file lo dice.

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
- Playwright `waitForFunction(fn, arg, options)`: le options sono il TERZO argomento.
- `chromium.launch()` = profilo effimero → cold/warm richiede `launchPersistentContext`.
- tsconfig: `erasableSyntaxOnly` (niente parameter properties) e ora `strict`.
- `.superpowers/` è gitignorato: il ledger SDD non sopravvive a `git clean -fdx`.
- Il dev server su :5173 può essere già acceso da un'altra sessione — `npm run dev` ripiega su
  :5174 e i driver puntano a :5173 di default.

## 5. Docket (user decisions pending)

Vedi `.harness/goals/fase-1b-matrice/docket.md` per il testo integrale.

- **#5b — APERTO, richiede ruling**: estendere il warm-up a tutte le celle (non solo cold)?
  Evidenza e raccomandazione nel docket. Bloccante per dichiarare qualunque rapporto cross-stack.
- **#7 — APERTO, richiede ruling**: il warm-up applicato è registrato in `BenchCell.anomalies`,
  ma `anomalies` significa "cosa è andato storto" — chi filtra `anomalies.length > 0` scarterebbe
  le celle misurate bene. Il campo pulito (`BenchCell.protocol`) è vietato dal constraint
  "no ad hoc field additions beyond `qualityScore`" in GOAL.md. Raccomandato: emendare e
  aggiungerlo in Fase 3, che tocca già lo schema per il bump a v3.
- Chiusi in questa sessione: **#3** (push approvato), **#4** (registrazione, nessuna azione),
  **#5** (strict applicato).
- Storico dei docket delle fasi precedenti: nel journal del goal.
