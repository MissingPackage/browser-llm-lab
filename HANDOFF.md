# HANDOFF — browser-llm-lab   (updated 2026-07-26, session 2)

## 1. Next decidable
**Goal attivo**: `fase-1b-matrice` (`.harness/goals/fase-1b-matrice/`) — GOAL.md + PHASES.md
scritti 2026-07-26 da brainstorming → goal-brief → goal-setup, chiudendo il docket #7. Design
di riferimento: `docs/superpowers/specs/2026-07-26-fase-1b-matrice-design.md`. **Gate
plan-check approvato** (2026-07-26, Cristiano: "approvato"). **Iterazione 1 fatta**: piano di
implementazione per la Fase 1 (adapter Transformers.js) scritto e committato su
`feat/fase-1b-matrice` — `docs/superpowers/plans/2026-07-26-fase-1b-matrice-adapter-transformersjs.md`
(4 task: adapter+StackId, BenchServer multi-stack, UI selector, driver e2e + run reale 4090).
Non ancora eseguito: in attesa della scelta di esecuzione (subagent-driven vs inline).

Contesto precedente: Fase 1b — Fondamenta **mergiata in `main`** (2026-07-26, merge commit su
richiesta esplicita di Cristiano): piano `docs/superpowers/plans/2026-07-26-fase-1b-fondamenta.md`
completo (8/8 task), branch `fix/fase-1b-fixin1b` integrato. Verificato post-merge: `npm test`
39/39, `tsc --noEmit` pulito, `npm run build` ok. `main` ora ha schema v2, probe esteso
(browser/features/anomalies, rilevazione software-adapter), repliche multiple con aggregazione,
UI aggiornata.

**Pushato su `origin/main`** (2026-07-26, `d2d3c43..4c72fb4`): `docs/superpowers/` è ora pubblico
su GitHub — la policy "solo locale" (docket #5) è stata rivista e il pre-push hook rimosso, vedi
docket #5 per il perché.

## 2. State delta (session 2, 2026-07-26)
- Scritto il piano Fase 1b — Fondamenta (8 task, TDD), verificato da loop-verifier: PASS su
  copertura FIX-IN-1B, coerenza tipi tra task, assenza placeholder, aderenza al formato
  writing-plans. **Ratificato 2026-07-26** (docket #5): `docs/superpowers/` ora tracciato in git
  locale, mai su GitHub (pre-push hook, vedi docket #5 per dettagli).
- **Eseguiti Task 1-4** (FIX-IN-1B backlog) su branch `fix/fase-1b-fixin1b`: escaping HTML in
  `render.ts`, exhaustiveness guard `WorkerToMain` in `main.ts` (guard verificato empiricamente
  con variante fittizia + tsc), test `dispose()` (verde al primo run — nessun bug preesistente),
  export disabilitato durante bench in corso. **Review avversaria** ha trovato un bug reale nella
  guardia di re-enable-on-error (`run === null` non copriva "primo bench fallisce prima di
  produrre celle", perché `run` è già non-null da subito dopo il probe) — corretto in
  `run === null || run.cells.length === 0`, stesso difetto anche nel piano, corretto lì pure.
  Verifica manuale in browser (playwright) del toggle Export.
- **Eseguito Task 5** (schema v2 — `DeviceProbe.browser/features/anomalies`, `SCHEMA_VERSION=2`
  + `probe.ts`: `parseBrowser`, enumerazione `adapter.features`, `detectSoftwareAdapter` a soglia
  128 MiB/vendor vuoto). Gap nel piano trovato ed emendato in-commit: Task 5 non elencava
  `tests/render.test.ts` tra i file da aggiornare (aveva una propria fixture `DeviceProbe` dal
  Task 1). Review avversaria: clean (solo 2 minor non azionabili — misclassificazione UA Edge
  mobile fuori scope, stringa anomalia in italiano coerente col resto del codebase).
- **Eseguito Task 6** (repliche multiple — `metrics.ts`: `aggregateReplicates`/`MetricAggregate`/
  `GenMetricsAgg`; `schema.ts`: `BenchCell.gen` diventa aggregato + `.replicates`/`.anomalies`;
  `benchServer.ts`: load una volta + generate x `replicateCount` default 3, flag `high-variance`
  a soglia stdev/mean 0.15). Stesso tipo di gap trovato in Task 5: il piano non elencava
  `src/render.ts` tra i file da toccare, ma il cambio di tipo lo rompe — applicato un fix
  **minimale di sola compilazione** (`.ttftMs.mean`, `.decodeToksPerSec?.mean`), senza toccare
  lo scope UI del Task 7 (badge anomalie, mean±stdev in vista). Review avversaria: clean, 2 minor
  non azionabili.
- **Eseguito Task 7** (render/UI — `fmtRate` mean±stdev, colonna anomalie con badge escaped;
  `main.ts` non toccato, il probe-box mostra già browser/features/anomalies via
  `JSON.stringify` esistente — **verificato dal vivo su GPU reale** (4090, vendor nvidia/
  lovelace, browser chrome 150.0.0.0, features senza `shader-f16` coerente coi landmine noti,
  anomalies: []). Review avversaria: trovato test anomaly-badge vacuo (asseriva solo substring,
  non il markup `<span class="anomaly">` né l'escaping) — rinforzato e verificato che il test
  rinforzato fallisce davvero su una regressione simulata.
- **Eseguito Task 8** (README — sezione "Fase 1b — fondamenta": schema v2, repliche, euristica
  software-adapter).
- **FINAL REVIEW whole-branch** (worktree isolato): trovata e corretta una contraddizione
  terminologica nel README — la sezione "## Note" preesistente definiva "Fase 1b" come sola
  matrice-piena, in conflitto con la nuova sezione Task 8 "Fase 1b — fondamenta"; corretta
  insieme a 2 riferimenti stale ("repliche multiple in 1b" → ora fatte). Nessun altro finding:
  coerenza cross-task confermata, nessun codice orfano dai gap del piano (Task 5/6 non
  elencavano render.ts), tipi `BenchCell` end-to-end puliti, nessuno scope creep.
- Stato branch pre-merge: `npm test` 39/39, `tsc --noEmit` pulito, `npm run build` ok.
- **Mergiato in `main`** su richiesta esplicita ("merge fix/fase-1b-fixin1b in main"): merge
  commit (non fast-forward, per lasciare traccia esplicita del punto di integrazione, come
  già fatto per `feat/fase-1a`). Gate post-merge rilanciati e verdi. Branch `fix/fase-1b-fixin1b`
  non cancellato (tenuto per ora, come `feat/fase-1a`).

## 3. State delta (session 1, 2026-07-25)
- Progetto creato da zero: spec approvata, piano fase 1a, 9 task eseguiti SDD (subagent + review avversaria per task).
- `main` = harness fase 1a completo e merged (`c9e05b4`): SPA Vite+TS, bench worker, adapter WebLLM (CreateMLCEngine in-worker), probe WebGPU, schema v1, metriche pure, UI+export. Suite 26/26, tsc pulito, build ok.
- Run E2E **reali sulla 4090** via `scripts/e2e-bench.mjs` (playwright headed): cold 55.6s / warm 1.6s load, ~106–118 tok/s decode su Qwen2.5-0.5B q4f32_1. Risultati in `results/*.json`.
- README con i finding live; final review whole-branch: READY TO MERGE, minors triagiati.
- `main` pubblicato su `origin`/GitHub; storico locale riscritto una volta per sostituire l'email privata bloccata da GH007 con la `noreply` del profilo. Tracking `origin/main` attivo.

## 4. Open threads
- Branch `feat/fase-1a` merged, può essere cancellato (tenuto per ora).
- Branch `fix/fase-1b-fixin1b` merged in `main`, può essere cancellato (tenuto per ora).
- FIX-IN-1B dal final review: **fatto**, in `main`.
- Varianza tok/s run-to-run ~10% → repliche multiple **fatte**, in `main`, con flag `high-variance` a soglia 0.15.
- Piano fase-1b-fondamenta **completo e mergiato** — vedi §1 per la decisione pendente (avvio "1b — matrice").

## 5. Landmines
- **`shader-f16` è per-browser, non per-GPU**: assente su chromium-playwright (q4f16_1 crasha → usare q4f32_1), presente su Firefox. Schema v2 ha ora `browser`/`anomalies`/`features` (in `main`) — verificato dal vivo: `features` su questa macchina/chromium non include `shader-f16`, coerente col finding sotto.
- **Firefox silent-fallback a CPU**: il FF 152 dell'utente girava su **llvmpipe** dichiarando `webgpu:true` (about:support: NVIDIA inattiva, acceleration blocked by platform) → la cella `*firefox152-LLVMPIPE-CPU.json` è un datapoint CPU, NON 4090. Il probe ora rileva il software adapter (128 MiB + vendor vuoto, in `main`). Run playwright-FF invece su GPU vera (nvidia-smi 100%): 9.9 tok/s.
- Headless shell = **SwiftShader**: mai accettare risultati come dati GPU (il driver e2e già rifiuta da solo).
- Playwright `waitForFunction(fn, arg, options)`: le options sono il TERZO argomento (bug già pagato una volta).
- `chromium.launch()` = profilo effimero → cold/warm richiede `launchPersistentContext` (già nel driver, profilo `/tmp/blab-e2e-profile`).
- tsconfig ha `erasableSyntaxOnly: true`: niente parameter properties nelle classi.
- `.superpowers/` è gitignorato: il ledger SDD (`.superpowers/sdd/progress.md`) non sopravvive a `git clean -fdx`.

- **Chrome flags & NVIDIA/Wayland**: enable-vulkan nel profilo corrompe il compositing (schermo nero/artefatti); force-enable-webgpu-interop crasha all avvio. Profilo quotidiano dell utente: flags azzerati il 2026-07-25, NON rimetterli.
- **Chrome GPU sandbox vs Vulkan ICD (Fedora/NVIDIA)**: col sandbox attivo il GPU process non legge gli ICD ("vkCreateInstance: Found no drivers", Permission denied su file leggibili) → WebGPU niente adapter. Riprodotto deterministicamente col toggle chromiumSandbox. Ricetta bench manuale: profilo dedicato + --ignore-gpu-blocklist + --disable-gpu-sandbox (scripts/bench-chrome.sh). I run playwright non lo soffrono (no-sandbox di default). Ritestare a ogni update di Chrome.
- Delta branded-vs-playwright RISOLTO: run manuale utente su branded = 116.9 tok/s (in linea). Era varianza; repliche multiple ora fatte (Fase 1b — fondamenta).

## 6. Docket (user decisions pending)
1. Espansione benchmark pubblico/community: deferred per scelta PI (2026-07-25), ripensare a banco maturo.
2. ~~Installare Chrome branded per testare `shader-f16`~~ FATTO 2026-07-25: f16 assente anche sul branded (pure con dev-features) → è Dawn/driver; q4f16_1 su questa macchina solo via Firefox.
3. Punteggio decode `null` vs `0` con <2 token: adjudicato dal controller (null); ratificare o ribaltare.
4. Guardia `completionTokens >= 2` + prosa piano allineata + probe never-throws + erasableSyntaxOnly: tre adjudication del controller in sessione, tutte documentate nel ledger — ratifica implicita se nessuna obiezione.
5. ~~`docs/superpowers/` gitignored vs tracciato~~ **REVISIONATO 2026-07-26** (Cristiano). Prima ratifica: tracciato in git locale ma mai su GitHub (pre-push hook). Conseguenza operativa emersa subito: blocca `git push origin main` per qualunque commit futuro che tocchi anche solo un file sotto `docs/superpowers/` insieme a codice — un cherry-pick "una tantum" non risolve, si ripresenta ad ogni push. Cristiano: "che senso ha se poi non possiamo più pushare... togliamo la regola, pazienza". **Deciso**: hook `.git/hooks/pre-push` rimosso, `docs/superpowers/` (piano fase-1b-fondamenta incluso) è ora pubblico su GitHub (push 2026-07-26, `d2d3c43..4c72fb4`). `.superpowers/` (ledger SDD) resta gitignored, non toccato.
6. ~~Merge di `fix/fase-1b-fixin1b` in `main`~~ FATTO 2026-07-26 su richiesta esplicita di Cristiano. Gate post-merge verdi.
7. **Chiuso 2026-07-26**: avvio "1b — matrice" formalizzato via brainstorming → design doc (`docs/superpowers/specs/2026-07-26-fase-1b-matrice-design.md`) → goal-brief → goal-setup. Ora è il goal attivo `fase-1b-matrice` (vedi §1). Decisioni prese durante il brainstorming: intero pacchetto in un design (non decomposto), ordine Transformers.js → wllama → qualità → sweep, sweep multi-device manuale per ora (automazione deferred), fascia Ceiling resta F3, punteggio qualità senza soglia pass/fail. Ricerca HF Hub ha confermato un gap strutturale: la fascia Large (Qwen2.5-7B, Llama-3.1-8B) non è eseguibile né su Transformers.js (nessun repo ONNX web-runnable) né su wllama (pesi Q4_K_M >4GB, tetto WASM) — documentato nel design, non bloccante. **Nuovo item aperto**: plan-check su `PHASES.md` (vedi `.harness/goals/fase-1b-matrice/docket.md`) — approvazione PI prima dell'iterazione 1.
