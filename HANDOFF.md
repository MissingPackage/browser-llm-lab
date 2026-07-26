# HANDOFF — browser-llm-lab   (updated 2026-07-26, session 2)

## 1. Next decidable
Piano `docs/superpowers/plans/2026-07-26-fase-1b-fondamenta.md`: **tutti gli 8 task completi**
(50/50 checkbox), branch `fix/fase-1b-fixin1b` PR-ready — 12 commit, `npm test` 39/39, `tsc
--noEmit` pulito, `npm run build` ok, review avversaria per-task + final review whole-branch
tutte clean (fix applicati inline, nessun residuo). **Non pushato, non mergiato.**

Due decisioni sono ora del PI, non mie (roadmap/merge, non un fix bounded):
1. **Merge di `fix/fase-1b-fixin1b`** — branch pronto, review completa, nessun blocco tecnico.
2. **Avvio "1b — matrice"** (adapter Transformers.js/wllama, sweep multi-device, modulo qualità
   — spec §Fasatura): è un'espansione sostanziale (due nuovi stack, non un fix incrementale) —
   merita probabilmente un passaggio brainstorming/spec-first prima di un piano di implementazione,
   non solo "scrivi il prossimo piano" in autonomia. Fermo qui in attesa di ruling.

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
- Stato branch finale: `npm test` 39/39, `tsc --noEmit` pulito, `npm run build` ok.
  Branch `fix/fase-1b-fixin1b` rebasato su `main` (che riceve separatamente i refresh
  HANDOFF/gitignore/ledger/plan-checkboxes, essendo bookkeeping non feature-work).
  **Non pushato, non mergiato — PR-ready, merge è docket (item 6).**

## 3. State delta (session 1, 2026-07-25)
- Progetto creato da zero: spec approvata, piano fase 1a, 9 task eseguiti SDD (subagent + review avversaria per task).
- `main` = harness fase 1a completo e merged (`c9e05b4`): SPA Vite+TS, bench worker, adapter WebLLM (CreateMLCEngine in-worker), probe WebGPU, schema v1, metriche pure, UI+export. Suite 26/26, tsc pulito, build ok.
- Run E2E **reali sulla 4090** via `scripts/e2e-bench.mjs` (playwright headed): cold 55.6s / warm 1.6s load, ~106–118 tok/s decode su Qwen2.5-0.5B q4f32_1. Risultati in `results/*.json`.
- README con i finding live; final review whole-branch: READY TO MERGE, minors triagiati.
- `main` pubblicato su `origin`/GitHub; storico locale riscritto una volta per sostituire l'email privata bloccata da GH007 con la `noreply` del profilo. Tracking `origin/main` attivo.

## 4. Open threads
- Branch `feat/fase-1a` merged, può essere cancellato (tenuto per ora).
- FIX-IN-1B dal final review: **fatto** (Task 1-4, branch `fix/fase-1b-fixin1b`, non mergiato).
- Varianza tok/s run-to-run ~10% → repliche multiple **fatte** (Task 6, branch `fix/fase-1b-fixin1b`, non mergiato) con flag `high-variance` a soglia 0.15.
- Piano fase-1b-fondamenta **completo** (8/8 task, branch PR-ready) — vedi §1 per le due decisioni pendenti (merge + avvio "1b — matrice").

## 5. Landmines
- **`shader-f16` è per-browser, non per-GPU**: assente su chromium-playwright (q4f16_1 crasha → usare q4f32_1), presente su Firefox. Schema v2 ha ora `browser`/`anomalies`/`features` (branch `fix/fase-1b-fixin1b`, non ancora mergiato in main) — verificato dal vivo: `features` su questa macchina/chromium non include `shader-f16`, coerente col finding sotto.
- **Firefox silent-fallback a CPU**: il FF 152 dell'utente girava su **llvmpipe** dichiarando `webgpu:true` (about:support: NVIDIA inattiva, acceleration blocked by platform) → la cella `*firefox152-LLVMPIPE-CPU.json` è un datapoint CPU, NON 4090. Il probe ora rileva il software adapter (128 MiB + vendor vuoto, branch `fix/fase-1b-fixin1b` non ancora mergiato). Run playwright-FF invece su GPU vera (nvidia-smi 100%): 9.9 tok/s.
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
5. ~~`docs/superpowers/` gitignored vs tracciato~~ RATIFICATO 2026-07-26 (Cristiano): "versioniamolo sul git locale ma teniamolo fuori da github". Fatto: rimosso da `.gitignore` (ora tracciato normalmente, commit incluso il piano fase-1b-fondamenta), aggiunto `.git/hooks/pre-push` che blocca qualunque push la cui tree introduca contenuto `docs/superpowers/` **oltre quello già presente in `origin/main`** (baseline = `origin/main`, non tree assoluta — altrimenti i 2 file storici già pubblici, spec e piano fase-1a, bloccherebbero ogni push futuro; testato con un remote fittizio prima di fidarsene). Hook locale, non tracciato — enforcement per-macchina, coerente con "solo locale". `.superpowers/` (ledger SDD) resta gitignored come da decisione precedente, non toccato da questa ratifica. **Conseguenza operativa**: da `ed36881` in poi, `git push origin main` fallirà finché il commit col piano fase-1b-fondamenta resta in `main` — per pubblicare serve prima un cherry-pick/rebase dei soli commit di codice su un branch pulito.
6. **Nuovo (2026-07-26)**: merge di `fix/fase-1b-fixin1b` in `main`. Branch PR-ready (8/8 task del piano fondamenta, review avversaria per-task + final review whole-branch tutte clean, `npm test` 39/39, `tsc`/`build` puliti). Nessun blocco tecnico — decisione di merge/timing lasciata al PI (non decisa in autonomia, per policy "NO merges to dev/main" del loop).
7. **Nuovo (2026-07-26)**: avvio del piano "1b — matrice" (adapter Transformers.js v3 + wllama, sweep multi-device M4/S22, modulo qualità-leggera — spec §Fasatura). A differenza delle fondamenta appena chiuse, introduce due stack interi nuovi: probabile candidato per un passaggio brainstorming/spec-first dedicato prima del piano di implementazione, non solo "il prossimo piano scritto in autonomia". In attesa di ruling PI su scope/priorità prima di procedere.
