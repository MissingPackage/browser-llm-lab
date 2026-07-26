# HANDOFF — browser-llm-lab   (updated 2026-07-26, session 2)

## 1. Next decidable
Piano `docs/superpowers/plans/2026-07-26-fase-1b-fondamenta.md`: Task 1-4 (FIX-IN-1B) **completi**
su branch `fix/fase-1b-fixin1b` (locale, non pushato, non mergiato). Prossimo passo concreto:
Task 5 del piano (schema v2 — `DeviceProbe.{browser,features,anomalies}` + probe esteso con
`adapter.features`/rilevazione software-adapter), poi Task 6 (repliche multiple + aggregazione),
Task 7 (render/UI), Task 8 (README) — nello stesso branch o uno nuovo, a scelta di chi esegue.
**Non tocca** adapter Transformers.js/wllama, sweep multi-device, modulo qualità: quello resta
un piano "1b — matrice" separato, da scrivere dopo che le fondamenta sono eseguite.

## 2. State delta (session 2, 2026-07-26)
- Scritto il piano Fase 1b — Fondamenta (8 task, TDD), verificato da loop-verifier: PASS su
  copertura FIX-IN-1B, coerenza tipi tra task, assenza placeholder, aderenza al formato
  writing-plans. Vive sotto `docs/superpowers/` (gitignored dal commit `347a7fe`, quindi non
  tracciato in git — vedi docket item 5).
- **Eseguiti Task 1-4** (FIX-IN-1B backlog) su branch `fix/fase-1b-fixin1b`: escaping HTML in
  `render.ts` (1bde4ee), exhaustiveness guard `WorkerToMain` in `main.ts` (188c886, guard
  verificato empiricamente con variante fittizia + tsc), test `dispose()` (b47abd7, verde al
  primo run — nessun bug preesistente), export disabilitato durante bench in corso (4378710).
  **Review avversaria** ha trovato un bug reale nella guardia di re-enable-on-error (`run ===
  null` non copriva "primo bench fallisce prima di produrre celle", perché `run` è già
  non-null da subito dopo il probe) — corretto in `run === null || run.cells.length === 0`
  (c319081), stesso difetto anche nel piano, corretto lì pure. `npm test` 28/28, `tsc --noEmit`
  pulito, `npm run build` ok, verifica manuale in browser (playwright) del toggle Export.
  Branch non pushato/mergiato (PR-ready, merge resta docket).

## 3. State delta (session 1, 2026-07-25)
- Progetto creato da zero: spec approvata, piano fase 1a, 9 task eseguiti SDD (subagent + review avversaria per task).
- `main` = harness fase 1a completo e merged (`c9e05b4`): SPA Vite+TS, bench worker, adapter WebLLM (CreateMLCEngine in-worker), probe WebGPU, schema v1, metriche pure, UI+export. Suite 26/26, tsc pulito, build ok.
- Run E2E **reali sulla 4090** via `scripts/e2e-bench.mjs` (playwright headed): cold 55.6s / warm 1.6s load, ~106–118 tok/s decode su Qwen2.5-0.5B q4f32_1. Risultati in `results/*.json`.
- README con i finding live; final review whole-branch: READY TO MERGE, minors triagiati.
- `main` pubblicato su `origin`/GitHub; storico locale riscritto una volta per sostituire l'email privata bloccata da GH007 con la `noreply` del profilo. Tracking `origin/main` attivo.

## 4. Open threads
- Branch `feat/fase-1a` merged, può essere cancellato (tenuto per ora).
- FIX-IN-1B dal final review: escaping innerHTML, exhaustiveness guard WorkerToMain, test dispose(), export re-disable, probe features — **piano scritto** (§1), non ancora eseguito.
- Varianza tok/s run-to-run ~10% → repliche multiple **pianificate** (§1, Task 6 del piano fondamenta), non ancora eseguite.

## 5. Landmines
- **`shader-f16` è per-browser, non per-GPU**: assente su chromium-playwright (q4f16_1 crasha → usare q4f32_1), presente su Firefox. Schema v1 non ha campi `browser`/`anomalies`: piano scritto per v2 (§1), non ancora eseguito.
- **Firefox silent-fallback a CPU**: il FF 152 dell'utente girava su **llvmpipe** dichiarando `webgpu:true` (about:support: NVIDIA inattiva, acceleration blocked by platform) → la cella `*firefox152-LLVMPIPE-CPU.json` è un datapoint CPU, NON 4090. Il probe 1b deve rilevare il software adapter (128 MiB + vendor vuoto). Run playwright-FF invece su GPU vera (nvidia-smi 100%): 9.9 tok/s.
- Headless shell = **SwiftShader**: mai accettare risultati come dati GPU (il driver e2e già rifiuta da solo).
- Playwright `waitForFunction(fn, arg, options)`: le options sono il TERZO argomento (bug già pagato una volta).
- `chromium.launch()` = profilo effimero → cold/warm richiede `launchPersistentContext` (già nel driver, profilo `/tmp/blab-e2e-profile`).
- tsconfig ha `erasableSyntaxOnly: true`: niente parameter properties nelle classi.
- `.superpowers/` è gitignorato: il ledger SDD (`.superpowers/sdd/progress.md`) non sopravvive a `git clean -fdx`.

- **Chrome flags & NVIDIA/Wayland**: enable-vulkan nel profilo corrompe il compositing (schermo nero/artefatti); force-enable-webgpu-interop crasha all avvio. Profilo quotidiano dell utente: flags azzerati il 2026-07-25, NON rimetterli.
- **Chrome GPU sandbox vs Vulkan ICD (Fedora/NVIDIA)**: col sandbox attivo il GPU process non legge gli ICD ("vkCreateInstance: Found no drivers", Permission denied su file leggibili) → WebGPU niente adapter. Riprodotto deterministicamente col toggle chromiumSandbox. Ricetta bench manuale: profilo dedicato + --ignore-gpu-blocklist + --disable-gpu-sandbox (scripts/bench-chrome.sh). I run playwright non lo soffrono (no-sandbox di default). Ritestare a ogni update di Chrome.
- Delta branded-vs-playwright RISOLTO: run manuale utente su branded = 116.9 tok/s (in linea). Era varianza; repliche multiple in 1b.

## 6. Docket (user decisions pending)
1. Espansione benchmark pubblico/community: deferred per scelta PI (2026-07-25), ripensare a banco maturo.
2. ~~Installare Chrome branded per testare `shader-f16`~~ FATTO 2026-07-25: f16 assente anche sul branded (pure con dev-features) → è Dawn/driver; q4f16_1 su questa macchina solo via Firefox.
3. Punteggio decode `null` vs `0` con <2 token: adjudicato dal controller (null); ratificare o ribaltare.
4. Guardia `completionTokens >= 2` + prosa piano allineata + probe never-throws + erasableSyntaxOnly: tre adjudication del controller in sessione, tutte documentate nel ledger — ratifica implicita se nessuna obiezione.
5. **Nuovo (2026-07-26)**: `docs/superpowers/` è gitignored dal commit `347a7fe` (2026-07-25), ma i file già esistenti (spec, piano fase-1a) restano tracciati in git da prima — solo i nuovi (incluso il piano fase-1b-fondamenta appena scritto) restano fuori dal controllo versione. Decisione da ratificare: va bene così (piani = lavoro effimero, non versionato) o i piani vanno tracciati esplicitamente con `git add -f` per continuità storica?
