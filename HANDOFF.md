# HANDOFF — browser-llm-lab   (updated 2026-07-25, session 1)

## 1. Next decidable
Avviare la **Fase 1b** (piano da scrivere, spec §Fasatura): adapter Transformers.js v3 + wllama,
modulo qualità-leggera, sweep multi-modello. Primo passo concreto: piano `docs/superpowers/plans/`
per 1b partendo dai FIX-IN-1B della final review (elencati in `.superpowers/sdd/progress.md`)
+ probe esteso con `adapter.features` (gap scoperto dal vivo).
Prima ancora, opzionale e a decisione PI: caccia a `shader-f16` su Chrome branded (docket #2).

## 2. State delta (this session)
- Progetto creato da zero: spec approvata, piano fase 1a, 9 task eseguiti SDD (subagent + review avversaria per task).
- `main` = harness fase 1a completo e merged (`c9e05b4`): SPA Vite+TS, bench worker, adapter WebLLM (CreateMLCEngine in-worker), probe WebGPU, schema v1, metriche pure, UI+export. Suite 26/26, tsc pulito, build ok.
- Run E2E **reali sulla 4090** via `scripts/e2e-bench.mjs` (playwright headed): cold 55.6s / warm 1.6s load, ~106–118 tok/s decode su Qwen2.5-0.5B q4f32_1. Risultati in `results/*.json`.
- README con i finding live; final review whole-branch: READY TO MERGE, minors triagiati.

## 3. Open threads
- Branch `feat/fase-1a` merged, può essere cancellato (tenuto per ora).
- FIX-IN-1B dal final review: escaping innerHTML, exhaustiveness guard WorkerToMain, test dispose(), export re-disable, probe features.
- Varianza tok/s run-to-run ~10% → 1b deve introdurre repliche multiple per cella.
- Nessun remote git configurato: repo solo locale.

## 4. Landmines
- **`shader-f16` è per-browser, non per-GPU**: assente su chromium-playwright (q4f16_1 crasha → usare q4f32_1), presente su Firefox (dove però il 152 release cappa i buffer a 128 MiB → 1.8 tok/s + `Device was lost`). Schema v1 non ha campi `browser`/`anomalies` per cella: gap da colmare in 1b (oggi il browser si desume solo dallo userAgent nel probe).
- Headless shell = **SwiftShader**: mai accettare risultati come dati GPU (il driver e2e già rifiuta da solo).
- Playwright `waitForFunction(fn, arg, options)`: le options sono il TERZO argomento (bug già pagato una volta).
- `chromium.launch()` = profilo effimero → cold/warm richiede `launchPersistentContext` (già nel driver, profilo `/tmp/blab-e2e-profile`).
- tsconfig ha `erasableSyntaxOnly: true`: niente parameter properties nelle classi.
- `.superpowers/` è gitignorato: il ledger SDD (`.superpowers/sdd/progress.md`) non sopravvive a `git clean -fdx`.

## 5. Docket (user decisions pending)
1. Espansione benchmark pubblico/community: deferred per scelta PI (2026-07-25), ripensare a banco maturo.
2. Installare Chrome branded (serve sudo) per testare `shader-f16` su NVIDIA — sblocca le quant q4f16_1 sulla 4090.
3. Punteggio decode `null` vs `0` con <2 token: adjudicato dal controller (null); ratificare o ribaltare.
4. Guardia `completionTokens >= 2` + prosa piano allineata + probe never-throws + erasableSyntaxOnly: tre adjudication del controller in sessione, tutte documentate nel ledger — ratifica implicita se nessuna obiezione.
