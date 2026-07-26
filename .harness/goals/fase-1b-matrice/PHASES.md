# PHASES — fase-1b-matrice

Derivato da `docs/superpowers/specs/2026-07-26-fase-1b-matrice-design.md` §Ordine di
implementazione. Fasi sequenziali (nessun `parallel-group`: le fasi 1-3 toccano tutte
`src/schema.ts`, quindi non hanno `owns` disgiunti).

| # | phase | done-when (mechanical) | authority delta | owns | status |
|---|-------|------------------------|-----------------|------|--------|
| 1 | Adapter Transformers.js | `npm test` verde (incl. nuovo conformance test transformersjs) + `tsc --noEmit` pulito + `npm run build` ok + almeno 1 run reale (non-SwiftShader) in `results/*.json` con `stack:"transformersjs"` | none | `src/adapters/transformersjs.ts`, relativo test, `src/schema.ts` (solo union `id`/`stack`), `src/main.ts`/`src/render.ts` (selettore stack minimo), `package.json` (dep `@huggingface/transformers`) | ready |
| 2 | Adapter wllama | `npm test` verde (incl. nuovo conformance test wllama) + `tsc --noEmit` pulito + `npm run build` ok + almeno 1 run reale in `results/*.json` con `stack:"wllama"` | none | `src/adapters/wllama.ts`, relativo test, `src/schema.ts` (estensione union), `package.json` (dep `wllama`) | ready |
| 3 | Modulo qualità + schema v3 completo | `npm test` verde (unit test perplexity + fallback 12-prompt) + `tsc --noEmit` pulito + `BenchCell.qualityScore` presente e tipato nello schema | none | `src/quality.ts`, `src/qualityPrompts.ts`, `src/schema.ts` (campo `qualityScore`, bump `SCHEMA_VERSION=3`), relativi test | ready |
| 4 | README + verifica finale whole-branch | Sezione "Fase 1b — matrice" nel README con gap Large documentato + suite completa verde (baseline 39 + nuovi test) + `tsc`/`build` puliti + entrambi i run (transformersjs, wllama) presenti in `results/` | none | `README.md` | ready |

Nota: l'esecuzione dello sweep manuale sui 3 device (M4/S22) resta **fuori da queste fasi**
(fuori scope del goal, per costruzione — vedi GOAL.md "must docket"). Il merge di
`feat/fase-1b-matrice` in `main` è docket-gated, non una fase autonoma.
