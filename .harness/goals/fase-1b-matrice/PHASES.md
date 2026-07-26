# PHASES — fase-1b-matrice

Derivato da `docs/superpowers/specs/2026-07-26-fase-1b-matrice-design.md` §Ordine di
implementazione. Fasi sequenziali (nessun `parallel-group`: le fasi 1-3 toccano tutte
`src/schema.ts`, quindi non hanno `owns` disgiunti).

| # | phase | done-when (mechanical) | authority delta | owns | status |
|---|-------|------------------------|-----------------|------|--------|
| 1 | Adapter Transformers.js | `npm test` verde (incl. nuovo conformance test transformersjs) + `tsc --noEmit` pulito + `npm run build` ok + almeno 1 run reale (non-SwiftShader) in `results/*.json` con `stack:"transformersjs"` | none | `src/adapters/transformersjs.ts`, relativo test, `src/schema.ts` (solo union `id`/`stack`), `src/main.ts`/`src/render.ts` (selettore stack minimo), `package.json` (dep `@huggingface/transformers`) | **done*** |
| 2 | Adapter wllama | `npm test` verde (incl. nuovo conformance test wllama) + `tsc --noEmit` pulito + `npm run build` ok + almeno 1 run reale in `results/*.json` con `stack:"wllama"` | none | `src/adapters/wllama.ts`, relativo test, `src/schema.ts` (estensione union), `package.json` (dep `wllama`) | ready |
| 3 | Modulo qualità + schema v3 completo | `npm test` verde (unit test perplexity + fallback 12-prompt) + `tsc --noEmit` pulito + `BenchCell.qualityScore` presente e tipato nello schema | none | `src/quality.ts`, `src/qualityPrompts.ts`, `src/schema.ts` (campo `qualityScore`, bump `SCHEMA_VERSION=3`), relativi test | ready |
| 4 | README + verifica finale whole-branch | Sezione "Fase 1b — matrice" nel README con gap Large documentato + suite completa verde (baseline 39 + nuovi test) + `tsc`/`build` puliti + entrambi i run (transformersjs, wllama) presenti in `results/` | none | `README.md` | ready |

**\* Fase 1 — COMPLETA** (2026-07-26), riserva chiusa. Condizioni verdi: `npm test` 47/47,
`tsc --noEmit` pulito, `npm run build` ok, run reali `stack:"transformersjs"` e `stack:"webllm"` in
`results/`, e **`npm run test:conformance` 8/8 + 8/8 (exit 0)** in browser reale sulla 4090.

La riserva era la clausola "incl. nuovo conformance test": i test unit iniettano tutti un engine
fake, quindi non caricavano un modello né asserivano su `capabilities()`. Ora c'è il contratto
condiviso (`src/conformance/contract.ts`) eseguito nel browser su entrambi gli adapter.
**Verificato che sappia fallire** (mutation test, 2026-07-26): reintrodotto a mano il bug
`callback_function`/`token_callback_function` → il contratto FALLISCE come deve
(`expected 16 timestamps, got 10`, exit 1) e isola l'adapter giusto (webllm resta 8/8). Mutazione
revertata e riverificato verde. Cattura in ~1 minuto il difetto che in Fase 1 era costato un run
GPU completo per essere scoperto.

**DONE-WHEN EMENDATO** (2026-07-26, ruling PI in docket #2): la clausola "verifiable via `npm test`"
di GOAL.md e di questa riga non è realizzabile per tutti gli adapter — WebLLM richiede WebGPU e non
gira in Node. Sostituita da: *ogni adapter passa lo stesso contratto di conformance eseguito **nel
browser** da uno script Playwright on-demand (`npm run test:conformance`)*; `npm test` resta la suite
unit veloce e offline. Un solo meccanismo per tutti gli stack, già pronto per wllama in Fase 2.

Nota: l'esecuzione dello sweep manuale sui 3 device (M4/S22) resta **fuori da queste fasi**
(fuori scope del goal, per costruzione — vedi GOAL.md "must docket"). Il merge di
`feat/fase-1b-matrice` in `main` è docket-gated, non una fase autonoma.
