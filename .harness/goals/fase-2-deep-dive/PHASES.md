# PHASES — fase-2-deep-dive

Decomposizione dell'iterazione 0 (2026-07-27). Sequenziale per costruzione (ruling PI:
branch unico `feat/fase-2-deep-dive`, ogni doc si appoggia sul precedente). Nessun
`parallel-group`.

Regola trasversale (dal contratto): i doc possono portare marker `[VERIFY]` durante le
fasi, ma il goal non chiude finché `grep -c "\[VERIFY\]" docs/deep-dive/*.md` non torna 0
(sweep in fase 7). Gli esperimenti di fattibilità (max 2 in tutto il goal) non hanno una
fase propria: nascono docket-born dentro le fasi 2–6 quando il passaggio creativo produce
un'idea a basso costo — l'autorità per eseguirli è già nel contratto, il tetto pure.

| # | phase | done-when (mechanical) | authority delta | owns | status |
|---|-------|------------------------|-----------------|------|--------|
| 1 | Skill `bottleneck-brainstorm` v1 (via writing-skills) | `.claude/skills/bottleneck-brainstorm/SKILL.md` esiste con frontmatter valido (name+description, grep); istruzioni contengono input/processo/output come da spec §skill; commit sul branch | none | `.claude/skills/bottleneck-brainstorm/` | done (2026-07-27, commit d79bc32; TDD: RED 2 run + GREEN 1 run, artefatti in baseline/) |
| 2 | Doc `compute-shader-dispatch.md` + dogfood skill | file esiste con heading letterali "Cosa fa", "Perché i numeri sono quelli", "Bottleneck & vie d'uscita" (grep); ≥1 citazione a `results/*.json` e ≥1 al bundle con versione pacchetto (grep); invocazione skill registrata in journal.md; eventuali raffinamenti skill committati | none | `docs/deep-dive/compute-shader-dispatch.md` | done (2026-07-27, iterazione 2; dogfood ok, nessun raffinamento) |
| 3 | Doc `buffer-limit-2gb.md` | stessi check meccanici della fase 2 (heading, citazioni, journal skill) sul file `buffer-limit-2gb.md` | none | `docs/deep-dive/buffer-limit-2gb.md` | done (2026-07-27, iterazione 3) |
| 4 | Doc `dequant-kernels.md` | stessi check meccanici della fase 2 sul file `dequant-kernels.md` | none | `docs/deep-dive/dequant-kernels.md` | done (2026-07-27, iterazioni 4-5: dump WGSL + doc) |
| 5 | Doc `kv-cache-layout.md` | stessi check meccanici della fase 2 sul file `kv-cache-layout.md` | none | `docs/deep-dive/kv-cache-layout.md` | done (2026-07-27, iterazione 6) |
| 6 | Micro-bench matmul: route SPA + run 4090 + doc | `npm run build` exit 0; `npm test` verde incl. unit test sulla matematica delle metriche; ≥1 JSON schema-versionato in `results/microbench/` da run reale 4090 (non SwiftShader — check `adapterInfo.vendor`); `docs/deep-dive/micro-bench-matmul.md` esiste con metodologia + numeri 4090 + slot M4/S22 marcati pending (grep "pending") | none | route SPA nuova (`src/microbench/` o equivalente), `docs/deep-dive/micro-bench-matmul.md`, `results/microbench/` | ready (parte 1 done, iterazione 7: motore+test+run 4090 valido; resta il doc) |
| 7 | `engine-design-notes.md` + closure sweep | file esiste e contiene i 5 filename degli altri doc (grep); `grep -c "\[VERIFY\]" docs/deep-dive/*.md` = 0; `npm test` + `tsc --noEmit` + `npm run build` verdi; `git diff goal-fase-2-start -- src/adapters/webllm.ts` vuoto; `ls experiments/` ≤ 2 entry | none | `docs/deep-dive/engine-design-notes.md` | ready |

Sizing: fasi 1 e 3–5 stimate 1–2 iterazioni; fase 2 (prima col dogfood della skill) e
fase 6 (unica con codice SPA) stimate 2–4 iterazioni; fase 7 stimata 1 iterazione.

Dopo la fase 7 il lavoro residuo è tutto docket-gated per costruzione: merge a `main`,
run manuali M4/S22, decisione promozione skill all'harness personale, eventuale terzo
esperimento. Stop-by-design.
