# Piano di split dei repo (deciso 2026-08-09, esecuzione ALLA PUBBLICAZIONE)

Ruling PI 2026-08-09 (chat): lo split in 3 repo si fa SOLO alla
pubblicazione — ricreare 3 harness ora è più costoso che tenere il lab
(bench app e harness engine condividono la build vite; separare ora =
chirurgia build + ri-validazione harness GPU). Parametri CONGELATI:

## I tre repo (privati alla nascita, pubblici alla release)

| Repo | Contenuto (path del lab) | Storia |
|---|---|---|
| `browser-llm-engine` | src/engine/**, tools/**, scripts/* (engine), tests/engine-*, docs/engine/**, results/engine/**, results/opfs-bench/** | filter-repo per path |
| `browser-llm-bench` | src/* (root: app bench), src/{adapters,conformance,microbench,prof}, results/{4090-linux-*,m4-pro*,s22-ultra*,microbench,dispatch-profile,methodology}, docs/deep-dive/** | filter-repo per path |
| `browser-llm-paper` | paper/** (nasce ORA nel lab proprio perché filter-repo lo estragga poi con la sua storia) | filter-repo per path |

## Regole d'esecuzione (quando si farà)

1. `git filter-repo` su clone freschi, MAI sul lab (che resta archivio
   privato intatto, con .harness/.claude/processo).
2. I mirror pubblici NON contengono `.harness/`, `.claude/`,
   `docs/superpowers/` (processo privato) — vanno in --path-rename/exclude.
3. Zone condivise: `results/opfs-bench` → engine (il cold-read è materia
   engine); `scripts/lib/hoststate.mjs` → copiato in entrambi;
   `package.json`/vite → riscritti per-repo (il lavoro rimandato).
4. Licenza motore: da scegliere alla release (Apache-2.0 / MIT — docket).
5. La visibilità diventa pubblica SOLO su ruling PI esplicito, repo per
   repo (il bench resta in standby anche da splittato).
