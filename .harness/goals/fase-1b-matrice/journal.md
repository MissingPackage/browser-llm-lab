# Journal — fase-1b-matrice

## Iterazione 1 (2026-07-26)

- Branch `feat/fase-1b-matrice` creato da `main` (post plan-check approvato).
- Scritto e committato il piano di implementazione per la Fase 1 (adapter Transformers.js):
  `docs/superpowers/plans/2026-07-26-fase-1b-matrice-adapter-transformersjs.md` (commit
  `848cee3`). 4 task: (1) `TransformersJsAdapter` + `StackId` + dep `@huggingface/transformers`,
  (2) `BenchServer` multi-stack routing (registry `Record<StackId, factory>`, protocol esteso
  con `stack`), (3) UI stack selector con filtro modelli, (4) driver e2e `STACK` env + run reale
  4090 + commit risultato in `results/`.
- Ricerca API Transformers.js fatta via skill dedicato (`huggingface-skills:transformers-js`,
  non a memoria) prima di scrivere codice nel piano — pattern DI identico a `WebLLMAdapter`
  (engine/isCached/now iniettabili) per tenere i test unitari lontani da rete/GPU reali.
- **Nota corretta durante la scrittura**: la spec parla di "Transformers.js v3", ma npm registry
  (controllato 2026-07-26) mostra l'ultima pubblicata è `4.2.0` (v3 fermo a `3.8.1`). Stessa API
  (`pipeline`/`TextStreamer`), solo la label di versione è invecchiata — non uno scope change,
  loggato nel piano (Global Constraints) e qui.
- Non ancora eseguito: in attesa della scelta di esecuzione (subagent-driven vs inline) da
  Cristiano.
