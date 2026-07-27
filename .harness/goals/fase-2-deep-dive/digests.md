# Digests — fase-2-deep-dive

## Iterazione 0 (2026-07-27) — setup

- Goal spine scaffoldato; contratto in GOAL.md, spec in
  `docs/superpowers/specs/2026-07-27-fase-2-deep-dive-design.md`.
- 7 fasi sequenziali: skill → 4 doc sotto-sistema (dogfood skill dalla fase 2) →
  micro-bench matmul (unica fase con codice SPA) → engine-design-notes + closure sweep.
- Primo target: fase 1, skill `bottleneck-brainstorm` v1 via writing-skills.
- Docket-born: plan-check (#1, pre-autorizzato condizionalmente in chat); gli esperimenti
  di fattibilità (max 2) nasceranno docket-born dentro le fasi 2–6.
- Stop-by-design dopo fase 7: merge, run M4/S22, promozione skill, terzo esperimento —
  tutto PI-gated.

## Iterazione 1 (2026-07-27) — fase 1 done

- Skill `bottleneck-brainstorm` v1 creata via TDD (2 RED + 1 GREEN, artefatti in baseline/),
  commit d79bc32 su feat/fase-2-deep-dive. Done-when fase 1 verde.
- Scoperta metodologica: la baseline non fallisce in capacità ma in forma → skill = contratto
  di output (tabella 5 colonne + instradamento esperimento/engine-notes/scartata).
- 3 findings load-bearing gratis dai run di test (1 GiB hardcoded, shader-f16 inutilizzato
  su S22, KV sized sul max) — salvati in baseline/, alimentano fasi 3-5 e docket #2.
- Verifier: FAIL su bookkeeping alla prima passata, sanato in-iterazione; re-check ok.
- Prossimo: fase 2, doc compute-shader-dispatch.md (primo dogfood ufficiale della skill).
