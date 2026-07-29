# Docket — engine-fase-b1 (decisioni PI pendenti)

2. **Dato per B2** (2026-07-29, fase 2 — osservazione fuori scope, non perseguita):
   col liv.2 funzionante, GPU busy ≈ 2.2 ms/token vs 8.1 ms/token wall di decode ⇒
   ~73% del tempo di decode è fuori GPU (readback argmax sincrono + encode). Il
   contratto B2 "floor dispatch ≤100" va probabilmente ri-inquadrato su questo dato
   (la leva più grossa potrebbe essere il sync, non il dispatch count). Da portare
   nel goal-brief di B2.

1. ~~**plan-check**~~ RISOLTO (2026-07-29, ruling PI in chat: "sul resto tutto ok"):
   PHASES.md approvato. Nello stesso ruling: soglia prefill da fissare con
   simulazioni + criterio di Pareto → eseguito (journal it.1), soglia = 3×,
   GOAL.md aggiornato. Fasi 1-2 sbloccate.
