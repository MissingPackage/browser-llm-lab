# Docket — engine-fase-b1 (decisioni PI pendenti)

3. **RULING RICHIESTO — spec B1** (2026-07-29, fase 1): approvazione di
   `docs/superpowers/specs/2026-07-29-engine-fase-b1-design.md`. Decisioni proposte
   da ratificare: (a) chiave prefix-cache su token-id (non testo: niente tokenizer,
   la chiave testuale ds4 esiste per la ritokenizzazione BPE); (b) lookup v1 = match
   esatto full-prefix, longest-prefix rimandata a fase C; (c) niente logits nel
   checkpoint (si ricalcola 1 forward, ~8 ms); (d) eviction LRU semplice (niente
   scoring a densità ds4 in v1); (e) contratto hard `pos === kvLen` su forwardToken
   (breaking per chi usava pos libere — i runner del repo usano già reset+sequenziale);
   (f) budget disco default 512 MB; (g) soglie §Soglie (prefill 3× già PI-ruled,
   resto = non-regressione fase A). Le fasi 3-6 restano blocked fino al ruling.

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
