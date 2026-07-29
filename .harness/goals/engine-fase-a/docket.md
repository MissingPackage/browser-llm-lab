# Docket — engine-fase-a

Decisioni PI pendenti del goal. Append-only; le decisioni prese migrano in GOAL/spec/journal.

1. **plan-check** (2026-07-29, iterazione 0): PHASES.md scritto — richiesta approvazione
   PI della decomposizione prima dell'iterazione 1 (gate di goal-setup per goal product).
   In particolare: (a) fasi 1+2 in parallelo ok? [di fatto superata: fase 2 eseguita,
   era contrattuale] (b) separazione correttezza-poi-velocità (fase 4 naive → fase 5
   L1-L3) ok, sapendo che allunga la strada al first-light di ~una fase ma misura le
   leve sul motore nostro?
2. **Ruling spec fase A** (2026-07-29, iterazione 2):
   `docs/superpowers/specs/2026-07-29-engine-fase-a-design.md` — decisioni chiave da
   approvare: solo Q4_0 in fase A; corpus di conformance in token-id (tokenizer
   rimandato); top-1 ≥99% come gate + agreement@8 e Δlogit riportati non gated; golden
   via llama-cpp-python; budget ≤100 dispatch con fallback attention dichiarato
   (rinegoziazione via docket); telemetria a 4 livelli con diagnostica mai-default.
   Sblocca le fasi 3-5.
