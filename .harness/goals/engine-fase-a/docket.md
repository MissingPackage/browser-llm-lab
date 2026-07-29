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
   Sblocca le fasi 4-5. [Agg. iterazione 3: spec corretta dopo il parse del GGUF reale —
   "Q4_0-only" diventa Q4_0+Q8_0+F32 (output.weight separato in Q8_0, bias q/k/v F32,
   niente tied embeddings nel file). Fase 3 avviata pre-ruling: authority `src/engine/**`
   su branch è esplicita nel contratto e gli artefatti (parser, dequant reference,
   golden) sono invarianti rispetto a entrambi i ruling pendenti; nessun merge.]
3. **Correzione del gate di conformance con evidenza di calibrazione** (2026-07-29,
   iterazione 4). Misurato: (a) motore GPU vs golden llama.cpp = 502/512 (98.05%);
   (b) cpuref f64 (matematica esatta) vs stesso golden = 502/512 con gli STESSI
   identici 10 mismatch (posizioni e token); (c) ⇒ motore GPU vs cpuref = 512/512.
   Il 99% del contratto è sopra il noise floor dell'oracolo (il suo vec_dot q4_0×q8_0
   quantizza le attivazioni: algoritmo diverso, ±0.4 logit sui near-tie, maxΔ 1.12).
   PROPOSTA: gate doppio — top-1 vs cpuref-f64 ≥ 99% (parità vera, oracolo-indipendente;
   oggi 100%) E top-1 vs golden llama.cpp ≥ 97% (sanity; oggi 98.05%). In alternativa:
   golden rigenerato da un oracolo f32-full-precision. Decidere prima del gate di fase.
