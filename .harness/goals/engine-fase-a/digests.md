# Digests — engine-fase-a

## Iterazione 0 (2026-07-29, goal-setup)

6 fasi: spec ∥ misure-di-contorno → GGUF+oracolo → forward naive con parità → L1-L3+
telemetria+tap → first-light. Spine deliberato: correttezza prima della velocità, così
le leve L1-L3 si misurano come delta sul motore nostro. Primo target dopo il plan-check:
fasi 1+2 in parallelo. Docket-born: plan-check (item 1); il merge su main resta fuori
dalle fasi by design.

## Iterazioni 1-6 (2026-07-29, ciclo unico esteso)

Fasi 1-6 eseguite su branch; FIRST LIGHT 123.0 vs WebLLM 116.5-117.8 same-day; parità =
matematica esatta; contatori 123/1/0 verificati da profiler esterno; telemetria −0.55%
overhead. VERIFIER GATE: PASS su 8/8 claim (agent loop-verifier, evidenza per claim nel
transcript; nota: due formati di conformance JSON nello stesso prefisso — l'ultimo per
nome è quello valido). Goal gated sui ruling PI docket 2-5: "serve input umano, non
altra esecuzione".
