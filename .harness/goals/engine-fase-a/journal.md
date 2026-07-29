# Journal — engine-fase-a

2026-07-29 — Iterazione 3 (fase 3, su branch engine/fase-a, avviata pre-ruling — vedi
docket 2): gguf.ts (parser v3, subset F32/F16/Q4_0/Q8_0), quant.ts (dequant reference
esatta Q4_0/Q8_0 + f16), shape.ts (inventario 291 tensori VERIFICATO sul file reale —
due correzioni alla spec: output.weight separato Q8_0, bias q/k/v F32), 119/119 test
(fixture sintetica CI + validazione file reale skipIf), gen-golden.py (llama-cpp-python
0.3.16 pinnato) → corpus 4 prompt token-id + golden 512 posizioni (argmax+top-32,
sha256 del GGUF registrato). Sanity: output oracolo sensati (p3 → "Paris").
Fasi 4-6 FERME in attesa dei due ruling (il plan-check 1b decide proprio la forma
della fase 4).

2026-07-29 — Goal aperto. Contratto approvato dal PI in chat (goal-brief, [ASSUMED]
approvati in blocco). Direction: docs/engine/direction.md.

2026-07-29 — Iterazione 0 (goal-setup): PHASES.md scritto (6 fasi, gruppo parallelo
1+2, spine correttezza-poi-velocità 3→4→5, first-light in fase 6). plan-check appeso al
docket: STOP by design in attesa dell'approvazione PI della decomposizione.

2026-07-29 — Iterazione 2 (fase 1): spec scritta
(specs/2026-07-29-engine-fase-a-design.md, 6 sezioni contratto verificate via grep).
Decisioni proposte: Q4_0-only, corpus token-id, top-1≥99%, golden llama-cpp-python,
budget ≤100 dispatch con fallback dichiarato, telemetria 4 livelli. Ruling richiesto
(docket 2). Con fase 2 già chiusa, TUTTO il lavoro residuo è PI-gated: plan-check
(docket 1) + ruling spec (docket 2) ⇒ stop-by-design da working protocol.

2026-07-29 — Iterazione 1 (fase 2 SOLTANTO — contrattuale, indipendente dalle domande
del plan-check): opfs-bench.mjs (write 2.2 GB/s, read API 7.5-11.7 GB/s warm — API non
bottleneck) + submit-callsites.mjs (M2: 7 submit/token = 2 free + 2 upload + 1 copyTo +
2 readback; 5 eliminabili, 2 fondibili ⇒ 1 submit/token per costruzione). estimates §6,
ledger A, direction §8.3 aggiornati. Fase 2 DONE per contratto. Fase 1 e spine 3→6
restano in attesa del plan-check.
