# Journal — engine-fase-a

2026-07-29 — Iterazione 4 (nucleo comune fase 4/5, su branch): (a) cpuref.ts — forward
CPU f64 con parità PIENA vs oracolo (16/16 smoke); (b) kernel WGSL completi + ktest
page: 11/11 PASS su 4090 al primo colpo; (c) gpuforward.ts — motore GPU end-to-end,
conformance 512 posizioni: **top-1 98.05%**, 8.5 ms/forward, 412 dispatch/token non
fusi. Bring-up bug memorabile: dispatchWorkgroups(151936) > limite 65535/dim ⇒ command
buffer rigettati IN SILENZIO (top-1 0.2%, zero errori) → griglia 2D + uncapturederror
fatale nel motore. I 10 mismatch sono tutti near-tie (margini 0.004-0.39, 9/10 rank-1
golden): l'oracolo CPU quantizza le attivazioni a Q8_0 (algoritmo ≠ f32 puro).
Calibrazione ESEGUITA: cpuref f64 vs golden = 502/512 con gli STESSI 10 mismatch del
motore GPU ⇒ **motore GPU = matematica esatta al 100% (512/512)**; il 98.05% è il noise
floor dell'oracolo (vec_dot q4_0×q8_0 quantizza le attivazioni). Proposta gate doppio
nel docket (item 3). Tap hidden-states implementati e verificati (taps=[11], 896 f32
non-zero, conformance invariata). NOTA per il plan-check: nel motore nostro L1+L2
sono risultati GRATIS by construction (buffer statici ⇒ bind group al load, 1
submit/token naturale) — la domanda naive-vs-diretto si riduce di fatto alla sola
fusione L3.

Stato DONE WHEN a fine 2026-07-29: spec scritta (ruling ✗); npm test verde ✓;
conformance harness fatto — 98.05% vs gate 99% (pende docket 3: con la proposta il
gate oggi PASSEREBBE: vs cpuref 100%, vs golden ≥97%); first-light ✗ (gated);
L1/L2 by construction, L3 ✗ (gated); telemetria ✗ (fase 5); tap ✓; OPFS ✓; M2 ✓.
Tutto il residuo pende dai docket 1-3.

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
