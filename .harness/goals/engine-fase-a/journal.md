# Journal — engine-fase-a

2026-07-29 — Goal aperto. Contratto approvato dal PI in chat (goal-brief, [ASSUMED]
approvati in blocco). Direction: docs/engine/direction.md.

2026-07-29 — Iterazione 0 (goal-setup): PHASES.md scritto (6 fasi, gruppo parallelo
1+2, spine correttezza-poi-velocità 3→4→5, first-light in fase 6). plan-check appeso al
docket: STOP by design in attesa dell'approvazione PI della decomposizione.

2026-07-29 — Iterazione 1 (fase 2 SOLTANTO — contrattuale, indipendente dalle domande
del plan-check): opfs-bench.mjs (write 2.2 GB/s, read API 7.5-11.7 GB/s warm — API non
bottleneck) + submit-callsites.mjs (M2: 7 submit/token = 2 free + 2 upload + 1 copyTo +
2 readback; 5 eliminabili, 2 fondibili ⇒ 1 submit/token per costruzione). estimates §6,
ledger A, direction §8.3 aggiornati. Fase 2 DONE per contratto. Fase 1 e spine 3→6
restano in attesa del plan-check.
