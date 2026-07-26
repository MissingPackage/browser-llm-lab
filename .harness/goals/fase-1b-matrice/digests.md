# Digests — fase-1b-matrice

## Iterazione 1 (2026-07-26)
Scritto il piano di implementazione della Fase 1 (adapter Transformers.js): 4 task TDD su
`feat/fase-1b-matrice`. API Transformers.js verificata via skill dedicato, non a memoria; corretta
una nota di versione della spec (dice "v3", npm pubblica 4.2.0 — stessa API). Nessun codice ancora.

## Iterazione 2 (2026-07-26)
Fase 1 eseguita subagent-driven e completata: adapter Transformers.js dietro l'`InferenceAdapter`
esistente, routing per-stack nel worker, selettore stack in UI, run reali sulla 4090. Gate verdi
(47/47, tsc, build). Due bug trovati dalle review e corretti, entrambi di misura: il piano cablava
il callback per-parola invece che per-token del `TextStreamer` (avrebbe contato parole come token),
e il default `stdout_write` della libreria stampava 675 righe dentro la finestra cronometrata del
decode (contaminazione in un bench comparativo). Il final review ha poi trovato che `quant` era
un'etichetta libera non falsificabile per il nuovo stack — ora c'è un guard che rifiuta il mismatch.

**Risultato**: primo confronto cross-stack difendibile — WebLLM 113.0 ±2.5 tok/s vs Transformers.js
55.5 ±2.1 tok/s (~2.04×) sulla stessa GPU, stesso modello, stessa sessione, condizioni equivalenti
verificate asse per asse dal review.

**Fermo per decisione PI**: docket #2 (il done-when chiede un conformance test che non è stato
scritto — gap simmetrico anche su WebLLM) e docket #3 (mergiare la sola Fase 1 o attendere fine goal).

## Iterazione (2026-07-27) — Fase 3
Fase 3 completata: modulo qualità (`src/quality.ts` + `src/qualityPrompts.ts`, 12 prompt
deterministici in 4 categorie + perplexity), schema bump a v3 con `BenchCell.qualityScore`
(opzionale, non ancora collegato a un run reale — vedi docket #10) e `BenchCell.protocol`
(docket #7, sostituisce l'abuso di `anomalies` per la nota di warm-up). Gate verdi: `npm test`
87/87, `tsc --noEmit` pulito, `npm run build` ok.

**Prossimo**: Fase 4 (README + verifica finale). Prima di dichiararla completa, decidere se
collegare `quality.ts` alla pipeline reale (docket #10) — altrimenti il README riporterà solo
metriche di velocità, come oggi.
