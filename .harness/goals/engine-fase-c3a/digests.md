# Digests — engine-fase-c3a

## it.1 (2026-08-01) — fase 1 DONE, fase 2 bloccata da ruling

- **Fatto**: telemetria di attribuzione nel motore GLM (opt-in, zero-overhead da
  spenta; timestamp-query su ogni pass, `mapAsync` dopo il submit come da
  root-cause di fase A) + TTFT e gap dalla funzione obiettivo nel report di
  bench + probe indipendente del floor di sync. Bench ri-eseguito a macchina
  quiescente: `results/engine/bench-glm-4090-b12-attrib-2026-08-01.json`.
- **Difetto trovato e corretto dentro l'iterazione**: il check di identità che
  ho aggiunto ha scoperto un double-count (`encodeCpuMs` includeva `ensureMs`).
  Corretto e ri-misurato; ora l'identità chiude allo 0.3%.
- **Il numero che cambia il goal**: wall 215.0 ms/token = `gpuBusy` 78.2 +
  stallo 53.8 + sync/CPU 83.0 ⇒ 63.6% fuori dalla GPU, ma i readback veri
  valgono solo 7.6 ms/token (probe). Con le leve 1 e 2 del contratto al 100%:
  **10.18 tok/s** contro un gate 13.43 che richiede ≤74.46 ms/token — e
  `gpuBusy` da solo ne occupa 78.2. **Le leve del contratto non bastano.**
- **Ma la GPU è sotto-clockata dalle bolle** che la leva 2 rimuove: 1746 MHz
  medi su 3105 di cap, utilizzo 34.6%. Al limite ottimistico ⇒ 15.63 tok/s,
  gate PASS. **Forbice 10.2-15.6, col gate dentro.**
- **Cross-check indipendente**: 2.22 GB di pesi/token su 576 GB/s reali ⇒ floor
  memory-bound 3.85 ms/token; `gpuBusy` è **20× sopra** (43 µs per dispatch su
  1816 dispatch/token). Quarta leva disponibile — granularità dei dispatch —
  ma è kernel engineering, ultima per direction §2.
- **Non regredito**: decode headline 4.678 (C2: 4.640), prefill 5.235 (5.221),
  `npm test` 220 passed + 2 skipped, `tsc --noEmit` pulito.
- **TTFT misurato per la prima volta: 88.06 s** contro il budget UX di 4 s
  (gap 22×). Il decode è a 6.41× dai 30 tok/s.
- **Prossimo**: STOP by design. La fase 2 (spec) non è decidibile senza il
  ruling di docket item 4 (quale ramo assumere) — e la risposta cambia il
  meccanismo che la spec deve scegliere. Aperti anche item 2 (clausola di
  fallback, ora quantificata) e item 5 (correzione di forma della riga PHASES).
- **Gate di protocollo scoperto**: verifier indipendente non lanciato (la policy
  di sessione vieta subagent non richiesti). Verifica fatta in proprio e
  meccanica; segnalato, non aggirato.

## it.2 (2026-08-01) — fase 2 DONE, ruling di spec pendente

- **Ruling recepiti**: quarta leva nel perimetro (GOAL emendamento 1), fase 4b
  in PHASES, clausola di fallback rimandata a fine fase 4, riga fase 1 corretta.
  Della (a) scartata ho tenuto il pezzo utile: la fase 4 chiude con una
  **ri-misura obbligatoria di clock e `gpuBusy`**, che dimensiona la 4b con un
  fatto invece che con l'assunzione ottimistica.
- **Spec scritta**: `docs/superpowers/specs/2026-08-01-engine-fase-c3a-design.md`,
  8 sezioni, budget per leva che somma esattamente al gate (74.4 ms/token).
- **La spec corregge il contratto su un punto di sostanza**: il contratto dice
  "eliminare i 46 readback", ma la misura dice che i readback valgono 7.6 degli
  83 ms/token — il resto è **frammentazione dei submit**. Quindi il criterio
  diventa **minimizzare i submit, non i readback**.
- **Pipelining scartato esplicitamente** (era una delle tre vie del contratto):
  nel decode il token t+1 dipende da t e il layer l+1 da l. Resta per il
  prefill e per la fase D.
- **[VERIFY] aperto**: `maxStorageBufferBindingSize` e `maxBufferSize` reali non
  sono mai stati letti (il codice negozia `min(limite, 2 GiB)` senza guardarlo).
  La via raccomandata per la leva 2 dipende da quel numero ⇒ probe da ~20 righe
  come primo task della fase 4.
- **Conto che cambia le aspettative sul prefill**: con M=16 il TTFT scenderebbe
  verso ~5-6 s, ancora sopra i 4 s. Il batching da solo non basta.
- **Prossimo**: STOP by design, ruling di spec a docket item 6 (5 decisioni).
