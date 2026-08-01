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
