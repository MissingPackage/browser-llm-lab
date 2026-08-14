# DIGESTS — engine-kquant

Tre-sei righe per iterazione. Chi legge solo questo file deve sapere dove sta
il goal e cosa e' in dubbio.

## it.0 — la spina (2026-08-14)

- Contratto chartered: doppia barra sul TTFT (**< 22.500 obbligatoria**,
  **< 18.000 nice to have**), segmenti `gemm:deltanet-out` e `gemm:ffn-down`
  entrambi **≤ 2.000 ms**, copertura del piano da 5,86× a **≥ 15,5×**.
- `PHASES.md`: sette righe. Fase 0 al banco (riga 1) → Q5_K in produzione
  (riga 2, −11,4 s proiettati) → Q4_1 (riga 3, −3,3 s) → le tre forme del 35B
  misurate ma NON cablate (riga 4) → misura di chiusura → gate → consuntivo.
- Quattro correzioni di fattibilita' trovate PRIMA di scrivere la tabella: lo
  split-K conta in blocchi da 32 e i K-quant hanno superblocchi da 256; la
  quantizzazione delle attivazioni si riusa tale e quale; serve un termine Σx
  che la q4_0 non ha; una cella del banco (35B down, K=512) sarebbe degenere a
  4 fette.
- Deciso da me: la famiglia di kernel si fa intera qui, il **cablaggio** del
  35B e' il goal successivo — non ha un byte di q4_0, ma il suo collo e' la
  residency, non il kernel.
- In dubbio: niente, per ora. Il plan-check e' registrato come approvato in
  modo implicito («procedi in loop»), e il primo digest e' il momento per
  cambiarlo.
