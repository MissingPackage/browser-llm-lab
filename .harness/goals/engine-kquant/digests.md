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

## it.1 — la fase 0 autorizza entrambe le famiglie (2026-08-14)

- **Metrica obiettivo ferma a 32.127 ms per costruzione**: la riga 1 non tocca
  il motore. Ha pero' potere di veto, e non l'ha esercitato.
- **Q5_K 28,07x · Q4_1 22,58x** sulla forma di produzione, a M=16, con zero
  celle scartate dal gate del banco. La regola di stop chiedeva 1,5x.
- **Il banco riproduce il segmento vero in millisecondi**: `24 x 395 x 1,2700 =
  12.039` contro i **12.169 ms** misurati in produzione, 1,1% di scarto. Il
  termine di paragone e' il percorso vecchio, non una sua imitazione.
- Proiezione **−15,2 s ⇒ TTFT ~16,9 s**, sotto anche la barra nice-to-have. E'
  una proiezione da microbench: la conferma sta nella riga 5.
- **In dubbio, e ora ha un'azione**: la quota Q4_1 del segmento `gemm:ffn-down`
  e' DEDOTTA, non misurata (il mio confronto era banco contro banco). La riga 3
  non chiude senza una categoria di misura propria per quei quattro siti.
- Verificatore indipendente: **PASS** con sei correzioni, tutte applicate —
  fra cui `396 → 395` chunk, che stava anche nel contratto.
