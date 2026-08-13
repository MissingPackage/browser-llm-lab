# DIGESTS — engine-ttft

## it.0 (2026-08-13) — decomposizione

- Goal aperto: TTFT a caldo (`prefill.ms + decode.firstMs`) ≤ 4 s sul prompt-idx
  0 (6333 token). Ruling A del PI: il load (10,89 s) ha una soglia sua, fuori
  scope.
- Il numero di partenza di HANDOFF era sbagliato due volte: prompt da 388 token,
  non 6k; e `ttftMs` include il load. Il TTFT a 6k non è mai stato misurato.
- 7 righe in PHASES. Prima misura, poi decisione, poi due riscritture: riga 0
  baseline onesta (due bracci, sequenziale vs `prefillChunk` M=16), riga 1 sonde
  prediction-gated con potere di chiudere il goal, righe 2-3 GEMM multi-riga e
  attenzione del prefill, riga 4 portabilità, 5 checkpoint, 6 chiusura.
- Il test di fattibilità (C7) ha cambiato tre done-when del contratto: il bench
  non usa il prefill a chunk che ha già in albero; M=8 è degenere per
  l'obiettivo (serve M ≥ 16); la clausola (e2a) è sul path 0.5B, non sul 4B.
- Docket-born: 5 item, di cui 3 chiedono un ruling (plan-check con gli
  scostamenti, la casa dello spec-dec MTP, la severità del controllo su
  `hostState`). Item 3 e 5 sono lavoro mio, ereditati dal goal precedente.
- STOP by design: `plan-check` aperto, l'iterazione 1 aspetta il sì del PI.

## it.1 (2026-08-13) — riga 0 chiusa: la baseline esiste, e dice 21,9×

- **Traiettoria**: l'obiettivo era senza numero (mai misurato), ora è **87.618 ms**
  di tempo al primo token a modello caldo contro un bersaglio di 4.000. Serve
  **21,9×**. Ciò che lo muove è la riga 1: decidere quali forme di kernel esistono.
- Il prompt da 6333 token non era mai stato misurato: ora lo è, coi tre tempi
  separati (caricamento 10,9 s · lettura del prompt 87,6 s · primo token 36 ms).
- **L'ipotesi della riga è caduta**: usare il percorso "a blocchi" già in albero
  rende il prefill **2,10× più lento**, non più veloce. Motivo: quel percorso usa
  il moltiplicatore vecchio, mentre il percorso una-posizione-alla-volta usa
  quello veloce riscritto il mese scorso. Il risultato conferma la diagnosi di
  it.0 invece di smentirla.
- Corretto un numero mio: avevo proiettato 192 s di prefill da una misura di
  codice vecchio. Il vero è 87,6 s — l'attenzione riscritta aveva già dato 2,20×
  anche qui.
- Gate tutti verdi: ktest 100/0, suite 531, tipi puliti, decode non regredito
  (47,79 contro un minimo di 45,53).
- Fatto per strada: il fix di `--out` (docket item 3) sul runner che la riga
  toccava comunque.
