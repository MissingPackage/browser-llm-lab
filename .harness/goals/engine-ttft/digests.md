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
