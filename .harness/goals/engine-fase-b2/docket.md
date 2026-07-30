# Docket — engine-fase-b2 (decisioni PI pendenti)

2. **RULING RICHIESTO — re-scope del contratto** (2026-07-30, fase 1): l'attribuzione
   ha refutato la premessa del goal. Dati (2 JSON in results/engine/, journal it.1):
   quota fuori-GPU al contesto bench = **20%** (non 73% — quello era un confronto a
   contesti diversi: gpuBusy@ctx64 vs wall@ctx570); GPU busy scala con kvLen
   (attention decode a 14 workgroup, ~8 µs/posizione/token: 2.43 ms @ ctx64 →
   6.46 @ ctx570); sync fissa ~1.6 ms/token; encode 0.05. Conseguenze: il gate
   "quota ≤50%" è già vero oggi (vacuo); la de-sync multi-step da sola vale ~1.22×
   (149 tok/s a K=8), non il 2× provvisorio. Opzioni:
   (a) **[RACCOMANDATA]** re-scope su "wall di decode al contesto bench verso il
       floor": leva primaria = attention decode split sul contesto (flash-decoding
       style, da 14 wg a 14×split), secondarie = multi-step K (−~1.4 ms) e fusioni
       floor GEMV. Gate possibile: decode ≥ 240 tok/s @ ctx bench (wall ≤ ~4.2:
       attn ~1.0 + gemv 2.4 + sync/8 ≈ 0.2 — plausibile, da confermare in spec)
       + guard-rail parità/prefill invariati. Le fasi 3-5 si ridisegnano di
       conseguenza (PHASES da rifare via ruling, come da protocollo).
   (b) tenere il contratto de-sync puro: gate onesto diventa ~1.2× (≈148 tok/s) —
       chiude presto ma lascia la leva grossa a un goal B3.
   (c) split: B2 = de-sync minimale (K multi-step), B3 = attention/kernel — due
       goal piccoli, più overhead di processo.
   NOTA metodo: qualunque opzione, la soglia va rifissata sui dati di attribuzione
   (Pareto), e il guard-rail anti-gaming sul gpuBusy va RIFORMULATO (ridurre il
   GPU busy diventa l'obiettivo, non un imbroglio: il guard giusto è sulla parità
   + non-regressione prefill/persistenza, più headline tok/s con baseline
   same-day, che rende visibile ogni trade).

1. ~~**plan-check**~~ RISOLTO (2026-07-30, ruling PI in chat: "Va bene. Puoi
   andare in loop su tutto" — PHASES.md approvato, loop autonomo autorizzato
   sull'intero goal). Testo originale:

   **plan-check** (2026-07-30, iterazione 0): approvazione di PHASES.md prima
   dell'iterazione 1. Punti da ratificare: (a) sequenza 1→2 (attribuzione PRIMA
   della spec, perché le soglie si fissano sui dati — nessun parallel-group);
   (b) fase 3 con percorso per-token B1 tenuto vivo come oracolo interno fino a
   fase 5; (c) split candidato di fase 3 (pipelining readback → feedback on-GPU)
   se sfora le 4 iterazioni. Il contratto GOAL.md è già approvato in chat
   (2026-07-30, goal-brief); qui si approva solo la decomposizione.
