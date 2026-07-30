# Docket — engine-fase-b2 (decisioni PI pendenti)

3. **RULING RICHIESTO — spec B2 v2** (2026-07-30, fase 2): approvazione di
   `docs/superpowers/specs/2026-07-30-engine-fase-b2-design.md`. Decisioni proposte
   da ratificare:
   (a) attention split CHUNK=64, griglia fissa (14,16), due pass con log-sum-exp —
       validato in isolamento: ~32 µs/layer piatto vs 144 del fuso a ctx 576
       (4.53×), parità identica al fuso vs CPU f64 (attn-bench JSON);
   (b) NIENTE switch runtime fuso/split: +0.2 ms/token accettati sotto ctx~128;
   (c) multi-step K=8 default con token feedback on-GPU ⇒ token_embd sale su GPU
       (~68 MB, kernel embedGatherQ4) — deviazione dalla scelta fase A
       "embedding CPU-side" (motivata: il readback per-token sparisce);
   (d) EOS mid-batch risolto con crop() (esatto by design B1); streaming a
       raffiche di K, K=1 ripristina il flusso B1;
   (e) SOGLIA decode: proposta ≥230 tok/s (sopra entrambi i plateau a leva
       singola: sync-only 149, kernel-only 185; margine ~8% sotto predizione 249)
       — alternativa ambiziosa ≥240 (margine 3.6%): SCELTA AL PI;
   (f) dispatch ≤100/token ARCHIVIATO con motivazione numerica (il floor GEMV è
       lavoro reale, non overhead dispatch; su 4090 lo split ne aggiunge 24 e
       vince comunque) — resta rimando §I con trigger mobile; nuovo bound sanity
       profiler ≤160;
   (g) conformance a K=1 (teacher-forcing), K>1 coperto dal gate token-identity.
   Le fasi 3-6 restano blocked fino al ruling.
   CAVEAT dal verifier (it.2, dichiarati per onestà del ruling): (i) il timing del
   microbench assume la serializzazione dei dispatch nello stesso pass via hazard
   RW — su Dawn oggi regge (e il fuso riproduce l'attribuzione: 144 µs × 24 ≈ 3.5
   ms vs ~4.0 attribuiti), ma è comportamento implementativo; (ii) la proiezione
   249 tok/s è additiva, non una misura del piano integrato — il margine dell'8%
   sulla soglia 230 esiste per questo (spec §Rischi 1).

2. ~~**RULING RICHIESTO — re-scope del contratto**~~ RISOLTO (2026-07-30, ruling PI
   in chat: "Opzione A, assolutamente. Non abbiamo fretta. L'obiettivo è quello di
   scrivere il miglior motore in assoluto (o almeno, uno dei migliori)"). Re-scope
   (a) adottato: GOAL.md riscritto (v2), PHASES ridisegnate (fasi 2-6, clausola di
   split pre-negoziata sulla fase kernel), guard-rail gpuBusy sostituito da
   parità+non-regressione+headline come da nota metodo. Testo originale:

   **RULING RICHIESTO — re-scope del contratto** (2026-07-30, fase 1): l'attribuzione
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
   NOTA dal verifier (it.1): nelle predizioni, "pipelined" rende solo +1% su
   "batched" perché encode (0.05 ms) ≪ gpuBusy — il fallback pipelining del
   readback previsto dal DONE-WHEN ha valore ~nullo su questa GPU (resta
   rilevante solo per device a encode alto, es. S22 ~18 ms).
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
