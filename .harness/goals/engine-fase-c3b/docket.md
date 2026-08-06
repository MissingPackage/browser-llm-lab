# Docket — engine-fase-c3b (decisioni PI pendenti)

1. ~~**PRE-AVVIO**~~ **RISOLTO (2026-08-07)**: numeri di non-regressione fissati
   nel contratto v2 da c3a docket item 21 (decode 5.211 / prefill 25.78 / TTFT
   17.88 a ctx ~500 / Qwen 326.2 / golden 98.828% / firma 14b / ktest 65 /
   suite 337+7); contratto riletto e RISCRITTO (v. item 2); tag
   goal-engine-fase-c3b-start creato. Testo originale: alla chiusura di C3a
   andavano fissati i numeri "il valore chiuso da C3a" e riletto il contratto.

2. **REGISTRAZIONE (2026-08-07, PI in chat: "Scegli tu su questi punti e poi
   procedi con il loop" — delega esplicita sui 3 punti del goal-brief).**
   Decisioni prese sotto delega, coi razionali:
   (a) **SPLIT in due goal.** Il contratto chartered 2026-08-01 copriva il
   paging in scarsita'; WP-0 (journal c3a it.20) ha misurato che il decode
   ottimistico COLLASSA in scarsita' (budget 50/25%: 100% token dirty, 27-68
   miss/token) e che il paging serve al segmento opposto della Pareto. Due
   meccanismi, due regimi, due matrici di bench ⇒ **C3b = decode ottimistico
   (questo contratto v2), C3c = paging** (contratto nuovo in
   .harness/goals/engine-fase-c3c/, chartered, parte a C3b chiuso col WP banda
   fredda come primo pezzo).
   (b) **Gate C3b = STRUTTURALE, niente gate tok/s.** sync/token <= 2 e
   submits/token <= 2 nel bench di PRODUZIONE + tassa di replay entro la
   proiezione WP-0. Razionale: WP-0 proietta 11.3 tok/s al tetto misurato
   (2596 slot) coi kernel di oggi — un gate 13.43 qui ripeterebbe C3a (gate
   hardware, non struttura). Il **floor 13.43 passa a C3c** con clausola
   pre-negoziata SCRITTA NEL CONTRATTO all'avvio (lezione c3a item 2/17a: la
   clausola si scrive prima, non a fine goal sotto pressione), perche' le leve
   che chiudono il gap (policy > LRU, GiB no-session, clock recovery) vivono
   li' o fuori dal nostro controllo.
   (c) **policy > LRU resta INTERA in C3c.** Belady dice P(dirty) ~dimezzabile
   anche nel regime near-total (gap 0.6pp a 2596), ma il meccanismo tier/AUTOPIN
   e' uno solo e spezzarlo su due goal costerebbe piu' del beneficio; coerente
   con (b), il tok/s che ne dipende vive dove vive la leva.
   Se una di queste tre non ti sta bene, il revert e' un edit di contratto —
   da fare PRIMA della fase di implementazione.

3. **RATIFICA COLLEGATA (pendente altrove, applicata qui in via provvisoria).**
   La banda di rumore del gate di non-regressione tok/s (c3a docket item 14,
   proposta: regressione = mediana sotto il riferimento oltre il 2% OPPURE
   sotto in modo statisticamente significativo a livello di repliche) e'
   APPLICATA IN VIA PROVVISORIA nel contratto v2 — senza, il confronto secco
   col massimo storico respinge rumore puro ~meta' delle volte (ratchet
   statistico, caso concreto in c3a item 14). La ratifica formale resta da
   dare su c3a item 14; se la respingi, il contratto torna al confronto secco
   e ogni commit gia' passato sotto la banda viene ri-dichiarato.
