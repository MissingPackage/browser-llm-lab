# Docket — engine-fase-c3c (decisioni PI pendenti)

<!-- Goal CHARTERED, NON AVVIATO (2026-08-07). Perimetro = ex-C3b chartered
     2026-08-01 (paging in scarsita'), riscritto dopo lo split registrato in
     .harness/goals/engine-fase-c3b/docket.md item 2. Avvio a C3b chiuso. -->

1. **RULING RICHIESTO (in fase di spec, non blocca l'avvio del WP banda
   fredda)** — budget instant-on ASSOLUTO. Il contratto chartered 2026-08-01
   chiedeva TTFT a freddo <= 4 s (budget UX): aritmeticamente fuori finche' il
   TTFT a caldo e' ~18 s (chiusura C3a). Il contratto v1 usa un target relativo
   [ASSUMED 1.25x il TTFT a caldo chiuso da C3b] col gap dai 4 s sempre
   riportato. Da decidere al momento della spec: ratificare il relativo,
   fissare un assoluto diverso, o tenere i 4 s come obiettivo di fase D.

2. **PRE-AVVIO (non una decisione pendente adesso)** — alla chiusura di C3b,
   prima di dare /goal su questo contratto: fissare i numeri "il valore chiuso
   da C3b" (decode/prefill/TTFT con la tassa di replay dentro, hit-rate LRU
   pura ai budget di prova, pattern c3a item 21); applicare l'esito della
   ratifica c3a item 14 (banda di rumore); rilettura del contratto, poi tag
   goal-engine-fase-c3c-start.
