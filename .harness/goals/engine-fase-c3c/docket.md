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

   **ESEGUITO (2026-08-08, chiusura C3b it.8) — numeri fissati:**
   *Sync path (riferimenti di non-regressione, banda ±5% = ruling c3b item
   9, ratifica c3a item 14 emendata):* decode 5.299 (max storico, c3b it.6)
   / prefill 25.78 / TTFT 17.88 s / Qwen 326.2 / golden 98.828% full-corpus
   b11 / cpuref 256/256 + 512/512 fase A / routing = firma 14b ai conteggi
   esatti (prefill 1 047 485/1 203 084, decode 208 441/235 520, router GPU
   1 438 591/1 438 604) / ktest 69 / suite 338+7 / tsc pulito.
   *Optimistic (il valore chiuso da C3b, profilo CON la tassa):*
   - b12 (2417 slot, 82.1% residenza): decode 11.60 tok/s (86.2 ms/token),
     TTFT 16.79, prefill 27.45; gate strutturale 2.188 (punto della tassa);
     P(dirty) 0.938, miss/token 4.78, replays/token 1.1875, repair 15.5
     ms/token; tassa 39.2 ms/token vs sim 34.5 (×1.14, entro ±25%).
   - tetto (2595 slot @12.88 GiB, 88.15%): decode 16.64 (60.1 ms/token),
     TTFT 12.60, prefill 36.58; gate 1.891 PASS; P(dirty) 0.8125,
     miss/token 2.234, replays 0.891, repair 3.70, stallo 3.88 ms/token.
     NOTA: al tetto il floor 13.43 e' GIA' battuto (16.64) al protocollo
     sessione minima — la clausola pre-negoziata del floor riguarda i
     budget sotto il tetto.
   *Hit-rate LRU pura ai budget di prova (harness routing, path sync):*
   b11 0.9575 (5 509 831/5 754 416), b12 0.9756 (routing-conformance
   2026-08-03).
   *Landmine per la spec C3c:* (i) P(dirty) del decode ottimistico e'
   SENSIBILE allo stato cache lasciato dal prefill (v1 non canonica del
   tetto: prefill decode-only ⇒ P(dirty) 0.922 e gate 2.203 agli stessi
   slot — il prefill chunked scalda la LRU con gli expert del contesto);
   (ii) preload optimistic in ordine (layer,expert) arbitrario
   (glmmodel.ts, commento con rimando alla landmine it.2); (iii) leva
   dichiarata per P(dirty): policy > LRU (WP-0: Belady ~dimezza) — intera
   in C3c per split ratificato (c3b item 2c); (iv) syncLogits 7.6 ms e'
   dell'era C3a, probe it.5 dice 0.08 — rimisura formale in apertura C3c;
   (v) 60 s fra run GPU consecutive, albero congelato nelle run lunghe.
