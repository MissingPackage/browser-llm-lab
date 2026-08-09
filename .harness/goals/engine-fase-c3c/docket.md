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

1-bis. **PROPOSTA DEPOSITATA (2026-08-08, fase 2 — spec §7).** Per il ruling
   item 1, coi numeri del WP banda fredda dentro: **(a) RACCOMANDATA —
   relativo 1.25× AUTO-ANCORATO** al TTFT a caldo misurato nella stessa
   config/sessione del report instant-on (scioglie la scelta 16.79 b12 vs
   12.60 tetto: l'ancora è la config della run, entrambe riportate);
   richiede overlap I/O-dietro-compute ≥ ~60% (proiezione senza overlap:
   1.28-1.61×). **(b)** relativo 1.4× (overlap parziale ~30% basta).
   **(c)** 4 s assoluto = obiettivo fase D (aritmeticamente fuori: il solo
   prefill a caldo è 12.6 s). Il ruling blocca SOLO la fase 7; fasi 3-6
   procedono. Dettaglio: spec §6-7.

4. **REGISTRAZIONE (2026-08-08, fase 2, non richiede decisione)** — spec
   depositata: `docs/superpowers/specs/2026-08-08-engine-fase-c3c-design.md`.
   Non tocca gate né soglie (floor 13.43, banda ±5%, correttezza invariati)
   ⇒ nessuno STOP di ruling oltre item 1 (regime di approvazione C3b).
   Correzione collaterale: KV vero = 108 288 B/token (nLayer 47 × keyLen
   576 × f32; probe kvBytes/525 conferma) — il "54 KB/token" di HANDOFF §5
   era il conto f16, stale di 2× (⇒ ctx 6k = 665 MB, non 361). HANDOFF
   corretto (ruling docs-update-when-stale).

3. **PLAN-CHECK — PRE-AUTORIZZATO (2026-08-08, PI in chat: "partiamo con
   c3c"; pattern c3b item 4 / c3a item 3).** PHASES.md è su disco al tag di
   apertura: 9 fasi sequenziali, authority delta = none ovunque, nessuna
   docket-born. Se vuoi rivederla, il punto giusto è PRIMA della fase 3
   (le fasi 1-2 producono tool di misura e spec, reversibili). Item 1
   (budget instant-on) arriva al ruling con la proposta della fase 2, coi
   numeri del WP banda fredda dentro; nota di rilettura contratto: il
   target relativo va ancorato a un riferimento a caldo esplicito — i
   candidati chiusi da C3b sono 16.79 s (b12) e 12.60 s (tetto/sessione
   minima), la proposta della spec deve sceglierne uno e motivarlo.

5. **REGISTRAZIONE (2026-08-08, it.3, nota verifier — non richiede decisione
   ora)** — il punto b12 rimisurato a codice fase-3 (run nonreg-c3c:
   decode 12.92 / TTFT 14.94) MIGLIORA il riferimento c3b (11.60 / 16.79)
   di ~11%, oltre la banda ±5% in direzione buona; spiegazione plausibile:
   sessione host (oggi la macchina rende come la sessione minima del tetto —
   la run ctx525 replica il tetto a −0.5%). Per il ruling "le metriche non
   peggiorano mai", 12.92/14.94 è il CANDIDATO nuovo riferimento b12: da
   fissare al più tardi in fase 8 (con host state comparabile dichiarato),
   non adottato unilateralmente qui.

1. **RISOLTO (2026-08-09, PI in chat: "ok (a)")** — budget instant-on =
   **relativo 1.25× AUTO-ANCORATO**: TTFT a freddo ≤ 1.25 × TTFT a caldo
   misurato NELLA STESSA config e protocollo del report instant-on (l'ancora
   è la config della run, entrambe riportate; gap dai 4 s UX sempre nel
   report — i 4 s assoluti restano obiettivo di fase D). Proposta 1-bis
   opzione (a) ratificata. La fase 7 è SBLOCCATA.

6. **REGISTRAZIONE (2026-08-09, it.8 — non richiede decisione ora)** — le
   selezioni del PREFILL non alimentano eusage (noteSelection è solo nel
   decode sync): colibri conta tutte le selezioni. Estensione candidata a
   valle (fase D o igiene di chiusura); non toccata in fase 7 per non
   muovere le baseline misurate di fase 5-6 a metà goal.

5-bis. **RISOLTO (2026-08-09, it.9, fase 8)** — item 5: il riferimento b12
   optimistic è FISSATO ai numeri freschi di fase 8: decode 13.172 /
   prefill 31.26 / TTFT 14.74 (host user-session-light dichiarato,
   strutturale 2.188 invariato). Ratchet "le metriche non peggiorano mai"
   applicato con host state comparabile dichiarato nel JSON.

7. **IGIENE (2026-08-09, it.9, nota verifier — fuori goal, non bloccante)**
   — nei campioni hostState del bench i campi power sembrano SCAMBIATI
   (before: 588 W a 0% util; after: 42.6 W a 100% util — implausibili
   entrambi): probabile misparsing dell'ordine dei campi nvidia-smi in
   scripts/lib/hoststate.mjs. Non tocca i gate (i tok/s vengono dal bench).
   Da sistemare a igiene, fuori dalle run di questo goal.

8. **INPUT HERO-DEMO M4 (2026-08-09, it.10, chiusura — PI-gated per
   hardware, nessuna azione senza ruling)** — cosa porta C3c alla demo:
   (i) su M4 48 GB la residenza è totale ⇒ regime `select:"gpu"` (it.17
   c3a): 1 submit / 0 sync, niente tassa; il paging C3c è il fallback
   sotto-48GB e il layer instant-on; (ii) instant-on: protocollo misura
   pronto (glm-instanton-run, eviction+fincore — su macOS il drop cache è
   `purge`, da adattare); budget 1.25× auto-ancorato ratificato; (iii)
   modello di banda parametrico in banda (bandmodel.ts): bastano i numeri
   del WP banda fredda su M4 (tools/opfs-cold, da rieseguire lì); (iv)
   riferimenti di non-regressione 2026-08-09 nel docket (item 2 + 5-bis);
   (v) landmine: campi power hostState (item 7), prefill in scarsità
   collassa (30→4 tok/s: su device piccoli il TTFT è prefill-bound).

9. **DIREZIONE POST-FASE-C (2026-08-09, PI in chat — brain dump ordinato e
   confermato con "vai")**: (0) identità = motore OPEN SOURCE, scopo
   avanzamento tecnologico (licenza da scegliere, split repo al rilascio);
   (1) research preview + writeup (blog+paper) SUBITO sui numeri di fase C —
   prerequisiti: confronto llama.cpp CUDA offload parziale stesso hardware
   (gate di onestà; PI: "già so che ci demolirà, giustamente"), split repo,
   licenza pesi; (2) goal GENERALIZZAZIONE prima della fase D: famiglia
   Qwen 3.6/3.8 (recon WP prima — architetture da misurare, non assumere),
   2-3 taglie, 2-3 baseline hardware consumer (8-12 GB = scarsità vera),
   prefill/TTFT dentro questo goal; (3) fase D DOPO, sui modelli target
   (spec-dec è per-modello). Ordine esecuzione immediato: (a) confronto
   llama.cpp, (b) recon Qwen, (c) goal-brief generalizzazione.

10. **SEQUENZA POST-FASE-C AGGIORNATA (2026-08-09, PI in chat)**: split repo
   → ALLA PUBBLICAZIONE (piano congelato in docs/publishing/split-plan.md);
   paper → COMPANION DEL RILASCIO, target ZENODO preprint, si scrive a
   valle della generalizzazione (bozza contratto parcheggiata in
   docs/publishing/paper-contract-draft.md); punto aperto non bloccante:
   timing del blog (prima del paper o insieme). Il prossimo goal operativo
   è la GENERALIZZAZIONE (engine-fase-q1, contratto approvato in sostanza
   dal PI "mi torna", ASSUMED da ratificare in spec come da pattern).
