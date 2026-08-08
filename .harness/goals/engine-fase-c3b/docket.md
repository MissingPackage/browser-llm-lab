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

3. ~~**RATIFICA COLLEGATA**~~ **RISOLTO (2026-08-08): banda ratificata a ±5%,
   v. item 9.** Testo originale:
   La banda di rumore del gate di non-regressione tok/s (c3a docket item 14,
   proposta: regressione = mediana sotto il riferimento oltre il 2% OPPURE
   sotto in modo statisticamente significativo a livello di repliche) e'
   APPLICATA IN VIA PROVVISORIA nel contratto v2 — senza, il confronto secco
   col massimo storico respinge rumore puro ~meta' delle volte (ratchet
   statistico, caso concreto in c3a item 14). La ratifica formale resta da
   dare su c3a item 14; se la respingi, il contratto torna al confronto secco
   e ogni commit gia' passato sotto la banda viene ri-dichiarato.

4. **PLAN-CHECK — PRE-AUTORIZZATO (2026-08-07, dalla stessa delega di item 2:
   "procedi con il loop"; pattern c3a item 3).** PHASES.md e' su disco al tag
   di setup: 7 fasi sequenziali, authority delta = none ovunque, nessuna
   docket-born. Se vuoi rivederla prima dell'implementazione, il punto giusto
   e' PRIMA della fase 2 (la fase 1 produce solo la spec, reversibile).

5. **REGISTRAZIONE (2026-08-07, it.1) — spec fase 1 depositata** (pattern
   plan-check pre-autorizzato, contratto DONE WHEN punto 1):
   `docs/superpowers/specs/2026-08-07-engine-fase-c3b-decode-ottimistico.md`.
   NON tocca gate ne' soglie del contratto (il gate strutturale <= 2 e la
   tolleranza +/-25% sono ereditati tali e quali) ⇒ nessun ruling bloccante.
   Scelte di meccanismo fissate dalla spec, sindacabili con un edit:
   (a) modo NUOVO `select:"optimistic"` — `select:"gpu"` it.17 resta intatto
   (riferimento e caso M4); (b) soglia precondizione [ASSUMED 0.88 = regime
   WP-0], mai abbassabile in produzione senza docket; (c) pin-for-replay
   (eviction del repair mai sugli slot Sel dei layer >= firstDirtyLayer) ⇒
   replay pulito per costruzione, max 1 replay/token, replay sporco = throw;
   (d) niente fallback dinamico a sync (warning strutturato oltre soglia,
   lo switch e' materia C3c).

6. **REGISTRAZIONE (2026-08-07, it.4) — emendamento della precondizione
   ottimistica (spec §2) + policy slot del bench.** Due decisioni di
   meccanismo prese sotto la delega di item 2, sindacabili con un edit:
   (a) **precondizione sul TOTALE (soglia 0.80), non per classe (0.88)**:
   il riparto proporzionale rendeva q4_1 il vincolo artificiale — 0.88 per
   classe = 13.5 GiB di slab, oltre il tetto fisico del device a QUALUNQUE
   host (il tetto misurato in sessione minima, 2596 slot, e' l'88.2% totale).
   P(dirty) dipende dalla frazione complessiva; WP-0 misura il collasso a
   <= 50% e piena funzionalita' a 82% (b12: 2419 slot, 9.55 tok/s proiettati
   > 5.211 sync). 0.88 resta il riferimento delle proiezioni, non il rifiuto.
   (b) **allocazione q4_1-first nel bench ottimistico**: q4_1 intera (256
   slot, ~1.3 GiB — azzera il miss surface di 4 layer MoE), il resto a q4_0;
   dichiarata nel report. Cosi' la run di fase 4 gira al protocollo b12
   canonico (12 GiB, parita' storica, zero rischio OOM) — l'alternativa era
   nessuna run possibile su questo device senza spegnere la sessione.

7. **FINDING + EMENDAMENTO SPEC (2026-08-07, it.4) — I3/I4: il replay puo'
   CASCARE, il repair diventa iterativo per prefisso.** Alla prima run di
   bench sul modello vero (b12 optimistic, warmup token 0) il replay e'
   uscito SPORCO (3 miss al layer 11) dopo il repair: l'invariante "replay
   pulito per costruzione" era FALSA — I4 vale solo per il PRIMO layer
   sporco (input = checkpoint bit-identico); a valle l'hidden riparato
   differisce da quello degradato del giro prima ⇒ selezioni diverse ⇒
   possibili nuovi miss. Ne' il ktest (mini-modello a 1 layer MoE: nessuna
   valle) ne' WP-0 (selezioni fisse dalla traccia oracolo) potevano vederlo.
   Emendamento (meccanismo, delega item 2): repair ITERATIVO — round di
   repair+replay finche' pulito, progresso stretto di firstDirty asserito
   (violazione = throw), cap teorico nMoe. Conseguenza da MISURARE: la tassa
   ha un termine in piu' (sync/token = 1 + P(dirty)*E[round|dirty]) e il
   confronto di fase 5 con WP-0 deve dichiarare che il sim NON modella la
   cascata (le sue selezioni non dipendono dall'hidden). Se il gate
   strutturale <= 2 fallisse per E[round] alto, e' un dato per il PI, non
   un fallimento da nascondere.

8. **RULING RICHIESTO — gate strutturale 2.188 > 2 al punto di budget b12:
   di chi e' il FAIL?** (2026-08-07, it.4, report
   `results/engine/bench-glm-4090-b12-optimistic-2026-08-07.json`, host
   quiescent). Il decode ottimistico in PRODUZIONE: **11.60 tok/s (da 5.211,
   +123%)**, TTFT 16.79 s (da 17.88), prefill 27.45 (da 25.78), gpuBusy
   39.4 ms/token (da 54.2: il clock recovery previsto in it.1 e' reale),
   **token generati 64/64 IDENTICI al bench sync** (qualita' invariata a ctx
   525). Il gate strutturale pero' misura **2.188 sync(=submit)/token > 2**:
   1 + P(dirty)·E[round|dirty] = 1 + 0.938·1.267 — P(dirty) 93.8% (sim WP-0
   a pari slot: 85.2%, gap da spiegare in fase 5) e la CASCATA dei round
   (item 7) che il sim non modella. Ai 2596 slot del tetto (sessione minima)
   l'aritmetica da' 1 + 0.65·1.27 = **1.82 PASS**; su M4 e' ~1.0. Il termine
   eliminato resta enorme: 47 → 2.19 sync/token (−95%).
   **Opzioni**: (a) [RACCOMANDATA] una run a SESSIONE MINIMA (azione host
   tua) per misurare il gate al tetto 2596: se PASS, la fase 4 chiude con
   quel report e il b12 resta il punto della tassa; (b) clausola 17a-style:
   FAIL dichiarato per budget/hardware, non per struttura; (c) emendare il
   gate sulle componenti. Il loop prosegue con la fase 5 (analisi della
   tassa: non dipende dal ruling).

9. **RULING PI (2026-08-08, in chat): banda di non-regressione fissata a
   ±5%.** Testuale: "Le bande è meglio fissarle al ±5%. C'è troppa
   variabilità nella macchina di sviluppo. Non abbiamo l'affidabilità di un
   server, purtroppo." Sostituisce il 2% provvisorio di item 3 (e ratifica
   c3a item 14 con larghezza emendata: 2% → 5%; il ramo "OPPURE
   statisticamente significativo su repliche" della proposta originale resta
   in piedi come criterio aggiuntivo). Si applica a TUTTE le metriche di
   rumore del blocco non-regressione (tok/s al ribasso, TTFT al rialzo); i
   gate di correttezza (cpuref, golden, firma routing, ktest, suite) restano
   secchi. Conseguenza immediata sulla fase 6 (run 2026-08-07 su disco):
   prefill 25.01 vs 25.78 (−3.0%) e TTFT 18.43 vs 17.88 (+3.1%) rientrano
   in banda ⇒ PASS; Qwen 325.3 vs 326.2 (−0.3%) PASS. Nota: c3a item 14b
   (riferimenti di conformance con ordine di somma diverso) NON è coperto
   da questo ruling e resta pendente.
