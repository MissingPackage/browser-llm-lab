# Docket — engine-fase-c3a (decisioni PI pendenti)

1. ~~**RULING — split della fase C3 e formulazione del gate prefill**~~ RISOLTO
   (2026-08-01, ruling PI in chat, sessione 18 — /goal-brief). Due decisioni
   prese insieme al contratto, registrate qui perche' modificano il perimetro
   ereditato da direction §7 e sciolgono una domanda lasciata aperta in C2.

   **(a) Split C3 → C3a + C3b.** PI: "ok lo split in 2 goal". Motivazione
   accettata: l'attribuzione misurata in C2 (docket C2 item 8) separa 158.9
   ms/token di STRUTTURA (pack CPU nel path caldo, 46 sync router/token,
   prefill sequenziale) da 56.1 ms/token di RESIDENZA. Le due masse si
   aggrediscono con leve diverse e chiudono su condizioni diverse; un goal
   unico "slab+tier+AUTOPIN+PILOT-real + gate tok/s" poteva chiudere a meta'
   su uno dei due assi. Perimetri:
   - **C3a** (questo goal): repack all'import, eliminazione/batching dei sync
     router, prefill batched M>1. Gate di chiusura = floor C1 13.43/56.58.
   - **C3b** (chartered, non avviato): budget slab ctx-aware, tier.h, AUTOPIN,
     PILOT-real/prefetch, modello di banda, instant-on, WP banda fredda browser.
   direction §7 emendata con lo split. Nomi c3a/c3b confermati dal PI
   ("ok sui nomi") per non rompere i riferimenti "C3" gia' scritti nei doc.

   **(b) Gate prefill: velocita', non sola capacita' — ma misurata sull'UX.**
   Il docket C2 item 6 chiudeva con una domanda esplicita non sciolta dal
   ruling 6a: "il PI decida se il gate prefill vada riformulato in C3 come
   requisito di capacita' e non di velocita'". Ruling PI: "su capacita' vs
   velocita' e' stato il dilemma iniziale del progetto. alla fine abbiamo
   deciso di puntare alla massima intelligenza possibile con una velocita'
   minima accettabile. 30+ tk/s come soglia normale e di piu' se thinking. per
   garantire l'esperienza utente ottimale".

   Traduzione nel contratto, in due livelli (entrambi scritti in GOAL.md):
   - **Gate di chiusura** (meccanicamente verificabile oggi): decode >= 13.43,
     prefill >= 56.58 tok/s — il floor oracolo CPU C1, come da ruling C2 6a.
   - **Target di programma, riportato a ogni bench e NON gateato**: decode
     >= 30 tok/s (>= 60 in regime thinking — i token di ragionamento sono
     latenza pura prima della risposta visibile) e **TTFT <= 4 s**.
   Il floor prefill 56.58 e' un `pp512` di llama-bench CPU: su p6 (461 token)
   vale 8.15 s di time-to-first-token, cioe' soddisfa il gate d'ingresso e non
   soddisfa nessuna esperienza utente. Lo stato attuale e' 5.221 tok/s = 88 s
   di TTFT. Da qui la doppia scrittura: C3a non puo' chiudere sotto il floor
   (era la deroga C2), ma non deve poter chiudere dichiarandosi arrivato.
   L'assenza dei numeri di gap dal report e' un FAIL della checklist.

   **Budget TTFT = 4 s** (non i 2 s proposti nel draft). PI: "il ttft va bene
   anche minore di 4 secondi. a 2 rischiamo di essere troppo stringenti e non
   chiudere mai. alla fine 4 secondi e' accettabile sul mio hardware che non e'
   top". Su p6 (461 token) 4 s equivalgono a ~115 tok/s di prefill, ~2x il
   floor CPU.

   **Soglie [ASSUMED] del draft confermate senza correzioni** ("ok, soglie
   confermate"): sync/token <= 2; pack < 1.0 ms/token nel path caldo; 60 tok/s
   come soglia UX in regime thinking; errore del modello di banda +/-15% (C3b);
   budget slab di prova 50% e 25% del parco (C3b).

2. **RULING RICHIESTO — clausola di fallback per la fase 4 (sync router)**
   (2026-08-01, iteration 0). Non blocca: le fasi 1-2 girano comunque; serve
   una risposta prima della fase 6.

   C1 aveva una clausola pre-negoziata (se il tap hidden sfora il timebox, il
   goal chiude in modalità ridotta e la via alternativa va a docket). C3a non
   ne ha. Il rischio è concreto e quantificato: la proiezione C2 dice che a
   residenza perfetta il decode arriva a **6.3-7.0 tok/s**, quindi tutto il
   resto del cammino verso 13.43 deve venire dall'eliminazione dei 46 sync,
   la cui resa non è mai stata misurata isolatamente (i 158.9 ms/token sono
   sync *più* 1.816 dispatch, senza split noto — è esattamente ciò che la
   fase 1 va a misurare). Senza clausola, l'esito "sotto il floor" produce una
   seconda deroga come in C2, cioè una decisione presa sotto pressione a fine
   goal invece che a mente fredda adesso.

   **Opzioni**:
   (a) [RACCOMANDATA] **clausola simmetrica a C1**: se dopo le fasi 3-5,
       ri-misurato a macchina quiescente, il decode resta sotto 13.43, C3a
       chiude con la misura, l'attribuzione aggiornata e la leva residua
       identificata a docket — senza scorciatoie sul path e senza auto-deroga;
       la decisione sul da farsi resta PI.
   (b) **nessuna clausola**: sotto il floor = FAIL del goal e ri-scope.
   (c) **clausola condizionata al numero di fase 1**: si decide dopo aver
       visto lo split sync/dispatch misurato (rimanda la decisione di
       un'iterazione, con l'informazione in mano).

3. ~~**PLAN-CHECK**~~ PRE-AUTORIZZATO (2026-08-01, PI in chat: "vai con goal
   setup e poi parti col loop"). PHASES.md è scritto e il loop parte senza
   attendere approvazione esplicita: 7 fasi, sequenziali (owns sovrapposti sul
   path caldo), fase 1 ready subito, fasi 3-6 gated dal ruling di spec (fase 2),
   fase 4 = rischio dichiarato del goal. Se leggendo PHASES.md la
   decomposizione non ti convince, la modifica passa da qui (item nuovo), non
   dal loop.

4. **RULING RICHIESTO — le tre leve del contratto non raggiungono il gate;
   scelta del ramo per la spec** (2026-08-01, it.1 fase 1). **BLOCCA la fase 2**
   (la spec deve scegliere il meccanismo dei sync, e la scelta dipende da
   questo). Numeri in `results/engine/bench-glm-4090-b12-attrib-2026-08-01.json`,
   journal it.1.

   **Misura.** Wall decode 215.0 ms/token si scompone in: `gpuBusy` **78.2**
   (36.4%) + stallo residenza **53.8** + sync/CPU **83.0** ⇒ **63.6% fuori
   dalla GPU**. Il probe indipendente dice che i 46 readback costano solo
   **7.6 ms/token** di floor irriducibile: meno del 10% degli 83 ms di
   sync/CPU è readback vero, il resto è latenza di submit e bolle.

   **Conseguenza.** Con le leve 1 e 2 del contratto al 100% di efficacia
   (repack che azzera il pack CPU −41.4; sync ridotti al floor −75.4):
   **98.2 ms/token = 10.18 tok/s**, contro un gate 13.43 che vuole ≤74.46
   ms/token. **`gpuBusy` da solo (78.2) eccede il budget del gate.** Le leve
   del contratto non bastano per costruzione — non è congettura, è il wall
   misurato meno i due termini che le leve rimuovono.

   **Osservazione discriminante già acquisita** (per questo ho campionato i
   clock durante la run): la GPU gira a **1746 MHz medi su 3105 di cap** con
   utilizzo **34.6%** — è sotto-clockata *dalle bolle che la leva 2 va a
   togliere*. Se `gpuBusy` scalasse col clock SM (limite ottimistico: i GEMV
   sono in parte memory-bound e il clock memoria è già al massimo) diventerebbe
   44.0 ms ⇒ **64.0 ms/token = 15.63 tok/s, gate PASS**.
   **La forbice vera è 10.2 – 15.6 tok/s, col gate 13.43 dentro.** La leva 2 ha
   un payoff di secondo ordine (far salire i clock) che nessuno aveva
   dimensionato e che può valere quanto quello di primo ordine.

   **Terzo dato, indipendente**: 2.22 GB di pesi letti per token contro 576
   GB/s di banda reale del device ⇒ floor memory-bound **3.85 ms/token**;
   `gpuBusy` è **20× sopra**. A 1816 dispatch/token siamo a 43 µs per dispatch
   su GEMV che ne dovrebbero costare ~2. Esiste una **quarta leva**
   (granularità/fusione dei dispatch) con margine enorme, non elencata nel
   contratto e messa per ultima da direction §2.

   **Opzioni**:
   (a) [RACCOMANDATA] **spec a due stadi, gate invariato**: la fase 4 attacca i
       sync e **ri-misura `gpuBusy` e i clock subito dopo**; il numero decide se
       serve la quarta leva. Costa un ciclo di misura, non un ciclo di
       implementazione, e scioglie la forbice con un fatto invece che con
       un'assunzione. Se dopo la leva 2 il gate resta irraggiungibile, la
       quarta leva entra in C3a via emendamento (item nuovo), non di soppiatto.
   (b) **ammettere subito la quarta leva** (fusione/granularità dei dispatch)
       nel perimetro C3a: massimizza la probabilità di prendere il gate, ma
       allarga lo scope a kernel engineering — che direction §2 deprioritizza —
       e allunga il goal di parecchie iterazioni.
   (c) **abbassare il gate di C3a** a un valore dentro la forbice pessimistica
       (es. 10 tok/s) e spostare 13.43 a un goal successivo: onesto, ma ripete
       la dinamica della deroga C2 prima ancora di provarci.
   (d) **chiudere C3a sulle sole leve 1-3 con misura** e aprire un goal
       dedicato alla guerra ai dispatch.

   Collegato all'**item 2** (clausola di fallback): se il PI sceglie (a), la
   clausola diventa più urgente, perché il ramo pessimistico della forbice è
   ora quantificato e non più ipotetico.

5. **CORREZIONE PHASES (piccola, non un ri-scope)** (2026-08-01, it.1). La riga
   della fase 1 chiede "run glmbench **exit 0**": è insoddisfacibile in fase 1,
   perché exit 0 = gate PASS e i gate passano solo in fase 6. L'esito corretto
   della fase 1 è **exit 4** (report scritto + gate FAIL). Chiedo di emendare la
   riga in "run glmbench che scrive il report (exit 0 o 4)". Stessa riga: il
   secondo JSON di attribuzione è stato consegnato **dentro** il report di
   bench (`attribution2`) invece che come file separato — sostanza consegnata,
   forma no; proposta: allineare il testo alla forma consegnata.
