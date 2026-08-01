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
