# DOCKET — engine-velocita-decode (già `engine-35b-residency`)

Le decisioni REGISTRATE E NON PRESE, in attesa di ruling del PI. Ordine e
meccanismo non stanno qui: quelli li decido io e finiscono nel journal.

## item 1 — host non quiescente: la baseline non è eseguibile adesso (io → PI, it.1)

**La riga 1 è ferma sulla sua seconda metà, e non è un problema di codice.**
La strumentazione è in albero e verde (vitest 1025 passed, tsc exit 0). Ciò che
manca è la misura, e l'host non è nello stato in cui una misura vale.

**Osservato**, `pgrep -af`: quattro istanze di `@playwright/mcp` con Chrome
vivo, fra cui un `--type=gpu-process` con `--enable-unsafe-swiftshader`. Più
`free -g`: 9 GB di RAM usati su 31, 21 disponibili. Il 35B chiede ~13 GiB di
VRAM e l'arena si dimensiona su ciò che resta.

**Perché non parto lo stesso.** Due ragioni, e la seconda è quella che pesa:
1. Il precedente esiste ed è documentato: `.harness/goals/engine-kquant/docket.md`
   item 2 registra cinque fallimenti consecutivi con tre sintomi diversi
   (`Failed to fetch`, `Target crashed`, device WebGPU perso) su un ambiente
   sporco, costati un gate intero.
2. **Una baseline presa su host non quiescente non è una baseline.** Il
   contratto pretende `hostState.declared = "quiescent"`, e la regola del
   progetto è che ogni misura dichiari il suo host. Prenderla adesso vorrebbe
   dire scrivere `"declared": "non quiescente"` su un artefatto che tutte le
   righe successive useranno come termine di paragone.

**Cosa serve, ed è una riga di comando che non tocco io**: chiudere le sessioni
Chrome/MCP di Playwright. Sono sessioni del PI, non mie, e non le termino senza
che me lo dica.

**Cosa NON è bloccato nel frattempo**: nulla di utile. La riga 2 dipende da
C0-4, C0-4 dipende dalla misura, la misura dipende da questo item. È uno
stop-by-design, non una pausa.

> **CHIUSO in it.2, e con una correzione a mio carico.** Il PI ha indicato la
> via giusta: accordarsi con la sessione che li stava usando
> (`personal-site-47`, screenshot di anteprime tema per il sito personale)
> invece di chiudere. Ha liberato la GPU in pochi minuti, le due run sono
> passate, e mi ha corretto un errore di diagnosi: **guardavo il pattern
> sbagliato**. `pgrep @playwright/mcp` conta i *server node*, che sono idle e
> non tengono VRAM; i processi da guardare sono i browser
> (`type=gpu-process`, `/opt/google/chrome/chrome`, `chromium`,
> `headless_shell`). Dei «quattro Chrome vivi» che avevo dichiarato al PI, uno
> solo era un browser reale. **Da riusare: per sapere se l'host è quiescente si
> contano i processi BROWSER, non i server MCP.**

**RULING:** _ (item chiuso senza bisogno di ruling)

## item 3 — la barra dei 30 tok/s non è raggiungibile nello scope chartato (io → PI, it.2)

**PI: serve un ruling. È funzione obiettivo, quindi non è mio.**

**Il fatto, misurato due volte su due alberi diversi:** il token PULITO del 35B
— zero miss, zero replay, nessuna tassa di residency — costa **43,585 ms =
22,9 tok/s** (`q35-optimistic-35b-cleantoken-2026-08-15.json`, oggi) e **43,736
ms** (`q35-vramplan-35b-it35.json`, 2026-08-11). Coincidono entro lo 0,7%.

**Conseguenza secca**: le righe 2 e 3 di questo goal aggrediscono la tassa di
residency. Anche portandola a **zero** — che nessuno sa fare — il 35B resta a
22,9 tok/s. **La barra dei 30 non è raggiungibile con lo scope che ho
chartato.**

**Dove sta il tempo adesso**: sul token pulito, `readbackWait` è il **93,5%**
(40,75 ms su 43,59), con `submitsPerToken: 1` e `readbacksPerToken: 1`. Non è
overhead: è la GPU che lavora su un solo pass, i 320 GEMV expert del token.

**La sola leva nota su quel termine** è la forma a gather K-quant della
consegna §4.2-4.4 (~2,6× di dispatch, ~1,27× di traffico pesi). **Questo
contratto l'ha messa FUORI SCOPE**, con questa motivazione: «oggi varrebbero il
16% del token». Quel 16% era calcolato sulla stima del token pulito a 21,1 ms
che ho poi ritrattato. **Sul numero vero la stessa leva vale sul 93,5%.** La
mia esclusione era sbagliata, e l'ordine della consegna §6 — residency prima,
kernel dopo — resta giusto su *cos'è primo nel profilo*, ma non basta a
raggiungere *questa barra*.

**Le tre uscite che vedo** (non scelgo io):
1. **Tenere la barra a 30 e allargare lo scope** alla forma a gather: il goal
   diventa residency + kernel, ~3-4 iterazioni in più, ed è l'unica strada che
   porta ai 30. Le righe 2 e 3 restano necessarie (senza, il regime sporco
   mangia qualunque guadagno sul pass).
2. **Tenere lo scope e abbassare la barra** a ciò che la residency può dare:
   il tetto è 22,9 tok/s, quindi una barra onesta starebbe sui ~20. Chiude un
   goal vero e misurabile, ma **lascia il 35B sotto la soglia di usabilità**
   della funzione obiettivo del progetto — cioè non risolve il problema per cui
   il goal esiste.
3. **Spezzare in due goal**: questo si chiude sulla residency con barra ~20, e
   il gather K-quant diventa il goal successivo con barra 30. Più righe
   chiuse, più consuntivi, e il 35B resta inusabile fino alla fine del secondo.

**La mia lettura, che non è un ruling**: la 1. La 2 chiude un goal senza
risolvere il problema, e la 3 fa la stessa cosa spendendo di più. Ma è la tua
funzione obiettivo.

**Nel frattempo NON parto con la riga 2**: il meccanismo giusto per la fetch
dipende da quanto margine serve, e quello lo decide questo ruling.

**RULING (PI, 2026-08-15):** uscita **1**, e con due aggiunte e un richiamo.

> «Tieni 30, allarga il goal alla forma gather (ti avevo detto che avremmo
> dovuto farlo anche nel goal dei k quant, ma non mi hai ascoltato mettendolo
> fuori scope). Aggiungi anche il raggruppamento delle richieste http se può
> dare un boost globale al motore. Non so quante volte ti ho ripetuto di non
> overingegnerizzare e applicare il principio di Pareto. Leve globali, massimo
> risultato, non piccolezze specifiche per il modello A o il quant B e robe
> simili. FACCIAMO IL MOTORE PIÙ VELOCE POSSIBILE. PUNTO»

**Applicato così:**
1. Barra tenuta a 30. Forma a gather **dentro**, ed è la riga 2 (era fuori
   scope).
2. Raggruppamento delle richieste I/O **dentro** come riga 2b, con il
   done-when che pretende l'effetto misurato **anche sul LOAD di 4B/9B/GLM** —
   è così che «se può dare un boost globale» diventa verificabile invece che
   sperato.
3. Il goal è stato **rinominato** `engine-35b-residency` → `engine-velocita-decode`:
   il nome vecchio mi avrebbe tirato verso il modello singolo a ogni
   iterazione.
4. **Regola meccanica messa in testa a PHASES.md e nei done-when: una leva vale
   solo se misurata su ≥ 2 famiglie.** Sulle intenzioni ho già fallito una
   volta in questo goal; questa è verificabile.

**Il richiamo, registrato senza attenuanti**: il PI aveva già chiesto la forma a
gather durante `engine-kquant` e l'ho messa fuori scope con «varrebbe il 16% del
token». Il 16% veniva da una stima non misurata. Sul numero vero vale il 93,5%.
Memoria scritta: `global-levers-not-per-model-optimizations`.

## item 2 — `blankNonCode` è duplicato in sette file di test (io → PI, it.1)

Trovato scrivendo `tests/engine-35b-repair-counters.test.ts`. La funzione che
bianca commenti/stringhe/template prima di scansionare il sorgente — ~90 righe —
è **copiata identica in sette file**: `engine-q35attnwiring`,
`engine-ktest-q41-wiring`, `engine-subgroups-feature`, `engine-ktest-q5k-wiring`,
`engine-prefillwiring-q5k`, `engine-prefillpbcat-q41`,
`engine-ktest-kquant35b-wiring`. `tests/helpers/` esiste già e ospita due
fixture, quindi la sede c'è.

**Non l'ho fatto**, e la ragione è la regola sui bench: un'estrazione tocca
sette file di test in un commit che deve restare leggibile come «la
strumentazione del repair». Nel mio test ho evitato la duplicazione con uno
stripper di soli commenti (15 righe), sufficiente lì perché gli identificatori
cercati non compaiono in letterali — ma è un'ottava variante, non una soluzione.

**Registrato secondo il ruling «sistemare, non recintare» (2026-08-14)**: il
difetto va tolto, non sorvegliato. Chiedo solo se toglierlo QUI (fuori brief,
sette file) o come item d'igiene a sé.

**RULING:** _

## item 4 — la regola che ho scritto io mi manda contro l'intento che voleva codificare (io → PI, it.11)

**PI: serve un ruling. È cosa conta come «leva globale», cioè funzione
obiettivo.**

**Il fatto**: il KFAN ha portato il decode del 35B da **22,58 a 28,90 tok/s**,
gate argmax 39/39, ma è cablato su **una famiglia sola** (`q35gpumodel.ts`).

**La regola che ho scritto** dopo il tuo ruling del 2026-08-15 — «una leva vale
solo se misurata su ≥ 2 famiglie» — mi manda a cablarlo anche sul GLM.

**Cosa costerebbe**, verificato leggendo i due generatori: il GLM non condivide
i kernel col 35B (usa `pairGemvSiluFastWgsl` con gate+up FUSI e
`gemvAccumFastWgsl`). Sono fattorizzati allo stesso modo, quindi il kfan ci
entra con le stesse tre modifiche — ma sarebbe la **terza scrittura a mano dello
stesso modo in una terza famiglia di generatori**, quattro siti di cui due
coperti, e quattro copie della stessa invariante che nessun compilatore
confronta. ~2 iterazioni.

**Perché esito**: it.10 ha misurato che **il primo termine non è più il MoE**.
Braccio kfan-ON: `ssmGemv` 6,98 ms/token contro `expert` 5,15. `ssmGemv` è la
proiezione DeltaNet — **non è MoE, e ce l'hanno 4B, 9B e 35B**, quindi una leva
lì è globale *per costruzione* e soddisfa la regola senza cablarla due volte.

Cablare il kfan sul GLM significa **completare una leva sul secondo termine
mentre il primo è scoperto**. Applicare la mia regola alla lettera va contro
l'intento che voleva codificare — «massimo risultato, leve globali, Pareto».

**Le tre uscite** (non scelgo io):
1. **Vai su `ssmGemv`** e lascia il kfan a una famiglia, dichiarandolo nel
   consuntivo. Attacca il primo termine, è globale per costruzione, e il debito
   GLM diventa un item d'igiene. *Rischio: il kfan resta una leva non
   generalizzata, cioè la cosa che il tuo ruling voleva evitare.*
2. **Chiudi prima il debito GLM** (~2 iterazioni), poi `ssmGemv`. Rispetta la
   regola alla lettera. *Costo: due iterazioni sul secondo termine.*
3. **Cambia la regola**: «≥ 2 famiglie» diventa «la leva è globale *per
   costruzione*, e il cablaggio su altre famiglie è un item d'igiene con la sua
   priorità». Più onesta rispetto all'intento, ma è una modifica a un criterio
   che hai imposto tu — non la faccio da solo.

**La mia lettura, che non è un ruling**: la 3, e poi la 1. Ma la regola l'hai
voluta tu proprio perché da solo avevo sbagliato questo giudizio una volta.

**RULING:** _

**RULING (PI, 2026-08-15):** nessuna delle tre uscite — **la premessa era
sbagliata, e l'ha corretta lui.**

> «La mia regola non era di maggiore di 2 famiglie. Semplicemente dicevo di
> riutilizzare quello che c'è già, costruire globalmente quando possibile e poi
> estendere, e propagare fix sul vecchio quando necessario. Proprio ieri in una
> sessione abbiamo avuto questa discussione, ti sei segnato la memoria fix don't
> fence e abbiamo anche aggiunto un workflow nuovo all'harness apposta per
> queste situazioni.»
>
> «Dai la priorità alle ottimizzazioni globali e riutilizzabili, poi quando
> aggiungeremo nuovi modelli le riutilizzeranno automaticamente e solo in quel
> caso, se non bastasse, o se avessero architetture diverse, faremmo delle
> ottimizzazioni specifiche per loro.»

**Perché le mie tre uscite erano tutte sbagliate**: stavano tutte dentro la
cornice della COPERTURA («quante famiglie ho toccato»), mentre il criterio è il
RIUSO («quante volte ho scritto la stessa cosa»). La risposta non era né cablare
il GLM né saltarlo: era **non scrivere il `kfan` quattro volte a mano**.

**L'aggravante, registrata perché è il vero difetto**: il principio era già in
casa. `fix-dont-fence` (2026-08-14) e il workflow `pattern-migration` sono stati
aggiunti il giorno prima **per questa identica situazione**. Avevo la memoria in
contesto e il workflow nella lista. Non è mancata la regola: **non l'ho
riconosciuta applicabile**, perché avevo catalogato il caso come «copertura di
una leva» invece che «pattern replicato a mano».

**Azioni prese:**
1. Memoria `global-levers-not-per-model-optimizations` **riscritta** con la
   regola vera e con la proxy sbagliata dentro, come reperto.
2. La regola «≥ 2 famiglie» **tolta** da `GOAL.md` e `PHASES.md` e sostituita.
3. Intervista chiesta a `harness-c1` su richiesta esplicita del PI: definire una
   mappa decisionale generale su riuso vs duplicazione, e soprattutto **il
   trigger** — perché una regola scritta ieri non si è attivata oggi.
4. Piano riordinato secondo l'ordine che ha dato: v. `PHASES.md` righe 2d e 3.

## item 5 — cosa chiude la riga 2d, e cosa chiude il goal (io → PI, it.17)

**PI: tre uscite, presentate in chat il 2026-08-15 sera e messe qui perché una
decisione che resta in chat, alla ripresa, è indistinguibile da un abbandono.**

**Lo stato**: decode 35B **22,58 → 28,90 tok/s (+28%)**, gate argmax 39/39.
Barra 30 = 33,33 ms/token, **mancano 1,27 ms**. La riga 2d è a **cinque
iterazioni su 2-3 stimate** e il valore che ha atterrato è sul **prefill** del
35B, che il goal ha dichiarato fuori scope.

**Il costo del passo che resta, misurato leggendo il codice**: `gemv`
(`q35gpumodel.ts:862`, il decode) emette UN dispatch; la forma split-K ne vuole
TRE (quantizza x → GEMM a fette → combine) più due buffer che nel decode non
esistono. E il 3,26× del banco è contro `base-batch-z`, non contro ciò che
`gemv` emette: **il guadagno va misurato, non ereditato.**

**Le tre uscite:**

1. **Costruire la rotta nel decode e misurarla** — 2-3 iterazioni. È l'unica
   che punta ai 1,27 ms mancanti con la leva già identificata. *Rischio: la
   riga arriverebbe a 7-8 iterazioni su 2-3, e il guadagno reale è ignoto
   finché non lo si misura contro il kernel giusto.*
2. **Chiudere il goal sul +28%** e fare della rotta nel decode un goal suo, con
   la sua misura e la sua barra. Il 35B resta sotto i 30, ma il risultato è
   reale, verificato e consegnabile — e la riga 2d chiuderebbe con ciò che ha
   davvero prodotto (il predicato, il q8_0 cablato) invece di trascinare.
3. **Andare sul secondo reperto**: `router` **2,88 ms/token** per scegliere 8
   expert su 256, cioè **72 µs a layer** per una riduzione minuscola. Il
   rapporto fra lavoro e tempo è così fuori scala che vale un'occhiata prima di
   progettare qualsiasi cosa — e 2,88 ms sono più del doppio dei 1,27 che
   mancano.

**La mia lettura, che non è un ruling**: la **3 prima della 1**. Il router è il
termine con il rapporto lavoro/tempo più assurdo dell'intero profilo, non è
stato guardato da nessuno, e da solo basterebbe alla barra. Guardarlo costa
un'iterazione; costruire la rotta nel decode ne costa tre e ha un guadagno che
non conosco. Pareto.

**CHIUSO in it.18 SENZA RULING — era una mia mis-escalation, e l'ho eseguita.**

L'intestazione di questo file dice *«ordine e meccanismo non stanno qui: quelli
li decido io»*, e le tre uscite erano esattamente un ordine di lavoro. Lo
step 5 del protocollo dà il test meccanico: *se il fallback che applicheresti
senza risposta coincide con la tua raccomandazione, non è un'escalation —
eseguila e registrala*. Avrei fatto la 3 comunque. Fatta.

**Esito**: il router è passato da **2,883 a 0,836 ms/token** e il decode del 35B
da **28,90 a 30,74 tok/s — sopra la barra**, con l'intero intervallo di
dispersione [30,55-31,11] sopra i 30 e il gate argmax 39/39 identico. La causa
era un `array<bool, 256>` privato letto 2.048 volte da un thread solo, difeso da
un commento che dichiarava la serializzazione «la specifica». Non lo era: il
massimo su un ordine totale non dipende da come si associa. Dettagli e prove nel
journal, it.18.

**Cosa NON ha deciso questa chiusura**: l'uscita 1 (la rotta split-K nel decode)
resta aperta come lavoro possibile, ma ha perso la sua urgenza — i 1,27 ms che
doveva inseguire non ci sono più.

## item 6 — `q35-mtp-draft-4b` FAIL nel ktest, e non è di questo goal (io → PI, it.18)

**Il fatto, discriminato e non ipotizzato**: `node .harness/tools/engine-ktest.mjs`
esce **110 PASS / 1 FAIL**. Il FAIL è `q35-mtp-draft-4b`: accept-rate GPU
**12/37 = 32,4%** contro il 50,0% del riferimento CPU f64 sulla stessa finestra.
Eseguito una seconda volta sull'albero **stashato** (senza la modifica di it.18)
stampa `12/37 = 32,4%`, cifra per cifra: **preesistente**.

L'ultimo verde noto è `111 PASS / 0 FAIL`, chiusura di `engine-kquant`
(journal di quel goal, righe 794 e 945). Fra lì e oggi stanno le iterazioni
13-17 di questo goal, di cui **it.16 ha cablato il q8_0 nel prefill senza mai
eseguire una run GPU** (la sua evidenza dice «Nessuna GPU»). È il sospettato
numero uno, non una diagnosi.

**Perché è un item e non un fix sul posto**: la testa MTP è del goal
`engine-fase-d` (fase 7-8), il caso confronta un accept-rate su golden e la
diagnosi richiede almeno un bisect fra it.13 e it.17 con un ktest da ~5 minuti a
tappa. **Costo stimato del fix: 1-2 iterazioni**, di cui ~30 min di GPU.

**Perché non può restare così**: la riga 6 di questo goal è un GATE DI MERGE che
pretende «`engine-ktest.mjs` tutti PASS». Con questo rosso il goal **non si può
chiudere**, qualunque cosa faccia la barra.

**Quello che farei senza risposta**: bisect fra it.13 e it.17 all'inizio della
prossima iterazione, prima di qualsiasi altra leva. Chiedo solo se preferisci
che vada invece sotto `engine-fase-d`, che è il goal proprietario di quel
codice.

> **CHIUSO in it.19 SENZA BISECT, e la domanda che avevo fatto era mal posta.**
> Non era una regressione del modello e non stava in nessuno dei commit it.13-17:
> **il test leggeva un altro campione.** `public/models/q35/golden-full.json` ha
> un nome, **due scrittori** (`q35-conf-run.mjs` e `q35-bench-run.mjs`, che ci
> copiano il golden della loro run, qualunque modello e qualunque
> `--golden-kind`) e **due lettori** — di cui uno, `q35conf.worker.ts`, verifica
> lo SHA, e l'altro, il ktest, non verificava niente. Il file conteneva il
> golden **smoke del 35B**, 39 token: il caso misurava l'accept-rate su 37
> confronti e lo confrontava col riferimento preso su 62. **L'avevo avvelenato
> io in it.18** lanciando l'A/B con `--golden-kind smoke`.
>
> **Corretto alla radice**: lo scratch dei bench si chiama ora `golden-run.json`,
> il ktest ha il suo fixture `golden-q35-4b-full.json` che nessun runner scrive
> e che `engine-ktest.mjs` copia dal repo a ogni avvio, e il caso MTP **verifica
> la finestra** invece di assumerla (`tokens.length !== W` ⇒ FAIL che nomina la
> causa vera). Sette casi statici in
> `tests/engine-golden-fixture-isolation.test.ts` pinnano che i due ruoli
> restino file diversi.
>
> **Gate**: `111 PASS / 0 FAIL`, `q35-mtp-draft-4b` **31/62 = 50,0%** — il
> riferimento, cifra per cifra.
>
> **Recidiva, non sfortuna**: `engine-fase-d` it.53 aveva già visto lo stesso
> sintomo («13/38 = 34,2% — un numero giusto su una finestra sbagliata») e
> l'aveva risolto puntando il file giusto **senza togliere la collisione**.

**RULING:** _ (item chiuso senza bisogno di ruling — la domanda su dove
bookkeeparlo è decaduta col bisect)

## item 7 — il GLM b12 non riproduce più il suo riferimento, e la causa è la LETTURA (io → PI, it.18)

**Il fatto, con l'A/B sullo stesso host**: `glm-bench-run.mjs` b12 optimistic,
stessa config del riferimento (prompt 6, ngen 64, reps 3, budget 12 GiB, ctxMax
525, chunked M=16, host quiescente):

    decode   riferimento 2026-08-15   15,330 tok/s
             albero di oggi CON it.18 11,33
             albero di oggi SENZA     11,35     <- non è di it.18
    prefill  37,542 -> 23,94

**Dove sta, e dove NON sta.** Non è calcolo:

    missesPerToken     4,78125  ->  4,78125     identico
    evictionsPerToken  4,78125  ->  4,78125     identico
    gpuBusy            ~38,5    ->  38,9 ms/token
    readMsPerToken     2,475    ->  17,5        <- 7x, ed è tutto qui

**Falsa pista mia, registrata**: avevo attribuito il crollo all'origine (avevo
lanciato su 5173 invece dei 5199 del riferimento, e OPFS è per-origine).
Rieseguito su 5199: 11,33. Non era la porta. L'OPFS del profilo c'è ed è pieno
(92 GB in `~/.cache/blab-glmroute-profile/Default/File System`).

**Le ipotesi vive, in ordine di costo**: (a) page cache dell'host — 92 GB di
OPFS contro 31 GB di RAM, e il riferimento seguiva altre run GLM che l'avevano
scaldata; (b) frammentazione/crescita dell'OPFS; (c) una regressione vera nel
reader fra il 2026-08-15 e oggi. **La (a) e la (b) non sono codice** e
cambierebbero cosa significa quel riferimento per tutti i confronti futuri.

**Costo stimato per discriminare: 1 iterazione** (una run ripetuta a caldo dopo
una prima che scalda la page cache, contro una a freddo — non serve bisect se la
(a) regge).

**Perché è un item e non un fix sul posto**: se la causa è (a) o (b), la
correzione non è una riga di codice ma **una decisione su cosa dichiara un
riferimento** — e quella è funzione obiettivo, non meccanismo. Anche qui la riga
6 è coinvolta: pretende «GLM b12 optimistic entro ±5% di 13,172 / 31,26 / 14,74»,
e 11,35 è fuori.

**Quello che farei senza risposta**: la run a due bracci del punto (a), perché
costa poco e taglia due ipotesi su tre.

> **MECCANISMO CHIUSO in it.20 — è la (a), e la prova sta dentro l'artefatto del
> riferimento.** Non è servita nessuna run nuova per diagnosticarlo:
>
>     riferimento 2026-08-15, STESSO file, STESSA run
>       warm-up  prefill   19,10 GiB / 6,414 s  =  2,98 GiB/s
>       warm-up  decode     1,51 GiB / 0,460 s  =  3,29 GiB/s
>       REPLICHE decode    24,21 MiB/token / 2,475 ms  =  9,55 GiB/s   <- 2,9x
>
> Lo stesso file, a minuti di distanza, tre volte più veloce della passata che
> l'aveva appena letto: **le repliche non hanno letto, hanno ripreso dalla page
> cache**. Oggi, su quattro run consecutive, warm-up e repliche coincidono a
> 1,31–2,34 GiB/s — la banda del dispositivo — e nessuna si è scaldata.
>
> **Il codice è escluso, verificato**: nessun commit fra `bb3d430` e HEAD tocca
> `glmsource.ts`/`glmmodel.ts`/`residency.ts`/`expertstore.ts`/`glmbench/`, e
> `bytesRead`/`misses`/`evictions` coincidono cifra per cifra fra le due date.
>
> **Corretto ciò che era mio da correggere**: il regime di lettura ora è
> **dichiarato** nell'artefatto (`readGiBs`/`readRegime` per fase e in headline)
> e stampato dal runner, con la soglia `OPFS_DEVICE_CEILING_GIBS` e la sua
> provenienza misurata. Cinque casi in `tests/engine-read-regime.test.ts`
> portano dentro i numeri veri dei due artefatti. Verificato su run vera:
> `readGiBs 1,336 · readRegime "disk"`
> (`bench-glm-4090-b12-readregime-2026-08-16.json`).

**QUELLO CHE RESTA E' TUO, ed è la sola cosa che non decido: il gate.** La riga 6
pretende «GLM b12 optimistic entro ±5% di 13,172 / 31,26 / 14,74». Quei numeri —
e i 15,330 che li hanno superati il 2026-08-15 — sono stati presi **senza che
nessuno sapesse in quale regime**. Tre uscite:

1. **Il riferimento si riprende a cache fredda** e la banda ±5% vale da lì. È
   l'unica che rende il gate riproducibile su qualunque host, ma **butta via il
   confronto storico**: tutti i numeri GLM precedenti diventano non
   confrontabili, e va detto nel consuntivo invece che nascosto.
2. **Il gate diventa condizionale al regime**: si confronta solo con riferimenti
   dello stesso `readRegime`, e una run `os-cache` non può chiudere un gate.
   Conserva la storia, ma da oggi in poi serve un riferimento per regime.
3. **Il gate smette di guardare il wall del decode** e guarda `gpuBusy`, che fra
   le due date si muove dello 0,8% (38,5 → 38,9 ms/token) invece che del 26%.
   Misura il motore invece dell'host — ma smette di misurare ciò che l'utente
   sente, che è il wall.

**La mia lettura, che non è un ruling**: la **2**, e la 1 subito dopo per il solo
GLM b12. La 3 è tecnicamente la più pulita e per questo la più pericolosa: il
57% del token GLM sta **fuori** dalla GPU, e un gate su `gpuBusy` dichiarerebbe
sano un motore che l'utente vede lento.

**Perché non la decido io**: è la definizione di cosa il progetto promette di
non far peggiorare — la funzione obiettivo, non il meccanismo. E il ruling
permanente sulle metriche («non peggiorano mai, banda ±5%») l'hai scritto tu.

> **RULING (PI, 2026-08-16)**, dato come priorità e applicato così:
>
> > «Per aiutarti a prioritizzare il lavoro cerca la leva che ci porti il più
> > vicino possibile ai 45 tok/s sul 35B. GLM al momento è secondario. È un
> > modello di 4 generazioni fa, dovremo comunque buttarlo e passare al 5.2 o al
> > nuovo 5.3 (quando avrà i pesi disponibili). Se funziona bene, altrimenti non
> > sprecherei troppo budget per lui.»
>
> **Conseguenza sui tre item aperti**, e vale per tutti e tre insieme perché la
> priorità è una sola: **il bersaglio è il decode CALDO del 35B**.
>
> - **item 7 (gate GLM)**: il GLM smette di essere un gate di risultato e resta
>   un gate di **non-regressione di correttezza**. Nessun ri-baseline, nessuna
>   run di banda: `readRegime` è già nell'artefatto, quindi chi in futuro
>   confronterà due numeri GLM saprà da sé se sono confrontabili. Budget: zero.
> - **item 8 (riga 2b)** e **item 9 (riga 3)**: entrambe vivono nel regime
>   **sporco** (miss, repair, replay). Il numero dei 45 tok/s è misurato **a
>   caldo, con zero miss**: quelle due righe **non lo toccano**. Deprioritizzate
>   — la 2b chiusa sulla misura (la risposta al «se» era no), la 3 sospesa.
>
> **Questo non chiude i tre item come questioni**: li mette fuori dal percorso
> verso la barra. Se un giorno il regime sporco torna a contare (chat vera,
> arena stretta), si riprendono da qui con le misure già fatte.

**RULING:** v. sopra (PI, 2026-08-16)

## item 8 — il done-when della riga 2b non è raggiungibile sul trasporto che ha (io → PI, it.32)

**PI: serve un ruling. È la taglia di una riga del contratto, non il suo
meccanismo.**

**Il fatto, misurato** (`q35-io-probe-2026-08-16.json`, curva
banda-contro-richieste-in-volo sullo stesso lettore e sullo stesso file, stessi
offset per ogni finestra, riscaldamento scartato):

    tetto del canale HTTP oggi      ~696 MB/s   (massimo a 48 richieste in volo)
    ginocchio della curva            2-4 richieste in volo
    repair reale                     539 MB/s
    prep reale                      ~250 MB/s

**Il done-when chiede «< 1,5 ms per fetch».** Un `readExpert` legge **1,7836 MB**
(misurato in it.31, non più assunto), quindi 1,5 ms significa **1.189 MB/s**.

    HTTP, al suo massimo misurato    696 MB/s  ⇒  2,56 ms   = 1,7x FUORI dal target
    OPFS, misurato in it.20        1.372 MB/s  ⇒  1,30 ms   ✓

**Anche portando il `prep` al massimo che il canale dà, il done-when resta fuori
di 1,7×.** Non è un problema di raggruppamento: è il trasporto.

**Perché è un ruling e non una decisione mia**: il contratto elenca la sorgente
non-HTTP fra i *candidati* — «coalescing, finestra di concorrenza, sorgente
non-HTTP (OPFS)» — cioè come un'alternativa fra pari da misurare. La misura dice
che **è l'unica strada al numero**, e questo cambia la taglia della riga:
raggruppare richieste è una fetta, spostare la sorgente è un'altra cosa. Il 35B
sono **19,46 GiB** che in OPFS vanno prima importati, con tutto ciò che comporta
(spazio su disco, tempo di import, la gestione della cache che it.20 ha già
mostrato essere delicata).

**Le tre uscite:**

1. **Tieni il numero e allarga la riga alla sorgente OPFS.** È l'unica che
   arriva a 1,5 ms. Costo: la fetta di import + il cambio di sorgente, e il
   goal cresce di parecchio.
2. **Abbassa il done-when a ciò che il trasporto può dare.** Portare il `prep`
   al livello del `repair` vale ~2× ed è la sovrapposizione della riga 3; un
   done-when onesto su HTTP starebbe sui **~2,6-3,3 ms**. Chiude una riga vera e
   misurabile senza inventare scope.
3. **Chiudi la riga 2b dichiarandola assorbita dalla riga 3.** it.31 e it.32
   concordano che la leva vera è tenere la pipeline piena attraverso i layer,
   che è il meccanismo già in contratto nella riga 3. Due righe che costruiscono
   la stessa cosa sono una scrittura doppia.

**La mia lettura, che non è un ruling**: la **3, e poi la 2 come done-when della
riga 3**. La 1 è vera ma è un goal suo — e il PI ha già detto una volta, in
questo stesso goal, che le leve globali vengono prima delle specifiche: il
cambio di sorgente tocca tutti i modelli e merita di essere deciso come tale,
non infilato in una riga che parlava di raggruppare richieste HTTP.

> **AGGIORNAMENTO it.33 — la domanda che avevi fatto ha una risposta misurata,
> e rafforza l'uscita 3.**
>
> Il tuo ruling diceva: *«Aggiungi anche il raggruppamento delle richieste http
> **se può dare un boost globale al motore**»*. Il done-when della riga 2b è
> stato scritto per rendere quel «se» verificabile. **Verificato: no, su questo
> trasporto.**
>
>     n      continuo   a raffiche   rapporto
>      8      516,4       460,9       1,12x
>     24      543,0       551,9       0,98x
>     48      665,3       547,9       1,21x
>     96      712,9       671,7       1,06x
>
> Stesso file, stesso lettore, stessi offset: cambia solo la forma. **A 24 le
> due coincidono.** Raggruppare le richieste cambia la banda dello 0,98-1,21×,
> cioè rumore.
>
> E il canale, in **ogni** configurazione provata (1-256 in volo, continua o a
> raffiche), sta fra 460 e 740 MB/s — mentre il `prep` reale ne ottiene 250.
> **Il collo non è nel come si chiedono i byte.** È dentro il path del motore, e
> lì la riga 2b non ha più niente da dire: quel pezzo è la riga 3.

> **RULING (PI, 2026-08-16) — DEPRIORITIZZATA.** La priorità data è «la leva che
> ci porti più vicino ai 45 tok/s sul 35B». Questa riga vive nel regime SPORCO e
> sul LOAD; il numero dei 45 è misurato **a caldo con zero miss**, quindi la riga
> non lo tocca. Chiusa sulla misura: il «se» del ruling precedente ha avuto la
> sua risposta (no, su questo trasporto), e la sorgente OPFS resta un goal suo se
> e quando il caricamento tornerà a essere la priorità.

**RULING:** deprioritizzata, chiusa sulla misura (PI, 2026-08-16)

## item 9 — il done-when della riga 3 è degenere sull'artefatto che nomina (io → PI, it.37)

**PI: serve un ruling. È un done-when, cioè contratto — e la soglia decide la
difficoltà del compito.**

**Il fatto**, letto da artefatti già in `results/` (nessuna GPU spesa):

    arena 12 (il riferimento)   miss 0   replays 0   replayLayers 0   ⇒ ratio 0,000
    arena  4                 miss 3283 replays 111 replayLayers 3702 ⇒ ratio 2,373

Il done-when chiede `replayLayers / (tokens × nLayer) ≤ 0,20` **«sull'artefatto
di riferimento»**, e l'artefatto di riferimento è l'arena da 12 GiB. **Lì il
working set ci sta, non c'è un miss, quindi non c'è un replay: il rapporto vale
zero e la clausola è già soddisfatta.** Tre-quattro iterazioni chiuderebbero
contro un numero a posto senza il lavoro.

**Precedente in questo stesso goal**: la riga 1 aveva la stessa forma di difetto
(`namedFrac` = 0/0 su passata pulita) e it.2 l'ha preso scrivendo la precisazione
in `PHASES.md`. Lì la degenerazione dichiarava *fallita* una riga riuscita; qui
dichiara *riuscita* una riga non fatta, che è la più pericolosa delle due.

**E il valore di partenza è sbagliato**: il contratto dice «oggi 0,87», misurato
**2,373** nel regime dove la clausola ha senso.

**Il bersaglio vero, misurato** (arena 4, braccio ottimistico): il token costa
556,0 ms di cui **393,4 di repair+replay, il 71%**; 39/39 token sporchi; 2,85
giri di replay per token.

**In più, la premessa della riga è smentita.** Il contratto dice «la riga 2 ne
cambia il segno, rimisurare i due bracci prima di scegliere». Rimisurati:

    arena 12   sync 133,35   ottimistico  42,77   ottimistico 3,12x meglio
    arena  4   sync 980,25   ottimistico 556,05   ottimistico 1,76x meglio

Il segno **non** è cambiato. Questo *restringe* la riga: non si sceglie fra due
path, si riduce il replay dentro quello che già vince.

**Le tre uscite:**

1. **Ri-ancora il done-when al regime sporco**: stessa clausola, artefatto ad
   arena strozzata **dichiarata**, valore di partenza 2,373. Serve una soglia
   nuova — 0,20 partendo da 2,373 è un compito diverso da 0,20 partendo da 0,87.
2. **Cambia la grandezza**: la clausola sui layer può scendere mentre il tempo
   no. Il termine che pesa è la **quota di `repair+replay` sul token** (oggi
   71%). Una soglia lì misura ciò che l'utente sente.
3. **Entrambe**: la clausola sui layer come gate strutturale, la quota di tempo
   come gate di risultato.

**La mia lettura, che non è un ruling**: la **3**, con la quota di tempo come
clausola che decide e quella sui layer come controllo che il meccanismo sia
cambiato davvero. **La soglia numerica la metti tu**: sceglierla io sarebbe
scegliere quanto è difficile il mio compito.

**Cosa faccio senza risposta**: eseguo la riga contro il regime sporco comunque
— contro quello a 12 GiB non c'è niente da eseguire — e riporto i numeri senza
dichiarare passato nessun gate.

> **RULING (PI, 2026-08-16) — DEPRIORITIZZATO.** La priorità data è «la leva che
> ci porti più vicino ai 45 tok/s sul 35B». La riga 3 vive nel regime SPORCO,
> e il numero dei 45 è misurato **a caldo con zero miss**: la riga non lo tocca.
> Resta chiusa sulla misura — i cinque esperimenti hanno risposto al «se» del
> ruling precedente, e la risposta era no. Si riprende da qui se il regime
> sporco tornerà a contare.

**RULING:** deprioritizzato (PI, 2026-08-16)

## item 10 — che formato accetta il motore sul disco (io → PI, it.41)

**Contesto**: `packExpertSlab` costa **7,11 s per sessione** sul 35B (misurato,
it.40) e il meccanismo per saltarlo esiste già nel modulo condiviso — il GLM lo
usa da un goal intero, il path Qwen no (it.41). Adottarlo richiede uno store di
slab, e lo store pone una domanda di prodotto.

**Misura che toglie il trade-off** (it.40): lo slab pesa **1,0005×** il GGUF —
19,45 → 19,46 GiB, dieci MiB su venti giga. Il repack riordina i byte, non li
aggiunge. Solo il q6_K si allinga a multipli di 4 (+0,95% su tre layer).

**RULING (PI, 2026-08-16):**

> «accetta entrambi e converte al primo caricamento.»

**Applicato così, e va scritto per esteso perché risolve l'obiezione che il PI
aveva fatto** («la mia idea era avere lo slab al posto del gguf, così non
chiediamo alle persone il doppio dello spazio»):

1. Il motore accetta **un file slab direttamente**: chi tiene allo spazio
   converte una volta, tiene solo lo slab, e il GGUF non gli serve più.
2. Il motore accetta **un GGUF** e lo converte al primo caricamento: è la via
   comoda, ed è esattamente ciò che `ensureSlabs` già fa per il GLM
   (temporaneo + rename, quindi un'interruzione non lascia un file valido a
   metà).

**Quindi la doppia copia è transitoria e sotto il controllo dell'utente, non
imposta.** Era l'obiezione giusta, e la misura di it.40 la scioglie: dato che lo
slab pesa quanto il GGUF, sostituirlo è a costo zero.

**La sotto-decisione che NON porto al PI perché è meccanismo**: dopo la
conversione il motore **non cancella** il GGUF dell'utente — dirà che può
farlo, non lo farà. Cancellare il file di un modello non è una cosa che un
motore fa da solo, e il costo di sbagliarsi è un download da 19 GiB.

**RULING:** entrambi i formati, conversione al primo caricamento (PI, 2026-08-16)

> **VINCOLO TROVATO in it.43, e tocca la SECONDA meta' del ruling.** La quota
> OPFS misurata per l'origine e' **10,00 GiB**, `persist()` viene **negata** e il
> tetto non si alza. Lo slab degli expert del 35B e' **17,07 GiB**: non ci sta.
>
> «Convertire al primo caricamento» richiede un posto dove scrivere, e in un
> browser quel posto e' OPFS. Quindi la seconda meta' del ruling, cosi' com'e',
> non e' eseguibile sul 35B.
>
> **La prima meta' regge e basta a se stessa**: il motore accetta uno slab
> DIRETTAMENTE, convertito offline da uno script e servito come il GGUF via HTTP
> Range. Zero OPFS, zero quota. **Ed e' il disegno che il PI aveva chiesto
> all'inizio** — «lo slab su disco invece del GGUF» — prima che io lo deviassi
> verso una cache. La misura mi ha riportato al suo disegno.
>
> **Non chiedo un ruling nuovo**: procedo sulla prima meta', che e' gia'
> approvata. Se un giorno servisse la conversione lato client, l'item torna
> aperto con la contraddizione di it.43 da chiarire per prima.

## item 13 — lo slab come FORMATO DISTRIBUIBILE, non come cache locale (PI → parcheggiato, 2026-08-17)

**Ruling del PI a valle della misura di it.51**: l'artefatto locale si toglie
(«così non rischiamo di inquinare i risultati»), il **motore si tiene** — costa
zero perché senza file è inerte e lo dichiara in `moeStats().slabSource`.

**La domanda che resta aperta, e non è di questo goal**: lo slab è un formato
*efficiente da distribuire* (per es. su Hugging Face) invece che da generare in
locale? Cioè: vale la pena pubblicare `modello.slabs.bin` accanto al GGUF?

**Cosa sappiamo già, e va letto prima di riaprirla:**
- il pack sparisce davvero (`packMs` 7.331 → **0**, it.51);
- ma la fetch peggiora del 4,6%, perché una richiesta da 1,77 MB non si
  sovrappone a nessuno mentre tre da 594 KB erano in `Promise.all`;
- **netto misurato: +3,7% tok/s** su un secondo artefatto da 17 GiB;
- non è stato provato lo slab letto in **2-4 sotto-range paralleli**, che è dove
  sta il ginocchio della curva banda/richieste-in-volo (it.33) e che
  recupererebbe i ~3,1 s persi. `slabFileRange` è già aritmetica: è una fetta
  piccola, ed è la sola cosa che potrebbe cambiare il segno del giudizio.

**Perché potrebbe valere come prodotto e non come ottimizzazione**: un formato
già impacchettato è *neutro rispetto al motore che lo consuma* — chiunque abbia
un'arena a slot lo userebbe senza pagare un repack. È materia da
`research-campaign` (una tesi, un pre-registro, un memo), non da una riga di
questo goal.

**NON riaprire senza**: (a) la misura dei sotto-range paralleli, (b) il costo di
banda/hosting di un secondo file per modello, (c) la domanda se il layout resti
stabile fra versioni del motore — oggi `SLAB_LAYOUT_VERSION` è un numero che
invalida tutto quando cambia, e un formato distribuito non può invalidarsi a
ogni refactor.

---

# APERTURA VERSO IL RILASCIO PUBBLICO (PI, 2026-08-17)

Il PI ha aperto la fase di preparazione al rilascio. Gli item qui sotto sono
DECISIONI, non lavoro: il lavoro discende da loro. Ordine di dipendenza dove c'è.

## item 14 — la licenza del repo (io → PI)

**Il repo non ha una licenza.** Senza, per default nessuno può usarlo: la
distribuzione pubblica è bloccata da questo prima che da qualunque codice.

**Cosa entra nella scelta**, con la mia lettura accanto:
- **Apache-2.0** — la stessa dei modelli Qwen che serviamo. Ha la clausola
  brevettuale esplicita, ed è quella che l'ecosistema (llama.cpp è MIT,
  transformers.js Apache-2.0) si aspetta da un motore. *È quella che
  consiglierei.*
- **MIT** — più corta e più permissiva, nessuna clausola brevettuale.
- **AGPL** — obbligherebbe chi lo integra a pubblicare: incompatibile con
  l'adozione che un motore cerca.

**Serve anche**, e non è la licenza: un file di ATTRIBUZIONE che dichiari che
non ridistribuiamo pesi, e che i GGUF citati sono di terzi (Qwen Apache-2.0;
i quant di bartowski/unsloth sono loro ripacchettamenti).

### RULING PI 2026-08-17: APACHE-2.0. **Item 14 CHIUSO.**

La clausola brevettuale e l'allineamento con Qwen hanno deciso. Cosa ne
discende come LAVORO, dentro lo split (item 18):
- `LICENSE` (testo Apache-2.0 integrale) alla radice del repo motore;
- `NOTICE` — l'attribuzione: **non ridistribuiamo pesi**, i GGUF citati sono di
  terzi (Qwen Apache-2.0; i quant `bartowski`/`unsloth` sono ripacchettamenti
  loro, con i loro SHA già pinnati in `q35shape.ts`);
- l'intestazione di licenza NON va messa in cima a ogni file: i commenti di
  questo motore sono la sua parte migliore e un blocco legale ripetuto in testa
  li seppellisce. `LICENSE` + campo `license` nel `package.json` bastano.

Il repo bench e il repo paper ereditano la stessa scelta salvo ruling contrario
al momento della loro pubblicazione (entrambi restano in standby).

## item 15 — i pesi si scaricano da Hugging Face (PI, deciso)

> «I pesi possiamo distribuirli facendoli scaricare da huggingface. Così siamo
> sicuri che vengano scaricate le versioni su cui abbiamo già ottimizzazioni.»

**Deciso.** Cosa ne discende, ed è lavoro vero:
- il registro dei modelli deve portare **URL HF + SHA-256 + formati attesi**, non
  un path locale. Lo SHA c'è già (`Q35_SHA256`); l'URL no;
- **CORS e Range su HF**: `huggingface.co/.../resolve/main/...` risponde 302 verso
  un CDN. Il nostro lettore fa richieste Range: va verificato che reggano il
  redirect e che gli header CORS lo permettano da un'origine diversa. **Non
  misurato**: oggi leggiamo da localhost;
- **il tempo di primo utilizzo** (13 GB su rete vera) non è mai stato cronometrato
  e diventa la metrica di prodotto più visibile;
- la quota OPFS (10,00 GiB, `persist()` negata, it.43) **non basta** a cacheare il
  file: o si ri-scarica ogni sessione, o si chiede all'utente un file locale.
  Questa è la domanda aperta più concreta.

## item 16 — tutto ciò che è pubblico è in INGLESE (PI, deciso)

> «Mi raccomando che sia tutto in inglese su quello che distribuiamo
> pubblicamente.»

**Deciso.** Il confine va però tracciato, perché oggi TUTTO è in italiano —
codice, commenti, documenti, artefatti:

    pubblico (inglese)   README, docs/ del repo motore, commenti del sorgente
                         distribuito, messaggi d'errore rivolti all'utente,
                         il paper
    interno (italiano)   .harness/ (journal, docket, GOAL/PHASES), i consuntivi
                         di sessione, questo file

**Il costo è grosso e va detto**: i commenti del motore sono la parte migliore di
questo repo e sono lunghi. Tradurli meccanicamente li rovinerebbe. Proposta:
si traducono **al momento dello split** (item 18), file per file, come parte
della ripulitura — non prima, o si traduce due volte.

## item 17 — la forma di distribuzione del motore (io → PI)

Un motore che gira nel browser non si distribuisce come «app + server API»,
perché non c'è un server. Le forme possibili, non alternative fra loro:

1. **pacchetto npm** (`import { createEngine } from "…"`) — è la forma vera per
   chi integra. Il modello lo sceglie l'applicazione ospite.
2. **pagina demo ospitata** — la prova che funziona. Costo: nessuno se i pesi
   arrivano da HF (item 15); serve COOP/COEP per `crossOriginIsolated`.
3. **Space su Hugging Face** — demo accanto ai pesi, con la loro CDN. È la sede
   naturale per un progetto che consuma GGUF.
4. **estensione/PWA** — offline vero, ma la quota OPFS resta il muro.

**Ciò che manca in tutte e quattro**: una **API stabile**. Oggi la superficie è
`createQ35GpuModel` con venti opzioni e nomi interni (`kfan`, `splitk`,
`vramCeilingBytes`). Un pacchetto pubblico ha bisogno di un contratto piccolo e
di un nome, e di dichiarare cosa NON è stabile.

### RULING PI 2026-08-17: **(1) PACCHETTO npm + (3) SPACE SU HF**. Item 14 chiuso, questo APERTO SUL NOME.

Le due forme scelte sono complementari e non ridondanti: l'npm è la forma per
chi integra, lo Space è la demo che vive accanto ai pesi (item 15) e quindi non
ha bisogno di un hosting nostro né di configurare COOP/COEP a mano. La **(2)
pagina demo ospitata** decade — lo Space la sussume. La **(4) estensione/PWA**
resta fuori: la quota OPFS (10,00 GiB contro 13 GB di file) è un muro che non
si aggira con la forma di distribuzione.

**IL PI HA SOSPESO IL NOME**: «magari pensiamo ad un nome migliore prima di
chiamare il repo browser llm engine». Il nome NON è cosmetico qui — è
simultaneamente il nome del pacchetto npm, dello Space, del repo e del simbolo
importato, e cambiarlo dopo la pubblicazione costa una deprecazione. Lo split
(item 18) **non parte finché il nome non è deciso**: è l'unico suo input
mancante. Candidati e criteri: sezione «item 20».

Resta vero, e diventa lavoro appena il nome c'è: la superficie pubblica oggi è
`createQ35GpuModel` con ~20 opzioni dai nomi interni (`kfan`, `splitk`,
`vramCeilingBytes`). Un pacchetto ha bisogno di un contratto piccolo E di dire
per iscritto cosa NON è stabile.

## item 18 — separare le repo adesso (io → PI)

Il ruling del 2026-07-30 dice motore e benchmark separati alla pubblicazione. Il
PI chiede se conviene farlo **ora**.

**La mia lettura: sì, e prima della traduzione.** Tre ragioni:
1. lo split DEFINISCE cosa è pubblico, e senza quel confine l'item 16 non è
   eseguibile (non si sa cosa tradurre);
2. farlo ora costa poco: il motore è `src/engine/**` + i test, e le dipendenze
   verso l'harness sono poche. Farlo dopo la ripulitura significa ripulire anche
   ciò che poi si butta;
3. il repo oggi contiene 25 GB di artefatti, worktree e risultati: il repo
   pubblico deve nascere **senza quella storia**, non ripulito dopo.

**Il rischio da dichiarare**: due repo raddoppiano il costo di ogni cambiamento
che li attraversa, e per un po' li attraverseranno tutti. Va deciso chi è la
fonte di verità durante la transizione.

### RULING PI 2026-08-17 (chiusura sessione): SÌ, E PRIMA DI TUTTO. **Item 18 CHIUSO come decisione.**

«Farò lo split come prima cosa in una sessione fresca». I parametri restano
quelli congelati in `docs/publishing/split-plan.md` (tre repo, `filter-repo` per
path, su cloni freschi, MAI sul lab).

**FONTE DI VERITÀ durante la transizione** — il rischio che l'item stesso
dichiarava, e che il ruling non copriva: **resta il lab**. I tre repo nascono
per estrazione e, finché il PI non dice il contrario, si RIGENERANO dal lab
invece di ricevere commit propri. Un repo pubblico che diverge dal lab prima di
avere un pubblico è solo un secondo posto dove sbagliare.

**ESECUZIONE, stato al 2026-08-17:**
- ✅ *igiene dei worktree*: i 5 worktree SDD (198 MB) sono stati verificati file
  per file contro main — non «zero commit», che era già noto, ma **contenuto**:
  `git show main:<f> | diff - <worktree>/<f>` su ogni file sporco. Main è
  strettamente più avanti ovunque (il worktree diceva `F32_ONLY` dove main ha
  `ROUTER=[F32,BF16]`; diceva «MAI ESEGUITI SU GPU» dove main ha il ktest 4/4;
  inlinava `blankNonCode` dove main importa `tests/helpers/source-scan.ts`).
  Nessun lavoro perso. **Rimozione BLOCCATA dal classifier di auto-mode**
  (`git worktree remove --force` + `git branch -D`): serve autorizzazione.
- ❌ *`git-filter-repo` NON è installato* — né comando git né modulo Python.
  `uv tool install git-filter-repo` **bloccato dal classifier**. È il primo
  passo del piano e senza non si parte.
- ⛔ *BLOCCANTE VERO*: **il nome** (item 20). Lo split non parte senza.

## item 19 — la matrice dei dispositivi (PI, in carico al PI)

> «Farò le prove su m4 pro 48 gb e s22 ultra e le includeremo nei risultati del
> paper.»

Tutti i numeri di questo progetto vengono da UNA macchina. Perché quelle prove
siano confrontabili serve, da parte nostra: un runner che gira senza il nostro
harness, un artefatto che dichiari il dispositivo, e una soglia di successo
dichiarata prima (il 35B su 48 GB unificati è un regime diverso; sull'S22 il
modello non ci sta e la domanda è quale modello ci sta).

## item 20 — il nome del motore (PI, sospeso da lui sull'item 17)

> «Magari pensiamo ad un nome migliore prima di chiamare il repo browser llm
> engine.»

Il nome è l'ULTIMO input mancante dello split (item 18) e non è cosmetico: è la
stessa stringa in quattro posti — repo GitHub, pacchetto npm, Space HF, e il
simbolo che chi integra scrive nel proprio sorgente. Cambiarlo dopo la
pubblicazione costa una deprecazione.

**IL VINCOLO npm È MOLTO PIÙ DEBOLE DI QUANTO SEMBRI** (verificato 2026-08-17,
`npm view`): quasi ogni parola singola risulta occupata, ma da SQUAT MORTI —
`atrium` 0.0.0, `quorum` 0.0.0-1, `arena` 0.0.2, cioè pacchetti mai pubblicati
davvero. Liberi: `synod`, `gguf-web`, `webgguf`, `ggufjs`, `residency`.
Occupati: `conclave` 1.0.0, `tessera` 0.15.5, `ardesia` 0.3.0, `bottega`,
`caldera`, `slabs`. **La via d'uscita standard è il pacchetto SCOPED**
(`@scope/nome`), che azzera del tutto la contesa. Repo GitHub e Space HF stanno
in namespace PERSONALI (`MissingPackage/*`, `CriM91/*`): lì qualunque nome è
libero. Quindi npm non deve dettare la scelta.

**IL FATTO CHE HO TROVATO GUARDANDO I REPO DEL PI** — `ardesia` non è un
candidato, è una FAMIGLIA già esistente: `ardesia-gguf` («LoRA-over-GGUF
training stack») e `ardesia-unsloth` («config-driven SFT→DPO→GRPO»), entrambi
Python, entrambi ADDESTRAMENTO. Riusare `ardesia` per un motore di INFERENZA nel
browser blurra due cose diverse: chi trova l'uno si aspetta l'altro. La
continuità di brand costa qui più di quanto renda.

**Il registro di nomi del PI**, letto dai suoi repo: `nightshift`,
`arcana-screen`, `gitgud` — parola singola, evocativa, MAI descrittiva.

**I candidati, con la ragione accanto:**
- **`conclave`** ⭐ — la MoE *è* un conclave: a ogni token una piccola assemblea
  scelta di expert si riunisce, delibera, si scioglie. È l'unico nome che dice
  l'ARCHITETTURA su cui questo motore è specializzato, invece della piattaforma
  su cui gira. Non collide con `ardesia-*`. npm scoped.
- **`tessera`** — doppio senso reale: tassello di mosaico E biglietto/gettone
  (token). Rende bene lo slab streammato. Ma legge come struttura dati, e non
  dice MoE.
- **`synod`** — stessa metafora di `conclave`, e LIBERO su npm senza scope. Ma
  più opaco per un lettore non italiano/cattolico.
- **`gguf-web` / `webgguf`** — liberi, e cercabili da chi digita «run gguf in
  browser». La contro-argomentazione: descrittivo = dimenticabile, e questo
  motore non è «gguf nel browser», è «modelli più grandi della VRAM che girano
  lo stesso».

**RULING:** _

### RULING PI 2026-08-17: **`webgguf`**, sotto il namespace personale. **Item 20 CHIUSO.**

Il PI ha rovesciato la linea evocativa con un argomento che accolgo: «i nomi
evocativi finiscono per non avere successo su GitHub». Su un repo che cerca
adozione la discoverability batte la memorabilità, e `conclave` non contiene
nessun termine che qualcuno digiti.

**`webgguf`**: libero su npm SENZA scope, **zero repo omonimi su GitHub**.

**NAMESPACE: personale (`MissingPackage/webgguf`), e la decisione è REVERSIBILE**
— un repo si trasferisce a un'org dopo, con redirect permanenti dal vecchio URL;
il pacchetto npm è unscoped e non ne risente. L'org si crea se e quando bench e
paper diventano pubblici davvero, non prima.

### IL VICINATO, misurato (2026-08-17, GitHub API) — perché cambia la DESCRIPTION

Il PI ha chiesto «qualcuno è arrivato prima di noi?». La risposta è no, ma il
motivo per cui è no va scritto, perché detta come ci si presenta.

| repo | ⭐ | cos'è davvero |
|---|---|---|
| `mlc-ai/web-llm` | 18.567 | browser, WebGPU, ma modelli **compilati MLC — non GGUF** |
| `huggingface/transformers.js` | 16.261 | browser, WebGPU, ma **ONNX** |
| `Michael-A-Kuykendall/shimmy` | 5.755 | «GGUF-native WebGPU» ma è un **server Rust con API OpenAI** |
| `ngxson/wllama` | 1.167 | GGUF nel browser, ma **WASM/CPU**: niente WebGPU |
| `huggingface/ratchet` | 767 | WebGPU + quant GGUF nel browser — **il più vicino**; fermo dal 2026-05-26 |
| `airframe` / `flarellm` / `Sipp` | 7-107 | motori Rust WebGPU GGUF, non in scheda |

I 18 repo `gguf-web*` che avevano spaventato il PI sono **webui che parlano con
un server** llama.cpp/Ollama: interfacce, non motori. Non sono concorrenti — ma
sono lo scaffale in cui si finisce presentandosi come «GGUF in the browser».

**LA CASELLA VUOTA È LA NOSTRA**: GGUF letto via Range in una scheda, WebGPU,
expert PAGINATI per far girare modelli più grandi della VRAM. Il differenziatore
non è il formato, è **un 35B MoE a 34,97 tok/s in una scheda**, dove web-llm in
pratica si ferma sui 7-8B densi.

**CONSEGUENZA OPERATIVA, non un commento**: la description del repo e la prima
riga del README aprono sul 35B nella scheda, NON su «run GGUF in your browser».
Il rischio di questo progetto non è che qualcuno sia arrivato prima — è essere
invisibile accanto a 18.567 stelle dicendo la cosa che dicono già in venti.

## item 21 — la mappa delle capability: due browser, due strade disgiunte (io, registrazione)

Misurato il 2026-08-17 sull'adapter REALE (non dalle note, non a memoria).
Registrato perche' e' il vincolo che decide quali ottimizzazioni sono
perseguibili, e su quale macchina.

| feature | Chrome 151/Linux/NVIDIA | Firefox 153, stessa scheda | M4 Pro, S22 |
|---|---|---|---|
| `shader-f16` | **NO** | **SI** | SI |
| `subgroups`, `subgroup-size-control` | **SI** | NO | ? |
| `chromium-experimental-subgroup-matrix` | **SI** | NO | ? |
| `timestamp-query` | SI | SI | SI |
| feature totali | 22 | 11 | — |

**`shader-f16` non e' un problema di driver.** Le quattro condizioni che Dawn
pretende (`PhysicalDeviceVk.cpp:373-378`) sono TUTTE vere qui: `shaderFloat16`,
`shaderInt16`, `storageBuffer16BitAccess`, `uniformAndStorageBuffer16BitAccess`,
su NVIDIA 610.57.04. L'adapter risponde comunque `false`, sotto CINQUE
combinazioni di flag (`allow_unsafe_apis`, `webgpu-developer-features`,
`disallow_unsafe_apis`, `use_dxc`, baseline). **Il cancello sta sopra Dawn, nel
build Linux di Chrome.** Nessun flag lo apre.

**Firefox la espone ma non e' una piattaforma di misura**: 9,92 tok/s dove
Chrome ne fa 89,5 sullo stesso modello — e **identico al 9,91 misurato il
2026-07-25**, quindi e' stabile e reale, non rumore. Tre settimane non l'hanno
mosso. (Caveat: il runner non verifica l'adapter su Firefox perche' nasconde il
vendor; potrebbe essere rendering software. Si accerta solo da `about:support`.)

**Conseguenza che nessuna delle due righe dice da sola**: f16 e subgroup-matrix
**non sono ottenibili nello stesso browser su questa macchina**. Non e'
«aggiungiamo f16»: sono DUE strade di ottimizzazione su DUE piattaforme.

### Il subgroup-matrix, e il vincolo che lo lega a f16

Il driver ha `VK_KHR_cooperative_matrix` (rev 2), `VK_NV_cooperative_matrix2` e
`VK_NV_cooperative_vector`: i tensor core ci sono a livello Vulkan, e Chrome
espone la feature WebGPU.

**MA** (`PhysicalDeviceVk.cpp:1774-1779`): le configurazioni con componenti
**F16 richiedono `ShaderF16`**. Su Chrome/Linux, dove f16 non c'e', il
subgroup-matrix sarebbe utilizzabile **solo con componenti F32** — ammesso che
esistano config F32 su Lovelace, che NON e' stato verificato.

Cioe': i due vincoli non sono indipendenti. La macchina che ha il
subgroup-matrix e' la stessa che non ha f16, e senza f16 il subgroup-matrix
perde la meta' delle sue configurazioni.

**RULING / PRIORITA': _** (il PI ha chiesto di metterlo fra le cose da fare
«se porta ulteriori possibilita' di ottimizzazione». La condizione e' oggetto
della consulenza aperta — vedi sotto — perche' i GEMV del decode sono
memory-bound, e una unita' matriciale non aiuta chi aspetta la memoria.)

### Consulenza aperta (PI, 2026-08-17)

> «Utilizza un agente fable come consulente esperto e consultati con lui per
> definire il da farsi sulle ottimizzazioni. Non perdere di vista l'obiettivo
> che e' avere il miglior motore web per llm (possibilmente in formato gguf)
> al mondo!»

Domande poste: (1) il subgroup-matrix vale, dato che il decode e' memory-bound
e su questa macchina sarebbe F32-only — e cambia risposta sul prefill, che e'
compute-bound? (2) una classifica delle ottimizzazioni per impatto sulla
funzione obiettivo, incluse le 4 di WebLLM e le nostre non sfruttate, con cosa
NON fare; (3) quanto vale f16 su un motore i cui kernel sono memory-bound su
pesi gia' a 2-6 bit (f16 tocca attivazioni e KV, non i pesi) — e vale spostare
lo sviluppo su Mac; (4) cosa stiamo trascurando che decide se questo motore e'
il migliore al mondo o solo un buon esperimento.

### ESITO DELLA CONSULENZA (agente fable, 2026-08-17) — verificata sui dati prima di essere accolta

**SUL SUBGROUP-MATRIX (la domanda del PI): NON metterlo fra le cose da fare, oggi.**

- *Sul decode, per costruzione*: le cooperative matrix pagano dove c'e' riuso di
  tile (GEMM, M>=8). A M=1 non c'e' tile — ogni peso serve UNA volta. Non e' che
  aiuta poco: non c'e' proprio il problema che risolvono.
- *Sul prefill, che sarebbe il candidato giusto*: il probe gia' committato
  (`webgpu-subgroup-matrix-probe-2026-08-10.json`) dice che la feature e'
  inutilizzabile qui — le config u8 non compilano in Tint, le f16 vogliono
  `shader-f16` che Chrome/Linux non da', e le eventuali f32 su NVIDIA non
  mappano sui tensor core (che vogliono f16/tf32). Si otterrebbe al piu' cio'
  che un GEMM a tile in workgroup storage ottiene in WGSL **core**.
- *E il WGSL core lo copre gia'*: il memo headroom proietta il prefill a 150-300
  tok/s col GEMM multi-riga PORTABILE.
- *Ed e' `chromium-experimental`*, un solo browser, dietro flag.

**RIVALUTARE quando**: (a) esce da experimental, (b) il GEMM a tile portabile e'
in produzione e misurato, (c) `shader-f16` compare su >= 2 piattaforme target.
**Tenere il probe nel rituale di release di Chrome**: costa 10 minuti.

**DUE PREMESSE CORRETTE, entrambe verificate da me sui file prima di accoglierle:**

1. **«i GEMV del decode sono memory-bound» e' FALSO in senso operativo**, ed era
   una mia glossa. `micro-bench-matmul.md:63,79`: la banda reale della scheda e'
   **435 GB/s misurati**, il GEMV **f32 la satura al 100%**, i quantizzati stanno
   al **20%**. Il collo e' il **costo ALU della dequant** e la forma del kernel
   (loop scalari per nibble, load u32 scalari, occupancy a M=1) — NON la DRAM.
   Il 17,2 GB/s contro ~500 non prova che siamo memory-bound: prova il contrario.
   Riferimento esterno sullo stesso silicio: llama.cpp Vulkan fa ~500 G pesi/s
   dove noi ne facciamo 133.
   **Conseguenza**: f16 e subgroup-matrix attaccano la densita' di calcolo, che
   non e' il vincolo. Il vincolo si attacca con vec4, coalescenza, multi-riga,
   occupancy.

2. **L'oracolo del prefetch e' 82,67% @K=8 sul 35B, non 91,92%** — quello e' di
   GLM (`direction.md:307`, spiegato: softmax-256 senza bias contro
   sigmoid+bias-64). MECCANISMI.md riportava il numero di GLM parlando del 35B
   in DUE punti, ed e' da li' che l'ho preso io e che l'ha preso l'HANDOFF.
   **Corretto in MECCANISMI.md §5 e §7.** Miss ridotti ~6x, non ~12x.

**LA CLASSIFICA, per «bit di intelligenza comprabili per iterazione»** — la
chiave e' che a 34,6 tok/s abbiamo ~13% di margine sopra il pavimento, e ogni ms
guadagnato non serve a salire di tok/s: serve a essere **speso in bit**.

  0. **micro-bench delle varianti GEMV** (1 giorno, zero run di modello): decide
     se la pendenza del punto 2 esiste, PRIMA di toccare il motore.
  1. **prefetch lookahead** — l'unica leva che compra BIT invece di ms. Il
     Q4_K_S (17,07 GiB, 65% residente) fa ~18 tok/s; il Q2_K ci sta tutto e fa
     34,6. La differenza fra i due e' fedelta' pagata alla quota VRAM. E'
     l'unica leva che puo' portare una quant PIU' RICCA sopra i 30. In piu'
     ripara il transitorio (turno freddo 11,6, 7.930 miss al primo turno). Ed e'
     la leva che nessun concorrente ha: **WebLLM non ha MoE affatto**.
  2. **riscrittura dei GEMV quantizzati** (vec4 + subgroupAdd + 2-4 righe/WG +
     probe `dot4I8Packed`): headroom dimostrato **3,7x** sullo stesso silicio.
  3. **attenzione a contesto lungo** (split-K sul contesto + KV f16 via
     `pack2x16float`, che e' **WGSL core e non richiede shader-f16**). Il kernel
     di scansione KV gira al **1,4% del picco** e a ctx 6333 il 4B crolla da
     25,9 a **9,95 tok/s**. Vedi rischio (b).
  4. **i ~11 ms di round-trip non attribuito** — ma con la diagnosi giusta:
     it.28 ha gia' smontato l'ipotesi CPU (la CPU vera e' 1,94 ms = 4,5%).
     Sonda prima, costruzione dopo.
  5. letture slab in sotto-range paralleli (igiene, -3,1 s sul load).
  6. **spec-dec MTP: PARCHEGGIATA.** Il checkpoint B l'ha misurata piu' LENTA
     coi kernel attuali, e l'1,29x proiettato e' sotto quello che la leva 2
     promette da sola.

**LE 4 LEVE DI WEBLLM, giudicate:**
- **f16** -> marginale sul nostro collo (i pesi sono gia' a 2-6 bit e f16 non li
  tocca; su NVIDIA l'fp16 scalare non e' piu' veloce dell'fp32). L'unico termine
  dove dimezzare i byte paga e' la **KV a contesto lungo**, e quella si dimezza
  con `pack2x16float` **oggi, in WGSL core, ovunque**.
- **TVM/autotuning** -> **NON adottare**: contraddice l'identita' del motore
  (zero dipendenze, GGUF in place, nessuna compilazione), che e' meta' del
  vantaggio. Ma il PROBLEMA che TVM risolve per loro e' il nostro rischio n.1.
  La risposta zero-dipendenze e' una **fabbrica di kernel parametrica +
  autotune on-device al primo caricamento**, col microbench harness che abbiamo
  gia', eseguito sul device dell'utente e persistito.
  Nota: sul 0.5B denso li battiamo 2,7x — **il loro compilatore non produce
  kernel migliori dei nostri a mano su questo hardware**. Il confronto che conta
  non e' WebLLM: e' **llama.cpp nativo**, 3,4x sopra di noi sul 4B.
- **quattro backend di cache** -> prodotto, non regime. Media priorita'.
- **service worker** -> accessorio del precedente.

**DA NON FARE, esplicito**: subgroup-matrix ora; migrare lo sviluppo su Mac;
adottare TVM; vocab ridotto sulla lm_head (rompe il ratchet di fedelta' per 2,1
ms); inseguire l'ampiezza di catalogo di WebLLM (il fossato e' il MoE
piu'-grande-della-VRAM, non 231 modelli); e tutto cio' che MECCANISMI.md ha gia'
seppellito coi numeri (policy tier, raggruppamento I/O, idot nel decode, slab).

**SUL MAC: NO, e la mossa giusta e' l'inversa.** Trasferire lo sviluppo
perderebbe i probe subgroup-matrix, il confronto con llama.cpp Vulkan sullo
stesso silicio e tutta la base di misure — per una feature che vale poco sul
nostro collo. Tenere la 4090 come banco PRIMARIO e aggiungere M4/S22 come
dispositivi di **verifica**. Piu' due azioni economiche: (a) aprire un crbug su
`shader-f16` Chrome/Linux/NVIDIA — il dossier e' gia' pronto (le quattro
condizioni di Dawn sono soddisfatte, il cancello sta sopra); (b) tenere il
codice **f16-ready** (la KV pack2x16 lo e' per costruzione) cosi' quando il
gate cade si accende un flag invece di riscrivere un motore.

**I TRE RISCHI STRATEGICI, in ordine di gravita':**

**(a) UN SOLO DISPOSITIVO MISURATO** — ed e' scritto nel nostro stesso
VALUTAZIONE §10. «Il miglior motore web del mondo» e' un claim di PIATTAFORMA:
kernel tarati a mano su Ada possono essere mediocri su Apple Silicon e pessimi
su Adreno. Il fattore 9 fra Chrome e Firefox *sulla stessa macchina* e' la
misura di quanto poco sappiamo. WebLLM ha costruito TVM esattamente per questo.
Costo d'ingresso minimo: **far girare ktest + chat-smoke su M4 e S22 questa
settimana, prima di ottimizzare alcunche' per loro**. E verificare che il motore
giri su **Chrome stable SENZA `--enable-unsafe-webgpu`**: un motore dietro flag
non e' un prodotto web.

**(b) IL NUMERO DI PUNTA E' A CONTESTO CORTO**: 34,6 tok/s a ~8k misurati contro
262k dichiarati dal modello, con un kernel di attenzione all'1,4% del picco. Se
qualcuno fuori dal progetto rifa' la misura con un prompt da 16k e il regime
crolla, **il claim non si difende**. Misurare la pendenza del 35B costa UNA run.

**(c) IL PRIMO UTILIZZO NON E' MISURATO NE' RISOLTO** (VALUTAZIONE §10 lo
ammette). Il fossato — modelli piu' grandi della VRAM — implica strutturalmente
file enormi: senza una storia di storage, ogni visita e' un pull da 13 GB e
nessuno torna. E' l'unico punto dove le leve 3-4 di WebLLM sono un vantaggio
vero. Prima verifica economica: **la quota da 10 GiB e' del disco o dell'origin?**
Chrome concede tipicamente ~60% del disco libero.

**RULING DEL PI: _** (la classifica e i tre rischi cambiano l'ordine del lavoro
di rilascio: non li applico senza ruling.)
