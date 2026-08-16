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

**RULING:** _
