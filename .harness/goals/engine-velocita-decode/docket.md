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
