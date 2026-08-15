# DOCKET — engine-kquant

Le decisioni REGISTRATE E NON PRESE, in attesa di ruling del PI. Ordine e
meccanismo non stanno qui: quelli li decido io e finiscono nel journal.

## item 1 — plan-check (PI)

`PHASES.md` scritto in it.0 e mostrato al PI in chat: sette righe, fase 0 in
testa con regola di stop, gate di merge dichiarato come tale, quattro
correzioni di fattibilita' (C0-1..C0-4) trovate nel test-fit e registrate nel
file prima della tabella.

**Il PI ha autorizzato la marcia nello stesso messaggio che ha chiuso il
brief**: «Procedi in loop fino alla chiusura del goal». Tratto quella frase come
approvazione del piano e parto dalla riga 1 — che e' anche la piu' reversibile
del goal (sole misure, nessuna riga di produzione toccata), quindi il costo di
un ripensamento sulla tabella e' il piu' basso possibile proprio all'inizio.

**Resta aperto in un senso solo**: se il PI vuole cambiare una riga, il momento
e' il primo digest. Registro qui che l'approvazione e' stata implicita e non
esplicita, invece di dichiarare il gate superato in silenzio.

**RULING:** _

## item 2 — il ktest non gira piu' in questa sessione, e NON e' il kernel nuovo (io → PI, it.6)

**Il gate di conformita' su GPU vera e' INDISPONIBILE da ~01:00 del 2026-08-15.**
Non lo dichiaro superato, e non dichiaro chiusa la riga che ne dipende.

**Cronologia, misurata:**
- **00:52 — ktest 103 PASS / 0 FAIL**, coi due casi nuovi del q5_K (riga 2)
  verdi a maxRel 2,61e-7 e 4,28e-7. E' l'ultima run valida.
- Da li' in poi, **cinque fallimenti consecutivi** con **tre sintomi diversi**:
  `ERROR: Failed to fetch` (due volte, a 65 e 67 PASS), `Target crashed`,
  `Failed to execute 'mapAsync': A valid external Instance reference no longer
  exists` (device WebGPU perso), e due TIMEOUT a 600 s con «pagina non
  leggibile».

**ATTRIBUZIONE, per esclusione e non per congettura:**
- **NON e' il banco q4_1 nuovo**: disattivata la sola registrazione
  (`results.push(...testPrefillGemmQ41MultiRow)`), la run fallisce **lo stesso**,
  allo stesso punto. E' l'osservazione che discrimina, ed e' costata una run.
- **NON e' il lock del profilo**: fallisce anche su un profilo Chrome nuovo.
- **NON e' il server**: `curl` risponde 200 prima e dopo ogni run, e il log di
  vite non porta errori.
- **NON e' (solo) il Chrome del Playwright MCP** lasciato aperto da un
  subagente dei workflow (2h57 di vita, 157 MiB di GPU, avviato subito dopo
  l'ultima run riuscita — la cronologia combaciava). Ucciso: la run successiva
  fallisce ancora.
- **Il punto di rottura e' stabile**: sempre subito dopo
  `q35-mtp-head-real-blk32`, cioe' il primo caso che carica pesi reali del 4B.
  Con e senza il banco nuovo.

**Ipotesi residua, non verificata**: pressione di memoria host/GPU accumulata in
una sessione lunga (tre run di banco headed, una di conformita', sei tentativi
di ktest). Il discriminante sarebbe un ambiente pulito — riavvio della sessione
o della macchina — e **non e' una cosa che decido io** su una macchina di
qualcun altro.

**COSA RESTA VERO E COSA NO:**
- riga 2 (Q5_K): kernel **verificato su GPU vera** alle 00:52. Regge.
- riga 3 (Q4_1): kernel scritto, cablato, con floor test e pavimenti derivati
  (`[q41 idot] floor rel=1,693e-5, tol 2e-4 = 11,8x sopra`), suite senza GPU
  verde a **836 passed | 10 skipped** — ma **MAI ESEGUITO SU GPU**. La riga 3
  resta APERTA, e il suo caso ktest e' scritto e registrato: la prossima
  sessione lo esegue e basta.

**RULING RICHIESTO:** _

**COSA FACCIO SE NON ARRIVA RISPOSTA**: alla ripresa, prima di qualunque altra
cosa, ri-eseguo il ktest su ambiente fresco. Se passa, la riga 3 si chiude e il
goal prosegue con la riga 4; se fallisce di nuovo allo stesso punto, il difetto
e' del banco `q35-mtp-head-real-blk32` o dell'infrastruttura, e diventa lavoro
suo — non una tassa su ogni goal che passa di qui.

### CHIUSO il 2026-08-15 — l'ipotesi residua era quella giusta

**Il PI ha eseguito il discriminante** (riavvio della macchina) e il ktest e'
tornato verde al primo tentativo utile: **105 PASS / 0 FAIL**, adapter
`nvidia lovelace`, `q35-mtp-head-real-blk32` PASS e tutti i 30+ casi che lo
seguono PASS. Nessuna riga di codice cambiata fra l'ultimo fallimento e questa
run: l'unica variabile mossa e' stata l'ambiente.

**Attribuzione definitiva**: stato accumulato dell'host/GPU in una sessione
lunga, non un difetto del banco `q35-mtp-head-real-blk32` ne' del kernel q4_1.
Non diventa lavoro suo. Non c'e' una tassa da pagare su ogni goal.

**Cosa resta come regola operativa** (mio, non serve ruling): quando tre sintomi
diversi convergono sullo stesso punto di rottura E la disattivazione del codice
nuovo non cambia l'esito, l'attribuzione all'ambiente e' gia' completa. Da li'
il passo giusto e' chiedere il riavvio, non ritentare: i due tentativi in piu'
che ho fatto in it.6 erano speranza, non misura.

**RULING: non serve piu'.** Item chiuso dall'evidenza.
