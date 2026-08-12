# HANDOFF — browser-llm-lab   (aggiornato 2026-08-12)

## 1. Next decidable

**Fase 7 in corso: far predire due token per volta invece di uno.** Il modello
Qwen porta con sé una testa ausiliaria (MTP) che, guardando lo stato interno,
tira a indovinare il token successivo-successivo: se indovina spesso, ogni
passata ne produce due.

L'ho fatta girare in CPU: indovina il **74%** delle volte nel regime che conta,
quello in cui il decode gira davvero — sul testo che il modello sta producendo
lui. Sul testo umano (più difficile, e non è il caso d'uso) fa il 50%.

(Il 31,8% di ieri era rumore, non un difetto: 22 posizioni, dove un solo colpo
vale 4,5 punti. E l'anomalia che sembrava un bug non lo era — spostare di uno le
posizioni non può cambiare niente, quella rotazione guarda solo le *distanze*
fra posizioni. Dettagli in journal it.51, insieme al confronto con
l'implementazione di riferimento di vLLM: combacia voce per voce.)

**La testa gira già sulla GPU** e dà gli stessi numeri del riferimento CPU
(scarto relativo 2,6 su 10 milioni): il blocco è cablato e testato, e non è
servito scrivere un solo kernel nuovo.

**Prossimo passo, non serve una tua decisione**: infilare la testa nel modello
vero — caricarne i pesi accanto agli altri, darle la sua memoria di contesto — e
poi chiudere il ciclo: la testa propone un token, il modello lo verifica nella
stessa passata, si tiene solo se coincide. Gate secco: i token accettati
identici a quelli di oggi. Riferimenti rieseguibili: `Q35_MTP=1 npx vitest run
tests/engine-q35-mtp-accept.test.ts` (~5 min, CPU) e `node
.harness/tools/engine-ktest.mjs` con `npx vite` acceso (~2 min, GPU).

## 2. Mappa

**Destinazione.** Far girare in browser il modello più capace possibile
restando usabile: almeno **30 token/secondo** e **primo token entro 4 secondi**.

**Distanza adesso**, e nessuna configurazione ci arriva: GLM-4.7-Flash **13,4
tok/s** con **14,5 s** al primo token; Qwen 35B **22,6**; Qwen 4B **25,9**. Con
la testa MTP al 74% il 4B arriva a ~39 e il 35B a ~34; anche allo scenario
prudente (50%) sono ~34 e ~29,4. Il costo del draft è ~15% di una passata, non
il 3% che avevo scritto ieri: rifà anche la proiezione sul vocabolario.

**Decisioni prese** (indice: il contenuto vive nel posto indicato, non qui)

- Il checkpoint di misura ridotto a vero merge gate — `PHASES.md` riga 6
- Il repack dei pesi resta dov'era, reso 6 volte più veloce — docket, "la fase 2 ha centrato l'obiettivo ma non la lettera"
- Il decode ottimistico è il percorso attivo; la soglia d'ingresso esclusa coi numeri — docket, "i tre numeri per tarare la soglia"
- Prefetch e politica a livelli esclusi perché misurati inutili — journal it.37
- I riferimenti di velocità Qwen di ieri misurano il percorso vecchio: da rifare — docket, "i riferimenti misurano il path sync"
- Il confronto col motore nativo non è a parità — docket, "il gap nativo e la page cache"

**Nebbia** (non ancora deciso né specificato)

- Quanto valga davvero il 74%: 23 posizioni danno 52-90%, e su un prompt solo
- Se il 35B abbia lo stesso accept-rate: mai misurato (in CPU costerebbe ore)
- Se GLM possa avere una testa equivalente: mai valutato
- Cosa debbano misurare i "riferimenti" — il percorso storico (confrontabile) o
  quello di oggi (vero). Nessuno dei due da solo basta

**Fuori scope**

- I benchmark comparativi fra stack: un goal chiuso, l'altro in pausa dichiarata
- Il raggruppamento degli expert e il pipelining del decode: registrati, non aperti
- Tutti i numeri destinati alla pubblicazione: si rimisurano al tag di release

## 3. Landmines

- **I JSON in `results/` possono mentire in silenzio.** I runner uscivano con
  successo anche quando la GPU falliva, scrivendo numeri plausibili. Ora c'è una
  sentinella su tutti e quattro, ma i file scritti PRIMA di oggi non sono
  protetti.
- **Gli 8 riferimenti Qwen del 2026-08-11 (`*fase-d-it43*`) misurano il percorso
  vecchio**: non usarli per la pubblicazione.
- **La cache del filesystem falsa il confronto col motore nativo del 35-41%.**
  Nessuno dei due artefatti dichiara il proprio stato di cache.
- **Mai confrontare misure fra macchine senza stato dell'host dichiarato**: la
  variazione fra due run identiche è ~2,4%, più della distanza dal riferimento.
- **Un tetto di 16 GiB di VRAM non esiste su questa scheda** (16.376 MiB
  totali): la richiesta sfonda.
- **Un campione da 22 posizioni non distingue niente**: ±1 colpo vale ±4,5
  punti, e it.50-51 ci ha perso due iterazioni. Prima di concludere da un
  conteggio di successi, guarda rango e log-prob del bersaglio: stessa
  informazione, varianza molto più bassa, spesso costo zero.
- **Leggere i parametri di un runner prima di spenderci sopra minuti di GPU.**
  Due volte in una sessione ho lanciato run con flag sbagliati dedotti invece
  che letti.
