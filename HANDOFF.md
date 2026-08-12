# HANDOFF — browser-llm-lab   (aggiornato 2026-08-12)

## 1. Next decidable

**Fase 7 in corso: far predire due token per volta invece di uno.** Il modello
Qwen porta con sé una testa ausiliaria (MTP) che, guardando lo stato interno,
tira a indovinare il token successivo-successivo: se indovina spesso, ogni
passata ne produce due.

**Il ciclo funziona ed è dimostrato corretto; quello che manca è la
convenienza.** Indovina il 50% delle volte sul testo umano e il **74%** nel regime in cui
il decode gira davvero — sul testo che il modello sta producendo lui. Il valore
sulla GPU coincide col riferimento CPU nel conteggio esatto (31 su 62), e un
draft costa **5,5 ms** contro i 34,5 di un token: il 16%. (Il 31,8% di due
iterazioni fa era rumore di campionamento, e l'anomalia che sembrava un bug non
lo era: journal it.51.)

**Il ciclo è chiuso e il gate secco è passato**: generando 16 token con la
proposta della testa si ottengono gli **stessi 16 token** della generazione
normale, uno per uno — con 5 proposte rifiutate e disfatte per davvero (il
pezzo difficile: 24 layer su 32 hanno una memoria interna che una proposta
sbagliata corrompe in modo plausibile).

**Ma oggi il meccanismo è più lento, non più veloce.** Tre misure sullo stesso
host, a parità di token prodotti: generazione normale **34,6 ms/token**,
speculativa una posizione alla volta **48,8**, speculativa a due posizioni in
un colpo **41,2**. Far leggere i pesi una volta sola ha recuperato il 16%, e
non basta.

**E il conto non torna**, che è la cosa importante: verificare due posizioni
insieme costa 60,4 ms contro i 34,6 di una sola, cioè 1,75 volte per il doppio
del lavoro — ma se i pesi si leggessero davvero una volta dovrebbe fermarsi
intorno a 1,15. Qualcosa nella passata costa più di quanto dovrebbe.

**Prossimo passo, non serve una tua decisione**: *misurare* invece di
ottimizzare a naso. Il progetto ha già una sonda che attribuisce il tempo ai
singoli segmenti sulla scheda; va puntata sulla passata di verifica. I tre
sospetti sono nel docket, nessuno dei tre è ancora escluso.
Riferimenti rieseguibili: `Q35_MTP=1 npx vitest run
tests/engine-q35-mtp-accept.test.ts` (~5 min, CPU) e `node
.harness/tools/engine-ktest.mjs` con `npx vite` acceso (~3 min, GPU).

## 2. Mappa

**Destinazione.** Far girare in browser il modello più capace possibile
restando usabile: almeno **30 token/secondo** e **primo token entro 4 secondi**.

**Distanza adesso**, e nessuna configurazione ci arriva: GLM-4.7-Flash **13,4
tok/s** con **14,5 s** al primo token; Qwen 35B **22,6**; Qwen 4B **25,9**. La
predizione doppia funziona ma va ancora resa conveniente (§1): col 50% e la
verifica a pesi letti una volta il 4B proietta **~36 tok/s**. Il 35B non è
coperto: il meccanismo non esiste sui modelli a esperti.

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
