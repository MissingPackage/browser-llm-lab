# HANDOFF — browser-llm-lab   (aggiornato 2026-08-12)

## 1. Next decidable

**Fase 7 in corso: far predire due token per volta invece di uno.** Il modello
Qwen porta con sé una testa ausiliaria (MTP) che, guardando lo stato interno,
tira a indovinare il token successivo-successivo. Se indovina spesso, ogni
passata del modello produce due token invece di uno.

L'ho fatta girare in CPU: indovina il **74%** delle volte nel regime che conta —
quello in cui il decode gira davvero, cioè sul testo che il modello sta
producendo lui. Sul testo umano (più difficile, e non è il caso d'uso) fa il 50%.

**Il 31,8% di ieri era rumore di campionamento, non un difetto**: 22 posizioni
di campione, dove un solo colpo vale 4,5 punti. Su 62 posizioni i numeri sopra.

**L'anomalia di ieri è chiusa e non era un bug.** Spostare di uno le posizioni
non poteva cambiare niente: quel meccanismo di rotazione guarda solo le
*distanze* fra posizioni, mai i loro valori assoluti, e nella testa scalano
tutte insieme. Il parametro inutile è stato rimosso. L'attenzione dentro la
testa funziona — azzerandola l'accuratezza scende da 50 a 31 — e la nostra
implementazione combacia voce per voce con quella di riferimento di vLLM, letta
oggi.

**Prossimo passo, non serve una tua decisione**: portare la testa sulla GPU e
chiudere il ciclo vero — la testa propone un token, il modello lo verifica nella
stessa passata, si tiene solo se coincide. Il gate della fase è secco: i token
accettati devono essere identici a quelli prodotti oggi. Se il 74% regge, il 4B
passa da 25,9 a ~39 token/s (col 50%, ~34): il primo superamento
dell'obiettivo. Riferimento CPU rieseguibile: `Q35_MTP=1 npx vitest run
tests/engine-q35-mtp-accept.test.ts` (~5 min, niente GPU).

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

- Quanto valga davvero il 74%: 23 posizioni danno un intervallo 52-90%, e il
  campione è un prompt solo
- Se il 35B abbia lo stesso accept-rate: mai misurato (il riferimento CPU sul
  35B costerebbe ore; si misurerà sulla GPU)
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
- **Leggere i parametri di un runner prima di spenderci sopra minuti di GPU.**
  Due volte in una sessione ho lanciato run con flag sbagliati dedotti invece
  che letti.
