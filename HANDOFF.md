# HANDOFF — browser-llm-lab   (aggiornato 2026-08-12)

## 1. Next decidable

**Goal nuovo: `engine-kernel-decode`** (contratto e fasi in
`.harness/goals/engine-kernel-decode/`). Obiettivo: **30 token/s a contesto
realistico** (≥ 6000 posizioni) sul modello 4B, dove oggi ne fa **9,95**.

**Perché quel numero e non 25,9.** La misura di it.59 ha scoperto che il
riferimento storico era preso a contesto corto: 38,72 ms/token a 388 posizioni
contro **100,52 a 6333**, cioè 10,4 µs per ogni posizione di contesto. A
contesto vero il token è per il **65% scansione della cache delle chiavi**, che
gira a 6,3 GB/s — l'1,4% di quanto la scheda sa fare.

**Le tre leve, in ordine di peso a contesto vero**: l'attenzione (65,8 ms su
100,5), i moltiplicatori quantizzati (27,1 ms, a un quarto della velocità che
llama.cpp ottiene sulla stessa scheda e sullo stesso file), e il multi-riga —
che però è la leva del *tempo al primo token*, quindi sta nel goal dopo.

**Prossimo passo, e questo richiede un tuo sì**: `PHASES.md` è scritto e in
attesa di `plan-check` (docket item 1). Cinque righe: fase 0 di sole sonde
prediction-gated che può chiudere il goal se le leve non esistono, poi
attenzione, poi moltiplicatori, poi il checkpoint dei 30 tok/s o l'esclusione
coi numeri, poi la chiusura.

**Già trovato scrivendo il piano** (docket item 2): `gpulimits.ts` crede che il
path Qwen non dipenda dal contesto e chiede 30.848 byte fissi di memoria di
gruppo, mentre il kernel di attenzione ne usa `4·ctxMax + 256`. Combaciano fino
a **ctxMax 7648**; sopra, la pipeline non si crea. Il motore ha un tetto di
contesto che il modulo dei limiti non dichiara.

## 2. Mappa

**Destinazione.** Far girare in browser il modello più capace possibile
restando usabile: almeno **30 token/secondo** e **primo token entro 4 secondi**.

**Distanza adesso**, e nessuna configurazione ci arriva: GLM-4.7-Flash **13,4
tok/s** con **14,5 s** al primo token; Qwen 35B **22,6**; Qwen 4B **25,9**. La
predizione doppia era l'ultimo moltiplicatore previsto ed è esclusa dai numeri
(§1): con i kernel di oggi nessuna configurazione arriva a 30, e la strada che
ci arriverebbe (~42 tok/s) passa da una famiglia di kernel che non esiste.

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
- **L'attenzione a chunk del prefill** (`attnDecodeWgsl` con `batch`) ha gli
  STESSI tre difetti di quella del decode — ridondanza GQA 4x, letture scalari,
  `scores` in memoria di gruppo — e la stessa riscrittura li toglie. Va al goal
  sul TEMPO AL PRIMO TOKEN, non a quello sul decode: stesso kernel, obiettivo
  diverso (it.59, 2026-08-13)
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
