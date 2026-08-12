# HANDOFF — browser-llm-lab   (aggiornato 2026-08-12)

## 1. Next decidable

**Fase 7 chiusa, e la predizione doppia è esclusa dai numeri.** L'idea era: una
testa ausiliaria propone il token successivo-successivo, il modello lo verifica
insieme al prossimo, e ogni passata ne produce due. Il meccanismo è **costruito
e dimostrato corretto** — 16 token generati con la proposta sono identici, uno
per uno, ai 16 generati normalmente, con le proposte sbagliate disfatte per
davvero. La testa indovina il 50% delle volte sul testo umano e il 70-74% su
quello che il modello produce da sé.

**Non paga lo stesso**: 41,8 ms per token contro 34,4 della generazione
normale. E la causa l'ho **letta nel codice**, non dedotta: i nostri kernel
"batch" mettono la riga su una dimensione del dispatch e **ogni riga rilegge i
pesi**. Fondono le chiamate, non riusano i dati. Verificare due posizioni costa
1,7-1,9 volte una, mentre la predizione doppia paga solo se costa ~1. L'unica
leva rimasta (fare la proiezione finale una volta invece di due) vale ~5 ms e
arriva a 38,6: sopra il sequenziale anche nel caso migliore, quindi non l'ho
costruita.

**Il premio, per chi riprende**: con un moltiplicatore matriciale a più righe
che riusa davvero i pesi, la stessa passata starebbe a ~23,5 ms per token, cioè
**~42 token/s** — sopra l'obiettivo. È una famiglia di kernel che qui non
esiste, ed è un progetto suo, non una fetta.

**Il checkpoint di misura è chiuso**: l'esclusione è ora un artefatto committato
(`results/engine/specdec-4090-2026-08-12T21-49-18-513Z.json`) con lo stato
dell'host dichiarato, il verdetto scritto accanto ai numeri e un runner che lo
rigenera. La ri-misura conferma entro il rumore: 1,18 volte più lento contro
1,19 della prima.

**Prossimo passo, non serve una tua decisione**: la fase 9, chiusura del goal —
checklist del contratto voce per voce, non-regressione GLM fresca, e il triage
finale del docket. Il codice della testa resta in albero: è gated e testato, e
torna utile il giorno in cui quei kernel esistono.

**Da sapere**: il goal si chiuderà **senza aver raggiunto i 30 token/s**. La
parità fra i due modelli — che era il contratto principale — è raggiunta e
verificata; il moltiplicatore che doveva superare la soglia non c'è.

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
