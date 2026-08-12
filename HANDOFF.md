# HANDOFF — browser-llm-lab   (aggiornato 2026-08-12)

## 1. Next decidable

**Fase 7 in corso: far predire due token per volta invece di uno.** Il modello
Qwen porta con sé una testa ausiliaria (MTP) che, guardando lo stato interno,
tira a indovinare il token successivo-successivo. Se indovina spesso, ogni
passata del modello produce due token invece di uno.

L'ho fatta girare in CPU e indovina il **31,8%** delle volte — utile, ma sotto
le attese per questa architettura.

**C'è un'anomalia da spiegare prima di andare avanti.** Sospettavo che il blocco
della testa ricevesse le posizioni sfasate di uno; ho aggiunto lo sfasamento e
misurato: stesso 31,8%, e soprattutto **le predizioni sono identiche una per
una** (0 diverse su 23). Non può essere: spostare di una posizione cambia
l'angolo di rotazione di circa un radiante, e 23 scelte su 248.320 possibilità
non restano tutte uguali per caso. O il parametro non arriva al calcolo, o
l'attenzione dentro quel blocco non sta facendo nulla — e quale delle due non
l'ho stabilito.

**Prossimo passo, non serve una tua decisione**: azzerare il contributo
dell'attenzione nel blocco della testa e rimisurare. Se il numero non cambia,
l'attenzione è inerte (e allora il problema è lì); se cambia, il bug è nel
parametro che ho appena aggiunto. Comando di partenza: `Q35_MTP=1 npx vitest run
tests/engine-q35-mtp-accept.test.ts` (~2 min, niente GPU).

## 2. Mappa

**Destinazione.** Far girare in browser il modello più capace possibile
restando usabile: almeno **30 token/secondo** e **primo token entro 4 secondi**.

**Distanza adesso** — nessuna configurazione ci arriva:

| modello | token/s | primo token |
|---|---|---|
| GLM-4.7-Flash | 13,4 | 14,5 s |
| Qwen 35B (MoE) | 22,6 | — |
| Qwen 4B | 25,9 | — |

Se il 31,8% della testa MTP regge, il 4B arriva a ~34 tok/s — il primo
superamento dell'obiettivo — e il 35B a ~29,8.

**Decisioni prese** (indice: il contenuto vive nel posto indicato, non qui)

- Il checkpoint di misura ridotto a vero merge gate — `PHASES.md` riga 6
- Il repack dei pesi resta dov'era, reso 6 volte più veloce — docket, "la fase 2 ha centrato l'obiettivo ma non la lettera"
- Il decode ottimistico è il percorso attivo; la soglia d'ingresso esclusa coi numeri — docket, "i tre numeri per tarare la soglia"
- Prefetch e politica a livelli esclusi perché misurati inutili — journal it.37
- I riferimenti di velocità Qwen di ieri misurano il percorso vecchio: da rifare — docket, "i riferimenti misurano il path sync"
- Il confronto col motore nativo non è a parità — docket, "il gap nativo e la page cache"

**Nebbia** (non ancora deciso né specificato)

- Se il 31,8% sia reale o un errore di posizioni
- Quale accept-rate serva perché la predizione doppia paghi sul 35B, dove la
  testa costa di più
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
