# HANDOFF — browser-llm-lab   (aggiornato 2026-08-13)

## 1. Next decidable

**Goal `engine-kernel-decode` CHIUSO: l'obiettivo di prodotto è superato sul
4B.** Il decode a contesto realistico (6333 posizioni) è passato da **9,95 a
47,93 token/s** — **4,82×**, contro una soglia di 30 — e il contesto ormai non
si paga quasi più: 0,15 µs per posizione contro 10,4.

Due leve, entrambe scelte da una fase di sole misure che aveva il potere di
chiudere il goal: l'**attenzione** col contesto spezzato su più gruppi di lavoro
e softmax in streaming (2,72×), e i **moltiplicatori quantizzati** con letture
vettoriali, due righe per gruppo e riduzione di sottogruppo dove è dimostrabile
che è sicura (1,77× ulteriore). La stessa fase ha **escluso** la fusione delle
teste GQA: misurata più lenta.

**Goal `engine-ttft` APERTO e IN CORSO** (2026-08-13, ruling A del PI): il primo
token entro 4 secondi **a modello caldo** — `prefill.ms + decode.firstMs` — sul
prompt da 6333 token. Il caricamento del modello (10,9 s) ha una soglia sua e
non appartiene a questo goal. Piano approvato; **riga 0 CHIUSA (it.1)**.

**LA BASELINE ORA ESISTE, ed è la prima misura end-to-end del 4B sul codice di
oggi**: TTFT a caldo **87.618 ms** contro un bersaglio di 4.000 ⇒ **serve
21,9×**. Scomposto: caricamento 10.892 ms · lettura del prompt 87.582 ms
(6332 token a 72,30 tok/s) · primo token 36 ms. Decode 47,79 tok/s a contesto
6333, che conferma il 47,93 del checkpoint e passa il gate di non-regressione.
Artefatto: `results/engine/ttft-baseline-4b-prompt0-2026-08-13.json`.

**Il percorso "a blocchi" è 2,10× PIÙ LENTO, misurato.** L'ipotesi della riga 0
— instradare il bench su `prefillChunk`, già in albero e già bit-esatto,
potrebbe muovere la metrica senza scrivere kernel — è **refutata**: 34,36 contro
72,30 tok/s. Il motivo conferma la diagnosi invece di smentirla: `prefillChunk`
instrada su `gemvQuantWgsl` con `batch: true` (M GEMV replicate sull'asse z,
riuso dei pesi ZERO) e il GEMV veloce del goal precedente **rifiuta `batch` per
costruzione** (`wgsl.ts:216-249`), mentre `step()` quei kernel veloci li ha.
Conseguenza per la riga 2: non basta "usare il chunking" — il riuso va portato
AL kernel veloce.

**Prossimo passo, senza gate**: riga 1, le sonde. Decide se le leve esistono
(GEMM multi-riga con riuso vero, attenzione a chunk del prefill) e **se il
bersaglio esiste**: 6333 token su 4e9 parametri sono 50,7 TFLOP, cioè 12,8
TFLOP/s sostenuti in 3,96 s, e il picco fp32 di questo device in WebGPU non è
mai stato misurato. È l'unica riga con potere di chiudere il goal.

**Tre decisioni aperte nel docket di `engine-ttft`**: dove vive lo spec-dec MTP
ora che accelera una metrica già raggiunta · se una dichiarazione `hostState`
smentita debba far fallire la run o solo annotarla · se la clausola di
portabilità passi da «sotto 16.384 B sempre» a «dichiarare, negoziare,
degradare», ora che si è verificato che il tetto di WebGPU **è negoziabile** via
`requiredLimits` (puro JavaScript, nessun permesso: Chrome Android compreso).

**Il 35B non ha ricevuto nulla da questo goal, ed e' misurato**: non ha un solo
tensore Q4_0 (Q8_0 251 · Q4_K 117 · Q6_K 4), e la forma nuova dei
moltiplicatori e' q4_0-only per costruzione. Portarlo dove sono 4B e 9B vuol
dire **dare alle famiglie K-quant e Q8_0 la stessa fase 0 che ha avuto la
q4_0** — una misura loro, non un'estensione a intuito del kernel esistente. E'
un goal suo, accanto a quello sul tempo al primo token. Nella chat del
2026-08-13 il 35B ha reso 9,58 tok/s: numero a FREDDO (primo turno dopo il
load, 13 GiB di tetto per un modello da 20,9), contro i 22,6 di riferimento a
caldo — il divario e' paginazione, non kernel.

**Cinque cose aperte che ho registrato e non deciso** (docket del goal chiuso):
il conductor troncava le patch a 16 KB — **CORRETTO alla fonte dalla sessione
harness** (`59e19d3`), installato qui · il done-when sulla portabilità chiede
più di quanto quella fase potesse dare: ereditato da `engine-ttft` riga 4, e il
suo senso cambia ora che il tetto WebGPU è verificato **negoziabile** · `--out`
assoluto/relativo: erano **nove** runner con due convenzioni opposte, il primo
(`q35-bench-run`) **corretto in it.1**, gli altri otto a docket · due call-site
GLM nel ktest non congelati · `hostState.declared` è una promessa che nessun
runner verifica.

## 2. Mappa

**Destinazione.** Far girare in browser il modello più capace possibile
restando usabile: almeno **30 token/secondo** e **primo token entro 4 secondi**.

**Distanza adesso**: Qwen 4B **47,93 tok/s a contesto 6333** (era 9,95) —
**sopra i 30 dell'obiettivo**. Il TTFT a caldo sul prompt da 6333 token è
**87,6 s MISURATO** (it.1) contro i 4 richiesti: serve 21,9×. È la metà
dell'obiettivo di prodotto ancora aperta, ed è il goal `engine-ttft`.
GLM-4.7-Flash resta residency-bound
(~13 tok/s, TTFT 14,7): nessuna delle leve di questo goal lo tocca. Il 35B non
è stato rimisurato dopo i kernel nuovi.

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
  `scores` in memoria di gruppo — e la stessa riscrittura li toglie. È ora la
  riga 3 del goal `engine-ttft`: non più fuori scope
- Tutti i numeri destinati alla pubblicazione: si rimisurano al tag di release

## 3. Landmines

- **I JSON in `results/` possono mentire in silenzio.** I runner uscivano con
  successo anche quando la GPU falliva, scrivendo numeri plausibili. Ora c'è una
  sentinella su tutti e quattro, ma i file scritti PRIMA di oggi non sono
  protetti.
- **Gli 8 riferimenti Qwen del 2026-08-11 (`*fase-d-it43*`) misurano il percorso
  vecchio**: non usarli per la pubblicazione.
- **`ttftMs` INCLUDE il caricamento del modello** (`q35conf.worker.ts:246`:
  `loadMs + prefillMs + firstMs`). Un TTFT citato senza i tre termini scomposti
  non dice cosa sta misurando — ed è così che 10,89 s di load sono finiti in un
  obiettivo di prodotto sul prefill.
- **`--prompt-idx` ha default 4 = 388 token, non il prompt da 6k.** È il flag da
  cui veniva il numero sbagliato corretto in it.0. I prompt ≥ 6000 token sono
  l'idx 0 (6333) e l'idx 5 (6128).
- **Il prefill "a chunk" è più LENTO del sequenziale** (34,4 contro 72,3 tok/s,
  it.1): instrada sul GEMV vecchio, mentre `step()` usa quello veloce. Un bench
  che non dichiara `prefillPath` non dice quale dei due ha misurato.
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
