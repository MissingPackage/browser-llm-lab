# HANDOFF — browser-llm-lab   (aggiornato 2026-08-14)

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

**Goal `engine-ttft` APERTO e IN CORSO** (2026-08-13): portare il tempo al primo
token **a modello caldo** — `prefill.ms + decode.firstMs` — sul prompt da 6333
token **il più in basso che questa macchina consenta**. Righe 0 e 1 CHIUSE.

**IL BERSAGLIO DEI 4 SECONDI È STATO TOLTO DAL PI** (2026-08-13, dopo la riga 1):
«va bene se non arriviamo a 4 secondi su questa macchina, scendiamo il più
possibile». Poiché «il più possibile» non è graduabile da un verificatore, la
forma meccanica è in tre clausole: TTFT a caldo **< 21.905 ms** (un quarto della
baseline) · **esaurimento delle leve**, ognuna in produzione o esclusa coi numeri
· **contabilità del tetto residuo** per segmento. Autorizzato anche il dot
product intero come leva di questo goal.

**Baseline (it.1)**: TTFT a caldo **87.618 ms**; la barra meccanica è 21.905 ⇒
serve 4,0×. Scomposto: caricamento 10.892 · lettura del prompt 87.582 (6332
token a 72,30 tok/s) · primo token 36 ms. Decode 47,79 tok/s a contesto 6333.

**LA METRICA SI È MOSSA — riga 2, it.14 e it.15.** È la prima volta da quando il
goal è aperto:

| | prefill | TTFT a caldo | decode |
|---|---|---|---|
| baseline (it.1) | 72,30 tok/s | 87.618 ms | 47,79 |
| split-K f32 (it.14) | 110,19 | 57.485 | 49,59 |
| **via intera (it.15)** | **123,26** | **51.392** | 48,00 |

**1,70× sulla baseline; alla barra manca 2,35×**, e la leva che resta è
l'attenzione del prefill (**6,76×** misurata) — la riga 3.

**Il riavvio della macchina ha risolto i crash**: `ktest` fa 100 PASS / 0 FAIL
sullo stesso codice che prima faceva morire la pagina. La causa era
infrastrutturale, non il cablaggio — e **cade anche la conclusione sul
`sdd-conductor`**: le sue 5 morti erano lo stesso segnale 144 che uccideva un
ktest sano, non un limite del veicolo.

**Il buco trovato in it.15 vale più del guadagno**: `prefillgemmplan.ts`
esisteva già — completo, coi suoi test — e `q35gpumodel.ts` **non lo importava**
(zero riferimenti). Il sito ri-derivava a mano la condizione e finiva sempre
sulla via f32, che il kernel stesso documenta come fallback. È la stessa forma
del difetto di it.7 del goal precedente («un terzo posto che decide la stessa
cosa»), e il commento che la nomina era tre righe sotto il codice che la
ripeteva. **Un piano non collegato non è un piano: è documentazione.**

**LE DUE RISPOSTE DELLA FASE DI SONDE (it.2), verificate a mano contro
l'artefatto**:

1. **Le leve esistono, e sono grosse.** La regola di stop NON scatta: a M=16 il
   moltiplicatore multi-riga passa da 2,6225 a **0,0609 ms = 43,1×** sulla forma
   attuale; l'attenzione del prefill da 12,2993 a **1,8207 ms = 6,76×** a
   contesto 6333. I byte di peso per token prefillato scendono da 13,27 MB a
   0,83 = **16×**, cioè il massimo teorico a M=16 e il doppio di quanto il
   done-when della riga 2 chiede.
2. **Il bersaglio dei 4 secondi NON è raggiungibile su questa macchina.**
   Proiezione dalla formula fissata prima di misurare, coi tempi misurati:
   5.776 ms di moltiplicazioni + 2.888 di attenzione = **8.665 ms**, ed è un
   PAVIMENTO — non conta i 24 layer DeltaNet, le norm, il RoPE, i dispatch. È
   10,1× sulla baseline e 2,2× sopra i 4 s. **Dopo il ruling questo non è più un
   fallimento**: la riga 5 non sceglie fra «raggiunto» ed «escluso», contabilizza
   la discesa e il tetto. Il pavimento proiettato sta comodamente sotto la barra
   dei 21.905.

**IL MODELLO MENTALE DEL GOAL ERA SBAGLIATO, ed è la scoperta che conta** (docket
item 10). Il piano diceva «il prefill è limitato dalla banda sui pesi, il riuso è
la leva». Vero sulla forma attuale, **falso su tutte le forme candidate**: appena
il riuso c'è, il collo si sposta sull'**occupancy** e la banda misurata crolla a
108 GB/s su un device che ne fa 300+. Due conferme indipendenti: `splitk` batte
`regs` di 2,13× a parità di corpo e di workgroup storage, solo con 576 workgroup
invece di 144; e la fusione GQA taglia il traffico KV di 4× ed è **più lenta**,
perché scende da 256 a 64 workgroup su 76 SM. **La riga 3 non deve adottare la
fusione GQA "perché sul decode ha funzionato": sul prefill la misura dice il
contrario.**

**Il tetto negoziabile non è una leva**: spazzando il limite concesso su
{16.384, 24.576, 32.768, 49.152} il throughput non si muove in modo utile. La
riga 1 aveva concluso «piatto entro ±5%»; **rigirando lo sweep in it.4 lo spread
è 7,6% e 8,6% su due forme su tre** — la conclusione regge nella sostanza ma è
più debole di come è stata scritta, e chi la cita deve saperlo.
La forma vincente chiede **4.096 B a M=16**, sotto il pavimento di spec — quindi
il conflitto mMax-vs-shared del docket item 1 **non esiste** per la forma che
vince, e la portabilità della riga 4 si ottiene gratis. Il legacy dell'attenzione
invece non crea nemmeno la pipeline sotto i 32.768 B.

**IL RIAVVIO HA RISPOSTO: ERA L'INFRASTRUTTURA** (it.14, 2026-08-14). Server
avviato staccato, stesso ramo, stesso comando: **ktest 100 PASS · 0 FAIL ·
exit 0**. Il cablaggio dell'assemblatore era sano; ciò che moriva ieri era il
reaping al confine di turno. **Mergiato in main (`7bc6b55`).** Di conseguenza va
riletta anche la conclusione «il conductor non chiude un task di questa classe
qui»: 5 morti su 5 erano lo stesso segnale 144, non un limite del veicolo.

**Stato della metrica: TTFT a caldo 87.618 → 56.984 ms = 1,54×.** Misurato col
comando identico alla baseline (`--prompt-idx 0 --n-decode 64 --vram-gib 8
--declared quiescent --prefill-m 16`), artefatto
`results/engine/ttft-riga2-4b-splitk-m16-prompt0-2026-08-14.json`:
prefill **34,36 → 111,16 tok/s = 3,24×** sul braccio chunked M=16 (1,54× sul
sequenziale, che era il path migliore di allora) · `prefill.ms` 56.961 ·
`firstMs` 23 · decode 47,17 tok/s a ctx 6333 (−1,30% su 47,79: dentro la banda).
Barra del contratto **< 21.905 ms** ⇒ **manca 2,60×**.

**LA PROSSIMA COSA DA FARE, ED È UN BUCO DEL CABLAGGIO, NON UNA RIGA NUOVA.**
Il done-when (a0) della riga 2 dice «se `idot` vince, è LEI la forma che va in
produzione». `idot` ha vinto (1,745× sopra la f32, it.6). Ma
`q35gpumodel.ts:746` chiama `prefillGemmQ4SplitKWgsl` — la via **f32**, che il
commento del kernel stesso (`wgsl.ts:3795`) dichiara «FALLBACK DICHIARATO ...
non come alternativa preferibile». `prefillGemmQ4SplitKIdotWgsl`
(`wgsl.ts:3718`) è in albero, portata riga-per-riga dal banco, col suo test di
divergenza testuale — **e nessun call-site la raggiunge**. `gpulimits.ts`
conosce già il flag `prefillGemmIdot`: il punto d'innesto c'è, mancano la scelta
a runtime sulla language feature e il ramo che la usa. Manca anche la clausola
(d) della riga 2, la copertura dei siti con worklist.

**Le tre leve sono tutte MISURATE**: moltiplicatore multi-riga `splitk` **38,0×**
a pesi freddi — **IN PRODUZIONE da it.14**, ma nella variante f32 · via intera
q4_0×q8_0 **1,745×** sopra di lui, quantizzazione delle attivazioni compresa
(costa il 5,6%) — **in albero, non chiamata** · attenzione del prefill in
streaming **6,76×** a contesto 6333 — **in albero, non chiamata** (riga 3:
`attnDecodeWgsl` con `batch` instrada ancora al legacy).
**Il divario fra i 43,1× del banco sul kernel e gli 1,54× end-to-end dice dove
sta ora il tempo: il moltiplicatore non è più il termine dominante, lo è
l'attenzione.** È l'argomento per fare la riga 3 prima o insieme all'idot.

**Ruling del PI del 2026-08-13 sulla barra del riuso**: da ≥ 8× a **≥ 5,5×
sull'inventario per-layer INTERO**. Il ≥ 8× era irraggiungibile a qualunque M
praticabile (tetto 8,67×, servirebbe M ≥ 92) perché la forma nuova è q4_0-only e
l'11,54% dei byte del 4B — 24 `ssm_out` Q5_K + 4 `ffn_down` Q4_1 — resta sul
percorso vecchio. **Il residuo è scope del goal K-quant** e va nominato nel
consuntivo di questo, non lasciato implicito.

**Il modello mentale del goal era sbagliato, ed è la scoperta che vale più dei
rapporti**: il piano diceva «il prefill è limitato dalla banda sui pesi». Vero
sulla forma attuale, falso su tutte le candidate — appena il riuso c'è, il collo
si sposta sull'**occupancy** e la banda misurata crolla a 108 GB/s su un device
che ne fa 300+. Due conferme indipendenti: `splitk` batte `regs` di 2,13× a
parità di corpo e di memoria condivisa, solo con 576 gruppi di lavoro invece di
144; e la fusione GQA taglia il traffico KV di 4× ed è **più lenta**, perché
scende da 256 a 64 gruppi su 76 processori. **La riga 3 non deve adottare la
fusione GQA "perché sul decode ha funzionato": sul prefill la misura dice il
contrario.**

**Il veicolo `sdd-conductor`: la conclusione di ieri è SOSPESA, non confermata.**
5 lanci, 5 morti, tutte al confine del turno — ma il riavvio ha dimostrato che
quel confine uccideva anche un ktest sano. Il veicolo non è stato riprovato dopo
il riavvio, quindi «non chiude un task di questa classe qui» oggi non ha
evidenza: ha solo una causa alternativa più semplice. Il lavoro sincrono
continua a reggere (tutte le misure GPU, tutti i gate, tutti i merge).

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
Il secondo termine è un'aspirazione di prodotto, non una promessa di questa
macchina: su un prompt da 6k il 4B ha un pavimento misurato di ~9,4 s, e il PI
ha riscritto l'obiettivo del goal in «il più in basso possibile» (2026-08-13).

**Distanza adesso**: Qwen 4B **47,17 tok/s a contesto 6333** (era 9,95) —
**sopra i 30 dell'obiettivo**. Il TTFT a caldo sul prompt da 6333 token è
**57,0 s MISURATO** (it.14, era 87,6 in it.1) contro una barra di 21,9 s: manca
2,60×, e il pavimento misurabile della macchina è ~9,4 s. È la metà
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
- **Il server di sviluppo muore da solo: TRE volte in una sessione**
  (2026-08-13, tutte exit 144 = ucciso da segnale, non crash). Evidenza: il log
  si ferma su «ready» senza una riga di errore, nessun OOM in `dmesg`, 19 GB
  liberi. L'ipotesi — il supervisore dei comandi in background lo reap al confine
  di turno — è **CONFERMATA per i confini di turno**: i tre morti erano tutti lì, e
  avviato staccato ha superato il confine successivo (verificato con `curl`).
  **NON sopravvive però all'uscita del processo Claude Code**: alla ripresa di
  sessione era di nuovo giù. `setsid` sposta il problema, non lo toglie.
  **Avviarlo così, e ri-avviarlo a ogni ripresa di sessione:**

      setsid nohup npx vite --port 5199 > /tmp/vite-5199.log 2>&1 < /dev/null &

  Verificare comunque PRIMA di ogni run che costa GPU, **con `curl` e non con
  `pgrep`**: `curl -s -o /dev/null -w "%{http_code}" http://localhost:5199/<entry>.html`.
  `pgrep -f "vite --port 5199"` FA MATCH SULLA PROPRIA RIGA DI COMANDO e risponde
  sempre "vivo" — ci sono cascata il 2026-08-13, dichiarando vivo un server morto.
  Il gate dei kernel almeno NOMINA la causa (exit 2, «nessun server su …»); i
  runner di bench no, e per loro il sintomo resta un fallimento di caricamento
  pagina, cioè una causa travestita.
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
