# HANDOFF — browser-llm-lab   (aggiornato 2026-08-14)

## 1. Next decidable

**GOAL ATTIVO: `engine-kquant`, riga 1 in corso (fase 0 al banco).** Chartered
2026-08-14. Contratto e spina: `.harness/goals/engine-kquant/{GOAL.md,PHASES.md}`.

**it.1: LA FASE 0 HA DETTO SI' A ENTRAMBE LE FAMIGLIE CABLABILI.** A M=16, sulle
shape vere, contro il kernel di produzione importato: **Q5_K 28,07x** (1,2700 →
0,0452 ms) e **Q4_1 22,58x** (2,3483 → 0,1040). La regola di stop chiedeva 1,5x.
Zero celle scartate dal gate del banco. Artefatto:
`results/microbench/ttft-riga1-4090-linux-2026-08-14T18-54-05-813Z.json`
(⚠ porta ancora il `kind` del goal precedente: scritto prima della correzione
`--tag`, e sara' superseduto dalla run di it.2 — non l'ho ritoccato a posteriori).
**Il banco riproduce il segmento vero in millisecondi**: 24 x 395 x 1,2700 =
12.039 contro i 12.169 misurati. Proiezione **−15,2 s ⇒ ~16,9 s**.
**Da misurare, non dedurre**: la quota Q4_1 di `gemm:ffn-down` oggi e' stimata
dal banco; la riga 3 non chiude senza un `pbCat` proprio per quei quattro siti.

Toglie ai pesi non-q4_0 le M riletture per chunk: `ssm_out` **Q5_K** (37,9% del
prefill) e `ffn_down` **Q4_1** (il 71% dei byte del segmento `gemm:ffn-down`)
passano alla forma multi-riga. **Barra: TTFT a caldo < 22.500 ms** (oggi
32.127), **nice to have < 18.000**; copertura del piano da 5,86× a **≥ 15,5×**;
`gemm:deltanet-out` e `gemm:ffn-down` entrambi **≤ 2.000 ms**. Proiezione
−14,7 s ⇒ ~17,4 s, e quel numero e' anche il **pavimento**: sotto i ~9,4 s
(il tempo fuori dai pass GPU) non si scende togliendo byte ai pesi.

**Il 35B e' il goal SUCCESSIVO, deciso dal PI.** Le forme di kernel che gli
servono (Q4_K, Q6_K, Q8_0) nascono qui — misurate al banco e verificate col
ktest — ma **non vengono cablate**: il 35B non ha un byte di q4_0 (expert Q4_K
17,67 GB), pero' il suo collo e' la **residency**, non il kernel, e non ha una
baseline fresca. Sara' wiring + residency. **Il 9B ha la stessa identica
struttura del 4B**: questa leva vale li' senza una riga di codice in piu'.

---

**Goal `engine-ttft` CHIUSO** (2026-08-14, ruling del PI). Il tempo al primo
token a modello caldo sul prompt da 6333 token: **87.618 -> 32.127 ms = 2,727x**.
La barra meccanica del contratto (< 21.905 ms) **non e' stata raggiunta**: manca
1,467x. Dieci clausole su dodici soddisfatte; le due che cadono sono la stessa —
la barra e la sua gemella `prefill.tokS > 289` — e **la causa e' misurata e
stava fuori dalla portata del goal**.

Consuntivo voce per voce, con l'artefatto accanto a ogni clausola:
`docs/engine/ttft-consuntivo-2026-08-14.md`.

**I DUE GOAL SUCCESSIVI, decisi dal PI. L'ORDINE FRA LORO E' L'UNICA COSA
APERTA** — e i due assi sono diversi, quindi non e' una preferenza ma una
scelta:

1. **K-quant — vale il 37,9% del tempo del prefill.** `gemm:deltanet-out` e' il
   primo termine con soli 24 dispatch, uno per layer DeltaNet, perche' `ssm_out`
   e' **Q5_K** e cade sul fallback legacy: riuso dei pesi ZERO, riletti 16 volte
   per chunk, **89,9 GB/s** su un motore che ne ha dimostrati ~300. **Il ruling
   del riuso lo aveva registrato come una CODA** («l'11,54% dei byte resta sul
   percorso vecchio»): la quota di byte sottostimava il peso proprio perche' la
   forma legacy rilegge i pesi M volte. **E' la leva piu' grande rimasta sul
   tempo al primo token.**
2. **0.5B — non muove la metrica di prodotto, muove la RAGGIUNGIBILITA'.** E' il
   path di conformita', e il suo valore e' girare su device che concedono il
   minimo di spec WebGPU (16.384 B di memoria di gruppo). Scope fissato dal PI:
   **«migreremo il possibile»** — tre siti su quattro adottano la forma
   split-K esistente; il quarto (`gemvResidualFast`, down-proj del **decode** a
   M=1) e' **non-migrabile per costruzione** e va dichiarato tale, non contato
   come buco. Veicolo: workflow `pattern-migration`, che ora produce
   `nonMigrable` + una lettura in chiaro invece di una percentuale.

**PRIMA DI TOCCARE IL 0.5B, leggere questo**: i consumatori sopra la garanzia
WebGPU sono **quattro**, non uno, e alimentano **un solo valore condiviso**
(`QWEN_WORKGROUP_STORAGE_BYTES`, che il motore chiede come tetto del device).
Migrarne un sottoinsieme fa SCENDERE quel massimo mentre gli altri chiedono il
valore vecchio ⇒ `createComputePipeline` fallisce **su ogni device**. La
costante e' gia' stata resa un `Math.max` **calcolato** dalle formule accanto ai
kernel (it.24) proprio per questo: quella correzione e' il prerequisito, ed e'
gia' in albero.

**LE LEVE DI `engine-ttft`, tutte in produzione e misurate prima/dopo**:
moltiplicatore multi-riga `splitk` (prefill 34,36 -> 111,16 tok/s) · via intera
`dot4I8Packed` (-> 123,26) · attenzione del prefill in streaming (-> 196,41).

**ESCLUSE COI NUMERI — l'eredita' piu' utile per chi riprende:**
- **fusione delle teste GQA sul prefill: PIU' LENTA** (2,0879 contro 1,8207 ms).
  Sul decode aveva funzionato: il verso e' opposto, e chi riprova deve saperlo.
- **il tetto di memoria di gruppo negoziabile non e' una leva** (spread 0,1-2,3%).
- **la ricorrenza DeltaNet NON e' il collo**: 5,0% del tempo contro il 47,3% dei
  dispatch. **Contare i dispatch non e' misurare il tempo** — l'avevo concluso
  dal solo conteggio, ed era gia' finito in un HANDOFF prima che il cronometro
  lo smentisse.

**Dove finisce il tempo** (`results/engine/q35-ttft-kernel-checkpoint-4b-2026-08-14.json`):
70,9% dentro i pass GPU, **29,1% fuori** (encode CPU, submit, buchi fra submit —
non attribuito piu' finemente, e lo dichiaro). **1,578 TFLOP/s contro il picco
fp32 misurato di 9,26 = 17%**: non e' un'efficienza, e' la prova che **il collo
non e' l'ALU**.

**Gate alla chiusura**: tsc pulito · vitest **680 | 10** · ktest **101 PASS /
0 FAIL** · top-1 contro l'oracolo llama.cpp **1012/1024 = 98,828% su ENTRAMBI i
bracci** · sequenze generate **identiche 8/8** · decode **48,15** tok/s (soglia
45,5).

**Goal `engine-kernel-decode` CHIUSO** (2026-08-13): decode a contesto 6333 da
9,95 a 47,93 tok/s, 4,82x, sopra la soglia di prodotto di 30.

## 2. Mappa

**Destinazione.** Far girare in browser il modello più capace possibile
restando usabile: almeno **30 token/secondo** e **primo token entro 4 secondi**.
Il secondo termine è un'aspirazione di prodotto, non una promessa di questa
macchina: su un prompt da 6k il 4B ha un pavimento misurato di ~9,4 s, e il PI
ha riscritto l'obiettivo del goal in «il più in basso possibile» (2026-08-13).

**Distanza adesso**: Qwen 4B **48,15 tok/s a contesto 6333** (era 9,95) —
**sopra i 30 dell'obiettivo**. Il TTFT a caldo sul prompt da 6333 token è
**32,1 s MISURATO** (it.25, era 87,6 in it.1) contro una barra di 21,9 s: manca
**1,47×**. Il goal `engine-ttft` ha il consuntivo pronto e la barra NON
raggiunta; la causa è misurata e sta **fuori dalla sua portata** — il 37,9% del
prefill è un `ssm_out` **Q5_K** sul percorso vecchio, e le leve di questo goal
sono q4_0-only per costruzione. **La prossima leva sul tempo al primo token è il
goal K-quant**, che finora era registrato come una coda.
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
- **`q35-bench-run.mjs --prefill-m` NON attiva il gate di conformità: misura la
  velocità.** Quel runner passa SEMPRE `?bench=`, e la pagina prende il ramo
  bench prima di arrivare al gate. Il gate del prefill a chunk lo lancia
  **`q35-conf-run.mjs --prefill-m 16`**. Ci sono cascato in it.20 e ho prodotto
  un JSON il cui NOME diceva `q35-prefillchunk` e il cui `kind` era
  `q35-bench-4b-fullresident` sul prompt-idx 4 — cioè le due landmine qui sopra
  sommate. Rimosso in `9c04728`. **Prima di credere a un artefatto, leggi il suo
  `kind`, non il suo nome di file.**
- **UN `git checkout` DI UN RAMO VECCHIO PUÒ FAR NASCERE UN AGENTE FANTASMA nel
  tuo working tree.** Successo il 2026-08-14, ricostruito dal reflog e dal log
  del watchdog: alle 00:00:51 il checkout di
  `wip/riga2-cablaggio-non-verificato` ha riscritto
  `.harness/loop-state.json` con la copia `status: "active"` di quel ramo
  (`next_wake` 23:12:44); alle **00:01:34**, 43 secondi dopo,
  `harness-loop-watchdog.timer` (poll ogni 2 min) l'ha letto, ha visto un
  risveglio scaduto da 48 minuti e ha **rianimato una sessione che era stata
  fermata apposta** con `ScheduleWakeup{stop:true}` prima del riavvio. Due
  sessioni Claude hanno lavorato in parallelo sullo stesso albero e sulla stessa
  GPU per ~15 minuti, e me ne sono accorto solo perché `q35gpumodel.ts` è
  cambiato sotto di me.
  Il watchdog **non legge male**: legge `active` perché glielo ha appena scritto
  git. Il difetto era che un file di RUNTIME stesse in git — corretto in
  `c21648e` (`.gitignore` + `git rm --cached`).
  **RESIDUO APERTO: 19 rami portano ancora `status: "active"`** — tutti i
  `wip/riga2-*` e i `worktree-wf_*`. Toglierlo da main impedisce a main di
  ri-armare la miccia, **non** impedisce a un checkout di quei rami di
  riarmarla. La difesa robusta (rifiutare uno stato scaduto da N poll, e non
  saltare il controllo «transcript fresco» sul ramo `headless_dead`) sta in
  `~/Projects/harness/tools/loop-watchdog.sh`, **altro repo, non toccato**.
  Sintomo da cui riconoscerlo: file sorgente che cambiano senza che tu li abbia
  scritti. `ListAgents` mostra la sessione peer; `git log -1 --format=%h
  .harness/loop-state.json` contro `updated_epoch` dice se il file è stato
  riscritto da un checkout invece che dall'hook.
- **ASPETTARE UN WORKFLOW IN BACKGROUND FA DUPLICARE LA TUA SESSIONE. Sempre, e
  allungare il risveglio non serve.** Il watchdog usa il silenzio del transcript
  come test di morte; una sessione che aspetta un workflow è silenziosa **per
  costruzione**, perché il workflow non scrive sul transcript di chi l'ha
  lanciato. Misurato tre volte il 2026-08-14: 00:56:05 («transcript fermo da 28
  min») e 02:03:28 («fermo da 62 min») hanno duplicato la stessa sessione
  `ae3ad6a9`, la seconda volta **nonostante il risveglio fosse stato allungato a
  60 minuti apposta**. Allungare sposta l'istante, non toglie la condizione: il
  ramo `headless_dead` del watchdog **salta del tutto** il controllo «transcript
  fresco ⇒ sta lavorando», e quel ramo è vero per sempre una volta che il
  watchdog ha spinto una prima volta su quella directory.
  I duplicati **non sono innocui e non sono inutili**: hanno committato lavoro
  buono (l'integrazione di it.17 e l'attribuzione del gate rotto sono pulite),
  ma nessuno li coordina, lavorano sullo stesso albero e sulla stessa GPU, e
  chi li ha generati non può verificarli mentre corrono.
  **Finché il watchdog non è corretto: niente `/loop` con workflow in volo.**
  Il lavoro lungo va fatto con un umano presente, o `ScheduleWakeup{stop:true}`
  prima di lanciare il workflow — al prezzo che il loop poi non riparte da solo.
