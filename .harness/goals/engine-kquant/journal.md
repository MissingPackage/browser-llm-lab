# JOURNAL — engine-kquant

Una voce per iterazione: cosa ho fatto, cosa ho misurato, cosa ho deciso e
perche'. Le decisioni che NON sono mie stanno nel docket.

## it.0 — decomposizione (2026-08-14)

Contratto in `GOAL.md` (chartered 2026-08-14, con la doppia barra del PI:
22.500 obbligatoria, 18.000 nice to have). Spina in `PHASES.md`: sette righe.

**Test-fit dei done-when prima della tabella — quattro correzioni**, tutte
scritte in testa a `PHASES.md`:

- **C0-1**: `prefillGemmSplitsFor` (`wgsl.ts:4150`) conta in blocchi da 32; i
  K-quant hanno il superblocco da 256 come unita' indivisibile (scale e min a
  6 bit condivisi). Va reso parametrico sulla famiglia, o il conto dei byte su
  cui poggia il goal smette di valere.
- **C0-2**: `prefillQuantXQ8Wgsl` quantizza per blocchi da 32 e i sotto-blocchi
  K-quant sono anch'essi da 32 ⇒ **la via intera e' disponibile ai K-quant
  senza un secondo quantizzatore e senza dispatch aggiuntivi**. Riuso, non
  lavoro nuovo.
- **C0-3**: K-quant e q4_1 hanno un termine costante per blocco che moltiplica
  Σx (non Σ(q·x)): nella via intera si ottiene con
  `dot4I8Packed(xq, 0x01010101)`. Ipotesi da confermare al banco, non promessa.
- **C0-4**: cella degenere evitata — il down-proj degli expert del 35B ha
  K=512 = 2 superblocchi per riga, quindi lo split-K a 4 fette **non esiste**
  su quella shape. Celle a `splits ∈ {1,2}`. Shape 35B verificate sull'header
  dump: dModel 2048, dFfnExpert 512, nExpert 256, 40 layer.

**Decisione mia, registrata e non escalata** (l'ordine e il meccanismo sono
miei): la famiglia di kernel si fa per intero in questo goal — Q5_K, Q4_1,
Q4_K, Q6_K, Q8_0, tutte misurate al banco e verificate col ktest — ma **solo
Q5_K e Q4_1 vengono cablate e misurate end-to-end**, perche' sono le uniche
che esistono nel 4B. Il 35B e' il goal successivo (deciso dal PI) e sara'
wiring + residency, non lavoro di kernel: 17,67 GB di expert su 16 GiB di
scheda, un piano diverso (`moeprefillplan.ts`) e nessuna baseline fresca da cui
partire. Cosi' quel goal parte da forme MISURATE invece che da forme da
inventare, e questo goal conserva UNA condizione di completamento.

**Reperto per il goal 35B**, dall'header dump 2026-08-10: il 35B non ha **un
byte** di q4_0 (expert Q4_K 17,67 GB · Q6_K 0,66 · attn Q8_0 1,09 · linear_attn
Q8_0 0,27 · head Q8_0 0,54). La via veloce attuale ne copre lo **0%**.

**Reperto per la scaletta dei modelli**: il **Qwen3.5-9B** ha la stessa identica
struttura del 4B (`linear_attn:Q5_K` 276,8 MB su 24 tensori, `ffn/shexp:Q4_1`
125,8 MB su 4). Questa leva vale li' **senza una riga di codice in piu'**.
Segnalato al PI in chat; non entra nel contratto (nessuna baseline 9B esiste, e
aggiungerlo raddoppierebbe le run di verifica).

Prossima: riga 1, fase 0 al banco.

## it.1 — riga 1: la fase 0 dice SI' a entrambe le famiglie cablabili (2026-08-14)

**Comando esatto** (letto prima di spendere, come impone il protocollo):

    setsid nohup npx vite --port 5199 > /tmp/vite-5199.log 2>&1 < /dev/null &
    curl -s -o /dev/null -w "%{http_code}" http://localhost:5199/ttbench.html   # 200
    BASE_URL=http://localhost:5199 node scripts/tt-microbench-run.mjs --label 4090-linux --host quiescent

Artefatto: `results/microbench/ttft-riga1-4090-linux-2026-08-14T18-54-05-813Z.json`
(18 celle `gemm-kquant-multirow`, **zero scartate**).

**I NUMERI, a M=16 — che e' il punto di lavoro del prefill (`PREFILL_M`):**

| famiglia | legacy (produzione) | splitk-idot | splitk-f32 |
|---|---|---|---|
| Q5_K `[4096, 2560]` | 1,2700 ms | **0,0452 ms = 28,07x** | 0,2058 ms = 6,17x |
| Q4_1 `[9216, 2560]` | 2,3483 ms | **0,1040 ms = 22,58x** | 0,1412 ms = 16,63x |

**La regola di stop (>= 1,5x) e' superata di un ordine di grandezza su
entrambe: entrambe si cablano.** La via f32 e' sempre piu' lenta dell'intera ma
sempre sopra la legacy a M >= 8: regge come fallback dichiarato.

**IL CONTROLLO CHE VALE PIU' DEI NUMERI: il banco riproduce la produzione.** Il
braccio legacy Q5_K misura **91 GB/s** di traffico pesi a M=16; il checkpoint di
`engine-ttft` misura **89,9 GB/s** sul segmento `gemm:deltanet-out` in
produzione. Due strumenti diversi, stessa cifra a meno dell'1,2%: il braccio di
paragone non e' un'imitazione del percorso vecchio, e' il percorso vecchio.

**PROIEZIONE sul segmento** (395 chunk — il numero si LEGGE dal checkpoint
(`"chunks": 395`, 6320 token), non si ricalcola da ceil(6333/16) = 396):
- Q5_K: 24 layer x 395 x 0,0452 = **429 ms** contro 12.169 ⇒ **−11,7 s**
- Q4_1: 4 tensori x 395 x 0,1040 = **164 ms** contro ~3.710 ⇒ **−3,5 s**
  ⚠ **QUESTO SECONDO NUMERO E' CIRCOLARE, e il verificatore ha ragione**: i
  3.710 ms sono `4 x 395 x 2,3483`, cioe' il MIO braccio di banco, non una
  misura di produzione. L'unica ancora vera e' che 3.710 < 4.971 (il segmento
  misurato), il che lascia 1.261 ms ai 28 tensori q4_0 gia' multi-riga — cioe'
  117 GB/s contro i 738 GB/s che lo stesso checkpoint misura sul q4_0 di
  `gemm:qkv`. Sei volte di scarto, e nessuno l'ha spiegato: o quei dispatch
  sono limitati dal lancio (120 dispatch con 72 workgroup minimi), o
  l'attribuzione al Q4_1 e' troppo generosa e il −3,5 s e' gonfiato.
  **AZIONE, non nota**: la riga 3 non chiude senza una categoria di misura
  PROPRIA per i quattro siti Q4_1 (`pbCat` separato in `q35gpumodel.ts`), cosi'
  il checkpoint della riga 5 attribuisce quel tempo invece di dedurlo.
- TTFT 32.127 − 15,3 s ≈ **16,8 s**, cioe' sotto ANCHE la barra nice-to-have
  (18.000). Con la penale di lettura fredda misurata sul q4_0 (+13% su
  `splitk-coldw`) resta ~17,0 s.
**Non e' un risultato: e' una proiezione da microbench.** Vale finche' non la
smentisce il checkpoint della riga 5, ed e' li' che va verificata.

**M=1 DICE UNA COSA CHE SERVIRA' AL CABLAGGIO**: a M=1 la forma split-K Q5_K in
f32 e' PIU' LENTA della legacy (0,1491 contro 0,0829). Il combine dei parziali
non si ammortizza su una riga sola. Il piano non deve instradare M=1 sulla forma
multi-riga — oggi non lo fa per costruzione (la via veloce vive solo nel piano
gemello del prefill), ma se un domani qualcuno la offrisse anche al decode,
questo e' il numero che glielo vieta.

**DIFETTO INCONTRATO SUL PERCORSO, TOLTO (non recintato)**: il driver del banco
scriveva sempre `ttft-riga1-*` come nome file e `microbench-ttft-riga1` come
`kind`, anche ora che il banco porta le celle di un ALTRO goal. E' la landmine
del progetto («leggi il `kind`, non il nome del file») vista dal lato di chi
PRODUCE l'artefatto. Aggiunto `--tag`, che muove nome e `kind` INSIEME cosi' non
possono divergere, e l'unione dei `kind` in `mbSchema.ts` resta chiusa (un tag
nuovo si aggiunge al tipo, non si inventa sulla riga di comando).
**Conseguenza onesta**: l'artefatto di questa iterazione porta ancora il `kind`
vecchio, perche' e' stato scritto prima della correzione. Non lo riscrivo a
posteriori — un artefatto di misura ritoccato dopo il fatto non e' piu' una
misura. La riga 1 si chiude in it.2 con una run che porta tutte e cinque le
famiglie e il tag giusto, e QUELLA supersede questa.

**Gate**: `npx tsc --noEmit` pulito · `npx vitest run` **695 passed | 10
skipped** (erano 680, +15 dal test nuovo senza GPU). **Zero file sotto
`src/engine/`**: le due union di tipo allargate stanno in
`src/microbench/mbSchema.ts` (la frase precedente diceva `src/engine/` ed era
sbagliata — correzione del verificatore). Questa riga non tocca il motore per
contratto, e il diff lo dimostra.

### Verificatore indipendente (it.1): PASS, con sei correzioni — tutte applicate

Le tre che cambiano qualcosa:

1. **Avevo scelto l'evidenza piu' debole.** Invece del confronto fra bande
   (91 contro 89,9 GB/s) c'e' un controllo diretto IN MILLISECONDI che non
   avevo fatto: `24 x 395 x 1,2700 = 12.039 ms` contro i **12.169 ms** misurati
   sul segmento in produzione — **1,1% di scarto**. E' la grandezza che il mio
   stesso ruling dice di guardare («quota di byte ≠ quota di tempo»): il banco
   riproduce il segmento nel tempo, non solo nella banda.
2. **Il gate di checksum confronta la somma CON SEGNO** (`ck.sum`); `ck.abs` e'
   registrato e mai confrontato. Una permutazione di righe o un errore
   simmetrico ci passerebbe attraverso. Per una fase 0 va bene — la correttezza
   vera sta nel ktest delle righe 2-4 — ma il messaggio di commit di it.1 dice
   «entrambi verificati dal gate di checksum» e **avrebbe dovuto dire "non
   smentiti da"**. Lo correggo qui perche' un messaggio di commit non si
   riscrive.
3. **Un termine che la proiezione non modella**: la forma veloce emette DUE
   dispatch dove la legacy ne emette uno (+~9.480 sul deltanet, +~1.580 su
   ffn-down). Quel costo cade nei 9.346 ms «fuori dai pass GPU» che la
   proiezione tratta come invarianti. Sono decine di ms, non secondi — ma
   appartengono al conto, non alle note.

E una che mi RAFFORZA: la penale di lettura fredda e' quantificabile e
innocua. Il braccio veloce legge 7,21 MB in 0,0452 ms = 159 GB/s, il 16% del
tetto DRAM di questa scheda; anche serializzando del tutto la lettura fredda
(7,15 µs su 45,2) si arriva a 497 ms contro una barra di 2.000.

Rilievo di processo, giusto: **ho committato prima che il verificatore
rientrasse**, invertendo i passi 4 e 6 del protocollo. Il contenuto verificato
era byte-identico, ma l'ordine era sbagliato: da it.2 il commit segue il gate.

Prossima: it.2 — Q4_K, Q6_K, Q8_0 al banco (shape 35B dall'header dump, celle a
`splits` 1 o 2 dove il superblocco non si divide in 4) e run finale con
`--tag kquant-fase0`.

## it.2 — riga 1 CHIUSA: cinque famiglie misurate, tre previsioni cadute (2026-08-14)

Pre-registrazione scritta PRIMA della run:
`docs/deep-dive/kquant-fase0-prereg-2026-08-14.md` (P1..P5, regola di stop per
famiglia, e la decisione registrata di misurare le tre non cablate sulla sola
via intera).

    BASE_URL=http://localhost:5199 node scripts/tt-microbench-run.mjs \
      --label 4090-linux --host quiescent --tag kquant-fase0

Artefatto: `results/microbench/kquant-fase0-4090-linux-2026-08-14T19-14-34-680Z.json`
— `kind: microbench-kquant-fase0`, **26 celle, zero scartate**.

**LA TABELLA DELLA REGOLA DI STOP (M=16, barra 1,5x):**

| famiglia | shape | legacy | multi-riga intera | rapporto |
|---|---|---|---|---|
| Q5_K | `[4096, 2560]` | 1,2700 ms | 0,0453 ms | **28,03x** |
| Q4_1 | `[9216, 2560]` | 2,3474 ms | 0,1040 ms | **22,57x** |
| Q8_0 | `[2048, 4096]` | 1,0468 ms | 0,0302 ms | **34,65x** |
| Q4_K | `[2048, 512]` | 0,0700 ms | 0,0168 ms | **4,16x** |
| Q4_K | `[512, 2048]` | 0,0496 ms | 0,0095 ms | **5,20x** |
| Q6_K | `[512, 2048]` | 0,0494 ms | 0,0092 ms | **5,36x** |

**Tutte e cinque le famiglie superano la regola di stop.** Le due del 4B si
cablano (righe 2 e 3); le tre del 35B restano misurate e non cablate, ed e'
quello il loro scopo.

### Il consuntivo delle previsioni — e vale piu' dei numeri

- **P1 (>= 10x su tutte e tre le ereditate): CADUTA.** Q4_K sta a 4,16-5,20x e
  Q6_K a 5,36x. Solo Q8_0 la rispetta.
- **P2 (Q8_0 la meno migliorata): CADUTA, ed e' rovesciata** — e' la PIU'
  migliorata delle sei (34,65x).
- **P3 (K=512 rende meno di K=2048, stessa famiglia): CADUTA.** Q4_K a K=512 fa
  5,20x contro 4,16x, ed e' anche piu' veloce in assoluto (0,0095 contro
  0,0168 ms).
- **P4 (nessuna cella scartata): CONFERMATA.** Zero, su 26.
- **P5 (memoria di gruppo sotto i 16.384 B garantiti): CONFERMATA.** Il massimo
  e' 5.632 B (Q6_K, che ha un array in piu' per le somme a mezzo sotto-blocco).

**COSA HANNO SBAGLIATO LE TRE PREVISIONI CADUTE, ed e' la stessa cosa: avevo
attribuito il guadagno al FORMATO, e il guadagno e' una proprieta' della
SHAPE.** La leva non e' l'unpack: e' quante volte la forma legacy rilegge la
matrice, e quanto costa rileggerla. I tensori grandi (7-15 MB) rendono 22-35x;
quelli degli expert del 35B sono da **0,59-0,87 MB**, entrano in cache, e le
riletture della forma legacy se le serve la L2 quasi gratis — da cui 4-5x.
P3 in particolare ragionava sulle fette e si dimenticava di N: a K=512 la shape
ha N=2048, cioe' 32 workgroup x 2 fette = 64, contro gli 8 x 4 = 32 dell'altra.
Occupancy, di nuovo — la stessa causa che in `engine-ttft` aveva fatto perdere
la fusione GQA.

**CONSEGUENZA PER IL GOAL 35B, e va scritta adesso perche' e' contro-intuitiva:
il 4-5x misurato qui e' un LIMITE INFERIORE, non una stima.** Il banco misura
UN tensore da 0,59 MB in isolamento, quindi caldo in L2. In produzione il 35B
ha 117 tensori di expert per 17,67 GB: nessuno di quei pesi e' in cache quando
serve, e li' la rilettura M volte costa il prezzo pieno. Chi riprende quel goal
non deve leggere "4x" come il tetto del guadagno.

**Gate**: `npx tsc --noEmit` pulito · test senza GPU **18 passed** (erano 15:
tre nuovi inchiodano la decisione «due famiglie cablate con entrambe le vie,
tre con la sola intera» e il pad da 212 byte del Q6_K).

### Verificatore indipendente (it.2): FAIL — e aveva ragione

**Il rilievo che conta: ho chiuso una riga su un done-when che avevo ristretto
io.** Due restrizioni, nessuna registrata dove andava:
1. le tre famiglie ereditate misurate senza il fallback f32 — e il contratto
   dice, alla lettera, «ogni via intera nuova va accompagnata dal suo fallback
   f32 DICHIARATO». E' un CONSTRAINT del PI, non una preferenza di meccanismo:
   il test «se non arrivasse mai un ruling farei cosi'» **non si applica a cio'
   che il contratto copre in senso opposto**;
2. quelle tre misurate al solo M=16, mentre il done-when dice «a M = 1, 8, 16,
   su TUTTE le famiglie» — restrizione che non stava nemmeno nel prereg.

**Correzione presa in it.3: non escalo, eseguo.** Chiedere il permesso di
saltare tre kernel costava piu' che scriverli (mezza iterazione). La riga 1
resta aperta fino a li'.

**E tre affermazioni di it.2 erano sbagliate:**
- **«il relDiff del Q6_K e' il piu' alto di tutti»: FALSO.** Il piu' alto e'
  il **q8_0** (1,376e-2 contro 1,123e-2) — cioe' il kernel senza unpack, senza
  offset e senza termine costante, quello di cui avevo scritto «se sbaglia lei
  sbaglia il banco». Il residuo scala col numero di termini accumulati
  (K·N = 8,4 M prodotti contro 1,05 M), non con la complessita' del formato.
  Il numero che assolveva il Q6_K era nell'artefatto e non l'ho guardato.
- **P2 non era falsificabile come l'avevo scritta**: confronta «chi migliora di
  piu'» fra formati misurati su shape che differiscono di 10x in byte. Il
  rovesciamento e' del disegno dell'esperimento, non del mondo. L'ho graduata
  come scoperta ed era rumore metodologico.
- **P5 e' confermata a M=16 soltanto** per le tre famiglie nuove, mentre il
  prereg diceva «a ogni M ≤ 16». Ora e' vera come scritta (it.3 le misura a
  tutti e tre gli M).

## it.3 — riga 1 CHIUSA sul done-when SCRITTO, non su uno ristretto (2026-08-14)

Il FAIL di it.2 si chiude eseguendo, non negoziando: tre fallback f32 scritti
(Q4_K, Q6_K, Q8_0) e tutte le famiglie misurate a **M = 1, 8, 16** come il
contratto chiede. Piu' i tre difetti trovati dal verificatore, tolti.

    BASE_URL=http://localhost:5199 node scripts/tt-microbench-run.mjs \
      --label 4090-linux --host quiescent --tag kquant-fase0

Artefatto: `results/microbench/kquant-fase0-4090-linux-2026-08-14T19-29-20-014Z.json`
— **54 celle, zero scartate**, e `kind`/`goal`/`prereg` ora dicono tutti e tre
`engine-kquant` (v. sotto).

**LA MATRICE COMPLETA — rapporto legacy/veloce, `idot | f32`, barra 1,5x:**

| famiglia, shape | M=1 | M=8 | M=16 |
|---|---|---|---|
| Q8_0 `[2048,4096]` | 3,26 \| 3,21 | 22,68 \| 13,66 | **35,20 \| 17,63** |
| Q5_K `[4096,2560]` | 3,47 \| 0,56 | 19,85 \| 3,88 | **28,10 \| 6,17** |
| Q4_1 `[9216,2560]` | 4,51 \| 4,55 | 21,94 \| 14,45 | **22,57 \| 16,63** |
| Q6_K `[512,2048]` | 1,29 \| 0,41 | 4,38 \| 1,36 | **6,13 \| 1,66** |
| Q4_K `[512,2048]` | 1,38 \| 0,57 | 3,95 \| 1,53 | **5,23 \| 1,85** |
| Q4_K `[2048,512]` | 0,91 \| 0,20 | 3,20 \| 0,88 | **4,16 \| 1,14** |

**IL REPERTO CHE I FALLBACK HANNO COMPRATO, e che senza di loro non avrei
avuto: sul Q4_K la via f32 NON supera la regola di stop.** 1,14x sulla shape
gate/up e 1,85x sulla down, contro l'1,5x della barra. Su un device senza
`packed_4x8_integer_dot_product` la forma multi-riga, su quelle shape, **non
vale la pena**. E' esattamente il tipo di cosa che si scopre solo scrivendo il
fallback che si era deciso di saltare — il verificatore aveva ragione due
volte, non una.

**A M=1 la forma multi-riga PERDE** (0,91x sul Q4_K gate/up, 0,56x sul Q5_K
f32): il combine dei parziali non si ammortizza su una riga sola. Vale come
divieto per il cablaggio: il piano non deve offrire questa forma al decode.

**Cosa e' stato tolto (difetti, non note):**
1. `ttRunner.ts:933-934` — l'artefatto dichiarava `kind` giusto e `goal`/
   `prereg` del goal PRECEDENTE. Era la correzione di it.1 fatta a meta': ora i
   tre campi si muovono insieme dal `--tag`, e un tag senza provenienza
   dichiarata fa uscire il driver con exit 2 invece di scrivere un artefatto
   orfano.
2. `GOAL.md` — le shape del 35B nel done-when dicevano `K=2560`, che e' il
   dModel del **4B**. Refuso corretto contro l'header dump.
3. Il test che inchiodava la decisione ritirata ora inchioda il vincolo vero:
   **ogni** famiglia ha entrambe le vie, e **ogni** famiglia ha M = 1, 8, 16.

**LA RITRATTAZIONE PIU' IMPORTANTE — «il 4-5x del 35B e' un limite inferiore»:
NON LO REGGONO I DATI.** Tre ragioni indipendenti, tutte del verificatore:
- il crollo da 28x a 5x ha **due** cause di peso simile, non una: il legacy e'
  ~2,2x piu' veloce per peso sulle shape piccole (cache), ma la forma veloce e'
  ~2,3x piu' LENTA (32-64 workgroup su 128 SM: fame di parallelismo). 2,2 x 2,3
  ≈ 5. Solo la prima migliora in produzione;
- il legacy a `[512,2048]` legge 9,44 MB in 49,6 µs = **190 GB/s** su una
  scheda che ne fa ~1.000: quella cella non e' limitata dalla banda, quindi
  «in produzione la rilettura costa il prezzo pieno» presuppone un regime che
  la misura non mostra;
- **e la piu' seria**: per Q4_K e Q6_K il braccio legacy che ho misurato **non
  e' il percorso di produzione**. Il motore instrada gli expert in regime
  d'ARENA con `accum` (`q35gpumodel.ts`), e `wgsl.ts:2175`/`:2359` vietano per
  costruzione la combinazione `batch && arena`. La GEMV `batch: true` su
  binding diretto, su quei 117 tensori, il motore non la emette e non puo'
  emetterla. In piu', nel prefill MoE la premessa stessa della leva cade: token
  diversi selezionano expert diversi, quindi la "stessa matrice riletta M
  volte" non c'e'.
**Il 4-5x non e' ne' un tetto ne' un pavimento: e' valido per una shape su un
braccio ipotetico.** La scheda di consegna della riga 4 lo dira' cosi'.

**Gate**: `npx tsc --noEmit` pulito · test senza GPU **19 passed** · suite
intera verde. Zero file sotto `src/engine/`: la riga 1 non tocca il motore.

**RIGA 1 CHIUSA.** Prossima: riga 2 — Q5_K in produzione, la prima che muove
il TTFT (−11,7 s proiettati). Veicolo dichiarato in PHASES: `sdd-conductor`.

## it.4 — riga 2: il Q5_K E' IN PRODUZIONE (2026-08-15)

Veicolo: `sdd-conductor` (dichiarato in PHASES), spec in
`docs/engine/kquant-riga2-spec.md`. Cinque task, **zero bloccati**, 25 agenti,
~3,3 h. Il suo autoreport NON e' evidenza: sotto ci sono i gate che ho eseguito
io.

**LA TRAPPOLA DELLA RIGA, evitata perche' era scritta nella spec.** `ssm_out`
**non passa da `gemvB`**: il ramo K-quant di `loadW`
(`q35gpumodel.ts:487-505`) si costruisce da se' il proprio `pushB` con
`gemvQ5KWgsl({batch: true})` e non tocca mai quel bivio. Cablare `gemvB` — la
cosa che sembra ovvia — non avrebbe cambiato **una riga** del comportamento, e
il test di copertura sarebbe passato lo stesso perche' conta i SITI, non i
dispatch. Il cablaggio e' nel ramo K-quant, verificato nel diff.

**GATE, eseguiti da me dopo il rientro del workflow:**
- `npx tsc --noEmit` **pulito**
- `npx vitest run` **779 passed | 10 skipped** (erano 699)
- `node .harness/tools/engine-ktest.mjs` → **103 PASS / 0 FAIL** (erano 101), coi
  due casi nuovi presenti e verdi: `prefill-gemm-q5k-multirow-idot` maxRel
  **2,61e-7** e `prefill-gemm-q5k-multirow-f32` **4,28e-7**
- copertura, test `[6c]`: **10,9376x** sull'inventario per-layer INTERO a M=16
  (barra della riga: >= 10,9; era 5,8593). **196/248 siti = 96,914% dei byte**,
  52 eccezioni = 3,086%
- `[6d]` **cancellato con la sua ragione**, come chiedeva il suo stesso commento

**CONTROLLO END-TO-END, e cosa dimostra davvero**: conformita' col golden
llama.cpp sul prompt-idx 0 (6333 token) col prefill a chunk M=16 →
**top1 62/64 = 96,875%** (`results/engine/q35-conf-4b-riga2-check-2026-08-15.json`).
**Questo numero non e' il gate del contratto** (che e' >= 1012/1024 full-corpus,
riga 6, su codice finale): su 64 posizioni ±1 colpo vale ±1,6 punti, ed e' la
landmine del campione piccolo. **Cio' che dimostra e' l'unica cosa che serve
adesso**: un cablaggio sbagliato darebbe spazzatura, non il 96,9% d'accordo con
un oracolo esterno. La non-regressione fine si misura alla riga 6.

**Il gate a due bracci NON e' stato usato, e non per pigrizia**: la
bit-identita' fra prefill sequenziale e a chunk e' **caduta in it.15 del goal
precedente** per via della quantizzazione delle attivazioni a int8 — leva
autorizzata dal PI — ed e' **sospesa con ruling**, non abbandonata. Girarlo
oggi direbbe `bitIdentical false` per una causa che non e' la mia.

**NON ANCORA VERIFICATO, e lo dichiaro**: che il segmento `gemm:deltanet-out`
sia davvero sceso. Il piano dice che 196 siti su 248 prendono la via veloce, ma
il piano non e' il cronometro. La misura e' la riga 5, su codice finale — cioe'
dopo la riga 3, come impone il vincolo «i bench costosi si eseguono su codice
finale».

Corretta un'etichetta lasciata dal workflow: due commenti dicevano «riga 3»
dove il lavoro e' la riga 2.

Prossima: riga 3 — il Q4_1. Piu' corta della 2: `ffn_down` passa da `gemvB`,
che la rotta al piano la chiede gia'.

## it.5 — riga 3: il Q4_1 e' cablato, la conformita' e' in coda (2026-08-15)

Il workflow si e' fermato al **primo wave con la suite rossa**, e ha fatto bene:
aveva messo i kernel q4_1 e allargato `PREFILL_GEMM_KINDS`, e **cinque
asserzioni in altri file** codificavano ancora il mondo precedente (q4_1 =
legacy). Non ha riscritto test che non erano suoi — che e' la regola giusta.

**LA GUARDIA DELLA RIGA 2 HA PAGATO, ed e' il reperto di questa iterazione.**
Nel momento in cui il piano ha accettato il q4_1 ma `gemvB` non emetteva ancora
i suoi kernel, la doppia condizione `route.via !== "legacy" && kk === "q4_0"` ha
fatto **ricadere quei tensori sulla legacy** invece di leggerli col kernel del
q4_0 — nibble senza offset, scale con passo sbagliato: logit storti, nessun
errore WebGPU, nessuna eccezione. La guardia era stata scritta come cintura
teorica; ha intercettato il caso vero dopo un giorno.

**Completato a mano:**
- il sito q4_1 in `gemvB`, **esplicito e non un ternario**. Avevo scritto la
  versione col ternario e il gate strutturale l'ha bocciata: quel test legge il
  SORGENTE per verificare che il kernel sia emesso con gli stessi `opts` con cui
  si e' chiesta la rotta, e dietro una variabile la catena non e' piu' leggibile
  — il gate diventerebbe cieco proprio sulla proprieta' che esiste per
  difendere. Sei righe in piu' valgono un controllo che resta meccanico;
- la **categoria di misura propria** per i quattro siti (`gemm:ffn-down-q41`):
  e' cio' che il verificatore aveva chiesto in it.1, e va fatto PRIMA della
  misura di chiusura o il prima/dopo non e' confrontabile;
- le cinque asserzioni aggiornate alla verita' nuova, ognuna con la ragione
  scritta accanto invece che cambiata in silenzio.

**Copertura: 10,9376x → 15,5247x** sull'inventario per-layer INTERO a M=16.
**200/248 siti = 99,796% dei byte.** Resta legacy un solo kind: i 48 siti Q8_0
con N=32, esclusi coi numeri dal contratto (0,204%).

**Gate**: `npx tsc --noEmit` pulito · `npx vitest run` **791 passed | 10
skipped**.

**RIGA 3 NON CHIUSA, e lo dichiaro**: mancano i due casi ktest del q4_1 e il
loro floor test — cioe' l'unica parte che nessun test di aritmetica puo'
dedurre (il kernel gira davvero e da' i numeri giusti su una GPU vera?). Coda
affidata a un workflow con spec propria
(`docs/engine/kquant-riga3-coda-spec.md`), che e' una gemellazione del caso
q5_K e non un progetto nuovo.

## it.6 — il banco q4_1 c'e', il gate su GPU no (2026-08-15)

Il workflow della coda ha consegnato **tre task su quattro**: il caso e le
tolleranze (`PREFILL_Q41_KTEST_CASE`, quattro tolleranze derivate), il floor
test esteso al q4_1, e il test di composizione del `pbCat`. **Il quarto — il
banco vero dentro `ktest.worker.ts` — non l'ha implementato**, lasciando in
albero il suo test in fase ROSSA (22 falliti): e' la forma TDD portata a meta'.
L'ho scritto io, gemello di quello q5_K.

**I pavimenti del q4_1, misurati dal floor test e non scelti:**

    [q41 idot] senzaFMA 1,462e-5 · conFMA 1,693e-5 · floor 1,693e-5 · tol 2e-4 = 11,8x
    [q41 f32]  senzaFMA 1,489e-5 · conFMA 1,715e-5 · floor 1,715e-5 · tol 2e-4 = 11,7x
    [q41 MUTATA senza +m*Sigma(x)] rel 5,233 — quattro ordini sopra la tolleranza

L'ultima riga e' il discriminante che conta: togliere il termine costante del
formato fa esplodere l'errore, quindi il test **non passerebbe** su un kernel
che se lo dimentica. E i pavimenti del q4_1 sono ~40x quelli del q5_K
(1,7e-5 contro 4,4e-7), coerente con 288 termini per riga invece di 128 e con
un prodotto scalare che qui non e' interamente esatto.

**Difetto tolto sul percorso**: il test di cablaggio del q5_K contava le
chiamate ai generatori CONDIVISI (`prefillQuantXQ8Wgsl`, `prefillSplitKCombine`)
su tutto il file. Misurava «quanti banchi esistono», non «come e' cablato
questo», e sarebbe tornato rosso a ogni banco nuovo per una ragione che non ha
niente a che vedere con cio' che difende. Ristretto al corpo del suo banco.

**E POI IL GATE SU GPU E' SPARITO.** Cinque tentativi, tre sintomi diversi,
sempre subito dopo lo stesso caso — e **fallisce anche col banco nuovo
disattivato**, che e' l'osservazione che scagiona il kernel q4_1. Attribuzione
per esclusione (profilo, server, Chrome zombie del Playwright MCP lasciato
aperto da un subagente: tutti esclusi con la misura) in **docket item 2**.

**Mi sono fermato al quinto tentativo, e il protocollo dice al secondo.** Lo
registro: ogni ritentativo costava 10 minuti di GPU e nessuno di essi ha
prodotto un'osservazione nuova dopo il terzo — quello che ha scagionato il
banco. Da li' in poi stavo solo sperando.

**STOP BY DESIGN.** Tutto cio' che resta del goal — chiudere la riga 3, la riga
4 (i tre banchi del 35B), la riga 5 (il checkpoint che dice se il TTFT e'
davvero sceso) e la riga 6 (i gate di merge) — passa dal ktest o da una run di
modello, cioe' dalla stessa infrastruttura che ora non regge. Il discriminante
e' un ambiente pulito, e su una macchina di qualcun altro non lo decido io.

## it.7 — RIGA 3 CHIUSA: era l'ambiente, e il riavvio l'ha dimostrato (2026-08-15)

**Il PI ha eseguito il discriminante che avevo chiesto — riavvio della
macchina — e il ktest e' tornato verde al primo tentativo utile.**

    [ktest] adapter: {"webgpu":true,"adapter":"nvidia lovelace"}
    [ktest] OK — 105 PASS, 0 FAIL

**Fra l'ultimo fallimento di it.6 e questa run non ho cambiato una riga di
codice.** L'albero e' lo stesso commit (`aff171b`), lavoro pulito. L'unica
variabile mossa e' l'ambiente: e' la definizione stessa di esperimento
controllato, ed e' il PI ad averlo fatto girare.

**`q35-mtp-head-real-blk32` PASS** (max|Δ| 4,77e-6, maxRel 5,67e-4, L2rel
2,67e-7), e PASS tutti i trenta e passa casi che lo seguivano e che in it.6 non
venivano mai raggiunti. Il punto di rottura stabile non era una proprieta' del
banco: era il momento in cui la sessione lunga aveva finito il credito.

**I DUE CASI DELLA RIGA 3, i primi mai eseguiti su GPU vera:**

    prefill-gemm-q41-multirow-idot   PASS   max|Δ| 7,63e-5   maxRel 1,73e-5
    prefill-gemm-q41-multirow-f32    PASS   max|Δ| 7,63e-5   maxRel 1,51e-5

**E qui c'e' la cosa che vale piu' del PASS.** I pavimenti derivati dal floor
test erano `1,693e-5` (idot) e `1,715e-5` (f32). L'errore misurato sulla GPU
vera e' `1,73e-5` e `1,51e-5`: **siamo esattamente sul pavimento**, non a
meta' strada verso la tolleranza. Vuol dire due cose insieme:
- il margine 11,8x verso `tol 2e-4` e' reale ma **e' tutto pavimento**, non
  slack — il caso e' dominato dall'aritmetica del formato (288 termini per riga,
  contrazione FMA), non da un difetto del kernel;
- **derivare la tolleranza invece di sceglierla ha funzionato**: un numero
  scelto a occhio o l'avrebbe messa troppo stretta (falso rosso permanente) o
  troppo larga, e in nessuno dei due casi avremmo saputo quale dei due.

Il discriminante del floor test regge: togliere `m*Sigma(x)` dal kernel porta
l'errore a `5,233` idot / `4,192` f32, cinque ordini sopra la tolleranza. Il
banco **non** passerebbe su un kernel che si dimentica il termine costante del
q4_1.

**DONE WHEN della riga 3, voce per voce:**

| clausola | esito | evidenza |
|---|---|---|
| `[6c]` ≥ 15,5x con le 4 `ffn_down` Q4_1 fra i `multirow` | **si'** | 15,5247x, it.5 |
| caso ktest Q4_1 PASS | **si'** | 105 PASS / 0 FAIL, i due casi qui sopra |
| floor test esteso | **si'** | pavimenti derivati, mutazione a 5,233 |
| `gpulimits` verde | **si'** | dentro la suite |
| tsc + vitest verdi | **si'** | `tsc --noEmit` exit 0 · `vitest run` **836 passed, 10 skipped**, exit 0 |

**RIGA 3 CHIUSA.** Il Q4_1 e' in produzione e verificato su GPU vera.

**IL PREZZO DI it.6, scritto perche' non si ripeta.** Cinque run di ktest da
~10 minuti l'una. La terza — quella che disattiva il banco nuovo e vede
fallire lo stesso — aveva gia' chiuso l'attribuzione: **non e' il codice
nuovo**. Da li' in poi la mossa corretta era una sola frase al PI («riavvia»),
non altri due tentativi. La regola che ne esce, e che vale oltre questo goal:
*quando la disattivazione del codice nuovo non cambia l'esito, l'ipotesi
ambiente e' gia' provata; il ritentativo non e' una misura.* Docket item 2
chiuso con questa nota.

**Il difetto trovato per strada, e tolto**: il comando che avevo lasciato in
HANDOFF.md per la ripresa era **incompleto** — `node .harness/tools/engine-ktest.mjs`
senza `BASE_URL`, con vite sulla 5199 e il runner che di default parla alla
5173. Prima cosa fatta alla ripresa, primo comando, fallito. Costo reale zero
(il runner rifiuta subito e dichiara la porta, senza spendere GPU) e proprio per
questo istruttivo: il runner era gia' scritto bene, ero io ad aver scritto male
la consegna. Corretto in HANDOFF.md. La stessa landmine e' annotata nel prereg
della riga 1 del goal ttft — la conoscenza c'era, non era nel posto dove sarebbe
servita.

**PROSSIMA: riga 4** — le tre forme del 35B (Q4_K, Q6_K, Q8_0) misurate e
verificate ma **non** cablate, piu' la scheda di consegna al goal successivo.

## it.8 — RIGA 4 CHIUSA: tre formati portati, verificati, e NON cablati (2026-08-15)

**`[ktest] OK — 111 PASS, 0 FAIL`.** I sei bracci nuovi (due per formato) girano
su GPU vera e stanno tutti **sul pavimento derivato o appena sotto**:

    prefill-gemm-q4k-multirow-idot  maxRel 7,39e-5   floor 7,841e-5   tol 1e-3
    prefill-gemm-q4k-multirow-f32   maxRel 3,86e-5   floor 6,266e-5   tol 8e-4
    prefill-gemm-q6k-multirow-idot  maxRel 8,08e-5   floor 8,810e-5   tol 1e-3
    prefill-gemm-q6k-multirow-f32   maxRel 5,63e-3   floor 1,118e-2   tol 1,5e-1
    prefill-gemm-q80-multirow-idot  maxRel 5,96e-4   floor 7,611e-4   tol 1e-2
    prefill-gemm-q80-multirow-f32   maxRel 2,04e-4   floor 1,859e-4   tol 2e-3

E' la terza volta di fila che i pavimenti derivati predicono il silicio, ora su
tre formati che in produzione non erano mai girati. La regola «la tolleranza si
deriva, non si sceglie» ha smesso di essere una preferenza di stile.

### Il conflitto di fattibilita', trovato PRIMA di spendere

La riga 4 chiedeva due cose che oggi non stavano insieme: **il kernel in
produzione** e **il piano che non lo instrada**. Il predicato di ammissibilita'
del piano non e' una lista propria — e' derivato INTERAMENTE da
`PREFILL_GEMM_KINDS`, per design dichiarato (`prefillgemmplan.ts:174-193`: «IL
PREDICATO NON VIVE QUI: si SONDA il kernel»). Quindi «il kernel esiste ma non e'
instradato» non era esprimibile.

E non era un cavillo: **`prefillGemmCheck` controlla kind, K e splits ma NON N**.
Mettere `q8_0` fra i kind avrebbe instradato da solo i **48 siti
`ssm_alpha`/`ssm_beta` del 4B con N=32** — mezzo workgroup per dispatch — che il
contratto esclude coi numeri. Su q4_K e q6_K non sarebbe cambiato nulla (il 4B
non ne ha un byte); su q8_0 si', e in silenzio.

**Deciso da me** (meccanismo, non funzione obiettivo): flag **`wired` +
`wiredWhy`** in `PREFILL_GEMM_SPEC`, dove gia' vivono tutti i numeri che
dipendono dal formato, consumato in **una sede sola** dentro `kernelVerdict` di
`prefillgemmplan.ts` — quindi nessun secondo predicato fuori da quel file, che
e' cio' che il gate strutturale sorveglia. Il `wiredWhy` non e' ornamento: un
booleano nudo non e' diagnosticabile, e la ragione finisce in telemetria.

**Il rifiuto per cablaggio ha una frase DIVERSA da quello geometrico.** Dare a
«questa shape non si moltiplica cosi'» e a «questa shape si moltiplicherebbe
benissimo, ma nessun sito ci passa» la stessa diagnosi avrebbe messo in
telemetria un problema di geometria dove il kernel non c'entra niente.

### Il port e' un port, e c'e' il numero che lo prova

`PREFILL_GEMM_PORT_DIFFS` resta **`{}`** con sei kernel in piu': il testo
portato e' byte-per-byte quello che la fase 0 ha misurato, verificato nelle due
direzioni su **14 coppie**. Se avessi riscritto invece di portare, il record non
sarebbe potuto restare vuoto.

**La produzione del 4B non cambia di un dispatch**: `[6c]` invariato a 15,5247x,
200/248 siti, 48 eccezioni tutte q8_0. Verificato anche sul caso concreto (q8_0
K=2560 N=32 resta legacy), non solo sul flag. E il gate morde: girando
`wired: true` su q8_0 falliscono 8 test in 3 file — provato mutando e
revertendo.

### Il refuso nel contratto, corretto

`GOAL.md` diceva che il prefill del 35B «gira su un piano DIVERSO
(`moeprefillplan.ts`)». **E' falso**: `planMoeChunk` ha un solo consumatore di
produzione, `glmmodel.ts:1368` — il GLM, non il 35B. Il 35B ripete per riga la
catena del DECODE (`q35gpumodel.ts:2743-2778`): readback CPU dei router logits
per riga, `pinUnion` che calcola gia' l'unione ma solo per pinnare gli slot, poi
`prepLayer` + `encodeExperts` per riga — 40 round-trip per chunk, 512 dispatch
per layer. Corretto sul posto col precedente del refuso `K=2560` di it.2. Chi
avesse ereditato quella frase avrebbe chartato il goal 35B sul file sbagliato:
il piano CPU-side e' gia' parametrico su `{nExpert, nExpertUsed}` e `{256, 8}`
lo soddisfa per struttura — **il lavoro e' il ramo `moe` di `q35gpumodel.ts`**.

### Il caso q6_K via f32, che meritava di essere sospettato

Pavimento relativo **1,118e-2** e tolleranza **1,5e-1**: una tolleranza
relativa del 15% ha tutta l'aria della compiacenza. Non lo e', ed e' stato
verificato: il caso peggiore cade sull'uscita di modulo **piu' piccolo
dell'intera griglia** (|ref| = 4,395e-2 contro una media di 1,340e3), dove
l'errore assoluto vale 4,914e-4 — dentro il pavimento assoluto della stessa
via. E' una cancellazione, non un kernel impreciso. La tolleranza e' forzata dal
pavimento (13,4x sopra, sotto il tetto di 20x che il test impone) e il
discriminante regge lo stesso: la mutazione porta l'errore a >10x la
tolleranza. Su GPU vera il misurato e' **5,63e-3**, meta' dell'inviluppo.

**Una precisazione che correggo dal report dell'agent**: `compare` fa
`maxAbs <= absTol || maxRel <= relTol` — un OR sui MASSIMI, non elemento per
elemento (`ktest.worker.ts:124`). Quindi non e' vero che «li' il giudizio lo
porta il ramo assoluto»: con quel relativo il banco passa dal ramo relativo. Non
e' un difetto — la mutazione dimostra che discrimina lo stesso — ma la ragione
scritta accanto dev'essere quella giusta.

### La trappola che il goal 35B eredita, scritta in tre posti

**Il flag `wired` e' per FORMATO, non per shape.** Il giorno in cui il goal 35B
accendera' `q8_0` per i suoi tensori attn (N=4096), i 48 siti del 4B con N=32
entreranno **nello stesso istante**, perche' `prefillGemmCheck` continua a non
guardare N. Serve, in quel momento, un predicato sulla shape. Sta nella
`wiredWhy`, nella scheda di consegna e qui — ma una stringa non e' un gate, e
chi cabla deve saperlo prima di girare il flag.

### DONE WHEN della riga 4, voce per voce

| clausola | esito | evidenza |
|---|---|---|
| un caso ktest PASS per ciascuna delle tre forme | **si'** | 6 bracci, 111 PASS / 0 FAIL |
| test che il piano NON le instrada | **si'** | `engine-prefillgemmplan-notwired.test.ts`, 17 test, gate mutato e provato |
| scheda di consegna al goal 35B | **si'** | `docs/engine/kquant-consegna-35b-2026-08-15.md`, 423 righe |
| tsc + vitest verdi | **si'** | exit 0 · **998 passed, 10 skipped** (erano 836: +162) |

**RIGA 4 CHIUSA.**

### Difetti tolti sul percorso, non recintati

- il commento di `q35gpumodel.ts:530` diceva «se domani il q4_K entra
  nell'elenco»: e' entrato oggi. Riscritto col fatto, e con la ragione per cui
  la guardia doppia resta la difesa che conta anche ora che i tre kind sono
  `wired: false`;
- l'avviso in `engine-ktest-q41-wiring.test.ts` sul conteggio dei generatori
  condivisi a livello di file era **gia' falso** (il debito era stato pagato in
  it.6): sostituito con la constatazione e la regola per chi aggiunge il settimo
  banco. Verificato che nessun test conti piu' a livello di file.

**NON toccato di proposito**: `docs/engine/kquant-riga3-coda-spec.md` parla di
due formati su cinque, ma e' l'ordine di lavoro di UNA coda gia' eseguita, non
un riferimento vivo. Estenderlo lo trasformerebbe in un documento che non e'.

**PROSSIMA: riga 5** — la misura di chiusura. Checkpoint fresco sul prompt-idx 0,
le due barre di segmento a 2.000 ms, e il debito di
`build-ttft-checkpoint.mjs:108` (i byte del segmento derivati dal meter invece
che ricopiati). **E' la prima riga da it.4 che muove la metrica del goal**, ed e'
la prima misura di tempo dell'intero goal.
