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
