# Docket — engine-fase-d

1. **PLAN-CHECK — PRE-AUTORIZZATO (2026-08-10, PI in chat: "vai" sul
   contratto presentato).** PHASES.md è su disco al tag di apertura: 9 fasi
   sequenziali, authority delta = none ovunque, nessuna docket-born. Blocco A
   = fasi 1-6 (merge gate autonomo), blocco B = 7-8, chiusura = 9. Se vuoi
   rivederla, il punto giusto è PRIMA della fase 2 (la fase 1 è
   parametrizzazione pura, reversibile e coperta dal ktest bit-exact).
   REGOLA DI MISURA nel piano: bench pieni SOLO alle fasi 6 e 8.

2. **REGISTRAZIONE (2026-08-10, it.1)** — i campi piatti di `SlabLayout`
   (`gateQs`, `gateScales`, `upQs`, `upScales`, `downQs`, `downScales`,
   `qsBytes`, `gateScalesBytes`, `downScalesBytes`, `downKind`) restano come
   COMPAT DERIVATA per non toccare ~100 call site GLM in un colpo solo. Non
   sono una seconda fonte di verità (derivano dai tre `SlabTensorLayout`).
   Da rimuovere quando i call site migrano alla vista generica — lavoro di
   igiene interno al goal, non un debito verso l'esterno.

3. **ECCEZIONE DICHIARATA E PERMANENTE (2026-08-10, it.1)** — il router
   del CPUREF (`q35cpurefmodel.ts`) resta un'implementazione INDIPENDENTE:
   è il riferimento del differential testing, se usasse `routerSelect` un
   bug del router sarebbe invisibile al confronto. Il gate strutturale la
   elenca come eccezione, non come debito.

4. **DA CHIUDERE PRIMA DEL DONE DI FASE 1 (2026-08-10, verifier it.2)** —
   STATO al 2026-08-10 it.5: **(a) (c) (d) CHIUSI** in it.3 (verificati);
   **(b) CHIUSO in it.6, MA NON come previsto** — il gate è ora fatto di invarianti su ATTI SINGOLI
   (allocazione GPU; nomi dei tensori expert; clamp del router), ognuno con
   allowlist motivata, senza esenzioni per import e senza congiunzioni da
   spezzare. MA il verifier ha bocciato anche quella versione con 5 evasioni
   eseguite, di cui una (un router Qwen legittimo) NON è catturabile da
   nessuna scansione del sorgente: la differenza fra duplicazione e seconda
   famiglia è SEMANTICA. ESITO di it.6: l'invariante è stato spostato nel
   SISTEMA DI TIPI — marchio di conio su `SlotRef` (solo residency.ts può
   CONIARLO: 11 sonde ostili del verifier tutte rifiutate da tsc). ATTENZIONE
   alla formulazione (il verifier ha bocciato quella precedente): il marchio
   ferma chi CONTRAFFÀ uno SlotRef, NON chi lo ignora — q35gpumodel oggi
   gestisce un'arena completa senza SlotRef e tsc è verde. Diventa portante
   con it.7. Bypass noti senza cast: inflow any e spread di uno SlotRef
   genuino. Più il test di tipo
   `tests/types/slotref-brand.ts` che va rosso se il marchio sparisce. La
   scansione del sorgente resta come RATCHET su impronte note, con la
   pretesa ridimensionata PER ISCRITTO nel file. Evasioni N3/N4/N5 chiuse
   allargando la scansione a tutto src/ (ancorata a __dirname, estensioni
   incluse) e correggendo il predicato di allocazione. **Docket item 4
   CHIUSO.** Resta la migrazione di q35gpumodel (it.7), che è ciò che
   elimina DAVVERO la duplicazione: le tre voci DEBITO NOTO dell'allowlist
   spariscono lì. Rilievi minori del verifier it.3 chiusi in it.4: flushSlotTable
   dimensionata sulla shadow e non sulla costante GLM; getter compat non
   enumerabili (spread/JSON di un layout K-quant non esplodono);
   `slotsOverride` con chiavi validate contro le classi della config (prima
   costruiva una cache a zero classi in silenzio).

   (a) motore di `ExpertCache` cfg-driven (stati di classe, `expertSlots`,
   `arenaNeeds`, `ensure`, `repinPass`, `stats`, `destroy`): oggi c'è un
   GUARD che rifiuta le config non onorate, che è onesto ma non è la parità;
   (b) gate strutturale da firme TESTUALI a invariante non aggirabile
   (modello: il nome-API di `gpudevice.test`, che non si può eludere);
   (c) campi compat di `SlabLayout`: sui K-quant fabbricano un offset
   `scales` finto (bytes 0) — o si rimuovono o si fa fallire chi li usa su
   un layout K-quant, prima che q35 passi da `slotBindRanges`;
   (d) `expertSlots` ripartisce sul parco GLM: renderlo cfg-driven, altrimenti
   q35 girerebbe solo via `slotsOverride` = un bypass.

## item 5 — la fase 2 ha centrato l'OBIETTIVO ma non la LETTERA del done-when (PI)

**Cosa dice PHASES riga 2**: "il repack K-quant esce dal path on-miss
(import-time, come GLM `expertSlab`)".

**Cosa ho fatto invece**: ho reso il repack quasi gratuito DOVE STA. Misure
(micro-bench CPU sui due slab reali del 35B, 20 ripetizioni):

| | prima | dopo | fattore |
|---|---|---|---|
| pack q4_K | 2,94 ms | 0,44 ms | 6,7x |
| pack q6_K | 2,49 ms | 0,58 ms | 4,3x |

Due cause, entrambe reali: (a) `repackKQuant` ricostruiva le parole a 32 bit
con un `|=` per BYTE — ma su little-endian quell'aritmetica scrive
esattamente il byte al suo posto, cioe' e' una COPIA travestita; (b)
`packExpertSlab` allocava un array temporaneo per tensore e poi lo ricopiava
nello slab, toccando ogni byte TRE volte (zero-fill, copia, copia).

**La domanda per il PI**: il residuo e' ~0,5 ms/miss. Sullo smoke del 35B
sono ~1,7 s su ~45 di prompt (~4%). Spostarlo all'import costerebbe
**~18 GB di slab su disco** (il GGUF e' 20 GB) piu' una passata di import su
tutto il file. Il mio parere: NON conviene, e la riga 2 di PHASES andrebbe
riscritta come "il costo di pack per miss scende di >=4x, misurato" — che e'
l'obiettivo per cui la fase esisteva. Ma riscrivere un done-when e' un
cambio di contratto, non una decisione di meccanismo: la lascio qui.

**Se il PI dice "no, fallo all'import"**: si fa, e la fase 2 si riapre — il
lavoro fatto resta valido comunque (il pack veloce serve anche all'import).

Registrato it.8 (2026-08-10).

### RULING PI 2026-08-10: ACCOLTO

"Sono d'accordo con te su item 5. Hai fatto la scelta giusta."

Il done-when della fase 2 e' RISCRITTO in PHASES.md nella forma ratificata:
"il costo di pack per miss scende di >=4x, misurato PRIMA e DOPO nello stesso
JSON". Gli slab all'import NON si fanno: ~4% di residuo non vale ~18 GB su
disco. **Item 5 CHIUSO.** La fase 2 e' chiusa anche per contratto.


## item 6 — la telemetria read/pack/upload esiste su q35 ma non su GLM (io, fase 5/6)

Rilievo del verifier di it.8: `timing: true` è ora permanente sul path q35
(costa 4 `performance.now()` per MISS) ed è ciò che ha reso DECIDIBILE la
fase 2 — senza la scomposizione non avrei saputo che il pack era il 22%.
Su GLM la telemetria è opt-in dal chiamante e i worker non la accendono.
Alla fase 5 (policy) e alla 6 (checkpoint) serve lo stesso strumento sui due
path, altrimenti si confrontano due cose misurate diversamente.
NON è una decisione da PI: è mio, e lo faccio quando serve. Registrato qui
per non perderlo. it.9 (2026-08-10).

## item 7 — il costo di I/O per miss NON è misurato (io, fase 5)

`readMs` misura la finestra dentro `ensure`, dove per q35 l'I/O non c'è per
costruzione (la lettura Range sta nell'`await` prima). Il campo ora lo dice
nel tipo (it.9), ma il NUMERO manca: e la fase 5 deve decidere se il
prefetch si giustifica, il che dipende esattamente da quanto costa una
lettura. Va strumentato l'`await` in `runLayer` prima di attaccare la fase 5.
it.9 (2026-08-10).


## item 8 — il MoE fa 41 submit/token: serve il resolve su GPU (PI: e' una fase nuova?)

Emerso misurando la fase 3 (it.10). `decodeBatch` funziona sui densi
(-15,0 ms/token) ed e' `null` sul MoE, perche' la selezione degli expert
legge i logits del router su CPU a OGNI layer: sul 35B sono **41 submit e 41
readback per token**.

E' esattamente il problema che GLM ha risolto in fase C passando da 47 a 2
sync/token, e la meccanica c'e' gia' in `residency.ts` (`arena: true`,
`slotTable: true` + kernel d'arena che ricavano l'indirizzo dello slot da
soli). q35 oggi non la usa: binda sotto-range calcolati dalla CPU.

**Perche' e' PI-gated**: non e' una riga della fase 3, ed e' plausibilmente
il pezzo piu' redditizio rimasto nel goal — sul 35B il decode e' dominato da
quei 41 round-trip, non dal calcolo. Le opzioni:
(a) fase 3-bis nuova, subito dopo la 3, prima di prefill e policy;
(b) dentro la fase 5 (policy MoE), che gia' tocca la residenza;
(c) piu' avanti, accettando che il 35B resti sul path lento fino alla fase 5.

**Il mio parere: (a)**. Il prefill batched (fase 4) e la policy (fase 5) sul
MoE si misurano male finche' ogni token paga 41 round-trip: rischiamo di
ottimizzare sopra un collo di bottiglia che poi sparisce, cioe' di rifare le
misure due volte. Registrato it.10 (2026-08-10).

### RULING PI 2026-08-10: (a) — FASE 3-BIS

"(a), fai la fase 3-bis". Riga aggiunta a PHASES fra la 3 e la 4.
**Item 8 CHIUSO.**

**Correzione di FATTO (it.17, 2026-08-11)**: il "41 submit" di questo item era
una stima, e la stima era bassa della meta'. Misurati sul 35B con i contatori
di `perf()`: **81 submit/token** (ogni layer MoE ne fa DUE — segmento statico e
dispatch dinamici) e 41 readback/token (questi erano giusti). Non cambia
nessuna decisione ne' riapre l'item: l'ordine di grandezza e la conclusione
erano quelli. Sta qui perche' un numero citato tre volte in tre documenti
diventa vero per ripetizione se nessuno lo misura.


## item 9 — il done-when della fase 3 NON e' soddisfatto: -3,3 contro >= -5,1 (PI)

Il verifier ha bocciato it.10: il mio "-15,0 ms/token" era un ARTEFATTO DI
MISURA (braccio lento cronometrato a freddo, subito dopo il load; braccio
veloce a caldo, terza passata). Ri-misurato a caldo, interleavato, 3
ripetizioni, mediana e dispersione:

| 4B | ms/token | [min-max] |
|---|---|---|
| `step` con sync per token | 40,02 | 40,0-40,2 |
| `decodeBatch` | 36,69 | 36,7-36,8 |
| accodato senza sync (pavimento) | 35,44 | 35,4-35,5 |
| **delta** | **-3,33 (-8,3%)** | spread 0,19 / 0,10 |

La riga 3 di PHASES chiede "**atteso >= -5,1 ms** dal q1". Misurato -3,33.
La differenza NON e' rumore: la dispersione e' 0,1-0,2 ms.

**Perche' e' -3,3 e non di piu'**: a caldo la serializzazione totale vale
4,58 ms/token (40,02 - 35,44) e il batch ne recupera il 73%. Il -5,1 del
contratto veniva dalla decomposizione del gap di q1, misurata su un altro
regime: era una stima, e la stima era alta.

**Le opzioni**:
(a) accettare -3,3 come esito della fase 3 sui densi e riscrivere la riga
    come "delta MISURATO a caldo con dispersione, qualunque sia";
(b) tenere la riga com'e' e lasciare la fase 3 APERTA finche' non si trova
    l'altro 1,8 ms (il pavienot senza sync e' 35,44: sopra quello non si va
    senza toccare i kernel, che e' un'altra fase);
(c) considerare la fase 3 sui densi chiusa e ri-misurarla alla fase 6, dove
    il numero che conta e' end-to-end.

**Il mio parere: (a)**. Il -5,1 era una stima ex-ante, non un requisito del
prodotto; il numero misurato e' positivo, verificato e con dispersione
dichiarata, e il guadagno vero sui densi non e' li' ma nei kernel (il
pavimento a 35,4 ms/token con 562 dispatch). Registrato it.12 (2026-08-10).

### RULING PI 2026-08-11: (a) — ACCOLTO

"(a) sul docket item 9". Il done-when della fase 3 e' RISCRITTO in PHASES.md:
il delta si riporta **MISURATO a caldo con dispersione, qualunque sia**, e la
soglia ex-ante di -5,1 ms cade (era una stima della decomposizione del gap di
q1, misurata in un altro regime, non un requisito del prodotto). Il numero di
riferimento della fase 3 sui densi resta **-3,33 ms/token (-8,3%)**, spread
0,19/0,10, e il gate secco (argmax identico su 39 token) e' quello che
qualifica la fase. Il guadagno residuo sui densi sta nei kernel (pavimento
35,44 ms/token con 562 dispatch), non nel batching del decode.
**Item 9 CHIUSO.** La fase 3 e' chiusa sui densi anche per contratto; il MoE
resta alla fase 3b.

## item 10 — regola nuova per l'harness: il primo passaggio non si misura (io)

Osservazione del verifier di it.10, che vale oltre quella fase: il primo
decode dopo `createQ35GpuModel` costa ~8 ms/token in piu' degli altri
(compilazione pipeline, cache, prima allocazione). Misurare un braccio li' e
l'altro a caldo gonfia qualunque delta. **Ogni micro-bench del goal scarta
una passata e interleava i bracci, e riporta mediana + dispersione, non un
campione.** Applicato al bench della fase 3 (it.12); da applicare al bench
del prefill della fase 4, che e' formulato allo stesso modo. Mio, non PI.
Registrato it.12 (2026-08-10).

## item 11 — il budget dell'arena del 35B non e' ctx-aware (io, fase 6)

Emerso da un OOM in it.14. GLM calcola il budget slab con
`slabBudgetCtxAware(allocCeiling, ctx)`: tetto MISURATO meno non-expert, KV,
buffer di lavoro e riserva. Il 35B no: `createQ35GpuModel` prende
`arenaBudgetBytes` come parametro, default **12 GiB fissi**, e nessuno
controlla che ci stia. Su questo host (14,4 GiB liberi) 12 GiB + densi + KV
sfonda, e il fallimento e' `VK_ERROR_OUT_OF_DEVICE_MEMORY` alla createBuffer:
rumoroso, ma solo perche' c'e' il listener `uncapturederror` — i buffer
diventano invalidi e il modello continua a girare producendo numeri plausibili
(top1 1/5 nel run viziato).

Non e' una decisione da PI: e' mia, ed e' lavoro della fase 6 (il checkpoint
che misura il 35B ai tier). Registrato qui per non perderlo. Nel frattempo i
run di correttezza dichiarano il budget (`--arena-gib`, nuovo in it.14).
it.14 (2026-08-11).

### CHIUSO it.35 (2026-08-11) — derivato, non calcolato

Il budget si deriva da `--vram-gib` (tetto) meno cio' che il modello ha DAVVERO
allocato meno la riserva. Non e' la formula di GLM (`slabBudgetCtxAware`, che
sottrae termini calcolati e che il suo stesso commento racconta essere gia'
costata un OOM per un termine dimenticato): e' un contatore dentro i due helper
di allocazione, quindi comprende pesi, KV, scratch e piano di prefill senza che
nessuno debba ricordarsene — ed e' ctx-aware per costruzione, perche' `kCache` e
`vCache` si allocano con `ctxMax`. Misurato sul 35B: tetto 13 GiB, **allocati
1,94**, riserva 0,50, budget expert 10,56. Run verde (argmax 39/39, routing
identico, 0 miss). **Item 11 CHIUSO.**

## item 12 — due sensori che mentono in silenzio sul path nuovo (io, fase 6)

Rilievi di margine di it.17, registrati per non perderli. Nessuno dei due e'
una decisione da PI ed entrambi sono lavoro della fase 6 (il checkpoint che
pubblica i numeri del 35B), non della fetta corrente:

1. **`readTap` in modo ottimistico ritorna un `Float32Array` VUOTO** invece di
   fallire: il tap e' cablato solo nel ramo sync di `step`, quindi chi lo usa
   col path a submit unico riceve un array vuoto e non un errore. E' uno
   strumento di debug che tace quando dovrebbe urlare.
2. **`dispatchesPerToken` non e' il numero di dispatch per token.** E'
   `steps.length`, cioe' i soli step STATICI: non ha mai contato i 5 dispatch
   per expert (320 x 5 sul 35B), e col path ottimistico non conta nemmeno i 40
   router. Viene pubblicato in ogni JSON di conformance e di bench col nome che
   promette il totale — 782 sul 35B, quando i dispatch veri sono migliaia.

Piu' una CONNESSIONE fra due cose gia' note: il repair+replay portera' 62,8 MiB
di snapshot dello stato ricorrente (it.17), e il budget dell'arena del 35B non
e' ctx-aware (item 11) — cioe' quei MiB andranno a sommarsi a un tetto che
nessuno controlla, sullo stesso host dove 12 GiB gia' andavano in OOM. I due
item vanno chiusi insieme o il secondo si accorgera' del primo con un
VK_ERROR_OUT_OF_DEVICE_MEMORY.

Registrato it.17 (2026-08-11).

### CHIUSO it.38 (2026-08-11) — i tre sensori sistemati prima dei bench

`moeStats` riporta la `policy` attiva (era il rilievo di it.37: due run che
differiscono per la policy si distinguevano solo dal comando) · nasce
`dispatchBreakdown` con static/dynamic/total, e sul 35B il totale e' **2.102**
contro i 782 che `dispatchesPerToken` pubblicava · `readTap` nel path
ottimistico LANCIA invece di restituire un array vuoto. Piu' `vramPlan` nel
report. Fatti PRIMA del checkpoint A perche' i bench lunghi si fanno su codice
finale e un report che mente li invalida. **Item 12 CHIUSO.**

## item 13 — i tre numeri che la FASE 5 deve usare per tarare la soglia (io, fase 5)

Misurati in it.18 sul regime freddo del path ottimistico (smoke 35B, 39 token,
10 GiB), e sono INPUT della policy d'ingresso, non lavoro della 3b:

1. **Il replay rigioca l'80,7% del token** (32,3 layer su 40, media di 109
   replay): il primo layer sporco sta in basso, quindi "rigiocare da firstDirty"
   e "rifare il token" quasi coincidono.
2. **Il repair fetcha il +12,0%**: 3742 miss contro i 3341 del path sync sullo
   stesso prompt. Un expert riparato puo' non finire nella `Sel` definitiva,
   perche' il replay a valle — con l'hidden corretto — sceglie diversamente:
   910 selezioni contate come miss e mai usate. E' inerente al meccanismo.
3. **Il repair e' il 65,6% del token freddo** (484,3 ms/token di CPU su 738,6).

E una PREVISIONE MIA SMENTITA, che vale piu' dei tre numeri: in it.16 avevo
scritto che a cache fredda il path ottimistico nudo sarebbe stato una
regressione. Misurato: 738,6 ms/token contro 1192,9 del sync freddo. NON lo
prendo per buono — un campione per braccio, due run diversi, passata dominata
dall'I/O: per l'item 10 non e' una misura. Ma la soglia della fase 5 va tarata
su un bench fatto apposta (bracci interleavati nello stesso processo, cache
riportata a freddo fra i bracci), non sulla mia intuizione di it.16 e nemmeno
su questi due numeri.

Registrato it.18 (2026-08-11).

### CHIUSO it.36 (2026-08-11) — la soglia non serve

Misurato nello STESSO processo (`debugEvictAll` fra i bracci) e nei DUE ordini,
per limitare il confondente della cache di sistema: a freddo l'ottimistico fa
650,92 / 660,67 ms/token contro 1111,50 / 1098,40 del sync — **1,68x piu'
veloce**, con l'ordine che sposta l'1,2%. A caldo 43,57 contro 132,81 (3,05x).
Il replay costa (109 replay, +12% di fetch, 80,7% del token rigiocato) ma costa
MENO dei 77 round-trip per token del path sync. **La soglia e' esclusa coi
numeri** e i tre numeri di questo item restano come descrizione del regime
freddo. **Item 13 CHIUSO.**

## item 14 — il dispatch del router costa 3,5 volte un GEMV expert (io, fase 5 o leva kernel)

Misurato in it.19 con la sonda dei timestamp: **70,8 us per dispatch** contro i
20,9 di un GEMV expert e i 19,6 di uno statico. E' UN workgroup: `routerTopKWgsl`
fa il prefill dei logits in parallelo e poi lascia al thread 0 la softmax su 256
expert e la selezione top-8 (nUsed x nExpert confronti in seriale). Sul 35B sono
40 dispatch per token = **2,83 ms, il 4,9% del tempo GPU**.

Non e' lavoro della fase 4 (che toglie dispatch, non li rende piu' veloci) e non
e' una decisione da PI. Sta qui perche' e' il tipo di costo che si scopre una
volta e poi non si ritrova piu': con GLM (64 expert, top-4) era un ottavo del
lavoro seriale e nessuno l'aveva notato. Registrato it.19 (2026-08-11).

## item 15 — il collo del 35B non e' dove il contratto sta guardando (PI: si cambia l'ordine?)

**Cosa e' successo.** it.19 ha misurato la decomposizione del token e io ne ho
tratto "il token e' dispatch-bound, ~20 us per dispatch": su quella lettura la
fase 4 (prefill batched) proiettava 2,02x. it.21 ha fatto l'esperimento che la
testava — togliere 320 dispatch per token, bit-identici — e la lettura e'
CADUTA: −0,327 ms, cioe' **1,02 us per dispatch rimosso invece di 21**.

**Il numero che resta in piedi**: gli expert leggono 571 MB di pesi per token
(320 selezioni x 1,785 MB, byte misurati) in 33,1 ms = **17,2 GB/s efficaci**,
su una GPU che ne fa ~500. E' il 3% della banda, ed e' il 58% del tempo GPU del
token.

**L'ipotesi sul perche'** (coerente col numero, NON provata): `gemvQ4K`/`gemvQ6K`
spartiscono i superblocchi di UNA riga sui 64 thread del workgroup, e sul 35B
sono 8 superblocchi per gate/up (K=2048) e **2 per il down** (K=512) ⇒ **8 lane
attive su 64 e 2 su 64**. Il workgroup occupa uno slot per far lavorare due
thread. Su GLM il rapporto era diverso (K piu' grandi) e il problema non si
vedeva.

**Perche' e' PI-gated.** Non e' una scelta di meccanismo dentro una fase: e'
l'ordine del contratto. La fase 4 (prefill batched, 2-3 iterazioni) ora ha una
proiezione che NON regge piu' — era costruita su tempo ∝ dispatch — e vale
comunque zero sul DECODE, che e' dove sta la funzione obiettivo del goal
(intelligenza sopra ~30 tok/s: il 35B oggi fa 71,9 ms/token = 13,9 tok/s, e per
arrivare a 30 servono 33 ms). Riscrivere i kernel expert K-quant vale su
prefill E decode, ed e' l'unico numero della decomposizione abbastanza grande da
poter fare la differenza.

Le opzioni:
(a) fase 4-bis PRIMA della 4: kernel expert K-quant riscritti per occupazione,
    gate = ktest contro cpuref + argmax identico sul golden, misura ms/token
    prima/dopo. Poi la 4 col suo micro-bench, e la sua proiezione RIFATTA sui
    numeri veri;
(b) si tiene l'ordine: fase 4 come da contratto, e i kernel dopo;
(c) si chiude la fase 4 come "esclusa coi numeri" (il done-when ammette
    l'esclusione motivata) e si passa direttamente ai kernel.

**Il mio parere: (a)**. La fase 4 non e' sbagliata — il prefill batched serve al
TTFT — ma la sua taglia (2-3 iterazioni) e il suo valore vanno ridiscussi DOPO
aver visto se i kernel expert si possono muovere, perche' se si muovono
cambiano anche i numeri su cui la fase 4 si misura. E l'ordine sbagliato costa
due volte: si ottimizza sopra un collo che poi sparisce (e' testualmente
l'argomento con cui il PI ha accolto l'item 8).

Registrato it.21 (2026-08-11).

### RULING PI 2026-08-11: (a) — FASE 4-BIS PRIMA DELLA 4

"sono d'accordo con (a)". I kernel expert K-quant si affrontano PRIMA del
prefill batched. Riga **4-bis** aggiunta a PHASES fra la 3b e la 4, col
done-when scritto qui sotto e i riferimenti misurati di it.19/it.21 come "prima"
(17,2 GB/s efficaci, 33,1 ms/token sul segmento expert, 57,3 ms di tempo GPU).
La fase 4 resta in piano ma la sua proiezione va RIFATTA dopo la 4-bis, perche'
i numeri su cui si misura cambiano. **Item 15 CHIUSO.**

DICHIARATO PRIMA DI COMINCIARE, come per la 3c: **la 4-bis NON sara'
bit-identica per costruzione**. Distribuire il lavoro su piu' lane cambia
l'ORDINE delle somme f32 dentro il prodotto scalare, e in f32 l'addizione non e'
associativa. Il gate quindi non e' la bit-identita' ma: ktest contro cpuref con
la tolleranza di oggi, argmax identico sul golden smoke, routing invariato.

## item 16 — il segmento STATICO e' peggiorato di 1,62 ms con la 4-bis (io, fase 4-bis o 6)

Misurato in it.22: rifacendo la spartizione del lavoro nei kernel K-quant, il
segmento expert e' sceso da 33,115 a 8,743 ms/token (−73,6%) ma il segmento
STATICO e' salito da 14,523 a **16,144** (+11,2%). Non e' rumore: nelle tre
misure precedenti stava fra 14,52 e 14,53.

Ipotesi NON verificata: nel regime nuovo le scale del superblocco si
ri-estraggono una volta per UNITA' invece che una per superblocco, e sui GEMV
statici K-quant la ridondanza potrebbe superare il guadagno. Non mi convince —
sugli expert la stessa ridondanza c'e' ed e' stravinta 3,8 a 1 — quindi la
spiegazione va misurata, non assunta: serve la sonda per-kernel dentro il
segmento statico (oggi la categoria e' una sola e mescola attenzione, deltanet,
shexp, router GEMV).

Vale 1,62 ms su 44,26 di token (3,7%): non blocca niente, e infatti la 4-bis
chiude comunque a −27,64 ms. Registrato perche' un +11% su una categoria dopo un
cambio di kernel e' esattamente il tipo di cosa che si smette di vedere se non
la si scrive. it.22 (2026-08-11).

### CHIUSO it.24 (2026-08-11) — non e' attribuibile alla 4-bis, per costruzione

Enumerati i tipi dei tensori del 35B dal GGUF: attn (q/k/v/output/qkv/gate),
ssm_out e shexp sono **Q8_0**; router, alpha, beta, norm e conv1d sono **F32**;
i q4_K/q6_K stanno SOLO negli `*_exps.weight`. Il segmento statico non contiene
nemmeno un GEMV K-quant: passa tutto da `gemvQuantWgsl` e `gemvF32Wgsl`, il cui
testo emesso la 4-bis non ha cambiato di un byte. Stessi kernel, stessi dati,
stesso lancio ⇒ il +1,62 ms non viene dal cambio. Restano deriva fra run o un
effetto globale (il token e' passato da 71,9 a 44,3 ms: la stessa GPU fa molto
piu' lavoro al secondo, e clock e potenza non sono gli stessi) — la classe di
fenomeni per cui esistono `hostState` e i bracci interleavati. **Item 16
CHIUSO.** E l'ipotesi che avevo scritto (ridondanza nell'estrazione delle scale)
era sbagliata: la correzione sarebbe stata su un kernel che non partecipa.

## item 17 — un terzo del token non e' tempo GPU, e nessuna fase lo guarda (PI)

Misurato in it.23 sul 35B a caldo: il token dura **45,48 ms** (44,26 con la
sonda spenta) e i pass GPU ne spiegano **30,36**. I restanti **15,12 ms — il
33,2% — stanno fuori**: encode CPU dei ~2300 dispatch, submit, attesa del
readback, argmax su 151k logit in CPU, dequant della riga di embedding.

E' la seconda voce del token dopo gli expert (8,72), piu' grande del blocco
GEMV del deltanet (9,84) e cinque volte il router. **Nessuna riga del contratto
la tocca**: la fase 4 e' prefill, la 5 e' policy di residenza, la 7 e'
spec-dec, e la 6 e' un checkpoint di misura.

**Perche' e' PI-gated**: la funzione obiettivo del goal e' il decode sopra ~30
tok/s. Siamo a 22,0-22,6 e servono −12 ms. Quei 15,12 ms sono il posto dove
−12 ms sono aritmeticamente possibili senza toccare un kernel — e la fase 4,
che e' la prossima riga, ne vale zero (agisce sul TTFT).

Le opzioni, come per l'item 15:
(a) fase 4-ter PRIMA della 4: attaccare il tempo fuori dai pass — decomporlo
    (encode / submit / attesa / argmax / embed), poi l'argmax su GPU (esiste
    gia' per i densi: `decodeBatch` lo fa), l'embed gather su GPU, e
    l'overlap fra il readback di un token e l'encode del successivo;
(b) fase 4 come da contratto, e questo dopo;
(c) portarlo dentro la fase 6 (checkpoint), che gia' misura.

**Il mio parere: (a)**, con la stessa logica con cui il PI ha accolto l'item 15
— e in piu' qui c'e' un fatto che li' non c'era: il primo passo di (a) e' una
MISURA (decomporre i 15,12 ms), quindi la fase si puo' aprire sapendo dopo
un'iterazione se vale la pena, invece di scommettere 2-3 iterazioni.

Registrato it.23 (2026-08-11).

### RULING PI 2026-08-11: (a) — FASE 4-TER PRIMA DELLA 4

"ok con l'opzione che raccomandi" su item 17. Riga **4-ter** aggiunta a PHASES
fra la 4-bis e la 4: si attacca il tempo che NON e' tempo GPU. **Item 17
CHIUSO.**

Il primo passo e' una MISURA — decomporre i 15,12 ms in encode / submit /
attesa / argmax / embed — e la riga lo dice: se la decomposizione non trova
almeno ~8 ms aggredibili, la fase si chiude come "esclusa coi numeri" invece di
spendere iterazioni. E' la differenza che ho argomentato rispetto all'item 15:
qui si sa dopo UNA iterazione se vale.

## item 18 — restano ~11 ms di GPU che nessun pass contiene, e l'unica leva e' il pipelining (PI, ma non ora)

Bilancio della fase 4-ter (it.28-29): il token del 35B e' 43,3 ms, di cui ~41 di
attesa GPU e **1,94 di CPU**. Ma la somma dei pass cronometrati e' 29,5 ms:
**~11 ms di tempo GPU non stanno in nessun pass**, e quattro ipotesi sono state
misurate e refutate (snapshot dello stato 0,30 ms · checkpoint dell'hidden ~0 ·
`popErrorScope` = attribuzione non costo · spezzare il pass ≤1 ms, nel rumore).

La spiegazione residua piu' plausibile, NON verificata: la latenza del round-trip
GPU→CPU per token (wire Dawn + event loop), inerente a "un sync per token". Si
attaccherebbe solo col **PIPELINING** — encodare il token N+1 mentre il readback
di N e' in volo — che cambia la semantica del decode (oggi la contabilita' di
fine token e il repair vivono in quel readback) e vale la pena solo se prima si
MISURA che quei ms sono davvero latenza e non lavoro.

Non lo apro come fase: la 4-ter ha gia' consumato il suo mandato e si e' chiusa
per esclusione com'era previsto. Sta qui perche' e' il piu' grosso singolo
addendo rimasto nel token (11 su 43) e perche' la prossima volta che qualcuno
guardera' quel numero deve trovare scritto cosa e' gia' stato escluso.
Registrato it.29 (2026-08-11).

## item 19 — il `batch` dei gemv fonde i dispatch, non il traffico dei pesi (io, fase 4)

Misurato in it.32: il prefill a chunk del 4B da' **1,151x** (29,88 → 25,97
ms/token, M=8, primo chunk scartato) contro i ~2x che la proiezione di it.23
prometteva. La proiezione si reggeva su "il traffico dei pesi si legge una volta
per chunk", e **non e' vero per questi kernel**: il modo `batch` mette `wid.z` =
riga e ogni workgroup (riga, riga d'uscita) rilegge la propria riga di pesi.
Fonde i dispatch, non il traffico.

Per avere l'amortizzazione serve una GEMM vera — una tile di pesi caricata una
volta e moltiplicata per M righe — cioe' la famiglia
`rmsPairGemmSiluChunkFast`/`rmsGemmQkvChunkFast` del path Qwen 2.5, che esiste
ma per un'altra geometria e un altro layout di pesi.

Non lo apro come fetta adesso: la fase 4 ha il suo done-when soddisfatto (logits
bit-identici + micro-bench prima/dopo) e il guadagno c'e', anche se e' un terzo
di quello sperato. Sta qui perche' la prossima persona che legge "2x" nella
proiezione di it.23 deve trovare accanto il motivo per cui il misurato e' 1,15x.
Registrato it.32 (2026-08-11).

**AGGIORNAMENTO it.33**: il quadro e' meno cupo di cosi', e per una ragione che
non avevo previsto. A parita' di contesto la M quasi non conta (1,218x a M=8 e
1,236x a M=16 su 1024 token), il che CONFERMA che i pesi non si amortizzano. Ma
il guadagno **cresce col CONTESTO**: **2,019x a 6456 token** (M=8, 806 campioni).
La spiegazione plausibile e' che a contesto lungo domina l'attenzione e le M
righe leggono la STESSA KV da workgroup concorrenti — il riuso non lo fa il
kernel, lo fa la cache della GPU. Quindi la GEMM vera resta la strada per
amortizzare i PESI, ma il meccanismo attuale gia' amortizza la KV dove conta.

## item 20 — il CHECKPOINT A non e' eseguibile su questo host cosi' com'e' (PI)

Il primo braccio (non-regressione GLM b12) e' morto con
`VK_ERROR_OUT_OF_DEVICE_MEMORY`. **Non e' il codice**: il JSON del riferimento
porta `vramPeakMiB: 15160` e uno `hostState` con `memUsedMiB: 817`; oggi la
sessione ne tiene **1.977** e ne restano liberi **13.971**, cioe' **1.189 in
meno di quanti il run ne pretende** — ed e' entro 30 MiB il delta della sessione
desktop (1.160). La correttezza di GLM e' intatta (ktest 96/96, L2rel identico
all'ultima decimale): manca la PRESTAZIONE a parita' di host.

La riga 6 pretende "host DICHIARATO, GPU scarica", e la condizione non e'
soddisfatta. Non e' una scelta di meccanismo: e' la macchina di chi lavora.

Le opzioni:
(a) **liberare ~1,2 GiB di VRAM** (sessione desktop piu' leggera: chiudere
    browser/app che tengono la GPU) e ri-lanciare il b12 identico al
    riferimento — e' l'unica strada che da' il confronto ±5% che la riga chiede;
(b) **ri-baselinare a un budget che entra** (b10 o b11): si ottiene un numero
    onesto ma NON confrontabile coi 13.172/31,26/14,74, quindi la riga 6 andrebbe
    riscritta e il riferimento storico va dichiarato non riproducibile su questo
    host;
(c) rimandare il checkpoint a quando la macchina e' libera, e intanto fare i
    riferimenti q35 (che nei tetti disponibili ci stanno).

**Il mio parere: (a) se e' possibile liberare la VRAM, altrimenti (c)**. La (b)
costa il confronto storico, che e' la ragione per cui quel riferimento esiste:
un merge gate che cambia metro mentre lo si attraversa non e' un merge gate.

Registrato it.39 (2026-08-11).

**ITEM 20 — CHIUSO in it.40, SUPERATO DAI FATTI (nessun ruling servito).** La
domanda era se liberare ~1,2 GiB di VRAM sulla macchina di chi lavora. Ri-ancorato
da disco a inizio sessione, `nvidia-smi` dava **577 MiB usati / 15.372 liberi** —
la sessione desktop e' oggi PIU' LEGGERA di quando fu preso il riferimento (817).
La condizione della riga 6 e' soddisfatta a costo zero, quindi non e' stata scelta
ne' la (a) ne' la (b) ne' la (c): la premessa e' decaduta. Il braccio e' stato
rilanciato identico al pin e PASSA in banda +/-5% (13,437 / 31,813 / 14.490,8 ms
contro 13,172 / 31,265 / 14.744,9), con la meccanica del path ottimistico identica
all'ultima cifra (pDirty 0,9375, missesPerToken 4,781).

## item 21 — l'exit code di `glm-bench-run.mjs` non dice quello che un merge gate gli chiederebbe (io, fase 6)

I gate che il runner valuta sono il **floor CPU di llama.cpp** (13,43 decode /
56,58 prefill, funzione obiettivo direction §2) e lo **strutturale <= 2
submit/token**. Nessuno dei tre e' la banda di NON-REGRESSIONE, che e' il
confronto con le mediane del riferimento. Conseguenza misurata in it.40: il
riferimento b12 optimistic 2026-08-09 fallisce prefill e strutturale, quindi
**exit 4 e' l'esito normale di una run perfettamente in banda** — e oggi, che il
decode e' passato da FAIL a PASS sullo stesso gate, l'exit code e' rimasto 4
lo stesso.

Non e' un bug del runner: quei gate misurano un'altra cosa, e la misurano bene.
E' un rischio del CHECKPOINT A, che e' un MERGE gate: chiunque automatizzi
"checkpoint verde = exit 0" leggerebbe questa run come una bocciatura. Per ora
l'ho annotato nel commento d'uso del runner e la lettura della non-regressione
resta il confronto programmatico coi campi del JSON di riferimento.

Aperto come item perche' la strada giusta (un flag `--nonreg <json-di-riferimento>`
che confronti le mediane e faccia lui la banda, con un exit code suo) e' un cambio
al runner, e l'albero e' congelato fino a fine checkpoint. Non lo apro adesso.
Registrato it.40 (2026-08-11).

## item 22 — la `firma` della riga 6 resta scoperta per scelta di scope (PI, fase 6)

La riga 6 chiede, per il braccio GLM: "b12 optimistic in banda +-5% + golden AL
PIN + cpuref + **firma**". Dopo it.41 e it.42 lo stato e':

- b12 optimistic in banda **✓** (it.41: -0,36% / -0,03% / +0,03%, stdev dimezzata)
- golden AL PIN **✓** (it.42: 1012/1024 = 98,828125%, identico al riferimento)
- cpuref **✓** (it.42: 256/256)
- **firma ✗**

La `firma` e' la ROUTING conformance (`glm-routing-conformance`, ~95 min):
direction § la chiama per nome due volte — "full-corpus, routing = firma item
14b" e "firma routing invariata ai conteggi". Il PI, in sessione, ha scelto
"solo conformance ai logits" e poi ha confermato "vai coi logits full": e' una
decisione di scope PRESA, non una dimenticanza, e sta qui perche' lascia una
voce della riga 6 non soddisfatta.

**Cosa abbiamo al posto suo, e quanto vale.** Nel report dei logits il blocco
`residency` e' identico voce per voce al riferimento, `bytesUploaded` compreso
AL BYTE su 4.999.280 richieste e 211.117 sfratti. Una selezione di expert
diversa anche di poco muoverebbe miss, sfratti e byte caricati: e' evidenza
circostanziale fortissima che il routing sia invariato. Ma non e' il gate — il
gate confronta l'istogramma per expert chiave per chiave — e chiamare "firma"
questa inferenza sarebbe esattamente il tipo di scorciatoia che il ratchet
esiste per impedire.

**Tre esiti possibili, nessuno dei quali decido io**: (a) lanciare la routing
conformance prima di chiudere il checkpoint (~95 min, la riga 6 torna piena);
(b) riscrivere la riga 6 togliendo `firma` e dichiarando per iscritto che il
routing si considera coperto dai conteggi di residenza — cambio di contratto;
(c) rimandarla alla riga 9 ("non-reg GLM piena fresca"), accettando che il
CHECKPOINT A si chiuda con una voce scoperta e che, se saltasse li', non si
sappia se e' stata l'unificazione del core o la fase 7 (spec-dec MTP).

Parere: (a) se il checkpoint deve valere come merge gate; (c) e' difendibile
solo se si accetta di perdere la bisezione contro la fase 7. La (b) toglie un
gate mentre lo si sta attraversando.

Registrato it.42 (2026-08-11).
