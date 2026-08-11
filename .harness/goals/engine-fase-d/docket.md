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
