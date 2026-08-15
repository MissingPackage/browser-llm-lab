# JOURNAL — engine-35b-residency

## it.0 — charter (2026-08-15)

Goal chartato dai contatori di `results/chat/chat-35b-2026-08-15T16-25-12.json`.
`GOAL.md` + `PHASES.md` scritti. Sette righe, barra decode 35B ≥ 30 tok/s.
Tre correzioni di fattibilità nel test-fit (C0-1, C0-2, C0-3).

## it.1 — la riga 1, prima metà: i contatori (2026-08-15)

**Fatto.** Sette contatori nuovi in `q35gpumodel.ts`, propagati a
`chat.worker.ts` e `q35conf.worker.ts`. Commento di `perf()` corretto (—84%).
`tests/engine-35b-repair-counters.test.ts`, 6 casi. vitest 1025 passed, tsc
exit 0. Commit `2f1637b`.

**Ritrattato nello stesso ciclo (C0-4).** «Token pulito = 21,1 ms» non regge:
`readbackMs`/pass 37,4 ms > 21,8 ms di porzione non-tail, e `q35-vramplan-35b-it35`
misura 43,74 ms su un token con 0 miss. Commit `967f928`. La riga 2 non parte
finché il token pulito non è misurato.

**Fermo sul docket item 1**: host non quiescente, quattro Chrome/MCP di
Playwright vivi. Sono della sessione `personal-site-47`; il PI ha chiesto di
accordarmi con lei invece di chiudere. Messaggio inviato, in attesa.

## it.2 — la finestra: le due run, e C0-4 è CONFERMATA (2026-08-15)

`personal-site-47` ha liberato la GPU (stava facendo screenshot di anteprime
tema). Verificato da qui prima di partire: `pgrep -af
"type=gpu-process|/opt/google/chrome/chrome|chromium|headless_shell"` **vuoto**;
i cinque `@playwright/mcp` sono server node senza figli Chrome — il mio check
precedente guardava il pattern sbagliato, e la correzione viene da lei. Dev
server 200 su `curl`, 22 GB di RAM disponibili. Entrambe le run exit 0.

### Run A — il token pulito è 43,59 ms, non 21,1

`results/engine/q35-optimistic-35b-cleantoken-2026-08-15.json`, pass
`optimistic-warm`, **0 miss · 0 dirty · 0 replay** — un token pulito misurato:

    tokenMs          43,585 ms/token      (it.35, 2026-08-11: 43,736)
    readbackWaitMs   40,753  = 93,5%      (it.35: 40,977)
    encodeMs          1,188
    argmaxMs          0,399
    tailCpuMs         0,200
    mediana su 3      44,000               (it.35: 44,316)

**Coincide con la misura del 2026-08-11 entro lo 0,7%.** Due conseguenze:

1. **C0-4 è confermata e la mia stima di stamattina era sbagliata.** Il token
   pulito vale **22,9 tok/s**. Togliere il 100% della tassa di residency **non
   raggiunge la barra dei 30**. Le righe 2 e 3, da sole, non chiudono il goal.
2. **I kernel di `engine-kquant` non hanno mosso il 35B di un ms** (43,74 →
   43,59, dentro il rumore). Era previsto e ora è verificato: il 35B non ha un
   byte di q4_0 e `PREFILL_GEMM_WIRED_KINDS` ne copre lo 0%
   (consegna §2). Il decode del 35B non è mai stato toccato da quel goal.

**Il termine che decide il goal è cambiato**: sul token pulito `readbackWait` è
il **93,5%**, con `submitsPerToken: 1` e `readbacksPerToken: 1`. Non è overhead,
è la GPU che lavora 40,75 ms su un solo pass — i 320 GEMV expert per token. La
leva su quel termine **è quella che il contratto ha dichiarato fuori scope**: la
forma a gather K-quant della consegna §4.2-4.4, stimata a ~2,6× di dispatch.

### Run B — i sette contatori reggono su GPU vera, e `namedFrac` = 0,9995

`results/engine/q35-optimistic-35b-arena4-2026-08-15.json`, arena strozzata a
4 GiB, `--opt-cold`. Pass `optimistic-warm` (39 token, regime sporco al 100%):

    tokenMs        561,97 ms/token
    tailCpuMs      509,71  = 90,7%
      repairMs     391,8   /token  (15.278,8 totali)
        fetchRepairMs  275,4 /token (10.740,2) = 70,3% del repair
        flushMs          0,38 /token — trascurabile, misurato e non assunto
      replayPassMs   117,7 /token  (4.588,6)
      accounting       0,29 /token (11,42)
    namedFrac      0,9995        ← done-when della riga 1: ≥ 0,95. PASSATO.

**Il 43% anonimo è diventato lo 0,05%.**

**Validazione dei contatori — il controllo che li qualifica.** L'avvertenza è di
`personal-site-47`: con contatori nuovi lo zero è ambiguo per costruzione, e
«misurato zero» va distinto da «mai scritto» prima di fidarsi. Applicata così:

- **`fetchRepairCalls` deve valere esattamente `misses`** (il codice fa una
  `readExpert` per miss). Torna in tutte e tre le passate dove il path gira:
  **4227=4227 · 3283=3283 · 2752=2752**. Un contatore nel posto sbagliato non
  ci azzecca tre volte su tre.
- **Ogni zero ha un gemello non-zero in un'altra passata, per una ragione
  strutturale dichiarabile prima**: `fetchRepairMs` è 0 in tutto il run A
  perché lì la passata fredda è `sync`, che usa l'altro sito; `fetchPrepMs` è 0
  nelle passate ottimistiche. I due si accendono in modo complementare — più
  forte di «non è zero».

**Tre reperti nuovi, non richiesti dal done-when:**

1. **Il costo per fetch è 3,27 ms, non 5,98.** I 5,98 erano una mia derivazione
   dal buco anonimo della chat, che conteneva anche i pass di replay: la
   strumentazione ha corretto la derivazione una seconda volta. Su
   5.855.674.368 byte in 10.740 ms = **545 MB/s**.
2. **Il RAGGRUPPAMENTO delle richieste vale 2,1×.** `fetchPrepMs`/chiamata =
   18.981,7/2.752 = **6,90 ms** contro i **3,27 ms** del repair. Stessi byte,
   stesso server, stesso `readExpert`: cambia solo quante `readRange` stanno
   nella stessa `Promise.all` — 24 in `prepLayer` (8 expert × 3 tensori),
   qualche centinaio nel repair (tutti i miss del prefisso). **È un 2× che la
   riga 2 può prendere senza cambiare sorgente dei byte**, ed è un ingresso che
   il contratto non aveva.
3. **`namedFrac` è indefinito quando non c'è repair** (in A: `tailCpuMs` 0,1996
   con repair e replay a zero ⇒ 0/0,1996 = 0, che *sembra* un fallimento del
   done-when e non lo è: senza miss il 100% di `tailCpu` È contabilità). La
   clausola ≥ 0,95 va letta **solo nel regime sporco**. Da precisare nel
   done-when, altrimenti la riga 1 si dichiara fallita su una passata pulita.

---

## PRONTO PER LA FINESTRA — i due comandi, verificati dal sorgente
## [ESEGUITI in it.2 — lasciati per tracciabilità]

Preparati mentre aspetto, così la finestra si spende eseguendo. **Nessuno dei
due è stato eseguito.** Flag letti da `scripts/q35-conf-run.mjs:11-40` e da
`src/engine/q35conf/q35conf.worker.ts:575-637` — mai chiedendo `--help` al
runner, che non lo conosce e partirebbe coi default (landmine di `HANDOFF.md:62`).

**Precondizione**: server già vivo e verificato con `curl` (200 su
`http://localhost:5199/`), mai con `pgrep`. Un runner GPU alla volta: A poi B,
mai insieme.

### Run A — il discriminante di C0-4 (~2 min)

    node scripts/q35-conf-run.mjs --model 35b --golden-kind smoke \
      --optimistic --arena-gib 12 \
      --out results/engine/q35-optimistic-35b-cleantoken-2026-08-15.json

Riproduce la forma di `q35-vramplan-35b-it35.json` sull'albero di oggi. **39
token** (34 prompt + 5 generati, golden `smoke` — è da lì che vengono i 39 di
it.35, verificato: il `full` ne ha 6.461). Arena 12 GiB **esplicita**: il report
scrive `cfg.arenaGiB ?? 12`, quindi senza il flag dichiarerebbe 12 anche se il
motore ne usasse un altro — sarebbe una dichiarazione non misurata.

Protocollo del runner, già di grado riferimento: cold sync che scalda, poi
REPS=4 coppie interleavate sync/optimistic sulla stessa cache, **prima coppia
scartata**, mediana e dispersione riportate (`q35conf.worker.ts:592-605`).

**Cosa leggo**: `passes[optimistic-warm].cpu.tokenMs`, con `misses: 0` e
`dirtyTokens: 0` a confermare che è un token pulito.
- **~43 ms** ⇒ C0-4 è confermata, il contratto si riapre sulla quarta leva
  (`readbackWait` era il 94%) PRIMA della riga 2.
- **~21 ms** ⇒ le righe 2 e 3 bastano, si procede come chartato.

### Run B — validare i contatori nuovi su GPU vera (~2 min)

    node scripts/q35-conf-run.mjs --model 35b --golden-kind smoke \
      --optimistic --arena-gib 4 \
      --out results/engine/q35-optimistic-35b-arena4-2026-08-15.json

Stessa forma, **arena strozzata a 4 GiB**: con 39 token e un'arena che non
contiene il working set i miss ricompaiono anche su una run corta, che è il modo
economico di far girare il path di repair. Serve perché oggi la strumentazione è
scritta e **non ha mai girato una volta su hardware**: i test sono statici e non
si accorgerebbero di un contatore piazzato nel posto sbagliato.

**Cosa leggo**: `passes[*].repair` — `namedFrac` (≥ 0,95 è il done-when),
`msPerFetch` (oggi 5,98 dai contatori della chat), `accountingMs`,
`fetchRepairCalls` contro `misses` del blocco `moe` (devono coincidere: un
`readExpert` per miss).

### Cosa NON è in questi due comandi, e perché

La **baseline nel regime sporco vero** (89,7% di token sporchi) vuole una
generazione lunga, non 39 token: al `full` sono 6.461 token e il solo pass
sync-cold sarebbe ~2 ore. Non entra in una finestra di 5 minuti e non serve per
C0-4. Va pianificata a parte, con `--max-gen` scelto sui numeri di A e B.

---

## it.3 — riga 2: il conflitto `accum`/slot NON esiste, e il top-K è parametrico

### Il reperto di fattibilità, che vale più del codice di oggi

Il contratto e la consegna §4.4c dichiaravano un ostacolo di design sulla riga
più grossa del goal: «il down del 35B usa `accum: true` col peso letto dalla
`Sel`, mentre il contratto a slot pretende che il down SCRIVA non pesato e che
sia la combine a sommare in ordine k — **sono due contratti diversi sullo stesso
tensore**», con la bit-identità come gate secco. **Letto il codice, non è
vero.** Le due catene calcolano la stessa espressione nello stesso ordine:

    35B decode      encodeExperts (q35gpumodel.ts:2099-2108):
                    for k2 = 0..topK ASCENDENTE, ogni down fa
                    `y[r] = y[r] + sel.w * partial[0]`  (wgsl.ts:2215)
    contratto slot  moeCombineWgsl (wgsl.ts:3397-3399):
                    for k = 0..N_USED ASCENDENTE, `t = t + wBuf[..] * ySlots[..]`

**Stessa somma, stesso ordine k crescente, stessi f32.** L'unica differenza è
*dove* sta l'accumulatore (memoria fra dispatch nel primo, registro nel loop nel
secondo) e il fatto che il secondo passa `partial` per la memoria — ma
`partial[0]` è già f32, quindi store+load è senza perdita.

**Il rischio residuo è UNO e ha già la sua macchineria in casa: la contrazione
FMA.** `y + w*partial` e `t + w*y` possono contrarsi o no in modo diverso nei
due WGSL. È esattamente ciò che i *floor test* di `engine-kquant` riga 2 fanno:
ri-derivare la tolleranza dal testo WGSL generato con e senza contrazione.
**La riga 2 non ha l'ostacolo di design che le era stato attribuito**, e il suo
gate è un floor test invece che una riprogettazione.

*Nota su come l'errore era nato: il §4.4c descriveva correttamente due
MECCANISMI diversi e ne ha inferito due SEMANTICHE diverse. È la terza volta in
questo goal che un'inferenza non verificata regge una decisione — dopo i 21,1 ms
e dopo il 16% del gather. Il correttivo è lo stesso: leggere il codice invece di
leggere la descrizione del codice.*

### Il codice: `nUsed` parametrico

`pairGemvSiluGatherWgsl` e `gemvDownSlotsWgsl` avevano lo stride degli slot
scritto `4u` a mano in tre punti (`wgsl.ts:3296`, `:3351`, `:3375`) mentre
`moeCombineWgsl` era già parametrico. Ora tutti e tre prendono `nUsed`,
default 4.

`tests/engine-moegather-nused.test.ts`, 5 casi. Il primo è la non-regressione
GLM nella forma più forte disponibile senza GPU: **a default il testo WGSL
generato è IDENTICO** (`expect(pairD).toBe(pair4)`), quindi il GLM non cambia
di un byte e non serve una run per dirlo. Il terzo lega i tre kernel fra loro:
un disaccordo di stride **non lancia e non fa fallire la compilazione** — legge
slot sbagliati e produce numeri plausibili, la classe di difetto che i gate a
tolleranza lasciano passare.

**EVIDENZA**: `npx vitest run` **1030 passed | 10 skipped** (78 file, +5 casi) ·
`npx tsc --noEmit` exit 0. Nessuna misura GPU in questa iterazione: non serviva
(il GLM è identico per costruzione, il 35B non è ancora cablato).

### Cosa resta della riga 2

1. **Le versioni K-quant dei due kernel a gather** — il pezzo grosso.
   `pairGemvSiluGatherWgsl` ha il layout q4_0 cablato nel corpo (`gQs4` +
   `gScales`, nibble − 8); i K-quant hanno un buffer solo e superblocchi da 256.
   L'aritmetica da riusare è quella verificata da `engine-kquant` riga 4 —
   unpack, offset, termine `Σx` — **non** la sua geometria multi-riga.
2. **I sotto-range su slab K-quant**: `slotTensorRanges` (`residency.ts:189`)
   esiste già; la vista legacy `slotBindRanges` **lancia** sui K-quant.
3. **Il cablaggio sul 35B** al posto del `for m2` di `q35gpumodel.ts:2763-2774`.
4. **Il floor test FMA** + i casi ktest per formato + la misura su GLM E 35B.

---

## it.4 — riga 2: il gather K-quant esiste, e non è costato aritmetica

### Il pezzo «grosso» non lo era

La consegna §4.2.1 diceva: «le tre versioni K-quant dei due kernel a gather non
esistono … è qui che l'aritmetica verificata dalla riga 4 si riusa». Dava per
scontato che si dovesse **portare l'aritmetica K-quant dentro
`pairGemvSiluGatherWgsl`**, che ha il layout q4_0 cablato nel corpo.

Ho fatto il contrario, ed è molto meno codice: **ho aggiunto il modo `gather`
ai GEMV K-quant già verificati** (`gemvQ4KWgsl`, `gemvQ6KWgsl`). Quei generatori
erano già fattorizzati per questo — un corpo aritmetico unico e i punti di
variazione isolati (`head`, `pre`, `XR`, `YR`, `xRowPre`, `tailAcc`), messi lì
dal modo `batch`. **Il gather è un terzo asse sugli stessi punti**, non un
kernel nuovo.

Il diff generato, plain contro gather (q4_K, K=2048 N=512), è **esattamente**:

    testa      + `@binding(2) gather`, y da binding 2 a 3
    preambolo  + gk / mRow / kslot / xR   (4 righe)
    lettura x    `x[xBase+…]` → `x[xR + xBase+…]`   (2 righe)
    scrittura    `y[r]` → `y[(mRow*nUsed + kslot)*N + r]`

Unpack dei superblocchi, offset, termine `Σx`, riduzione: **non una riga
toccata.** `tests/engine-kquant-gather.test.ts` lo pretende nella forma più
forte disponibile: **normalizza** il sorgente gather riportandolo alla forma
plain e chiede **uguaglianza esatta**. Un'asserzione a campione («contiene
ancora `nibLo`») lascerebbe passare una modifica dentro il corpo; questa no.

**Conseguenza pratica**: i casi ktest del gather validano l'INDIRIZZAMENTO, non
ri-validano l'aritmetica — che è già passata su GPU vera in `engine-kquant`
riga 4. La riga 2 costa un cablaggio, non una riscrittura numerica.

### Le tre combinazioni che non esistono, e perché lanciano

`assertGatherCombo` le nomina con la ragione nel messaggio:
- **gather + arena**: nel gather l'expert lo fissa il DISPATCH (bindings a
  sotto-range, `slotTensorRanges`), l'arena lo risolve da `Sel`. Il gather è
  PLAIN per scelta, come i gemelli q4_0 del GLM;
- **gather + batch**: si contenderebbero `wid.z`;
- **gather + accum**: il contratto a slot vuole la scrittura **non pesata** —
  se il kernel pesasse, la somma dei k passerebbe dal `moeCombine` al kernel e
  l'ordine delle somme f32 non sarebbe più quello del decode, che è il gate di
  bit-identità. Il caso [5] del test verifica che nella riga di scrittura non
  compaia `sel.w`.

### DECISIONE PRESA, non escalata: niente fusione gate+up per ora

Il gemello q4_0 del GLM fonde gate+up+silu in UN kernel; il mio riuso no, perché
i GEMV K-quant sono per una matrice sola. Catena per expert:

    GLM q4_0 (fuso)     pair(gate+up+silu) · down · combine        = 3
    K-quant (oggi)      gate · up · silu · down · combine          = 5
    35B per riga (ora)  (gate·up·silu·down) × 8 expert × 16 righe  = 512/layer

**Cosa farei se nessuno rispondesse: questo.** Il salto che conta è da
**per-riga** a **per-expert** — da 512 dispatch per layer a ~2·|unione|+1 ≈ 203
— ed è ortogonale alla fusione. Fondere gate+up sarebbe un secondo ordine
(5 → 3 per expert su un termine già collassato) e costerebbe un kernel nuovo
con aritmetica nuova, cioè esattamente ciò che questa iterazione ha evitato.
Pareto: prima il collasso, la fusione solo se la misura della riga 4 la chiede.
Registrata qui, nessun ruling necessario.

**EVIDENZA**: `npx vitest run` **1043 passed | 10 skipped** (79 file, +13 casi)
· `npx tsc --noEmit` exit 0. Nessuna GPU: il path in produzione è invariato per
costruzione (caso [2]: senza `gather` il testo è quello di ieri byte per byte).

### Cosa resta della riga 2

1. **I sotto-range su slab K-quant**: `slotTensorRanges` (`residency.ts:189`)
   esiste; la vista legacy `slotBindRanges` **lancia** sui K-quant. Da usare nei
   call site nuovi.
2. **Il cablaggio sul 35B**, al posto del `for m2` di `q35gpumodel.ts:2763-2774`,
   con `planMoeChunk` (già parametrico su `{nExpert, nExpertUsed}`).
3. **Il floor test FMA** (il rischio residuo isolato in it.3) + i casi ktest di
   indirizzamento per formato.
4. **La misura su GLM E 35B** — la regola delle ≥ 2 famiglie.

---

## it.5 — FEASIBILITY: la riga 2 punta la leva giusta al bersaglio sbagliato

**Nessun codice questa iterazione, e la ragione è che scriverlo sarebbe stato
sprecato.** Il controllo di fattibilità della riga (protocollo: *feasibility
before conformity*) ha trovato un difetto di piano.

### L'aritmetica che lo dimostra

Dal token pulito misurato in it.2 (`optimistic-warm`, 0 miss, 1 submit/token):

    readbackWait                            40,753 ms/token
    dispatch expert per token   (8×4 + 1) × 40 layer  = 1.320
      → per dispatch                          30,9 µs
    byte di pesi expert per token                566,2 MB
      → banda implicita                         13,9 GB/s
      → a 576 GB/s quei byte costerebbero        0,98 ms

**Il decode è 41× sopra il pavimento di banda, e non è né compute-bound né
bandwidth-bound: è dispatch/occupancy-bound.** 1.320 GEMV minuscole in un
submit solo, ~31 µs l'una, 13,9 GB/s effettivi su una scheda da 576. Coincide
col reperto già in casa (consegna §3.3: «32-64 workgroup su 128 SM: fame di
parallelismo», banda efficace misurata 17,2 GB/s).

### Il difetto: il gather per RIGHE non tocca il decode

La forma a gather raggruppa **le righe di un chunk** per expert: un dispatch per
expert sulle sole righe che lo selezionano. **Nel decode le righe sono UNA.**
Con M=1 l'unione è 8 expert con una riga ciascuno ⇒ **8 dispatch per (layer,
op), esattamente quanti sono adesso. Guadagno zero.**

E infatti il ~2,6× della consegna §4.4 è calcolato «per layer e per chunk da
M=16»: **è una leva di PREFILL.** La barra di questo goal è sul DECODE. La riga
2 chiama la forma a gather «la leva sul 93,5%» — l'intenzione è giusta, il
meccanismo descritto no.

### La leva che invece lo tocca: collassare i k, non le righe

Nel decode l'asse con 8 elementi non è la riga, è **il top-K**. Oggi
`encodeExperts` (`q35gpumodel.ts:2099-2108`) fa `for k2 = 0..8` e dispaccia
gate/up/silu/down per ciascuno. Collassando i k in **un** dispatch con
`wid.z = k`: **1.320 → 200 dispatch per token**, e l'occupazione per dispatch
×8.

**È piccolo, ed è verificato sul codice:**
1. `selBuf` è già `array<Sel>` in storage, indicizzato da `moeIdx.selIdx`
   (`arenaSlotWgsl`, `wgsl.ts:3090`);
2. le `topK` entry di un (riga, layer) sono **contigue**:
   `selIdx = (row*nMoeLayer + m)*topK + k` (`q35gpumodel.ts:1377`);
3. ⇒ il preambolo diventa `selBuf[moeIdx.selIdx + wid.z]`. **Una riga.**
4. La corsa sull'accumulo (8 down che fanno `y[r] += w·partial` insieme) si
   risolve con **il contratto a slot di it.4**: ogni k scrive il suo slot,
   `moeCombineWgsl` somma in ordine k — e a M=1 con `nUsed=8` è *esattamente*
   la catena di somme del decode di oggi, quindi **bit-identica per
   costruzione**, non "attesa identica".

**Ciò che it.3 e it.4 hanno costruito serve tutto**, e non per fortuna: il
`nUsed` parametrico è il `nUsed=8` della combine, e il contratto a slot è ciò
che rende il collasso possibile senza corse. Cambia il bersaglio del prossimo
passo, non le fondamenta.

**L'ostacolo dichiarato**: il divieto `batch && arena`
(`wgsl.ts:2175`, `:2360`). Il collasso usa `wid.z` in regime d'arena, che è la
combinazione vietata. La consegna §4.2.4 lo aveva previsto: «va **rimosso con
una ragione scritta**, non aggirato: oggi è corretto perché protegge da una
combinazione che nessuno ha misurato». La ragione scritta ora c'è, ed è questa
sezione. *Nota: il divieto resta giusto per `batch` (righe) + arena — è il modo
`kfan` (k su wid.z) a essere legittimo, e va introdotto come modo A SÉ invece
che allentando la guardia esistente.*

### La correzione di piano, presa e registrata (meccanismo: è mia)

La riga 2 si sdoppia. La barra, i done-when e la regola delle ≥ 2 famiglie non
cambiano — cambia quale meccanismo attacca quale metrica:

- **2 (gather per righe)** — il PREFILL a chunk e il GLM. Fatto per due terzi
  (it.3, it.4). **Non muove la barra**, e va detto nel consuntivo invece di
  lasciarlo credere.
- **2c (kfan: collasso dei k)** — **è questa che muove la barra del decode.**
  1.320 → 200 dispatch/token. Vale su ogni famiglia MoE in regime d'arena: il
  GLM fa `gemvAccumFast` k=0..3, stessa struttura ⇒ le ≥ 2 famiglie ci sono.

*Terza inferenza non verificata smontata in questo goal, e stavolta prima di
spenderci: i 21,1 ms, il «16%», il conflitto accum/slot — e ora «il gather è la
leva del decode». Il correttivo è sempre lo stesso: l'aritmetica sui numeri
misurati, prima del codice.*

**EVIDENZA**: nessun test nuovo (nessun codice). `vitest` 1043 passed e `tsc`
exit 0 restano quelli di it.4, albero invariato.

---

## it.6 — riga 2c: `kfan` esiste, tre righe di differenza e zero di aritmetica

Modo `kfan` nei due GEMV K-quant in regime d'arena. **Introdotto come modo A SÉ,
non allentando il divieto `batch && arena`** — che per le righe resta giusto: in
`batch` x e y diventano matrici [M,K] e [M,N] e una `Sel` sola non basterebbe,
mentre il kfan tiene UNA riga e mette i k su `wid.z`, che è l'asse per cui le
entry di `Sel` sono già contigue.

Diff generato, accumulante contro kfan (q4_K, arena, K=2048 N=512) — **per
intero**:

    let sel = selBuf[moeIdx.selIdx];       →  selBuf[moeIdx.selIdx + wid.z]
    if (!ok) { return; }                    →  (rimosso: v. il miss)
    y[r] = y[r] + sel.w * partial[0];       →  y[wid.z*512u + r]
                                               = select(0.0, partial[0], ok)

**Nient'altro.** `tests/engine-kquant-kfan.test.ts` (15 casi) lo pretende con la
normalizzazione di it.4: si riporta il kfan alla forma accumulante e si chiede
uguaglianza **esatta**; una quarta riga di differenza fa cadere il test.

### Il punto di correttezza che il kfan introduce, e che non era ovvio

Il ramo d'arena esce con `if (!ok) { return; }` sul miss, ed **è giusto quando
il contributo si accumula**: un expert mancante somma zero. **Con gli slot no.**
Uscire lascerebbe lo slot col valore del **token precedente**, e `moeCombine` lo
sommerebbe: nessun crash, nessun NaN, nessun accesso fuori range — solo il
modello sbagliato, e solo sui token che hanno avuto un miss. È la classe di
difetto che un gate a tolleranza non vede.

Il kfan quindi **non esce**: scrive `select(0.0, partial[0], ok)`. Stesso
contributo nullo, ma DICHIARATO — la stessa filosofia del degrado definito di
`arenaSlotWgsl`. Il caso [3] del test lo verifica in tre modi (niente uscita
anticipata, la `select` c'è, `ok` resta definito).

### Perché la bit-identità qui è pretesa ESATTA e non a tolleranza

A M=1 con `nUsed = topK`, `moeCombineWgsl` somma `Σ_k w[k]·y[k]` in ordine k
crescente partendo da `sM` — che è **letteralmente** la catena di
`encodeExperts` (`for k2 = 0..topK` ascendente, ogni down accumula il suo
`sel.w * partial`). Stessi valori, stesso ordine, stessi f32. Resta il solo
rischio di contrazione FMA, isolato in it.3, che ha il suo floor test.

### Refactor incontrato sul path, fatto e non recintato

`arenaSlotWgsl` era una costante con `selBuf[moeIdx.selIdx]` cablato. È
diventata una funzione con l'indice come parametro (default invariato), e i
quattro call site sono passati a `arenaSlotWgsl()`. Il testo emesso dai
non-kfan è **identico** — caso [2] del test.

**EVIDENZA**: `npx vitest run` **1058 passed | 10 skipped** (80 file, +15 casi)
· `npx tsc --noEmit` exit 0. Nessuna GPU: il path in produzione è invariato per
costruzione e c'è il test che lo dimostra.

### Cosa resta della riga 2c

1. **Il cablaggio**: `encodeExperts` (`q35gpumodel.ts:2090-2109`) emette oggi
   `for k2` × 4 dispatch; deve diventare 4 dispatch con `z = topK` + la
   combine. Serve il buffer degli slot (`topK × N` f32 per layer) e il bind
   group con `y` = slot invece di `moeAcc`.
2. **`moeCombineWgsl` cablata sul 35B** con `nUsed = 8`, M = 1.
3. **Il floor test FMA** e i casi ktest di indirizzamento.
4. **La misura su GLM E 35B** — la regola delle ≥ 2 famiglie. Il GLM fa
   `gemvAccumFast` k=0..3: stessa struttura, stesso collasso.

---

## it.7 — la combine legge i pesi dalla Sel; il cablaggio è specificato e NON iniziato

### Fatto: `moeCombineWgsl({ weightsFromSel: true })`

Il kfan scrive gli slot; qualcuno deve sommarli. La combine esistente prende i
pesi da un `wBuf: array<f32>` esplicito — che sul 35B **non esiste**: i pesi di
mixing stanno già in VRAM dentro la `Sel` che il motore scrive per ogni layer.
Costruire il `wBuf` costerebbe **una `writeBuffer` per layer nel ciclo caldo, 40
per token**, per ricopiare dati già sul device.

La variante li legge da lì: `selBuf[moeIdx.selIdx + k].w`. Diff completo contro
la versione a `wBuf`: le due struct, due binding (`selBuf` a 3, `moeIdx` a 4) e
il termine del peso. **Nient'altro** — caso [8], per normalizzazione.

**È M = 1 per costruzione, ed è dichiarato**: l'indirizzo `selIdx + k` copre i
topK contigui di UN (riga, layer); per M > 1 servirebbe `nMoeLayer` per saltare
di riga in riga e il kernel non ce l'ha. Chi dispaccia usa `y = 1`. Il prefill a
chunk continua col `wBuf` esplicito — caso [9]: il default è invariato e il GLM
non cambia.

**EVIDENZA**: `npx vitest run` **1062 passed | 10 skipped** (+4 casi) ·
`npx tsc --noEmit` exit 0.

### NON fatto, e la ragione: il cablaggio non entra in questa iterazione

*Pre-limit hand-back del protocollo, con il dubbio concreto invece che
generico.* Ho letto la zona (`q35gpumodel.ts:1441-1490`, `:1852-1857`,
`:2150-2160`, `:2830-2835`) e il cablaggio **non è il `for k2` da solo**:

1. i buffer `gateE`/`upE`/`moeAcc` sono **condivisi da TRE path di encode** —
   il one-pass (`:1852`), `encodeExperts` del decode (`:2154`) e il prefill per
   riga (`:2832`). Il kfan ne vuole versioni per-k (`topK × dE` e `topK × d`);
   allargare quelli condivisi tocca anche i due path che non devono cambiare;
2. servono **tre pipeline nuove per classe di expert** (`mkExpertClass`,
   `:1450-1461`) più i bind group;
3. il `pSilu` va dispacciato su `topK × dE` invece che `dE`;
4. e nulla di tutto ciò è verificabile senza GPU.

**Un cablaggio a metà lascerebbe l'albero rotto per la prossima iterazione**, ed
è il caso in cui il protocollo dice di fermarsi sul confine pulito. L'albero è
verde e committato.

### LA SPECIFICA DEL CABLAGGIO, perché la prossima iterazione esegua e non ri-derivi

**Additivo**: buffer e pipeline NUOVI accanto agli esistenti, i tre path
attuali intoccati finché la misura non promuove il kfan.

    buffer nuovi (per il solo path decode)
      gateK   topK * dE * 4      (8 * 512 * 4  =  16 KB)
      upK     topK * dE * 4      (16 KB)
      ySlots  topK * d  * 4      (8 * 2048 * 4 =  64 KB)
      — trascurabili accanto agli 11,17 GB d'arena

    in mkExpertClass (`:1450`), per ogni classe:
      pGateK = gemvQ4KWgsl({ K: d,  N: dE, arena: kar(L.gate), kfan: {nUsed: topK} })
      pUpK   = gemvQ4KWgsl({ K: d,  N: dE, arena: kar(L.up),   kfan: {nUsed: topK} })
      pDownK = (dk === "q6_K" ? gemvQ6KWgsl : gemvQ4KWgsl)
               ({ K: dE, N: d, arena: kar(L.down), kfan: {nUsed: topK} })
      bgGateK = bg(xn, gateK) · bgUpK = bg(xn, upK) · bgDownK = bg(gateK, ySlots)
      — NB: `bg()` (`:1436`) prende (src, dst) e lega arena+selBuf+moeIdx: la
        firma regge così com'è, cambia solo il buffer di destinazione

    pSiluK    = pipe(siluMulWgsl(topK * dE))     // elementwise: basta la taglia
    bgSiluK   = mkBg(pSiluK, [gateK, upK])
    pCombine  = pipe(moeCombineWgsl({ D: d, nUsed: topK, weightsFromSel: true }))
    bgCombine = mkBg(pCombine, [x, moeAcc, ySlots,
                                selBuf, {buffer: moeIdxUni, offset: 0, size: MOE_IDX_BYTES}])
      — `sM` = `moeAcc`, che a quel punto contiene GIÀ lo shexp (l'axpy di
        `:1184`); `xM` = `x`. La combine fa `x += moeAcc + Σ_k w·y`, che è
        esattamente ciò che oggi fanno il down accumulante + `pAdd`.

    encodeExperts, versione kfan — da 33 dispatch per layer a 5:
      disp(E.pGateK, E.bgGateK, [dE, 1, topK], dyn0)
      disp(E.pUpK,   E.bgUpK,   [dE, 1, topK], dyn0)
      disp(pSiluK,   bgSiluK,   [ceil(topK*dE/64), 1, 1])
      disp(E.pDownK, E.bgDownK, [gg2[0], gg2[1], topK], dyn0)
      disp(pCombine, bgCombine, [ceil(d/64), 1, 1])
      — `dyn0` = l'offset dinamico di **k = 0** del layer: il kernel ci somma
        `wid.z`. Oggi `dyn` è calcolato per ogni k2 (`:2152`); qui serve solo
        quello di k2 = 0.
      — 5 × 40 = **200 dispatch/token** contro i 1.320 di adesso.

**Il gate che questo cablaggio deve passare**: `pAdd` sparisce dal path (la
combine fa anche il `+=`), quindi **`bgAddRes` non va più dispacciato nel ramo
kfan** — dispacciarlo sommerebbe `moeAcc` due volte. È il primo errore da
cercare se il primo run dà numeri doppi.

### Cosa resta della riga 2c

1. Il cablaggio qui sopra.
2. Il floor test FMA (rischio isolato in it.3).
3. I casi ktest di indirizzamento su GPU vera.
4. La misura su GLM E 35B — la regola delle ≥ 2 famiglie.
