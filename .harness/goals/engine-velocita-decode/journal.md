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

---

## it.8 — riga 2c: il kfan è cablato, spento, e il tranello ha il suo test

### Correzione trovata eseguendo: il cablaggio non andava dove diceva la specifica

La specifica di it.7 diceva «sostituire il `for k2` di `encodeExperts`». **È il
posto sbagliato.** `encodeExperts` (`q35gpumodel.ts:2184`) serve il path SYNC e
il prefill-per-riga; **il decode ottimistico — quello con
`submitsPerToken: 1` e i 40,753 ms che la barra misura — ha il suo loop
inline** (`:1891-1900`, dentro il ramo che costruisce l'unico submit del token).
Cablare `encodeExperts` non avrebbe toccato la metrica del goal, e i test
sarebbero passati lo stesso.

Trovata leggendo il codice prima di scriverlo, non dopo. *Quarta inferenza non
verificata di questo goal, e seconda presa in tempo.*

### Cablato

Nel ramo ottimistico, per layer: **5 dispatch invece di 33.**

    pGateK  [dE, 1, topK]   dyn0 = m * topK * MOE_IDX_STRIDE
    pUpK    [dE, 1, topK]   dyn0
    pSiluK  [ceil(topK*dE/64), 1, 1]
    pDownK  [gg2[0], gg2[1], topK]   dyn0
    pCombine [ceil(d/64), 1, 1]      dyn0
      → 200 dispatch/token contro 1.320

`dyn0` è l'offset di **k = 0**: il kernel ci somma `wid.z` per raggiungere la
sua `Sel`. Buffer nuovi e non allargati (`gateK`, `upK`, `ySlots`: 16+16+64 KB),
perché `gateE`/`upE`/`moeAcc` sono condivisi dai path che non vanno toccati.

### PARTE SPENTO, ed è metodo e non prudenza

`setKfan(on)` accende a caldo; il default è `false`. I due bracci vanno
confrontati **nello stesso processo, sulla stessa cache, a parità di
residenza** — è così che sono state decise sync-vs-optimistic (it.35) e il
token pulito (run A). Due run separate confronterebbero anche gli host.

### Il tranello, e il test che lo prende

Nel ramo sequenziale il down accumula in `moeAcc` e poi `pAdd` fa `x += moeAcc`.
Nel kfan la combine fa `x += moeAcc + Σ_k w·y`: **include già il `+=`**.
Dispacciare anche `pAdd` sommerebbe `moeAcc` — che contiene lo shexp — **due
volte**: nessun crash, nessun NaN, output plausibile ma sbagliato, e su un 35B
lo si scoprirebbe leggendo del testo leggermente storto. Un gate a tolleranza
non lo prende.

Casi [12]-[16]: il ramo kfan dispaccia i cinque pipeline attesi, **non contiene
`pAdd`**, il ramo sequenziale è intatto e lo dispaccia ancora, il default è
spento, e i tre buffer sono per-k e non allargamenti degli scratch condivisi.

**EVIDENZA**: `npx vitest run` **1067 passed | 10 skipped** (+5 casi) ·
`npx tsc --noEmit` exit 0 · diff +95/−3 su un file.

### Cosa resta della riga 2c — e da qui in poi serve la GPU

1. **La misura A/B**, che è anche il primo gate di correttezza: stesso processo,
   `setKfan(false)` poi `setKfan(true)`, argmax confrontati token per token.
   **Se il ramo kfan avesse il doppio-add, l'argmax divergerebbe subito.**
2. Casi ktest di indirizzamento su GPU vera.
3. Floor test FMA (rischio isolato in it.3).
4. La misura su GLM — la regola delle ≥ 2 famiglie.

**Il runner non ha un flag `--kfan`**: `q35-conf-run.mjs` passa opzioni via
query string (`optimistic`, `optcold`, `tier`, …) che `q35conf.worker.ts` legge.
Serve aggiungerlo lì, sul modello di `--optimistic`, e far girare la coppia di
passate interleavate come per sync/optimistic. È il primo passo della prossima
iterazione, prima di chiedere la finestra GPU.

---

## it.9 — IL NUMERO SI È MOSSO: 22,58 → 28,90 tok/s, e il gate ha preso due bug

**Prima misura in cui l'obiettivo cambia.** A/B nello stesso processo, stessa
cache, bracci interleavati, `--kfan` su `q35-conf-run.mjs`:

    kfan OFF   44,286 ms/token  =  22,58 tok/s
    kfan ON    34,601 ms/token  =  28,90 tok/s     speedup 1,280x
    argmax     39/39 IDENTICI   ← il gate, e viene prima del tempo
    submits/token 1 in entrambi

**Barra 30 tok/s = 33,33 ms/token: mancano 1,27 ms.** Non ci siamo, ma il
braccio è vivo e corretto.

### I due bug che il gate ha preso, e nessuno dei due sarebbe caduto da solo

**(1) Il mio: il down leggeva l'`h` del k = 0 per tutti e otto gli expert.**
Nel modo kfan avevo cambiato l'indice di `Sel` e la scrittura, ma **non
l'ingresso**: gate e up leggono tutti lo stesso `x` (l'hidden del token), il
down no — il suo ingresso è l'`h` che gate/up/silu hanno prodotto **per
ciascun k**. Senza offset di riga leggeva sempre lo slot 0.

Il modo in cui si presentava: **girava, era più veloce (1,295x) e dava argmax
DIVERSI** — 14/39 uguali, prima divergenza al token 2. *Un gate a sola velocità
avrebbe promosso un motore rotto e nessun test statico se ne sarebbe accorto.*
Fix: `kfan.xPerK`, vero solo per il down. Casi [17] e [18].

**(2) Preesistente, e mio da togliere perché sta sul path: `setInFlight(true)`
senza `finally`.** Fra il submit e il readback il flag I1 rende la slotTable
intoccabile; in mezzo c'è `popErrorScope`, **che lancia**. Senza `finally`
`inFlight` restava `true` per sempre e **ogni passata successiva del processo**
moriva su «ensure con token in volo».

La conseguenza pratica, misurata: il primo run dell'A/B ha riportato
`argmaxCompared: 0` e tre passate a zero token, e **l'errore riportato era
quello del braccio SANO** (`sync-warm`) — che era solo il primo a inciampare nel
flag lasciato sporco dal braccio kfan. La diagnosi puntava al path sbagliato.
Un errore di validazione in un braccio azzerava l'intero run e ne falsificava
l'attribuzione. `try/finally`: chi apre la finestra la chiude, sempre.

**(3) E la causa prima del punto 2**: `pCombine` era creata con `pipe()`, cioè
**layout AUTO**, che non dichiara offset dinamici — e io le passavo `[dyn0]`.
Errore di validazione WebGPU. Ora ha un bind group layout esplicito con
`hasDynamicOffset` su `moeIdx`, come i GEMV expert.

### La correzione al modello mentale: il dispatch costa 8,65 µs, non 31

    risparmio 9,685 ms  su  1.120 dispatch tolti  =  8,65 µs/dispatch

In it.5 avevo scritto **30,9 µs** dividendo TUTTI i 40,753 ms di `readbackWait`
per i 1.320 dispatch expert — ma il token contiene anche attention, norm,
router e coda, che non sono dispatch expert. **Il conto era una divisione, non
una misura**, e sovrastimava di 3,6x.

Conseguenza sul piano: togliere il 100% dei dispatch expert residui varrebbe
altri ~1,7 ms (200 × 8,65 µs). **Non basta a chiudere il divario di 1,27 ms con
margine, e soprattutto non è più lì il grosso**: dopo il kfan restano 34,6 ms
per token di cui il collasso non ha toccato niente. Il prossimo termine va
MISURATO (sonda `--gpu-time` per categoria), non dedotto — è la quarta volta in
questo goal che una divisione si spaccia per misura.

**EVIDENZA**: `npx vitest run` **1069 passed | 10 skipped** (+2 casi) ·
`npx tsc --noEmit` exit 0 · tre run GPU su host quiescente verificato
(`results/engine/q35-kfan-ab-2026-08-15{,b,c}.json` — il primo e il secondo
sono i due fallimenti, tenuti perché sono l'evidenza dei due bug).

### Cosa resta della riga 2c

1. **Il divario di 1,27 ms** dalla barra. Il prossimo termine si misura con
   `--gpu-time` (sonda per categoria: statico / router / expert / coda), che
   perturba e va dichiarata.
2. **La misura su GLM** — la regola delle ≥ 2 famiglie non è ancora soddisfatta.
3. Casi ktest di indirizzamento e floor test FMA.
4. La misura di chiusura a contesto lungo: questi 39 token sono lo smoke.

---

## it.10 — la ripartizione per categoria, PER BRACCIO: il primo termine non è più l'expert

### Prima la strumentazione, perché l'aggregato non bastava

Il primo run `--gpu-time` ha dato una ripartizione **cumulativa su tutto il
processo**, che mescola kfan-off e kfan-on: attribuire lo split sarebbe stata la
quinta divisione spacciata per misura di questo goal. `gpuTimeStats()` è
cumulativo come `perf` e `moeStats`, quindi si snapshotta prima/dopo ogni
passata e si differenzia — 15 righe nel worker, e il braccio si legge invece di
stimarlo.

### La misura (ms per token, sonda accesa — PERTURBA e va dichiarato)

    categoria     kfan OFF   kfan ON    delta
    expert          8,951     5,151     −3,800
    ssmGemv         7,767     6,980     −0,787
    router          3,221     2,883     −0,338
    attn            2,823     2,537
    shexp           2,380     2,156
    ssmOut          2,281     2,046
    tail            1,444     1,276
    routerGemv      0,702     0,660
    ssmRec          0,596     0,553
    norm            0,435     0,400
    resid           0,229     0,214
    static          0,047     0,045
    TOT GPU nei pass 30,878   24,902    −5,976
    ms/token (sonda) 47,455   37,005    −10,450
    gate: argmax 39/39 IDENTICI

**`ssmGemv` è il primo termine del decode del 35B: 6,98 ms contro i 5,15
dell'expert.** Non è MoE — è la proiezione DeltaNet, e ce l'hanno anche il 4B e
il 9B. Chi cerca ancora velocità nel MoE lavora sul secondo termine.

Due letture accanto:
- **La sonda costa**: 47,46 ms/token contro i 44,29 senza. Il rapporto fra i
  bracci regge (1,283 contro 1,280), il valore assoluto no.
- **GPU nei pass 24,9 su ~34,6 ms di parete = 72%**; il 28% resta fuori.
  Il kfan ha tolto 5,98 ms di GPU ma 10,45 di parete: **la metà del guadagno è
  CPU** — 1.120 dispatch in meno sono anche 1.120 encode in meno.
- **`router` 2,88 ms/token per scegliere 8 expert su 256**, 40 pass per token =
  72 µs a layer per una riduzione minuscola. Sproporzionato, e non l'ho toccato.

### Dove siamo rispetto alla barra

    28,90 tok/s misurati (senza sonda)  ·  barra 30 = 33,33 ms/token
    mancano 1,27 ms/token

### DECISIONE PRESA, non escalata: la riga 2c si chiude come «leva atterrata», e il prossimo passo è una riga NUOVA

Cosa farei se nessuno rispondesse: questo. Il kfan ha fatto il suo — 1,28x,
gate verde, e ha spostato il primo termine. Spremere altri 1,27 ms **dal MoE**
significherebbe lavorare sul secondo termine mentre il primo sta lì: è
esattamente l'errore che il PI ha corretto (leve globali, Pareto).

Le due candidate, e sono entrambe **globali per costruzione** — il che soddisfa
la regola delle ≥ 2 famiglie senza sforzarla:
1. **`ssmGemv` (6,98 ms, primo termine)**: la proiezione DeltaNet del decode.
   4B, 9B e 35B ce l'hanno tutti. *NB: `engine-kquant` ha ottimizzato
   `gemm:deltanet-out` nel PREFILL; questo è il GEMV del decode, path diverso.*
2. **`router` (2,88 ms)**: 72 µs a layer per un top-8 su 256. Il rapporto fra
   lavoro e tempo è talmente fuori scala che vale un'occhiata prima di
   progettare qualsiasi cosa.

### Cosa resta APERTO della riga 2c, e non va dimenticato

**La regola delle ≥ 2 famiglie NON è soddisfatta**: il kfan è cablato solo in
`q35gpumodel.ts`. Il GLM fa `gemvAccumFast` k=0..3 — stessa struttura, stesso
collasso possibile — ma non è cablato e non è misurato. Finché non lo è, la
riga 2c ha una leva misurata su UNA famiglia sola, che è precisamente ciò che il
ruling del PI vieta.

**EVIDENZA**: `npx vitest run` **1069 passed | 10 skipped** · `npx tsc --noEmit`
exit 0 · due run GPU su host quiescente
(`q35-kfan-gputime-2026-08-15{,b}.json`).

---

## it.11 — RE-PLAN: la riga 2c è a 2x della stima e sta per diventare un pattern replicato a mano

Nessun codice, e la ragione sta nel protocollo: al superamento della stima il
passo decidibile diventa **la ri-pianificazione, presentata al PI**, non
un'altra iterazione conforme. Riga 2c: **5 iterazioni su 2-3 stimate**, con la
barra non raggiunta e il primo termine che nel frattempo si è spostato FUORI
dalla riga.

### Il reperto di fattibilità sul GLM

Il GLM non condivide i generatori col 35B. Il 35B usa `gemvQ4KWgsl`/`gemvQ6KWgsl`
(gate e up separati); il GLM usa **`pairGemvSiluFastWgsl`** (gate+up+silu FUSI,
`wgsl.ts:3100`) e **`gemvAccumFastWgsl`** (`:3277`). Letti entrambi: sono
fattorizzati come i K-quant — `head`/`pre`/`body`/tail — quindi il kfan ci
entrerebbe con **le stesse tre modifiche** (indice di `Sel`, ingresso per-k,
scrittura nello slot al posto dell'accumulo pesato).

**Ed è esattamente il problema.** Sarebbe la **terza** volta che scrivo lo
stesso modo a mano in una famiglia di generatori diversa:

    gemvQ4KWgsl   kfan  ✅ it.6
    gemvQ6KWgsl   kfan  ✅ it.6
    pairGemvSiluFastWgsl  ❌
    gemvAccumFastWgsl     ❌

Quattro siti, due coperti: è la forma «pattern applicato al 50%» che in questa
repo ha una regola sua. E il costo non è solo la scrittura: sono quattro copie
della stessa invariante (lo stride degli slot, il miss che scrive zero, la
scrittura non pesata) che nessun compilatore confronta.

### E soprattutto: la riga 2c non è più dove sta il tempo

Misurato in it.10, braccio kfan-ON: **`ssmGemv` 6,98 ms/token contro `expert`
5,15.** Il primo termine del decode del 35B non è più il MoE. Cablare il kfan
sul GLM è **completare una leva sul secondo termine**, mentre il primo è
scoperto e — questo è il punto — **è globale per costruzione**: la proiezione
DeltaNet ce l'hanno 4B, 9B e 35B, quindi una leva lì soddisfa la regola delle
≥ 2 famiglie senza doverla cablare due volte.

### Perché NON decido da solo, e il test se lo sia chiesto davvero

Il discriminante del protocollo: *cosa farei se nessuno rispondesse?* Ho
oscillato, e l'oscillazione è il segnale. La regola «≥ 2 famiglie» **l'ho
scritta io** come forma meccanica del ruling del PI; adesso quella regola mi
manda a spendere due iterazioni sul secondo termine di un modello che non è il
collo, mentre il primo termine è globale e scoperto. **Se applico la mia regola
alla lettera vado contro l'intento che voleva codificare** — che era «massimo
risultato, leve globali, Pareto».

Questa non è scelta di meccanismo (mia): è cosa conta come leva globale, cioè
la funzione obiettivo. Va al PI.

### Stato per la ripresa

- Riga 2c: **leva atterrata e verde** — 22,58 → 28,90 tok/s, gate argmax 39/39,
  cablata su q35gpumodel, `setKfan` per l'A/B. **Debito: una famiglia sola.**
- Barra 30 tok/s: **mancano 1,27 ms/token.**
- Primo termine misurato: `ssmGemv` 6,98 ms/token (era `expert`).
- Secondo reperto non toccato: `router` 2,88 ms/token per un top-8 su 256,
  72 µs a layer — sproporzionato.

---

## it.13 — riga 2d, verifica di riuso: **il kernel non va scritto, va instradato**

Il done-when della riga pretende il riuso PRIMA della scrittura. La risposta è
netta e viene dagli artefatti, non dalla prosa della consegna (che su un punto
si è già rivelata sbagliata, §4.4c in it.3).

### Cos'è `ssmGemv`

`q35gpumodel.ts:1071`, `mark("ssmGemv")` copre **quattro GEMV per layer
DeltaNet**: `attn_qkv`, `attn_gate`, `ssm_beta`, `ssm_alpha`. Il 35B ha 30 layer
DeltaNet su 40 (`full_attention_interval: 4`) — coerente col contatore della
sonda, 30 pass/token. **120 GEMV per token, 6,98 ms = ~58 µs l'una.** Non è
overhead di dispatch (8,65 µs): è lavoro vero, fatto male.

### La forma veloce ESISTE, è in produzione, ed è misurata — non è wired

Dal banco della fase 0 di `engine-kquant`
(`results/microbench/kquant-fase0-4090-linux-2026-08-14T19-29-20-014Z.json`),
**Q8_0 `[2048, 4096]`, M = 1** — cioè esattamente il regime del decode:

    base-batch-z (cio' che la produzione emette)   0,2278 ms
    splitk-idot                                    0,0698 ms   = 3,26x
    splitk-f32                                     0,0710 ms   = 3,21x

Il kernel è **già portato in produzione** (`wgsl.ts`, sezione «PREFILL GEMM …
SPLIT-K — port dal banco») e **ktest-ato su GPU vera** dalla riga 4 di
`engine-kquant`. Non c'è aritmetica da scrivere.

**Perché il decode non lo usa**: in `gemvB` (`q35gpumodel.ts:900`) tutta la
rotta veloce sta dentro `if (prefillPart !== null)` — **è una via di prefill per
costruzione**. Il decode cade sul ramo dopo, che è la forma lenta e giusta.
Il piano (`prefillgemmplan.ts`) è già la sede unica della rotta, e già dichiara
che `q8_0` non è instradato con la sua ragione.

**Quindi la riga 2d non è un lavoro di kernel: è estendere al decode una rotta
che esiste, con il piano che già la decide.** È il riuso nella forma più pura
disponibile, ed è globale per costruzione — la rotta serve 4B, 9B e 35B insieme
perché è il piano a sceglierla, non il chiamante.

### Il vincolo, e va sulla SHAPE non sulla famiglia

Verificato sull'header dump del 35B (`q35-header-dump-2026-08-10.json`,
`meta`) e sul codice (`q35gpumodel.ts:439`, `:615`):

    attn_qkv    N = qkvDim = (2·linKHead + linVHead)·linHeadDim   GRANDE
    attn_gate   N = ssm.inner_size = 4096                          GRANDE
    ssm_beta    N = linVHead                                       PICCOLO
    ssm_alpha   N = linVHead                                       PICCOLO

**Due dei quattro sono degeneri per la forma split-K**, che ha 64 righe per
workgroup: con N piccolo il kernel gira quasi a vuoto. È lo stesso reperto che
la consegna §4.1 aveva registrato per il 4B (48 siti `ssm_alpha`/`ssm_beta` con
N=32) — **e ora si vede che vale anche sul 35B**, cioè non era una peculiarità
di un modello: è la shape di quei due tensori in questa architettura.

**Conseguenza operativa**: il predicato va sulla **shape**, e girare `wired` per
famiglia romperebbe il 4B. `prefillGemmCheck` oggi guarda kind, K e fette ma
**non N** — è la prima trappola dell'HANDOFF, e questa riga la incontra per
davvero invece che in teoria.

**E la buona notizia sui byte**: i due tensori grandi sono ~tutto il peso dei
quattro; i due piccoli sono trascurabili. Coprire i due ammissibili cattura
quasi tutti i 6,98 ms.

### Cosa fa la prossima iterazione

1. `prefillGemmCheck` impara a guardare **N** (il predicato mancante).
2. `q8_0` instradato, con il predicato di shape che esclude i siti a N piccolo —
   e un test che lo verifica **sul 4B**, dove l'esclusione è quella che conta.
3. La rotta esce da `if (prefillPart !== null)`: il decode chiede al piano
   come fa il prefill. **Questo è il pezzo globale**: una sede sola, tre
   famiglie.
4. Misura su 35B, 4B e 9B — ereditata, non riscritta.

**EVIDENZA**: nessun codice, nessun test nuovo, albero invariato e verde
(1069 passed, tsc exit 0 da it.11). Questa iterazione ha prodotto la verifica
di riuso che il done-when chiedeva, e ha evitato di scrivere un kernel che
esisteva già.

---

## it.14 — riga 2d: il predicato su N esiste, ed è nel posto che la repo aveva già dichiarato

### Fatto

`PREFILL_GEMM_ROWS_PER_WG = 64`, esportata **accanto alla griglia che la usa**
(`prefillGemmGrid` lancia `ceil(N / 64)` workgroup): due copie di quel numero
sarebbero due forme che divergono in silenzio. Il predicato consuma la costante
in `kernelVerdict` (`prefillgemmplan.ts`).

### Il primo tentativo era nel posto sbagliato, e sono stati i test a dirlo

Avevo messo il controllo in `prefillGemmCheck`. Due test sono caduti, ed
entrambi avevano ragione:

1. **`gpulimits.test.ts`** interroga `prefillGemmWorkgroupStorageBytes` **a
   N=1**, apposta, per provare che il fabbisogno di memoria di workgroup dipende
   SOLO da M. È una query di dimensionamento, non una rotta: gatearla sulla
   routabilità confonde due domande diverse.
2. **`engine-prefillgemmplan-notwired.test.ts` [w3]** pinna che
   «`prefillGemmCheck` non guarda N». Non difendeva il difetto: lo
   **documentava**, scritto dal goal precedente per rendere visibile il buco.

La distinzione che ne esce, e che vale oltre questo caso:

    prefillGemmCheck  = «questa forma si GENERA?»   kind, K, fette
    kernelVerdict     = «questa forma CONVIENE?»    + N

A N=32 il kernel è **corretto** — guarda `r < N` e produce il valore giusto.
Non è il kernel a dover rifiutare: è il piano a non instradare. E il piano è la
sede unica che la repo aveva già dichiarato per i predicati di ammissibilità
(`wgsl.ts:4396-4402`: «l'unico posto dove un predicato può stare senza diventare
una seconda soglia che diverge in silenzio»). **Il posto giusto era già scritto;
il primo tentativo l'ha ignorato e i test l'hanno riportato lì.**

### Il guard irraggiungibile, e il test che lo raggiunge

**Oggi il predicato su N non si attiva mai per il q8_0**: il controllo del
cablaggio viene prima, e il q8_0 non è cablato. Un guard che nessun test
attraversa è un guard di cui non sai se funziona — stessa classe del contatore
mai incrementato (l'avvertenza di `personal-site-47` in it.9).

`tests/engine-prefillgemm-nmin.test.ts`, 6 casi, lo esercita su **ogni kind
cablato**, dove il controllo lo raggiunge davvero: il confine esatto (63 no,
64 sì), le shape vere di 4B e 35B, il sensore che prova che non passa a vuoto,
e l'ordine dei rifiuti (prima il cablaggio, poi la shape — la stessa gerarchia
che il kernel tiene fra formato e geometria).

Aggiornato anche il commento di [w3]: resta vero che il contorno del kernel non
guarda N, ma ora i 48 siti sono esclusi **due volte** — dal flag di famiglia e
dalla shape. **È la seconda che li terrà fuori quando il q8_0 verrà cablato.**

**EVIDENZA**: `npx vitest run` **1075 passed | 10 skipped** (+6 casi) ·
`npx tsc --noEmit` exit 0.

### Cosa resta della riga 2d

1. **Girare `wired` per il q8_0** — ora è sicuro: la shape protegge il 4B.
2. **Far uscire la rotta dal ramo prefill** in `gemvB`, così che il decode
   chieda al piano come fa il prefill. È il pezzo globale: una sede, tre
   famiglie.
3. Misura su 35B, 4B e 9B — ereditata, non riscritta.

---

## it.15 — il flag `wired` del q8_0 è difeso da NOVE test, e questo cambia la stima

**Provato e tornato indietro, con l'albero verde** (1075 passed, tsc exit 0).

Girare `wired: true` per il q8_0 è **una riga**. Fa cadere **nove test in
quattro file**, tutti scritti dal goal precedente per pinnare che quel formato
NON è instradato:

    tests/engine-prefillgemm.test.ts
      [a]-port  «i kind con un kernel sono sei; quelli INSTRADATI restano
                 q4_0, q5_K e q4_1»
      [f]       conteggio: «kind instradati dal piano: expected 4 to be 3»
    tests/engine-prefillgemmplan-notwired.test.ts
      [w1] ×2   `PREFILL_GEMM_WIRED_KINDS` sottoinsieme proprio; e «la ragione
                 del q8_0 NOMINA il fatto che lo esclude: N=32 sul 4B»
      [w2]      «q8_0 K2048xN4096 (35B attn): il KERNEL accetta, il PIANO resta
                 legacy»
      [w3]      «...e il piano li lascia comunque sulla legacy, con la sua
                 ragione»
      [w5]      «a M=16 il rifiuto è del CABLAGGIO, a M=1 è dell'M»
    tests/engine-prefillgemmplan.test.ts
      [1]       «kind fuori dalle vie veloci: legacy, con una ragione ≥ 40
                 caratteri che NOMINA il kind»
    tests/engine-prefillgemm-nmin.test.ts
      [6]       il mio, di it.14: usava il q8_0 come esempio di kind NON
                 cablato — premessa che sparisce

**Non è un contrattempo: è buona ingegneria del goal precedente.** Quel flag è
la cosa che tiene il 4B lontano da una rotta sbagliata, e nove test lo dicono da
angolazioni diverse. Girarlo è un atto deliberato con un costo reale, non una
riga.

### Perché non l'ho fatto adesso

Nessuno dei nove è "sbagliato": ognuno pinna un valore che **cambia di
proposito**, e ognuno ha un'intenzione da preservare mentre il valore si
aggiorna. Alcuni cambiano premessa, non solo numero — `[w1]` pretende che la
ragione del q8_0 nomini ciò che lo esclude, e da domani quella ragione dirà
perché è INCLUSO; `[6]` mio ha bisogno di un altro kind non cablato (q4_K o
q6_K, che restano fuori).

Aggiornare nove caratterizzazioni **in fretta e a contesto profondo** è il modo
tipico di indebolire un test invece di aggiornarlo — e un test che passa senza
più difendere niente è peggio di un test rosso. *Pre-limit hand-back del
protocollo: il dubbio è concreto, il confine è pulito.*

### La stima della riga 2d si allunga, e va detto

Era 2-3 iterazioni, tre consumate, e il cablaggio del q8_0 da solo è
un'iterazione piena (flag + nove test + il ramo esplicito in `gemvB`). Poi resta
il pezzo grosso, che è anche quello che porta il valore: **far uscire la rotta
dal ramo condizionato al prefill**, perché oggi tutto questo serve solo la fase
di lettura del prompt — **il decode non chiede al piano.**

### Il reperto che vale oltre questo caso

`PREFILL_GEMM_SPEC[kind].wired` non è un booleano: è **un'interfaccia con nove
consumatori di test**. Chi lo gira paga la superficie, e la paga tutta insieme
perché i test sono giustamente distribuiti su quattro file per angolazione. È il
prezzo corretto di una decisione che cambia cosa il motore esegue — ma va
messo nella stima, non scoperto girando il flag.

**EVIDENZA**: albero invariato e verde, 1075 passed | 10 skipped, tsc exit 0.
Nessun commit di codice: l'unico prodotto di questa iterazione è il reperto.

---

## it.16 — il q8_0 è cablato: undici test aggiornati, nessuno indebolito

Il flag è girato, il ramo emittente c'è, e i test che lo difendevano sono
**undici** — due più dei nove previsti in it.15. I due in più sono gate
strutturali che leggono il sorgente del motore:

    engine-prefillgemmplan-notwired [w4]  «il wiring non nomina i tre formati
                                           non cablati» — legge q35gpumodel.ts
    engine-prefillwiring-q5k [a]          «gemvB apre DUE rami veloci»

### Come li ho aggiornati, e la differenza che conta

**Nessuno è stato allentato.** Tre sono diventati *più* stringenti:

- **[w3] notwired** — «i 48 siti del 4B restano legacy». Prima pretendeva che
  la ragione nominasse il **kind**; ora pretende che nomini la **shape**
  (`N=32`, `workgroup`) **e che NON nomini il cablaggio**. È il cambio di
  custode reso esplicito: se restassero legacy per la famiglia, vorrebbe dire
  che il cablaggio non è avvenuto. Quella riga è ora l'unica difesa di quei 48
  siti, e il test lo dice.
- **[w1] notwired** — la ragione del q8_0 doveva nominare ciò che lo
  *escludeva*; ora deve nominare i 48 siti che il predicato protegge, il
  predicato stesso, **e la misura che giustifica il cablaggio** (3,26x). Una
  `wiredWhy` che dicesse solo «è veloce» perderebbe l'informazione utile a chi
  un giorno toccasse il predicato su N.
- **[w4] notwired** — era una lista scritta a mano di tre generatori che «non
  devono comparire». Ora **si deriva dal flag**: il generatore di un formato
  cablato DEVE comparire (altrimenti il piano dichiara una via veloce che
  nessuno percorre), quello di un non cablato no. Non marcisce al prossimo
  cablaggio.

Il caso [w4] portava già la sua istruzione: *«se un giorno ci sarà, sarà perché
qualcuno ha cablato — e allora questo test va cambiato con la sua ragione, non
aggirato»*. Scritto dal goal precedente, letto oggi, seguito alla lettera.

### L'errore che ho fatto e corretto in due minuti

Sul file `engine-prefillwiring-q5k` ho fatto un replace globale `.toBe(2)` →
`.toBe(3)`. Ha rotto un conteggio che doveva **restare** 2 (`planPrefillGemm`
chiamato in due siti). **È esattamente il modo di indebolire i test che it.15
temeva**, e l'ha preso il test stesso. Revert e fix chirurgico: il file è
guidato da una lista dichiarata (`site.fmts`), e la modifica giusta era
aggiungere `q8_0` **lì** — dopodiché tre asserzioni si aggiornano da sole,
perché usano quella lista come atteso.

Che il primo tentativo grossolano sia stato preso in due minuti è il motivo per
cui it.15 ha rimandato invece di improvvisare a contesto profondo.

### Il codice

- `wired: true` per il q8_0, con la `wiredWhy` che porta il predicato e le due
  misure (35,2x a M=16, 3,26x a M=1).
- Ramo esplicito `kk === "q8_0"` in `gemvB`, **non un ternario**: il gate
  strutturale legge il sorgente e verifica che il kernel sia emesso con gli
  stessi `opts` con cui si è chiesta la rotta.

**EVIDENZA**: `npx vitest run` **1074 passed | 10 skipped** · `npx tsc --noEmit`
exit 0. Nessuna GPU: questo cablaggio tocca il **prefill**, e il suo effetto va
misurato — ma il decode non passa ancora dal piano, che è il pezzo dopo.

### Cosa resta della riga 2d

1. **Far uscire la rotta dal ramo condizionato al prefill** — è il pezzo che
   porta il valore sulla barra, ed è quello globale: una sede, tre famiglie.
2. La misura: TTFT del 4B (non deve regredire: i 48 siti restano legacy) e il
   decode del 35B.

---

## it.17 — due correzioni al modello, e il re-plan della riga 2d

### (1) `gemvB` NON è il path del decode — e l'errore era nella mia consegna

La consegna di it.16 diceva: «far uscire la rotta dal ramo condizionato al
prefill in `gemvB`, così che anche il decode chieda al piano». **Sbagliato.**
`loadW` restituisce due emettitori distinti (`q35gpumodel.ts:501-523`):

    push  → gemv(...)    il DECODE      (`:862-868`, sei righe)
    pushB → gemvB(...)   il PREFILL     (`:900+`, dove sta la rotta veloce)

`gemvB` **è** il path di prefill: il decode non ci passa affatto. Non c'è nessun
ramo da aprire — c'è una rotta da costruire in un'altra funzione.

*Quinta inferenza non verificata di questo goal, e la prima che ho scritto io
nella consegna di me stesso.* Le altre quattro: i 21,1 ms, il «16%», i «due
contratti diversi», i 30,9 µs per dispatch.

### (2) Il 3,26× è misurato contro un kernel che NON è quello del decode

Il banco confronta `splitk-idot` con `base-batch-z`, cioè
`gemvQuantWgsl({kind:"q8_0", batch:true})`. Il decode emette
`gemvQuantWgsl({kind:"q8_0", K, N, hasBias:false})` — **senza `batch`**, e senza
il `vec4Rows2` che `gemv` accende solo per il q4_0 (`:864`).

A M=1 le due forme fanno lavoro quasi identico, ed è plausibile che il rapporto
regga. **«Plausibile» è esattamente la parola che in questo goal è già costata
quattro volte.** Il guadagno del decode va misurato contro ciò che il decode
emette davvero, non contro il braccio del banco.

### Cosa invece è ESATTO adesso: le shape, derivate dal sorgente

`q35shape.ts:86-89` + `meta` dell'header dump (`inner_size` 4096,
`time_step_rank` 32, `state_size` 128, `group_count` 16), K = dModel = 2048:

    attn_qkv    N = (2·16 + 32)·128 = 8192   AMMESSO    66,32% dei pesi
    attn_gate   N = inner_size      = 4096   AMMESSO    33,16%
    ssm_beta    N = time_step_rank  =   32   ESCLUSO     0,26%
    ssm_alpha   N = time_step_rank  =   32   ESCLUSO     0,26%
                                             ammessi:   99,48%

**Due reperti:**
1. **Anche il 35B ha `ssm_alpha`/`ssm_beta` a N=32**, come il 4B. Il predicato
   di it.14 non protegge una peculiarità di un modello: protegge **quella
   coppia di tensori in questa architettura**, ovunque compaia. È la forma più
   forte che il ruling del PI potesse avere — un predicato costruito una volta
   che serve tutte le famiglie perché guarda la geometria, non il nome.
2. **I due ammessi sono il 99,48% dei pesi dei quattro.** Coprirli cattura
   praticamente tutto `ssmGemv`.

### Il costo vero del passo che resta

Dare al decode la via veloce non è spostare un `if`. `gemv` emette **un**
dispatch; la forma split-K ne vuole **tre** (quantizza x → GEMM a fette →
combine) e due buffer che nel decode non esistono: le parziali (N × fette
float) e la x quantizzata. Più il piano interrogato a M=1.

### RE-PLAN — la riga 2d ha superato la stima, e il protocollo dice di presentarlo

Stima 2-3 iterazioni, **cinque consumate**. Cosa hanno prodotto, tutto in albero
e verde:

    it.13  la verifica di riuso: il kernel esiste, non va scritto
    it.14  il predicato su N in `kernelVerdict`, con i suoi casi
    it.15  il reperto: il flag e' difeso da 11 test
    it.16  q8_0 cablato, 11 test aggiornati e tre resi piu' stringenti
    it.17  le shape esatte, e due correzioni al modello

**Ma il valore atterrato finora è sul PREFILL del 35B**, che questo goal ha
dichiarato fuori scope. La barra è sul decode, e il decode non è stato toccato.

**EVIDENZA**: nessun codice. `vitest` 1074 passed | 10 skipped e `tsc` exit 0
restano quelli di it.16, albero invariato.

---

## it.18 — LA BARRA È SUPERATA: 28,90 → 30,74 tok/s, e la leva era un commento sbagliato

**Il numero, misurato** (`results/engine/q35-router-par-ab-2026-08-16.json`,
stesso protocollo dei riferimenti: sync-cold che scalda, 4 repliche interleavate
sulla stessa cache, prima scartata, mediana e dispersione):

    braccio kfan ON   34,601 ms/token  ->  32,531 ms/token
                      28,90 tok/s      ->  30,74 tok/s        +6,4%
    dispersione       [32,143 - 32,732] = [30,55 - 31,11] tok/s
    braccio kfan OFF  44,286 -> 42,959 ms   (22,58 -> 23,28 tok/s)
    gate argmax       39/39 IDENTICI, routingDiff 0

**Barra 30 tok/s = 33,33 ms/token. Siamo a 32,531, e l'INTERO intervallo di
dispersione sta sopra la barra.** I 1,27 ms che mancavano erano lì.

### La decisione, presa e non escalata

Il docket item 5 chiedeva un ruling su tre uscite. **Era una mia
mis-escalation**: l'intestazione del docket dice *«ordine e meccanismo non
stanno qui: quelli li decido io»*, e le tre uscite erano esattamente un ordine.
La regola dello step 5 del protocollo è meccanica: *se il fallback che
applicheresti senza risposta coincide con la tua raccomandazione, non è
un'escalation*. La mia raccomandazione era la 3 (il router prima della rotta nel
decode). L'ho eseguita e la registro.

### L'ipotesi, e l'evidenza che l'ha nominata prima di toccare il codice

`routerTopKWgsl` (`wgsl.ts:3859`) portava questo commento:

> «La selezione gira su un thread solo: sono nExpert×nUsed confronti e la
> serializzazione E' la specifica — un top-k parallelo con riduzione
> cambierebbe l'ordine dei confronti e quindi il tie-break.»

**È falso, e il suo costo si leggeva già nel profilo di it.10.** Nel braccio
kfan-ON: `router` 2,883 ms su 40 dispatch = **72 µs a layer** per scegliere 8
expert su 256, contro **16,6 µs** del `routerGemv` che quei logit li produce
macinando 256×2048 MAC. Quattro volte più lento per 250 volte meno lavoro: non è
il costo del dispatch, è un thread su 64 che fa 8×256 confronti seriali leggendo
un `array<bool, 256>` in memoria **privata** con indice dinamico — cioè scratch,
non registri.

**Perché il commento è falso, e non è un'opinione.** Lo scan lineare con `>`
stretto calcola il massimo secondo l'**ordine totale su (punteggio, −indice)**:
punteggio più alto, a pari punteggio indice minore. Il massimo su un ordine
totale è associativo e commutativo — una riduzione ad albero che implementa
*quello stesso comparatore* restituisce lo stesso elemento comunque si associi.
Non c'era niente da conservare. L'unica divergenza vera è su NaN, dove `>` non è
un ordine totale, ed è dichiarata nel commento nuovo: con logit NaN il layer è
già perduto a monte.

### Cosa resta seriale, e perché non è pigrizia

Due somme, e sono le uniche righe in cui l'ordine **è** davvero la specifica
(l'addizione f32 non è associativa):

- `sum` = i probs dei selezionati, nell'ordine di selezione (NU addendi);
- `z` = il denominatore della softmax, in ordine di indice (NE addendi).

Tutto il resto del blocco softmax — il **massimo** (riduzione, esatta), gli
`exp`, la divisione — è parallelo e per-elemento. Quindi il kernel nuovo è
**bit-identico nei valori** al vecchio, NaN a parte. Non è una speranza: v. la
prova sotto.

### La prova della bit-identità, che è caduta gratis dal discriminante

Il ktest è stato eseguito **due volte, sullo stesso host, a dieci minuti di
distanza**: una con la modifica, una con `git stash` (per un altro motivo, v.
sotto). I quattro casi router hanno stampato **le stesse cifre**:

    router-top4-gpu-vs-cpu        7,38e-8 / 1,64e-7  minSepRetto 3,43e-5
    q35-router-resolve-gpu-vs-cpu 2,76e-8 / 2,32e-7  minSepRetto 1,92e-5, resolveErrati 0
    router-top4-near-tie          tenuto fino a eps=1e-6, primo flip a 1e-7
    router-resolve-slottable      4,40e-8 / 9,76e-8, top4 CPU {54,7,41,40}
    glm-model-2layer-shadow       router GPU set 6/6, ordine 6/6, wMaxRel 1,27e-7

`near-tie` è il caso che **impone** la separazione fra 4° e 5° e la stringe
finché f32 cede: è precisamente il tie-break, ed è invariato. Su 64+64 estrazioni
casuali, 8×8 near-tie costruiti e i layer VERI del GLM, zero cifre di
differenza.

### La misura causale: il router crolla, il resto non si muove

`results/engine/q35-router-par-gputime-2026-08-16.json`, braccio kfan-ON, sonda
per categoria accesa (perturba, ma perturba entrambi i lati allo stesso modo):

    categoria    prima    dopo     delta
    router       2,883    0,836   -2,047   <- 40 dispatch: 72 -> 20,9 us a layer
    ssmGemv      6,980    6,992   +0,012
    expert       5,151    5,152   +0,001
    attn         2,537    2,539   +0,002
    shexp        2,156    2,157   +0,001
    ssmOut       2,046    2,046   +0,000
    routerGemv   0,660    0,663   +0,003
    TOT GPU     24,902   22,872   -2,030

**Tutte le altre categorie stanno entro ±0,012 ms.** È la firma di una modifica
isolata: il delta del token pulito (−2,070 ms) e il delta del router (−2,047 ms)
coincidono entro il 1%. E i 20,9 µs rimasti sono ormai il pavimento del
dispatch (`routerGemv` ne fa 16,6): **su questo termine non resta quasi niente**.

### La seconda famiglia, misurata e non assunta

Il kernel è **uno solo** e lo montano entrambi i modelli MoE — `glmmodel.ts:1068`
e `q35gpumodel.ts:1758`. È globale per costruzione, non per copertura: non è
stata scritta una seconda volta.

Sul GLM il guadagno è piccolo e va detto col numero: `gpuBusy` 38,9 → 38,6
ms/token. La ragione è aritmetica — il GLM fa top-4 su 64 (256 confronti
seriali) contro il top-8 su 256 del 35B (2.048), otto volte meno da parallelizzare
— e comunque **il 57% del token GLM sta fuori dalla GPU**, quindi 0,3 ms di GPU
non si vedono nel wall. Decode 11,33 (con) contro 11,35 (senza): **nessuna
regressione**, il resto è rumore.

### DUE ROSSI PREESISTENTI, trovati eseguendo i gate e NON causati da qui

Entrambi discriminati con lo stesso metodo — `git stash`, rieseguire, confrontare
— invece che con un'ipotesi.

**(1) `q35-mtp-draft-4b` FAIL nel ktest.** accept-rate GPU 12/37 = 32,4% contro
il 50,0% del riferimento CPU f64. Sull'albero **stashato** stampa
`12/37 = 32,4%`: cifra per cifra lo stesso. È preesistente. L'ultimo verde noto
è `111 PASS / 0 FAIL` (chiusura di `engine-kquant`). Docket item 6.

**(2) Il GLM b12 non riproduce più il suo riferimento**, e neanche questo è mio.
Decode 11,35 tok/s sull'albero stashato contro i **15,330** di
`bench-glm-4090-b12-riga6-2026-08-15.json` (−26%). La causa **non è di calcolo**:
`missesPerToken` 4,78125 e `evictionsPerToken` 4,78125 sono identici cifra per
cifra, `gpuBusy` è 38,9 contro ~38,5 — a muoversi è solo la **lettura**,
`readMsPerToken` **2,47 → 17,5** (7×). Docket item 7.

*Falsa pista mia, registrata perché mi è costata un run da sette minuti*: avevo
attribuito il crollo alla porta (avevo usato 5173 invece dei 5199 del
riferimento, e OPFS è per-origine). **Sbagliato**: rieseguito su 5199 dà 11,33.
Il metodo giusto era quello che ho applicato dopo — stashare e rimisurare, non
ipotizzare.

### EVIDENZA

- `npx tsc --noEmit` exit 0
- `npx vitest run` **1074 passed | 10 skipped** (invariato: nessun test statico
  pinna questo kernel — v. il rilievo in fondo)
- `node .harness/tools/engine-ktest.mjs` **110 PASS / 1 FAIL**, il FAIL
  riprodotto identico sull'albero senza la modifica
- `results/engine/q35-router-par-ab-2026-08-16.json` (barra, host quiescente:
  1 processo browser, 653 MiB VRAM, GPU 0%)
- `results/engine/q35-router-par-gputime-2026-08-16.json` (causalità)
- `results/engine/bench-glm-4090-b12-routerpar-5199-2026-08-16.json` +
  `...-BASELINE-nostash-2026-08-16.json` (non-regressione GLM, A/B sullo stesso host)
- `results/engine/bench-glm-4090-b12-routerpar-2026-08-16.json` (il run su 5173,
  tenuto perché è il reperto che ha smentito la mia ipotesi sulla porta)

### NON VERIFICATO

- La barra della **riga 4** resta da fare: questo è il 35B su contesto 39
  posizioni (golden `smoke`), non il checkpoint su tutte e quattro le famiglie
  con `decodeContext` dichiarato. Il numero è confrontabile col 28,90 perché è
  lo stesso protocollo sullo stesso golden, non perché sia il consuntivo.
- 4B e 9B non hanno router (sono densi): la leva non li tocca, e il done-when
  «misurata su ≥2 famiglie» qui vale come «le due famiglie CHE HANNO un router».

---

## it.19 — il FAIL non era una regressione: era un file con un nome che mentiva

**Il gate di merge è tornato verde: `111 PASS / 0 FAIL`**, e
`q35-mtp-draft-4b` riporta **31/62 = 50,0%** — cifra per cifra il riferimento
CPU f64 di `engine-fase-d` it.51/53. Nessun bisect eseguito.

### Come si è chiusa in dieci minuti invece che in due iterazioni

Il docket item 6 stimava 1-2 iterazioni di bisect fra it.13 e it.17, ~30 min di
GPU. **Prima di spenderli ho letto il test**, e la prima riga utile era questa
(`ktest.worker.ts`, `testQ35MtpDraft4B`):

    const tokens = [...p.promptTokens, ...p.generated].slice(0, W);   // W = 64

Il verdetto stampava `12/37`. `tot = tokens.length − 2`, quindi
**`tokens.length` era 39, non 64**: il golden dava una finestra da 39 token. Non
un modello che sbaglia di più — **un altro campione.**

### La causa: un nome, due scrittori, due lettori

    public/models/q35/golden-full.json
      <- scrive  scripts/q35-conf-run.mjs      (il golden della SUA run)
      <- scrive  scripts/q35-bench-run.mjs     (il golden della SUA run)
      -> legge   q35conf.worker.ts             (ma verifica lo SHA: protetto)
      -> legge   ktest.worker.ts               (nessun controllo: esposto)

`copyFileSync(golden, ".../golden-full.json")` ci copia **qualunque** golden:
`--model 35b --golden-kind smoke` ci lascia 39 token. Il file trovato oggi era
byte per byte `golden-q35-35b-smoke-2026-08-10.json`. Il full del 4B che il
ktest si aspetta ne ha **6.461** per il prompt 0.

**L'ho avvelenato io in it.18**, lanciando l'A/B del 35B con `--golden-kind
smoke`; e it.2 aveva fatto lo stesso il 15. Il `111 PASS` di `engine-kquant` era
riproducibile solo perché l'ultima run di bench, per caso, era stata una 4B
full. **Il gate dipendeva da quale bench fosse girato per ultimo.**

### Il precedente, che rende questa una recidiva e non una sfortuna

`engine-fase-d` it.53 aveva già incontrato **lo stesso sintomo**: *«alla prima
esecuzione ho puntato `golden-smoke.json` [...] e l'accept è uscito 13/38 =
34,2% — un numero giusto su una finestra sbagliata»*. Allora fu risolto
puntando il file giusto. **La collisione è rimasta**, e ha riaperto la stessa
ferita da un'altra direzione (non il file puntato a mano: il file riscritto da
qualcun altro).

### La correzione, alla radice e non sul sintomo

1. **Lo scratch dei bench si chiama `golden-run.json`** — il nome dice che è
   volatile. I due runner ci copiano; `q35conf.worker.ts` legge lì (e continua
   a verificare lo SHA, la difesa che aveva già).
2. **Il ktest ha il suo fixture, `golden-q35-4b-full.json`, che nessun runner
   scrive.** A metterlo lì è `engine-ktest.mjs`, che lo copia dal repo a ogni
   avvio: un `copyFileSync`, e la dipendenza da «quale bench è girato per
   ultimo» sparisce.
3. **Il caso MTP verifica la finestra**: `tokens.length !== W` ⇒ FAIL che dice
   *«FINESTRA SBAGLIATA [...] è il golden sbagliato, NON una regressione del
   modello»*. È la difesa che mancava: un test che pinna un riferimento preso su
   una finestra deve dichiarare di aver ottenuto quella finestra.
4. Il file avvelenato è stato rimosso da `public/models/q35/`.
5. `tests/engine-golden-fixture-isolation.test.ts`, 7 casi, pinna che i due
   ruoli restino file diversi e che il golden nel repo copra davvero i 64 token.

### Il reperto che vale oltre questo caso

**Il difetto ha prodotto un FAIL con un numero plausibile in un gate di merge.**
Non un crash, non un NaN, non un file mancante: 32,4% contro 50%, esattamente la
forma di una regressione del modello. Mi è costato **mezz'ora di GPU per
escluderlo** — due esecuzioni intere del parco kernel con `git stash` — e la
diagnosi vera è arrivata leggendo il codice, non misurando.

*Il metodo che ha funzionato, e quello che ha sprecato tempo*: il `git stash` +
riesecuzione ha stabilito CHE non era mio (utile, e necessario per non
attribuirmi un rosso). Ma la CAUSA l'ha data la lettura di quattro righe di
test. In it.18 avevo la stessa informazione sotto gli occhi — `12/37` accanto a
un riferimento `31/62` — e ho letto «accept-rate diverso» invece di
«denominatore diverso».

### EVIDENZA

- `node .harness/tools/engine-ktest.mjs` → **`PASS 111 · FAIL 0`**, adapter
  nvidia lovelace; `q35-mtp-draft-4b` **31/62 = 50,0%**, 63 draft a 4,80 ms
- `npx tsc --noEmit` exit 0
- `npx vitest run` **1094 passed | 10 skipped** (erano 1087: +7 casi nuovi)
- i quattro casi router di it.18 invariati cifra per cifra anche in questa run

### NON VERIFICATO

- **Docket item 7 resta aperto**: il GLM b12 a 11,35 tok/s contro i 15,330 del
  riferimento, con `readMsPerToken` 2,47 → 17,5. È l'ultimo rosso del gate di
  merge (riga 6) ed è la prossima iterazione.
- Le run di conformance/bench NON sono state rieseguite dopo il rename dello
  scratch: il path cambia in tre file e il test statico lo pinna, ma la prova
  d'esecuzione arriverà con la prossima run che usa quei runner.

---

## it.20 — il riferimento del GLM si smentisce da solo: leggeva dalla RAM

**Il rosso non è una regressione del motore.** Il riferimento del 2026-08-15
dichiarava 15,330 tok/s di decode; oggi lo stesso bench ne dà 11,26–11,59 su
quattro run. La differenza non è codice: **le repliche del riferimento
leggevano dalla page cache del sistema operativo.**

### La prova sta dentro l'artefatto del riferimento

    riferimento 2026-08-15, STESSO file, STESSA run
      warm-up  prefill  19,10 GiB / 6,414 s  =  2,98 GiB/s
      warm-up  decode    1,51 GiB / 0,460 s  =  3,29 GiB/s
      REPLICHE decode   24,21 MiB/token / 2,475 ms  =  9,55 GiB/s   <-- 2,9x

Lo stesso file, a minuti di distanza, tre volte più veloce della passata che
l'aveva appena letto. **Nessun disco fa questo.** Le repliche non hanno letto:
hanno ripreso dalla page cache ciò che il warm-up ci aveva messo.

Oggi, quattro run consecutive, warm-up e repliche coincidono: **1,31–2,34
GiB/s**, cioè la banda del dispositivo. Nessuna delle quattro si è scaldata —
il working set eccede la cache disponibile su questo host adesso.

### I tre fatti che escludono il codice, e sono stati verificati, non assunti

1. **Nessun commit ha toccato il path di lettura** fra `bb3d430` (il commit del
   riferimento) e HEAD: `git diff --stat` non nomina `glmsource.ts`,
   `glmmodel.ts`, `residency.ts`, `expertstore.ts` né `glmbench/`. Il codice che
   esegue `readMs` è **byte-identico**.
2. **`bytesRead` è identico cifra per cifra** sulle due date: 20.512.309.248 nel
   prefill, 1.624.375.296 nel decode. Stessi byte, stesse `misses` (4,78125/
   token), stesse `evictions`.
3. Il braccio con e senza it.18 dà 11,33 contro 11,35: neanche la leva del
   router c'entra.

**Stessi byte, stesso codice, sette volte il tempo.** Restava solo da dove
arrivavano.

### La correzione: il regime di lettura si DICHIARA

Il difetto non è che la page cache aiuti — è che **un gate di merge confrontava
un numero che dipendeva dalla RAM libera del pomeriggio, e nessun campo
dell'artefatto lo diceva**. Un riferimento che non dichiara il suo regime di
lettura non è un riferimento.

- `readBandwidth(bytesRead, readMs)` in `residency.ts`, dove vive l'aritmetica
  della cache. Restituisce `{gibs, regime}` con `regime ∈ disk | os-cache |
  non-misurato` (mai NaN nel JSON, mai un regime inventato su zero byte).
- `OPFS_DEVICE_CEILING_GIBS = 4`, **con la provenienza scritta accanto**: le
  quattro passate di warm-up misurate sui due artefatti (2,31–3,29 GiB/s). Non
  una scheda tecnica — i numeri di questo host.
- `readGiBs`/`readRegime` in ogni fase del report GLM **e nella headline**, che
  è il livello che i gate leggono.
- Il runner lo stampa, e su `os-cache` aggiunge l'avviso in chiaro.
- `tests/engine-read-regime.test.ts`, 5 casi che portano dentro **i numeri veri
  dei due artefatti**: le repliche del riferimento devono uscire `os-cache`,
  quelle di oggi `disk`, i quattro warm-up `disk`. Chi tocca la soglia deve
  farli fallire o aggiornarli con la sua ragione.

### EVIDENZA

- `results/engine/bench-glm-4090-b12-readregime-2026-08-16.json`, run vera:
  headline **`readGiBs 1,336 · readRegime "disk"`**, warm-up prefill 2,34 e
  decode 1,31, entrambi `disk`. Console:
  `[glmbench] regime di lettura: disk (1.34 GiB/s)`.
- `npx tsc --noEmit` exit 0 · `npx vitest run` **1099 passed | 10 skipped**
  (erano 1094: +5)
- decode 11,258 tok/s: la quarta run di oggi, coerente con 11,26/11,33/11,35/11,59

### COSA RESTA AL PI, e perché non l'ho deciso io

Il meccanismo è chiuso; **cosa farne è funzione obiettivo**. Il gate della riga 6
dice «GLM b12 optimistic entro ±5% di 13,172 / 31,26 / 14,74», e quei numeri —
come i 15,330 che li hanno superati — sono stati presi senza sapere in quale
regime. Le opzioni stanno nel docket item 7. **Non tocco un gate che il PI ha
scritto.**

### NON VERIFICATO

- Non ho misurato il GLM a cache **calda per costruzione** (una run che rilegge
  subito ciò che ha appena letto): quattro run consecutive non si sono scaldate
  da sole, ma non ho forzato la condizione. Se il PI vuole un riferimento
  `os-cache` riproducibile, quella misura va progettata.
- Il q35conf non ha ancora la stessa dichiarazione: lì `readMs` è ~0 per
  costruzione (la fetch è asincrona e sta fuori dalla finestra, nota già in
  `residency.ts:251`), quindi il campo direbbe `non-misurato`. Va deciso se
  vale comunque metterlo per uniformità.

---

## it.21 — la rotta split-K vale il decode: 3,89× sulla shape che pesa

**Pre-registrata prima di guardare** (`docs/deep-dive/velocita-decode-2d-prereg-2026-08-16.md`),
con tre previsioni numerate e una regola di decisione fissata in anticipo. Due
corrette, **una sbagliata — ed è quella che ha corretto il modello.**

    shape          braccio        p50 ms    GB/s   % del picco (576)
    K2048 N=4096   base-decode    0,1616     55,1     9,6%
    K2048 N=4096   splitk-idot    0,0489    182,2    31,6%
    K2048 N=8192   base-decode    0,1324    134,7    23,4%
    K2048 N=8192   splitk-idot    0,0340    524,4    91,0%

- **P1** (`base-decode` ≈ `base-batch-z` entro ±15% a M=1): **corretta**, +1,4%
  e +1,1%. Il termine di paragone sbagliato non spostava niente — **il 3,26× di
  it.13 era numericamente giusto**, contro il kernel del decode fa 3,30×. Il
  dubbio di it.17 era corretto, la grandezza no.
- **P2** (2,5–3,5× a N=4096): **corretta**, 3,30×.
- **P3** (a N=8192 il vantaggio è MINORE, 1,5–3,0×): **SBAGLIATA**, è 3,89×.

### Perché P3 era sbagliata, e cosa cambia nel modello

Avevo previsto che a N=8192 lo split-K rendesse meno «perché il GEMV lancia ~N
workgroup e a 8.192 su 128 SM non gli manca parallelismo». I GB/s dicono che
**non era un problema di occupazione, era di banda**: il kernel del decode sta
al **9,6%** del picco a N=4096 e al 23,4% a N=8192 — migliora col numero di
righe ma resta lontanissimo dal pavimento — mentre lo split-K a N=8192 arriva al
**91,0%**, sostanzialmente ottimale.

**Il modello corretto**: un workgroup di 64 thread per riga di uscita non satura
il bus, e aggiungere righe non lo cura; lo cura **spezzare il K**, che dà a ogni
workgroup un accesso contiguo più lungo. Il numero di workgroup non era la
variabile. *È la stessa forma dell'errore del router (it.18): avevo diagnosticato
occupazione dove il problema era altro.*

### La decisione, per regola scritta prima

3,89× ≥ 2,0× sulla shape che porta il 66,32% dei byte ⇒ **la rotta si
costruisce**, e la stima di 2-3 iterazioni è confermata.

    per layer (attn_qkv + attn_gate)  0,2940 -> 0,0829 ms   3,55x
    ssmGemv nel motore                 6,99  ->  1,97 ms    -5,02 ms
    meno 90 dispatch/token aggiunti (quantX 1 per layer — la x è LA STESSA
      per i due tensori — più 2 combine) x 8,65 us          +0,78 ms
    NETTO                                                   ~4,24 ms/token
    token 32,531 -> ~28,3 ms = ~35,4 tok/s

**Riserva dichiarata**: è una proiezione da banco, e questo goal ha già pagato
quattro volte il prezzo di trattare una proiezione come una misura. Vale come
**criterio di spesa**, non come risultato: il risultato lo dirà l'A/B nel
processo, come per il kfan.

### Il difetto tolto sulla strada, e non era nel brief

Il braccio nuovo ha fatto fallire `ttkquant-fase0-varianti` — «esattamente un
braccio legacy, ed è il primo». **Il test aveva ragione**: proteggeva
l'univocità del denominatore, che è esattamente ciò che era mancato al 3,26×.

Non l'ho indebolito: ho reso preciso il modello. `KQuantVariant` ha ora un campo
`regime: "prefill" | "decode"` sui soli bracci di paragone, e la regola diventa
**un denominatore PER REGIME, mai due nello stesso**. Il test è più forte di
prima: pretende anche che ogni `legacy` dichiari il suo regime (`a.legacy ===
(a.regime !== undefined)`), che il decode esista **solo a M=1**, e — caso nuovo
— che `base-decode` sia `gemvQuantWgsl` senza `batch` **byte per byte** e che
NON coincida col paragone del prefill.

E il braccio è scritto **una volta sola** (`pushDecodeArm`) per tutte e cinque le
famiglie, invece che ricopiato nei tre rami: alla seconda copia la domanda è se
va fattorizzata, alla terza non è più una domanda.

*Nota di metodo, a mio carico*: la prima stesura l'ho fatta con uno script di
sostituzione a regex, che ha duplicato un blocco e sparso chiamate in una
funzione sbagliata. `git checkout` del file e rifatto con edit mirati. Su un
file da 2.400 righe la regex non è più veloce: è solo meno verificabile.

### EVIDENZA

- `results/microbench/velocita-decode-2d-4090-linux-2026-08-16T02-56-25-413Z.json`,
  host quiescente, p50 su campioni interleavati, warm-up scartato
- `npx tsc --noEmit` exit 0 · `npx vitest run` **1100 passed | 10 skipped**
  (erano 1099: +1 caso, e uno riscritto più stringente)

### NON VERIFICATO

- L'artefatto è stato prodotto **prima** che il braccio fosse esteso a q5_K e
  q4_1 (allora usciva solo per q4_K, q6_K e q8_0). Le loro celle `base-decode`
  arriveranno alla prossima run — `q5_K K=4096 N=2560` è `ssm_out`, cioè il
  quinto termine del decode (2,05 ms/token): vale guardarla.
- **Il conto che manca al piano**: i due buffer che la rotta vuole e che nel
  decode non esistono (parziali N×fette f32, x quantizzata). Sono VRAM sottratta
  all'arena expert, e su questo modello l'arena è il vincolo. Va nel piano della
  prossima iterazione, non scoperto costruendo.

---

## it.22 — il conto della VRAM: 130 KiB, e la mia preoccupazione era mal riposta

it.21 aveva lasciato al piano un conto esplicito: *«i due buffer che la rotta
vuole sono VRAM sottratta all'arena expert, e su questo modello l'arena è il
vincolo»*. **L'ho fatto, ed è un non-problema.** Lo scrivo senza gonfiarlo: era
giusto verificarlo prima di costruire, ed è giusto dire che l'esito è banale.

### I numeri, dalle stesse espressioni che il prefill usa già

`PREFILL_SPLITS_MEASURED = 4` (`wgsl.ts:4175`), M = 1, K = 2048:

    part   N=8192   4 x 1 x 8192 x 4 =  131.072 B = 128 KiB
    part   N=4096   4 x 1 x 4096 x 4 =   65.536 B =  64 KiB
    xq     K=2048   2048/4 x 4       =    2.048 B =   2 KiB
    xsc    K=2048   2048/32 x 4      =      256 B

    UN part condiviso dai due tensori      130,2 KiB   = 0,00104% dell'arena da 12 GiB
    DUE part, uno per tensore              194,2 KiB   = 0,00154%

**Il fattore che rendeva grosso il conto del prefill è `M_MAX = 16`. Nel decode
M = 1**, e con esso sparisce l'unico termine che contava.

**Consiglio: due `part`, non uno.** Costa 64 KiB in più e toglie una dipendenza
falsa: con un buffer solo, il GEMM di `attn_qkv` e quello di `attn_gate` devono
serializzare sulla scrittura anche se sono indipendenti. A 0,0015% dell'arena,
comprare l'indipendenza è gratis.

### Il buffer del prefill NON si può riusare, e questo sì che andava verificato

`prefillOn = M_MAX > 0` e `M_MAX = min(16, floor(opts.prefillM ?? 0))`
(`q35gpumodel.ts:703-704`): senza `prefillM` i tre buffer sono **null**. Gli A/B
del decode di questo goal (`--kfan --arena-gib 12`) non passano `--prefill-m`,
quindi in quelle run non esistono. **La rotta del decode deve allocare i suoi**,
incondizionatamente — il che, a 130-194 KiB, è la cosa giusta comunque: legare
una leva di decode al fatto che qualcuno abbia acceso il prefill a chunk sarebbe
un accoppiamento gratuito.

### Le tre verifiche che rendono il piano eseguibile

1. **I quattro tensori passano da `gemv`** (`q35gpumodel.ts:1117-1120`:
   `wqkv.push` / `wz.push` / `wb.push` / `wa.push`), ed è l'emettitore del
   DECODE — non `gemvB`. La correzione di it.17 regge sul codice.
2. **Leggono TUTTI lo stesso `xn`.** Quindi la quantizzazione di x si fa **una
   volta per layer** e la usano entrambi gli ammessi: 30 dispatch di quantX per
   token, non 60. È il numero che avevo già usato nella proiezione di it.21.
3. **Il predicato esiste già e va interrogato, non riscritto**: `kernelVerdict`
   (`prefillgemmplan.ts:209`) respinge `N < PREFILL_GEMM_ROWS_PER_WG`, che è
   esattamente ciò che tiene fuori `ssm_alpha`/`ssm_beta` (N=32) e dentro
   `attn_qkv` (8192) e `attn_gate` (4096). Il decode chiede allo stesso piano:
   una verità sola, ed è il lavoro di it.14 che paga qui.

### Una discrepanza che va detta, perché tocca la proiezione

Il banco dà 294 µs per i due tensori grandi di un layer; il motore misura
**233 µs per l'intero segmento `ssmGemv` di un layer**, cioè per tutti e
quattro. Il motore è più veloce del banco sullo stesso lavoro (~26%): condizioni
diverse, sonda diversa, stato di cache diverso.

**Conseguenza sulla proiezione di it.21**: la grandezza trasferibile è il
**rapporto** (3,55× per layer), non i millisecondi assoluti — ed è così che
l'avevo applicata (6,99 × 0,0829/0,294). Il conto regge; l'avrei sbagliato se
avessi sottratto i 294 µs del banco dai 233 del motore.

### IL PIANO DELLA ROTTA, in cinque fette — perché la prossima iterazione scriva codice invece di progettare

1. **I buffer**: due `part` (4 × N × 4 sulle due N ammesse) + `xq` + `xsc`,
   allocati sempre, ~194 KiB.
2. **`quantX` una volta per layer** su `xn`, prima delle quattro `push`.
3. **`gemv` interroga `planPrefillGemm`** e, se ammesso E il flag è acceso,
   emette i tre dispatch (quantX già fatto → GEMM a fette → combine) invece di
   uno. Non ammesso ⇒ resta la via di oggi, che è ciò che protegge i 48 siti del
   4B e i due tensori a N=32.
4. **Flag spento di default + `setSplitk()`** per l'A/B a caldo nello stesso
   processo: è il pattern che ha funzionato col kfan (it.9) e senza il quale
   l'A/B confronta due processi invece di due bracci.
5. **Gate**: argmax 39/39 identici fra flag OFF e ON, stessa cache, bracci
   interleavati — **prima** del tempo, come per il kfan. Poi `ssmGemv` dalla
   sonda `gpuCat` e il `decode.tokS`. Infine 4B e 9B: hanno gli stessi tensori
   DeltaNet, ed è lì che si vede se la leva è globale per costruzione o solo per
   intenzione.

### EVIDENZA

Iterazione di sola lettura e aritmetica: nessuna modifica al codice, albero
invariato (`52b3efa`). I numeri vengono da `wgsl.ts:4175`,
`q35gpumodel.ts:703-704, 754-772, 1103-1120` e `prefillgemmplan.ts:209-246`.
Nessuna GPU spesa: era la domanda giusta da fare a costo zero prima di
spenderne.

---

## it.23 — la rotta è in albero, spenta, e il braccio spento è provato no-op

Fette 1-4 del piano di it.22. **La fetta 5 (l'A/B su GPU col flag acceso) resta
alla prossima**: serve il cablaggio del flag nel runner, e non l'ho improvvisato
a fine iterazione.

### Il problema di progetto che il piano non aveva visto

Il piano diceva «flag spento + `setSplitk()` per l'A/B a caldo, come `setKfan`».
**Non funzionava così**: il kfan si accende a caldo perché i suoi dispatch si
codificano *per token* (`encodeExperts`), mentre il piano dei dispatch statici
si costruisce **una volta sola** al load. Una scelta fatta lì sarebbe immutabile
per la vita del modello, e l'A/B diventerebbe un confronto **fra due processi**
— cioè fra due stati di cache diversi.

**it.20 ha appena mostrato quanto costa quell'errore**: 15,3 contro 11,3 tok/s
sullo stesso identico codice, e la differenza era la page cache. Fare l'A/B su
due processi qui avrebbe prodotto esattamente quella classe di numero.

**La soluzione**: il piano porta **entrambi i bracci** e l'encoder ne salta uno.
`Step` ha ora un campo `arm?: "legacy" | "splitk"`, e un solo predicato —
`stepOn(st)` — decide. Costo: qualche pipeline e bind group costruiti e mai
eseguiti nel braccio spento, pagati una volta al load e mai per token.

**Il filtro è UNA funzione e non tre condizioni ricopiate**, ed è deliberato:
i cicli che sparano `steps` sono cinque, e uno che se ne dimenticasse
eseguirebbe *entrambi* i bracci — il secondo riscriverebbe l'uscita del primo,
nessun errore, numeri plausibili, e un A/B che confronta una cosa con se stessa.

### Cosa è entrato

1. **I buffer** (`decPart`, `decXq`, `decXsc`): le stesse espressioni del
   prefill con **M = 1**, ~131 KiB. **Nessuna condizione sulla famiglia** — a
   decidere è `planPrefillGemm`, cioè shape e formato, non il modello. Una
   guardia `arch === "qwen35moe"` avrebbe reso la leva specifica di un modello,
   che è precisamente ciò che il ruling del PI vieta.
2. **La rotta in `gemv`** — l'emettitore del DECODE, non `gemvB`: quantX →
   GEMM a fette → combine, e in coda il GEMV legacy marcato `"legacy"`. Solo
   `via: "idot"`: la via f32 è il fallback dichiarato del *prefill* e nel decode
   non è mai stata misurata; instradarla sarebbe ereditare un numero da un altro
   regime, l'errore che questo goal ha già fatto quattro volte.
3. **`setSplitk` / `splitkOn` / `splitkAvail`**. `setSplitk(true)` **lancia** se
   il piano non ha instradato nemmeno un tensore: accendere una rotta assente
   darebbe un A/B piatto e la lettura sbagliata («la leva non serve») invece di
   quella giusta («la leva non è cablata qui»).

**Ritirata una decisione di it.22, con la sua ragione**: il piano prevedeva DUE
`part` per togliere la dipendenza falsa fra i due GEMM. Ne è entrato **uno**:
quantizzando `x` dentro `gemv` — come fa il gemello del prefill — i due tensori
serializzano comunque su `decXq`, e il secondo `part` non comprerebbe niente.
La nota nel codice dice quando tornerebbe ad avere senso (se un giorno la
quantizzazione venisse issata a una volta per layer).

### Il test che ha protestato, e aveva ragione

`engine-prefillwiring-q5k` pretende che `planPrefillGemm` sia chiamato in
**esattamente due posti nominati** — «chi ne aggiunge un terzo sta ri-derivando
la rotta da qualche altra parte». Il terzo l'ho aggiunto io, ed è legittimo: è
un *altro regime*.

Non l'ho allentato. Ora pinna **tre** posti e in più che il terzo:
- stia dentro `gemv`, non altrove;
- chieda **`M: 1`** — con M diverso chiederebbe la rotta di un chunk di prefill
  dentro il path del decode;
- emetta la rotta per intero **coi buffer del decode** e **anche il GEMV
  legacy**, che è ciò che tiene l'A/B dentro un processo solo.

*Dettaglio che è costato due minuti e vale ricordarlo*: le prime asserzioni
guardavano le stringhe `"legacy"`/`"splitk"`. Il lettore di sorgente **bianca i
letterali** — giustamente, un test strutturale non deve leggere le stringhe —
quindi ora guardano gli identificatori.

### EVIDENZA

- `npx tsc --noEmit` exit 0 · `npx vitest run` **1101 passed | 10 skipped**
  (+2 casi rispetto a it.22)
- `node .harness/tools/engine-ktest.mjs` **111 PASS / 0 FAIL**
- **Il braccio spento è provato no-op su GPU**: `q35-model-4b-argmax` resta a
  **570 dispatch/token** con `argmax GPU == oracolo 6/6`, identico a prima del
  cambio. Se `stepOn` sbagliasse e girassero entrambi i bracci, quel conteggio
  salirebbe e l'argmax cadrebbe. È la verifica che il filtro funziona.

### NON VERIFICATO — ed è la prossima iterazione

- **Il flag ACCESO non è mai stato eseguito.** Nessun numero, nessun gate: la
  correttezza della rotta nel decode è per ora un'inferenza dal fatto che il
  banco la misura corretta a M=1 e che il prefill la usa. Serve `--splitk` su
  `q35-conf-run.mjs` (sul modello di `--kfan`) e poi, **nell'ordine**: gate
  argmax 39/39 fra OFF e ON, *poi* `ssmGemv` dalla sonda, *poi* `decode.tokS`.
- **`splitkAvail()` sul 35B non è stato letto**: non so ancora quanti tensori il
  piano abbia davvero instradato. Va guardato PRIMA di leggere un A/B, altrimenti
  un risultato piatto è ambiguo.

---

## it.24 — la rotta è veloce e NON è ancora corretta: gate 38/39

**Il gate viene prima del tempo, e non è passato.** Lo scrivo in testa perché il
numero di velocità è il migliore di tutto il goal e non va letto.

    GATE   argmax kfan-ON vs kfan+rotta   38/39   IDENTICI: NO
           prima divergenza: token 21

    tempo (che NON è un risultato finché il gate è rosso)
           optimistic-warm            42,692 ms   23,42 tok/s
           optimistic-warm-kfan       32,546 ms   30,73 tok/s
           optimistic-warm-kfan-rotta 27,358 ms   36,55 tok/s   1,19x

`results/engine/q35-splitk-ab-2026-08-16b.json`.

**La regola del progetto è esplicita e me la applico**: se `argmaxIdentical` non
è `true`, il ms/token del braccio non significa niente. Un motore che genera un
token diverso non è un motore più veloce — è un altro motore.

### Il primo A/B è servito lo stesso, e per il motivo per cui l'avevo protetto

La prima run ha riportato **`available: false`**: il piano non aveva instradato
nemmeno un tensore, e il braccio non è partito. Senza la guardia di it.23 avrei
letto un A/B piatto e concluso «la leva non paga» invece di «la leva non è
cablata». **È esattamente il caso per cui `splitkAvail` esiste**, ed è la prima
volta in questo goal che una difesa costruita in anticipo ha intercettato la
lettura sbagliata prima che la scrivessi.

### Perché il piano rifiutava: una clausola di contratto di un altro goal

`planPrefillGemm` respinge **ogni** kind a M=1 (`PREFILL_M1_LEGACY`). Quella
clausola sta nel done-when della riga 2 di `engine-ttft`, e il suo stesso
commento la dichiara **conservativa e non derivata**:

> al banco a M=1 l'unica cella più lenta della legacy è `q5_K` sulla via f32
> (0,56x), mentre `q5_K/idot` fa 3,47x e `q4_0/idot` 8,64x PIÙ VELOCI [...]
> restringerla alla cella misurata è una decisione del PI, non
> dell'implementatore.

**Non l'ho ristretta.** Ho aggiunto a `planPrefillGemm` un parametro `regime`
che vale `"prefill"` di default — quindi per tutti i chiamanti di prima il
comportamento è identico byte per byte, e i loro test lo pinnano — e il decode
chiede con `regime: "decode"`, portando la misura del suo regime (3,89x e 3,30x
a M=1 contro il kernel che il decode emette davvero, pre-registrata in it.21).

**La ragione strutturale della clausola non si applica al decode**, ed è il
punto che rende la scopatura difendibile e non un cavillo. La clausola dice «a
M=1 non c'è riuso dei pesi da ammortizzare, che è l'intero punto della forma
multi-riga». Vero — ma a M=1 lo split-K **non vince per riuso dei pesi: vince
per banda**. Il GEMV a un workgroup per riga sta al 9,6-23,4% del picco, lo
split-K arriva al 91,0%.

*Registro il confine*: è una decisione al limite fra meccanismo (mio) e
contratto (del PI). L'ho presa perché non tocca il prefill di una virgola — e se
il PI la vuole diversa, il posto dove intervenire è un parametro con un default,
non una clausola riscritta. Quattro casi nuovi pinnano che il prefill a M=1
resti legacy su tutti i kind, che il predicato sulla SHAPE valga in **entrambi**
i regimi (`N=32` resta legacy anche nel decode, che è ciò che protegge i 48 siti
del 4B), e che senza `dot4I8Packed` il decode non prenda la via intera.

### Il cablaggio del quarto braccio

`--splitk` su `q35-conf-run.mjs` implica `--optimistic` e `--kfan`: il braccio è
**kfan+rotta contro kfan**, non contro il decode nudo — altrimenti il rapporto
conterrebbe due leve e non ne isolerebbe nessuna. Interleavato con gli altri
tre, stessa cache, stesso processo.

### LA DOMANDA CHE APRE LA PROSSIMA ITERAZIONE, e non ha ancora risposta

38/39 con divergenza al token 21: **bug o pareggio ravvicinato?**

La rotta spezza K in fette e le ricombina, quindi cambia l'ordine delle somme
del prodotto scalare: la bit-identità **non è pretendibile** e infatti il gate
non la pretende. Ma un flip di argmax ammette due spiegazioni opposte:

1. **pareggio ravvicinato**: i primi due logit di quel token distano meno
   dell'errore di ri-associazione f32, e il flip è fisica;
2. **bug**: un indice sui parziali, una fetta letta due volte, un `xsc`
   sbagliato — e allora i 27,4 ms sono il tempo di un motore rotto.

**Non lo so, e «sembra un pareggio» è precisamente la parola che in questo goal
è già costata quattro volte.** Il discriminante, in ordine di costo:

- **il divario dei primi due logit al token 21** nei due bracci: se è dell'ordine
  di 1e-4 relativo è (1), se è largo è (2). `lastLogits()` esiste già
  sull'interfaccia del modello — serve leggerlo nel punto giusto;
- se resta ambiguo: **conformance top-1 contro l'oracolo** con la rotta accesa
  (il gate della riga 6 pretende ≥ 1012/1024). Un bug d'indicizzazione
  farebbe crollare quel numero; un pareggio ravvicinato no.

### EVIDENZA

- `npx tsc --noEmit` exit 0 · `npx vitest run` **1105 passed | 10 skipped**
  (+4 casi sul regime)
- `results/engine/q35-splitk-ab-2026-08-16.json` (il run con `available: false`,
  tenuto: è il reperto della guardia che ha funzionato)
- `results/engine/q35-splitk-ab-2026-08-16b.json` (l'A/B vero, gate 38/39)
- host quiescente, 4 repliche interleavate, prima scartata

### NON VERIFICATO

- **Tutto ciò che riguarda la velocità della rotta.** 36,55 tok/s è il numero di
  un braccio che non ha passato il gate di correttezza, e finché non lo passa
  non entra in nessun consuntivo.
- 4B e 9B non sono stati toccati: la leva è globale per costruzione (decide la
  shape, non il modello) ma non è stata misurata altrove.

---

## it.25 — non è un pareggio: è la quantizzazione delle attivazioni

**La domanda di it.24 aveva due uscite — pareggio ravvicinato o bug — e la
risposta è: nessuna delle due.**

### La misura

Sonda dei logit accesa (`--logit-probe`, costa zero readback: `step(..., true)`
i logit li legge già). Token 21, l'unico divergente su 39:

    braccio OFF (kfan)          top1 = 248046  v = 17,527882
                                top2 =   2899  v = 17,251318   gapRel 1,58e-2
    braccio ON  (kfan+rotta)    top1 =   2899  v = 17,169683
                                top2 = 248046  v = 17,099520   gapRel 4,09e-3
    swapped = true              deltaTop1Rel = 2,44e-2

### Perché `swapped: true` da solo avrebbe ingannato

I due bracci scelgono l'uno il secondo dell'altro: è il segno che avevo scritto
come «pareggio ravvicinato». **Ma i numeri accanto lo smentiscono.**

    divario da ribaltare (braccio OFF)      0,276564   (1,58%)
    spostamento del candidato 248046        0,428362   (2,44%)

**La perturbazione è PIÙ GRANDE del divario che deve ribaltare.** Non è un
pareggio che oscilla: è uno spostamento sistematico che lo supera. Con un
divario dell'1,58% — largo, non minuscolo — un flip non è fisica.

*Se avessi guardato solo `swapped` avrei chiuso «è un pareggio, si accetta» e
avrei promosso un braccio che sposta i logit del 2,4%. È la ragione per cui il
campo `deltaTop1Rel` sta accanto a `swapped` e non da solo.*

### E non è nemmeno un bug: è una proprietà DICHIARATA della rotta

Una ri-associazione f32 su K=2048 dà errori relativi dell'ordine di 1e-6..1e-4.
**2,4e-2 è due-tre ordini di grandezza sopra.** La spiegazione non è l'ordine
delle somme — è che la via `idot` **quantizza le attivazioni a int8**
(`prefillQuantXQ8Wgsl`): `pesi × q8_0`. Il GEMV legacy moltiplica attivazioni
f32 per pesi int8; la rotta quantizza anche la x. Un errore di ~1/256 per
elemento, accumulato su 30 layer, produce esattamente questo ordine di
grandezza.

**Il banco lo sapeva già e lo dichiarava**: `ttRunner.ts:568` usa una tolleranza
di **2e-2 per i bracci `splitk-idot`** contro 1e-3 per tutti gli altri. Venti
volte più larga, e per questa ragione. Il numero era in casa e non l'avevo
collegato.

### LA VIA D'USCITA È GIÀ MISURATA, e costa il 2%

it.23 aveva escluso `via: "f32"` scrivendo «nel decode non è mai stata
misurata». **Sbagliato: it.21 l'ha misurata**, ed era nella stessa tabella che
ho letto per decidere.

    K2048 N=8192   base-decode  0,1324    splitk-idot 0,0340 (3,89x)
                                          splitk-f32  0,0348 (3,80x)
    K2048 N=4096   base-decode  0,1616    splitk-idot 0,0489 (3,30x)
                                          splitk-f32  0,0497 (3,25x)

**La forma f32 costa il 2,4% in più dell'intera e non quantizza le
attivazioni.** Se l'ipotesi regge, prende quasi tutto il guadagno senza il
difetto — e il gate argmax dovrebbe tornare 39/39.

*Sesta inferenza non verificata di questo goal, e stavolta l'ho scritta io in
un commento di codice ieri.* Il correttivo è sempre lo stesso: il numero c'era.

### LA PROSSIMA ITERAZIONE — l'esperimento che chiude la domanda

Instradare `via: "f32"` nel decode (il ramo esiste già in `gemvB`, va portato in
`gemv`: niente quantX, due dispatch invece di tre) e rifare l'A/B.

- **argmax 39/39** ⇒ ipotesi confermata: la divergenza era la quantizzazione
  delle attivazioni, la rotta è corretta, e si prende la f32.
- **ancora 38/39** ⇒ l'ipotesi cade e torna in gioco il bug d'indicizzazione:
  allora il passo dopo è la conformance top-1 contro l'oracolo.

**In nessuno dei due casi i 36,55 tok/s di it.24 diventano un risultato**: quel
braccio quantizza le attivazioni e il suo gate è rosso.

### La domanda che resta al PI, e che nasce da qui

Se la f32 passa il gate, la rotta intera resta **più veloce del 2,4%** ma cambia
i token. La domanda non è di meccanismo: **il decode può quantizzare le
attivazioni?** Il prefill lo fa già ed è passato per il gate giusto (top-1 vs
oracolo ≥ 1012/1024). Se la risposta è sì, la via intera va misurata con QUEL
gate e non con l'argmax fra bracci. Non la decido io: è la definizione di cosa
il motore promette di calcolare. **La registro qui e la porto nel docket solo se
la f32 fallisce** — se passa, la domanda diventa un'ottimizzazione futura da
2,4% e non una decisione bloccante.

### EVIDENZA

- `results/engine/q35-splitk-logitprobe-2026-08-16.json` — host quiescente,
  4 repliche interleavate. **Il tempo di questa run NON è un riferimento**: la
  sonda scansiona il vocabolario per token e quel costo entra nel `ms`.
- `npx tsc --noEmit` exit 0 · `npx vitest run` **1105 passed | 10 skipped**
- Il flag `--logit-probe` è a parte proprio perché perturba: le run di velocità
  restano pulite.

---

## it.26 — IPOTESI CONFERMATA: gate 39/39 e 39,15 tok/s

**Il gate è verde, e questa volta il tempo si può leggere.**

    GATE   argmax kfan-ON vs kfan+rotta   39/39   IDENTICI: SI
           firstDivergentToken: nessuno · routingDiff 0

    optimistic-warm             42,771 ms   23,38 tok/s
    optimistic-warm-kfan        32,282 ms   30,98 tok/s
    optimistic-warm-kfan-rotta  25,543 ms   39,15 tok/s    1,264x
      dispersione [25,146 - 26,402] = [37,88 - 39,77] tok/s

`results/engine/q35-splitk-f32-ab-2026-08-16.json`. Barra 30, nice-to-have 45.

### L'ipotesi di it.25 era giusta, e la prova è che il flip è sparito

Cambiata **una** cosa: la rotta usa la forma **f32** invece della via intera.
Niente quantizzatore delle attivazioni, due dispatch invece di tre. Il token 21
non diverge più, e nessun altro. **La divergenza era la quantizzazione delle
attivazioni**, non un indice sui parziali: il cablaggio era corretto dal primo
giorno.

### La sorpresa: la f32 è più veloce, non il 2,4% più lenta

Al banco la f32 perdeva il 2,4% sul kernel (0,0348 contro 0,0340 a N=8192). Nel
motore **vince**: 25,543 contro i 27,358 del braccio intero di it.24.

La ragione è che il banco cronometra **il solo GEMM**, mentre il motore paga
anche i dispatch. La via intera ne vuole TRE per tensore (quantX → GEMM →
combine), la f32 ne vuole DUE: **60 dispatch in meno per token** (30 layer × 2
tensori ammessi). *Le due run non sono perfettamente confrontabili — sono due
processi e due stati termici — ma il segno è coerente col conto dei dispatch, e
il numero che conta è comunque quello del braccio che passa il gate.*

**Il reperto riusabile**: un kernel più veloce al banco può essere più lento nel
motore se porta con sé un dispatch in più. Il banco misura il kernel, non la
rotta.

### Cosa NON ho deciso, e sta in una costante con la sua misura accanto

`DEC_SPLITK_IDOT = false` in `q35gpumodel.ts`, con il perché scritto al
call-site: la via intera resta **cablata e spenta**. Non è una riga cancellata —
se il PI decide che il decode può quantizzare le attivazioni, si riaccende
cambiando una parola, e il gate da usare non sarà l'argmax fra bracci ma la
conformance top-1 contro l'oracolo (che è come il prefill ha ottenuto lo stesso
permesso). **Quella domanda non la porto nel docket**, come dichiarato in it.25:
la f32 ha passato il gate, quindi il residuo è un'ottimizzazione futura e non
una decisione bloccante.

### Dove siamo, in una riga

    22,58  ->  28,90  ->  30,74  ->  39,15 tok/s
    base       kfan       router     rotta split-K
                          parallelo  (f32, senza quantizzare le attivazioni)

**1,73× dall'inizio del goal**, barra 30 superata con margine, nice-to-have 45
a portata: mancano 5,85 tok/s, cioè 3,4 ms/token.

### EVIDENZA

- `results/engine/q35-splitk-f32-ab-2026-08-16.json` — host quiescente,
  4 repliche interleavate, prima scartata, **gate argmax 39/39** e
  `routingDiff 0`
- `node .harness/tools/engine-ktest.mjs` **111 PASS / 0 FAIL** dopo il cambio a
  `gemv`; `q35-model-4b-argmax` resta a **570 dispatch/token** con argmax 6/6
- `npx tsc --noEmit` exit 0 · `npx vitest run` **1105 passed | 10 skipped**

### NON VERIFICATO

- **La rotta è SPENTA di default** (`splitkEnabled = false`), come il kfan. Il
  39,15 è il numero di un braccio acceso a caldo, non di ciò che il motore fa
  oggi da solo. Accendere i due flag di default è una decisione della riga di
  chiusura, non di questa.
- **4B e 9B non sono stati misurati con la rotta accesa.** La leva è globale per
  costruzione (decide la shape, non il modello) ma «per costruzione» non è una
  misura, ed è precisamente l'errore che il PI ha già corretto una volta in
  questo goal.
- `ssmGemv` dalla sonda non è stato riletto: so che il token cala di 6,7 ms, non
  ho la conferma per categoria che venga da lì.
