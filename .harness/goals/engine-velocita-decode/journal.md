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
