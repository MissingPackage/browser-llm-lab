# Fase 4 / strato 1 — arena expert a binding fisso, slot come offset aritmetico

Addendum di design alla spec C3a (`2026-08-01-engine-fase-c3a-design.md`
§3.2-bis), prodotto in it.15 (2026-08-03). Meccanismo ratificato dal ruling di
spec (docket item 6); questo documento fissa i layout e lo slicing. Redatto da
agente Plan su lettura dei file veri; rivisto e approvato dall'orchestratore.

## 0. La geometria vera, calcolata (non assunta)

| voce | q4_0 | q4_1 |
|---|---|---|
| slab (`moe.ts` `mkLayout`) | 5 308 416 B | 5 505 024 B |
| parco | 2 688 | 256 |
| slot a budget 12 GiB (riparto `ExpertCache`) | 2 216 | 203 |

**Fatto n.1.** Oggi `slabBufferCap()` usa `maxBindingBytes` = il negoziato =
`q6kHeadBytes` (250 MiB), NON 2 GiB: i buffer di classe attuali (~4,29 GB, 3
q4_0 + 1 q4_1) non sono bindabili per intero. La correzione di it.5
(`residency.ts:13-21`) è giusta per il regime a sotto-range e si INVERTE nel
regime arena (il binding È il buffer): va dichiarato nel codice.

**Fatto n.2.** Finestra d'arena al tetto NVIDIA (2 147 483 644 B):
W(q4_0)=404 slab/binding, W(q4_1)=390 ⇒ 6+1 binding a budget 12 GiB, 7+1 a
parco completo (4c). Margine di progetto: `ARENA_BUFFERS_MAX = 8`.

**Fatto n.3.** `maxStorageBuffersPerShaderStage` negoziato oggi = 7; le
pipeline arena chiedono nBuf+3 ≤ 11 (adapter: 16). Da negoziare col
consumatore dichiarato.

## 1. Architettura

Ogni classe ha N buffer d'arena bindati FISSI in un bind group statico; una
struttura `Sel` in VRAM (entry per `(layer MoE, k)`) porta `{id, slot, w,
flags}`; i kernel GEMV expert leggono `Sel[selIdx]` con `selIdx` da uniform a
dynamic offset (valore CPU-noto: è `(layer,k)`, mai l'expert) e trasformano lo
slot in indirizzo aritmetico. Chi riempie `Sel` è intercambiabile: CPU
(writeBuffer, regime attuale) o router+resolve su GPU (a residenza totale).
I kernel expert NON cambiano fra i due regimi.

Dettagli:
1. **Arena bindings**: in modo arena `ExpertCache` cappa il buffer di classe a
   `min(maxBufferBytes, maxBindingBytes)` → un buffer = un binding. nSlots
   invariato; VRAM identica; buffer 4 → 7.
2. **Indirizzamento**: binding `array<vec4<u32>>` (offset slab tutti multipli
   di 256 ⇒ allineati); scale lette dallo stesso binding con selezione di
   componente `arena[w>>2u][w&3u]` (precedente in produzione:
   `unpack2x16float(gScales[gb>>1u])[gb&1u]`).
   `bufIdx = slot / SLABS_PER_BUF`, `base = (slot % SLABS_PER_BUF) * SLAB_W`
   (costanti compile-time ⇒ mul+shift). Range indici < 2^30, no overflow.
3. **Selezione del binding**: `fn ld4(b,i)` con switch sui nBuf binding —
   branch uniforme sull'intero dispatch (un expert per dispatch): scalare,
   senza divergenza.
4. **`Sel` + slotTable**: `Sel` è l'indirezione unica; la `slotTable`
   (expertKey→slot) serve solo al resolve GPU ⇒ entra nello slice B.

### Alternative scartate (sintesi)
- Dispatch per binding con early-out: +2200 dispatch/token quasi vuoti (resta
  fallback se R1 patologico).
- Duplicazione testuale del corpo per buffer: è la mitigazione di R1, non la
  partenza.
- Doppio binding qs/scale per buffer: 17 > 16 binding, muore sul limite.
- Arena separata per le scale: tocca gli owns della fase 3 (chiusa).
- Buffer 4 GiB con 2 sotto-range: 809 slab non si spezzano in 2 range legali.
- Fusione 4 expert in 1 dispatch gate/up: il down accumulante sarebbe una race
  (ordine somme ⇒ viola identità). Rimandata alla leva 4, dichiarata.
- binding_array/bindless/dynamic offset GPU-driven: non in Dawn / richiedono
  valore CPU (§3.0-bis).

## 2. Layout esatti

### 2.1 `Sel` (storage, VRAM)
```
struct Sel { id: u32, slot: u32, w: f32, flags: u32 }   // 16 B
selBuf: array<Sel, nMoeLayer*nUsed>                      // 184 entry = 2 944 B
selIdx = moeLayerIdx*nUsed + k
```
`slot = 0xFFFFFFFF` = MISS; `w` già ×1.8 (sostituisce i 4 `wExp[k]`);
`flags` bit 0 = miss risolto (skip). CPU-select: 1 writeBuffer da 64 B/layer.

### 2.2 `MoeIdx` (uniform, dynamic offset)
```
struct MoeIdx { selIdx: u32, tableBase: u32, moeLayer: u32, _pad: u32 }
stride 256 B (minUniformBufferOffsetAlignment) × 184 entry = 47 104 B
```
Bind group: 4 STATICI in tutto ({q4_0,q4_1}×{gate/up,down}), costruiti al
load; `slotBgCache` sparisce.

### 2.3 Binding pipeline arena
0..nBuf-1 arena (buffer interi) | nBuf: x | nBuf+1: out | nBuf+2: selBuf |
nBuf+3: MoeIdx uniform con `hasDynamicOffset: true`. Storage/stage ≤ 11.

### 2.4 Costanti per classe nel WGSL
SLAB_W (1 327 104 q4_0 / 1 376 256 q4_1), SLABS_PER_BUF, 6 offset in word da
`SlabLayout` (MAI riscritti a mano — arrivano dal layout di fase 3).

### 2.5 `slotTable` (slice B)
`array<u32, nLayer*nExpert>` = 12 032 B; mantenuta da `ExpertCache.ensure` via
writeBuffer con shadow Uint32Array + flush dell'intervallo sporco una volta
per layer PRIMA delle dispatch; ordine obbligatorio writeBuffer(slab) → poi
writeBuffer(tabella); chiave globale layer*nExpert+e (l'eviction non rispetta
i layer).

## 3. File e funzioni (firme)

- `wgsl.ts`: `pairGemvSiluFastWgsl(opts { K, N, arena?: ArenaOpts })` e
  `gemvAccumFastWgsl(opts { kind, K, N, arena?: ArenaOpts })` — STESSA
  funzione, opzione aggiuntiva; senza `arena` il testo emesso è IDENTICO a
  oggi (byte per byte). Il corpo aritmetico esiste una volta sola nel
  sorgente. `routerTopKWgsl(opts { …, resolve?: { nExpert, nUsed } })` —
  blocco opzionale in coda, senza `resolve` output testualmente identico a
  it.9 (slice B).
  `ArenaOpts { nBuf, slabWords, slabsPerBuf, qsWords, scalesWords,
  gateQsWords, gateScWords, upQsWords, upScWords, nUsed }`.
  Le struct `Sel`/`MoeIdx` sono DUE costanti condivise (`SEL_STRUCT_WGSL`,
  `MOE_IDX_STRUCT_WGSL`) usate sia dai kernel d'arena sia dal resolve: le
  scrivono due generatori, e un layout che divergesse non darebbe errore.
- `residency.ts`: `ExpertCacheOpts += { arena?: boolean; slotTable?: boolean }`;
  in modo arena `slabsPerBuffer = floor(min(maxBufferBytes, maxBindingBytes) /
  layout.bytes)` col commento sull'inversione del cap di it.5;
  `SlotRef += { idx }`; nuova `arenaGeometry(cls)`; `slotTableBuffer()` +
  manutenzione in `ensure` (slice B); `slotBindRanges()` resta invariato.
- `gpulimits.ts`: `MAX_STORAGE_BINDINGS_PER_STAGE` → `MAX_STATIC_STORAGE_BINDINGS`
  (valore invariato); `ARENA_BUFFERS_MAX = 8`;
  `expertArenaBindings = nBuf+3`; `EngineNeedsOpts += { arenaBuffers?,
  arenaWindowBytes? }` ⇒ 2 LimitNeed nuovi coi consumatori
  ("catena expert arena, nBuf binding" / "finestra d'arena: N slab in un
  binding" — è il requisito che alza il binding size da 250 MiB).
- `tests/gpulimits.test.ts`: la scansione testuale resta per i binding
  letterali; caso nuovo che GENERA il WGSL arena a nBuf=8 e conta i @binding
  nella stringa prodotta vs `expertArenaBindings(8)`.
- `glmmodel.ts`: mkPipe con layout esplicito opzionale ("auto" resta default
  per i ~20 kernel non-arena); pipes expert → varianti arena con BGL espliciti
  (`hasDynamicOffset` NON esprimibile con layout:"auto" — ragione tecnica);
  wExp eliminato → selBuf + moeIdxUni; slotBgCache/slotBgs eliminati → 4 bind
  group statici; ciclo k → `setBindGroup(0, expBg[cls][stage], [selIdx*256])`;
  ensure + 1 writeBuffer da 64 B; `GlmModelOpts += { select?: "cpu"|"gpu"|
  "shadow", cache.arena?: boolean }`; dispatchesPerTokenPlanned invariato in
  slice A (+1/layer MoE in B); destroy aggiornata.
- Worker (glmbench/glmconf/glmroute/ktest): solo `negotiateLimits` con
  arenaBuffers/arenaWindowBytes calcolati da una funzione esportata da
  residency.ts (non ricopiata). In slice B `glmroute` passa `select:"shadow"`
  e aggiunge al report `gpuRouterAgreement` (schemaVersion 1 → 2).

### 3-bis. Deroghe dello slice B, ratificate in it.16

Il documento descrive ciò che il codice fa: qui i tre punti in cui l'attuazione
diverge da quanto scritto sopra, con la ragione.

1. **Regione ombra indirizzata da entry di uniform, non da offset di binding.**
   §4 dice "raddoppia il buffer": il buffer È raddoppiato, ma la regione ombra
   NON si binda a offset — `nSel·16 = 2 944 B` non è multiplo di
   `minStorageBufferOffsetAlignment` (256), quindi un `{buffer, offset}` sarebbe
   illegale. `moeIdxUni` porta invece `nMoeLayer` entry in più, una per layer
   MoE, col `selIdx` già spostato nell'ombra: il WGSL del resolve è lo STESSO
   che servirà allo slice C — cambia la entry che il dynamic offset seleziona,
   non il kernel.
2. **`tableBase` è il layer ASSOLUTO × nExpert.** In slice A la uniform portava
   l'indice MoE; la chiave della slotTable è `expertKey`, che usa il layer vero
   (l'eviction non rispetta i layer). Campo inerte fino a qui, corretto in B.
3. **Readback dell'ombra: staging dedicata, mappata insieme all'hidden.** Non
   "viaggia dentro" la mapAsync esistente (buffer diversi non si mappano
   insieme): `selStaging` è propria, e le mapAsync di coda partono in
   `Promise.all` dopo lo stesso submit ⇒ un round-trip host solo, come prima.
   La copia è dell'INTERA Sel (5 888 B): oltre all'ombra si rilegge la regione
   di produzione, esposta come `GlmRouting.vram` accanto a `GlmRouting.gpu`. È
   ciò che i kernel expert hanno letto davvero, e confrontarla con la decisione
   di `routerSelect` chiude R5 dal lato che decide — gate nel ktest
   (`vramChecked` completo, zero difformità) e nel report di glmroute.

### 3-ter. Attuazione dello slice C (it.17)

Come sopra: i punti in cui il codice dice qualcosa di più — o di diverso — da §4.

1. **Residenza totale = preload al load, non pin.** In modo `gpu` la CPU non vede
   la selezione, quindi non può caricare né pinnare *durante* il token: il parco
   del modello entra in VRAM in `createGlmModel` (un `ensure` per expert dei layer
   MoE) e la `slotTable` si pubblica con UN flush. Niente pin, perché con
   `nSlots >= parco` (precondizione verificata *prima* di costruire la cache, così
   un rifiuto non lascia GB appesi) la free list non si esaurisce e il ramo di
   eviction non è raggiungibile: è geometria, non policy — e si asserta
   (`evictions === 0`, più `misses` e occupancy per classe == parco). Dopo il load
   la slotTable NON cambia più: nessun flush per token, in nessun punto.
   **Caveat di scala, dichiarato**: il preload è sincrono e bloccante, ed è
   dimensionato sul ktest (64 expert, ~350 MB). Sul modello vero sono 2 944 read
   OPFS in fila più altrettante `writeBuffer` senza yield: stallo di minuti e
   pressione sulla staging interna della coda (firma R4). La 4c non riusa questo
   ciclo com'è — le serve il percorso file-slab (`expertSlab`, senza pack CPU) con
   preload chunked/asincrono e pubblicazione della slotTable per blocchi.
2. **Nessuna entry di `MoeIdx` in più.** In shadow il router usa le
   `nMoeLayer` entry dedicate (selIdx nell'ombra); in gpu usa la entry
   `(layer, k=0)`, che *esiste già* e porta esattamente `selIdx = m·nUsed` e il
   `tableBase` giusto. Cambia il dynamic offset, non il buffer, non il WGSL.
3. **L'identità con A NON è bit-a-bit, e non poteva esserlo.** La differenza è UNA
   e sola: il peso di mixing in `Sel`, che in modo gpu lo calcola il router in f32
   mentre in cpu lo calcola `routerSelect` in f64. È intrinseca all'interruttore —
   il senso dello slice è che la CPU non calcoli più nulla.
   **Cosa porta davvero il claim** (le misure che il caso cpu non fa già):
   `hidden` `maxRel` 2,50e-7 gpu-vs-cpu (bit-identici 8 309/12 288), id degli
   expert identici **per k e nell'ordine** su tutte le posizioni, pesi `maxRel`
   1,27e-7 — lo stesso numero dell'ombra dello slice B, che è la firma del
   narrowing f32 e non di un'altra causa. Gate derivati dalla misura, non dal
   limite teorico: 1e-6 su hidden (~4×, ed è la soglia R2) e 1e-6 sui pesi (~8×).
   **Cosa NON lo porta**: l'uguaglianza degli argmax gpu-vs-cpu è *implicata* —
   entrambi i run sono già gated a `argmax == argmax(ref f64)` — e va letta come
   leggibilità del report, non come copertura.
   Sui LOGIT nessuna di queste soglie si applica: la testa amplifica di ~1e3
   (GEMV su 2048 termini con cancellazione, pavimento 1e-3 al denominatore) e i due
   run distano 2,56e-4, cioè quanto il run cpu dista già dal riferimento f64
   (1,91e-4). La soglia per disuguaglianza triangolare (2·5e-3) sarebbe a sua volta
   implicata dai gate esistenti: si prende invece la banda della misura, 1e-3
   (~4×), che è un vincolo indipendente.
4. **`selMiss`** (R6) conta le entry MISS della regione di PRODUZIONE in tutti i
   modi con readback di Sel; in modo gpu > 0 ⇒ il forward alza un errore (token
   invalido) che cita il seam evict-post-resolve. L'ombra non entra nel conto: lì
   i MISS sono attesi.

## 4. Slicing

### Slice A — arena e offset, la CPU comanda ancora (it.15)
Kernel arena + ExpertCache arena + BGL espliciti + Sel/uniform + collasso wExp
+ rimozione slotBgCache. Sel riempita dalla CPU dopo gli ensure. Niente
tabella, niente router GPU. GATE:
1. ktest `expert-arena-vs-slotrange`: stesso slab bindato a sotto-range
   (kernel attuale) e dentro un'arena da 3 slab al terzo offset (nBuf=2
   forzato): uscite f32 IDENTICHE BIT-A-BIT (gate/up e down, entrambe le
   classi). Fallback R2 dichiarato sotto.
2. ktest `glm-model-*` invariato ma con maxBindingBytes di test abbassato a
   3 slab per forzare nBuf ≥ 3 (senno' nBuf=1 e lo switch non ha archi).
   Gate invariati + PLANNED invariato + misses > 0.
3. unit node: round-trip idx → (bufIdx, base) vs slotBindRanges per OGNI slot
   delle due classi.
4. bench GLM: routerSyncs/submits/dispatches per token IDENTICI al baseline
   (asserzione esplicita: lo strato 1 non riduce i sync da solo);
   `gpuByCatMs.experts` non peggiore di it.13 iso-clock oltre il 5%.

### Slice B — selezione su GPU, in ombra (it.16)
routerTopK+resolve, slotTable, modo `shadow`: router GPU scrive Sel in regione
ombra mentre la CPU comanda come in A; confronto GPU-vs-CPU al tail (copia
2×2 944 B su staging dedicata, mappata insieme all'hidden — §3-bis.3). Misura
la fedeltà del router GPU sul corpus vero di glmroute (31 274 posizioni) invece
delle 64 estrazioni sintetiche di it.9. GATE: `gpuRouterAgreement` ≥ 99.99%
set-match e zero difformità sulla Sel di produzione riletta;
setMatch decode/prefill verso l'oracolo IDENTICI all'artefatto 07-31 (decide
ancora la CPU); dispatch +1/layer MoE dichiarato, submit e routerSyncs
invariati (asserito nel ktest `glm-model-shadow-invariance`).

### Slice C — l'interruttore (it.17)
`select:"gpu"`: salta copy logits, submit per layer, mapLogits, routerSelect,
ensure; routing[] letto da Sel al tail. Ammesso SOLO con residenza totale
(errore esplicito altrimenti, mai degradazione silenziosa). Verificabile OGGI
nel ktest: mini-modello con `slotsOverride {q4_1: 64}` ⇒ residenza totale per
costruzione, modo gpu contro gli stessi riferimenti f64. GATE: identità con A
+ `telemetry().routerSyncs === 0` + `submits === 1` per token.

## 5. Rischi (probabilità, discriminatore, mitigazione)

- **R1 switch nel loop interno costa** (media): discriminatore = ktest arena a
  nBuf=1 vs nBuf=8 sullo stesso dato (la differenza È il costo dello switch);
  mitigazione = duplicazione testuale del corpo per buffer (meccanica).
- **R2 identità bit-a-bit non regge** (media-bassa, contrazione FMA per
  modulo): se cade, guardare QUANTO: ≤1e-7 = riassociazione benigna ⇒ gate
  declassato a maxRel ≤ 1e-6 + L2rel modello invariata, dichiarato nel
  journal; ordini sopra = bug d'indirizzo.
- **R3 limiti più alti degradano il device** (bassa, spec §3.6.2):
  discriminatore = gpuByCatMs.attn (non tocca l'arena: se peggiora anche lei
  è la negoziazione); mitigazione = finestra ridotta a parcoClasse/8 = 1,78
  GiB.
- **R4 frammentazione/OOM con 7 buffer da 2,14 GB** (bassa-media): fallimento
  a createBuffer/buildMs; mitigazione = budget −1 slab per classe o buffer da
  2·W slab con 2 range.
- **R5 ordine writeBuffer slab/tabella invertito** (bassa, conseguenza alta):
  il ktest mini-modello con misses>0 ed evizioni per token lo vede come L2rel
  che esplode dopo il riempimento; flushSlotTable in un solo punto.
- **R6 select gpu a residenza parziale** (media, errore naturale della 4c):
  contatore telemetria NUOVO `selMiss` letto nel tail; il modo gpu rifiuta di
  partire senza residenza totale; flags rende il degrado definito.
- **R7 hasDynamicOffset/allineamento**: error scope già attivo.
- **R8 router f32 flippa dove f64 no** (bassa, NON è un bug: it.9 tiene a
  1e-6): gpuRouterAgreement dello slice B sul corpus vero decide PRIMA
  dell'interruttore.

## 6. Cosa NON cambia (e dove si dichiara)

routerSyncs=46 e submits=47 per token in A e B (asserzione nel bench +
tabella nel journal); percorso ensure (firma, LRU, pinned, contatori);
formato slab/file slab (fase 3); path Qwen (zero modifiche — unici consumatori
delle funzioni toccate: glmmodel e ktest); categorie di telemetria (routerTopK
entra in `router`); GlmRouting.weights Float64Array; conteggio dispatch: A
invariato, B +46/token, C come B — la fusione gate/up ×4 esplicitamente NON
fatta qui.

**Aggancio strato 2 (LOOKA)**: slotTable + Sel sono l'interfaccia che il
prefetch userà — caricare uno slab e pubblicarne lo slot sono asincroni e non
toccano né bind group né encode. Lo strato 1 non gli lascia nulla da smontare.
