# Studio codebase: JustVugg/colibri (letto da /tmp/study-colibri, clone shallow 2026-07-29)

## Sintesi

Colibrì esegue GLM-5.2 (744B MoE, 75 layer sparsi × 256 expert + testa MTP = 19.456 expert)
su ~25 GB RAM: parte densa int4 residente (~9,9 GB), **expert (~19 MB l'uno) streamati da
disco**. Cuore: un file C (`c/colibri.c`, 7.125 righe) + header. Tier per expert: VRAM
(opzionale) → pin RAM appreso da `.coli_usage` → LRU per-layer → disco. Quattro meccanismi
nascondono il disco: batch-union (ogni expert unico letto 1 volta per batch), PIPE (pool I/O
che sovrappone pread e matmul), PILOT (il router di L+1 sullo stato post-attention di L
predice il 71,6% del top-8 vero → prefetch un layer avanti), pin/repin guidati da heat
persistente. Il placement decide solo la VELOCITÀ, mai l'output (byte-identico).

## Architettura (com'è organizzato)

- `c/colibri.c` (7.125 righe): engine GLM-5.2 completo — config, load, MLA attention, MoE,
  router, cache/tier, PILOT, PIPE/URING, MTP, spec-decode, server, CLI. Commenti bilingui
  IT/EN con misure e numeri di issue: il codice È la documentazione di ricerca.
- Header: `quant.h` (1.319: kernel int8/int4/int3/E8, AVX2/AVX-512/NEON), `st.h` (549:
  safetensors + fd/mmap/O_DIRECT/mirror), `uring.h` (137), `tok.h` (426), `kv_persist.h`
  (121), `telemetry.h` (204: dashboard + persistenza usage), `tier.h` (60: LFRU puro,
  testabile), `sample.h`, `grammar.h`, `compat.h` (Windows shims).
- `olmoe.c` (1.049) e `inkling.c` (1.501): motori Stadio-A separati per validare la
  matematica contro oracoli HF prima di scalare (PILOT/HOT/LRU prototipati lì).
- `iobench.c` (69): microbench del pattern I/O reale (letture random 19 MB parallele).
- Totale `c/*.c+*.h`: 15.638 righe, zero dipendenze runtime (no BLAS/libnuma/liburing:
  syscall raw). GPU (CUDA/Metal) e dashboard web = strati opzionali sopra lo stesso core.

## 1. PILOT — router lookahead

- **Input della predizione**: lo stato residual **post-attention del layer L** (pre-MoE):
  `layer_forward_rows` chiama `pilot_prefetch(m, li+1, x, S)` subito dopo l'add del residuo
  attention, S≤8 (`colibri.c:4284`). `pilot_prefetch` (`colibri.c:4007-4069`) esegue la
  STESSA pipeline del routing vero sul layer futuro — `rmsnorm(post_ln L+1)` → `matmul(router
  L+1)` → `sigmoid+bias` → top-K greedy, K=`PILOT_K` (8 hint-only, 6 real, `colibri.c:6768`).
- **Thread dedicato**: i predetti non residenti entrano in un **ring lock-free 1P/1C da 4096
  slot** (`pilot_q`, `colibri.c:3778`) consumato dal pthread `pilot_worker`
  (`colibri.c:3918-3934`). Motivo misurato (`colibri.c:3774-3777`): con coda disco satura il
  `fadvise(WILLNEED)` inline bloccava ~0,5 ms × 169k chiamate = +92 s/48 token. Ring pieno ⇒
  scarta (un hint perso non è un errore).
- **Due modalità**: default **hint-only** (`expert_prefetch` → `fadvise(WILLNEED)` sui 3
  tensori+scale, `colibri.c:2099-2107`, `st.h:461-466`); `PILOT_REAL=1` = **pread veri** in
  `ecache[L+1]` (`pilot_realload`, `colibri.c:3784-3832`; batch io_uring 3835-3916).
  Invariante (`colibri.c:810-826`): il pilota scrive SOLO layer > `g_cur_moe_layer`; `moe()`
  prende possesso del layer e aspetta i load in volo su di esso (`colibri.c:2811-2819`) ⇒
  mai matmul su slot mezzo-caricati. Slot in load marcati `eid=-(eid+2)` (`colibri.c:3878`).
- **Su miss**: zero costo di correttezza — l'expert vero è demand-loaded dal path normale;
  la predizione sbagliata costa solo banda. Guard LFRU (`PILOT_EVICT_GUARD`,
  `colibri.c:3797-3811`): una speculazione non evicta un residente "warm" (≥2 accessi e
  più caldo, isteresi 25%+4).
- **Dove si misura il 71,6%**: modalità `LOOKA=1`, contatori-only. `la_predict`
  (`colibri.c:3713-3764`) calcola le predizioni (kind 0 skip-attn, 1 PILOT stale, 2
  two-step); `moe()` FASE A le confronta col routing vero (`la_hit/la_tot`,
  `colibri.c:3037-3049`); report finale a `colibri.c:7117-7122`. Documentato in
  `docs/tuning.md:90-99`: 71,6% recall vs 41,3% "stessi expert del token precedente";
  `colibri.c:3708` riporta 75,8% per kind 1 (misura più recente); `PILOT_TWO` +2,3% recall
  (`colibri.c:3717-3750`). Onestà: "su host disk-saturated il PILOT hint-only può essere
  netto-negativo" (`docs/tuning.md:98-99`).
- Esiste anche **COUPLE** (`colibri.c:666-680, 3937-4005`): predizione senza router da
  tabella offline di co-attivazione (L,e)→top expert di L+1/L+2 (lift mediano 1,8×, p99
  40×, struttura che trasferisce tra workload). Stesso ring del pilota, zero matmul.

## 2. Learned pinning / routing heat

- **File `.coli_usage`**: testo, una riga `"<layer> <expert> <count>"` per expert con
  count>0; scritto atomicamente (tmp+rename) da `stats_dump_q` (`telemetry.h:183-191`),
  salvato **a ogni turno** (`usage_save`, es. `colibri.c:6021`). Caricato **additivamente**
  all'avvio (`usage_load`, `telemetry.h:195-201`: `eusage[l][e]+=cnt` — la storia si
  accumula tra sessioni).
- **Tre contatori distinti** (Model, `colibri.c:201-203`): `eusage` (persistente, PIN/STATS),
  `eheat` (calore di sessione, decade con `tier_decay` = right-shift a ogni passata di
  repin, `tier.h:56-58`), `elast`+`eaccess_clock` (recency per LFRU). Tutti incrementati in
  FASE A del routing (`colibri.c:3012-3021`).
- **Pin all'avvio**: `pin_load` (`colibri.c:6263-6428`) ordina per count decrescente, calcola
  quanti expert entrano nel budget (`PIN_GB`, oppure `expert_avail` = RAM − residente − slack
  onesto, `colibri.c:6468-6476`) e carica i top-N in slot `pin[layer]` mai evictati (prefisso
  VRAM se CUDA). `PIN=auto` preferisce la storia viva `.coli_usage` al profilo congelato
  `stats.txt` (`colibri.c:6974-6989`).
- **AUTOPIN (default)**: senza `PIN`, con storia ≥5.000 selezioni pinna con quota
  proporzionale alla FIDUCIA: `conf=hist/200000` (cap 1), budget = 50% spazio expert × conf
  (`colibri.c:7010-7018`). Più uso → più storia → pin più grande e più giusto.
- **REPIN live (opt-in `--repin N`)**: tra i turni `repin_pass_limit` (`colibri.c:5361-5444`)
  usa `tier_pick_lfru` (`tier.h:35-54`: score `heat<<8|recency`, frequenza primaria) per
  scambiare il pin più freddo col non-pinnato più caldo; isteresi 25%+4 anti ping-pong, max
  4-16 swap/passata (~20 MB l'uno), poi `tier_decay`. `.coli_usage` resta intatto.
- **Quanto migliora**: `docs/benchmarks.md` #12 (Ryzen AI 9, 128 GB): "46.7 GB auto-learned
  PIN → expert hit 66%"; da ~0,05-0,1 tok/s cold a 6,84 tok/s con residenza piena (6×5090).
  Nessuna curva sistematica hit-rate-vs-ore: solo datapoint.

## 3. Cache esperti: LRU per-layer + batch-union

- **Struttura**: `ESlot **ecache; int *ecn; int ecap` (`colibri.c:196`) — array di ESlot per
  layer (76 righe: 75 sparsi + MTP), `ecn[l]` slot usati, `ecap` tetto comune. Ogni `ESlot`
  (`colibri.c:162-168`) = 3 QT (gate/up/down) che sono **viste dentro un unico `slab`**
  (una pread coalescente) + `fslab` scale + `used` (clock LRU globale `m->eclock`).
- **Dimensionamento**: `cap_for_ram` (`colibri.c:6481+`) deriva `ecap` dal budget RAM (88%
  di MemAvailable se non dato) sottraendo denso, pin, KV pool, working set `ws[64]` e 2,5 GB
  di riserva page-cache (misurato: strangolarla fa crollare le pread da ~800 a ~180 MB/s,
  `colibri.c:6503-6506`). A runtime `rss_guard` (`colibri.c:5316-5360`) confronta l'RSS
  MISURATO ogni ~16 token: se sfora libera slot LRU in-place (eid=-1, mai compattare: il
  pilota tiene puntatori in `ecache[]`) e abbassa `ecap`.
- **Lookup/eviction**: scansione lineare pin→ecache (`colibri.c:3183-3189`); hit ⇒ bump
  `used`; miss ⇒ load in `ws[q]`. Fine blocco: "promozione LRU (swap buffer)"
  (`colibri.c:3647-3653`) — gli slab di `ws[]` vengono SCAMBIATI con la vittima (`used`
  minimo), zero copie. L'eviction dalle speculazioni ha in più il guard LFRU (§1).
- **Batch-union** (`moe()`, `colibri.c:2780-2784, 3051-3057`): FASE A route tutte le S
  posizioni (router in un matmul batch); FASE B costruisce `uniq[]` con bitmap `seen[E]`;
  FASE C/D processa blocchi di 64 unici — ogni expert caricato una volta e matmul-ato per
  tutte le posizioni che lo usano; shared expert = un matmul a S righe (FASE E). Mentre un
  blocco computa, il successivo riceve WILLNEED (`colibri.c:3265-3276`); `EXPERT_BUDGET`
  (decode-only, miss-aware) taglia i miss meno pesati (`colibri.c:3058-3157`).

## 4. I/O

- **Pattern di lettura**: le 3 matrici di un expert sono adiacenti nel container → **UNA
  `pread` coalescente da ~19 MB** nello slab (`expert_load_impl`, `colibri.c:1636-1662`:
  ordina per offset, verifica contiguità; fallback 3 pread; scale a parte). `pread_full`
  gestisce short-read/EINTR (`colibri.c:1439-1457`). Alternative: `COLI_MMAP=1` (viste mmap
  + WILLNEED + pre-touch, `colibri.c:1525-1567`), `DIRECT=1` (O_DIRECT allineato 4K,
  `colibri.c:1644-1655`, default OFF); `DROP=1` = fadvise(DONTNEED) post-lettura.
- **Pool async (PIPE, default ON)**: 8 pthread I/O (`PIPE_WORKERS`, max 16) con cursore
  lock-free generation-tagged `(gen<<8)|index` anti-ABA (`colibri.c:1905-2055`): il main
  dispatcha i miss del blocco e fa matmul aspettando `ready[q]` solo per l'expert che serve
  ADESSO. `URING=1`: io_uring possiede la concorrenza (batch 64 load, `colibri.c:1719-1903`).
- **Dual-SSD**: `COLI_MODEL_MIRROR=<dir>`. Ogni expert assegnato a UN drive da hash
  deterministico di (layer,eid) vs soglia `g_mir_share`/256 (`expert_route`,
  `colibri.c:1389-1394`); determinismo obbligatorio: WILLNEED e pread demand devono colpire
  lo stesso fd/page-cache (niente doppia cache). Split da `COLI_DISK_WEIGHTS=p,m` o
  **misurato all'avvio** (`mirror_probe_bw`: 8 pread O_DIRECT da 19 MB sparse, OMP,
  `colibri.c:6146-6172`). Mirror validato byte-identico, mai scritto, fallback al primario
  su errore (`mir_pread`, `colibri.c:1408-1422`). README: 9+3 GB/s ⇒ ~+33%.
- **Banda**: nessuna costante hard-coded, si misura (probe; `iobench.c` per l'utente).
  Datapoint (`docs/benchmarks.md`): decode cold ≈ **11 GB/token** (75×8 expert); dev box
  ~1 GB/s ⇒ 0,05-0,1 tok/s cold; GB10 5,58 GB/s O_DIRECT ⇒ 0,50 warm / 3,33 con
  CACHE_ROUTE; Optane 3,27 GB/s ⇒ 0,16 cold.

## 5. Formato pesi: container int4 "gs64"

- Container safetensors con tensori GIÀ quantizzati: per ogni `X.weight` (int4 packed, 2
  valori/byte) esiste `X.weight.qs` = scale f32. `qt_resolve_fmt` (`colibri.c:1027-1049`)
  VALIDA i byte contro layout noti (container non fidato ⇒ refuse, mai overflow): fmt 1=int8,
  2=int4 per-row, 4=int4 grouped, 5=int3-g64, 6=E8/IQ3.
- **gs64** = fmt 4 con group size rilevato da `detect_group_size` (`colibri.c:1005-1018`):
  scale bytes = `O*ng*4`, `ng=ceil(I/gs)`, gs sondato in {16..256}. Layout riga:
  `q4[o*ceil(I/2)]` nibble low/high consecutivi, valore = nibble−8, scala per gruppo di 64.
- **Dequant nel kernel, mai materializzata**: `matmul_i4_grouped` (`quant.h:168-202`) per
  riga o e gruppo g: unpack nibble → sub 8 → FMA con x → parziale × `scl[g]` (AVX2, coda
  scalare). Path S=1: attivazione quantizzata int8 + `dot_i4i8` VNNI/AVX-512 (`quant.h:541+`,
  `g_idot=1` default).
- Modello pubblico raccomandato: **int4-g64 con testa MTP int8** (~372 GB; README:240-250:
  la per-row int4 dava errori, g64 "cured every failing case").

## 6. KV: 576 float/token, persistita

- GLM-5.2 usa MLA: per token si tiene SOLO il latente compresso `kv_lora=512` + `k_rot`
  `qk_rope=64` ⇒ **576 float/token/layer** invece di 32.768; k_nope e value ricostruiti al
  volo con `kv_b` (commento Model `colibri.c:190-192`; dimensioni confermate a
  `colibri.c:4230-4231`). ≈182 KB/token sui 78 layer (`docs/tuning.md`).
- **Persistenza** (`kv_persist.h`): file `<SNAP>/.coli_kv` append-only, magic `COLIKV1`,
  header dimensioni+`nrec`, un record/posizione = `[tok i32][Lc+Rc dei 78 layer][Ic DSA]`
  (`kv_disk_append`, `kv_persist.h:55-84`). Crash-safe: si appendono solo posizioni nuove,
  `nrec` riscritto per ULTIMO (crash a metà ⇒ file coerente col vecchio nrec). Header
  validato contro il modello al load (`kv_disk_load:91-94`); la riga KV MTP non si salva
  (`kv_start=-1`). Chat riaperta senza re-prefill, "validated byte-identical".

## 7. MTP draft head

- GLM-5.2 ha una testa multi-token nativa stile DeepSeek-V3 (layer extra `n_layers` con
  propri expert): `mtp_draft` (`colibri.c:4543-4579`) concatena `[emb(tok); h_norm]` →
  `eh_proj` → `layer_forward` del layer MTP → argmax(lm_head), iterato per G draft.
  `mtp_absorb` (`colibri.c:4583-4604`) assorbe nella KV della testa le coppie VERIFICATE in
  un passaggio batch. Verifica in `spec_decode` col forward batch del main; la speculazione
  pilota anche l'I/O (PILOT predice per tutte le posizioni del draft, `colibri.c:3769-3770`).
- **int4 collassa l'acceptance a 0-4%, int8 ok**: documentato in `README.md:172-176,249-250`
  ("int4 heads collapse to 0-4% acceptance, #8"), `docs/windows.md:27,108`; misura in issue
  #8 (2,2-2,8 tok/forward con testa int8, `docs/benchmarks.md:20`). Nel codice l'int8 della
  testa è assunto strutturalmente (slot 2×, `colibri.c:161,257`); `c/tools/repair_mtp_int8.py`
  ripara container int4. Analisi della CAUSA: non trovata nel repo (solo il fatto misurato).
- Acceptance sui bench CPU: 52-57% (`docs/benchmarks.md:92,99`). Componibili: draft n-gram
  (`ngram_draft`, `colibri.c:4524`) e grammar-forced draft (GBNF).

## 8. Anatomia e cosa il narrow focus ha evitato di scrivere

- Core: `colibri.c` 7.125 righe; con tutti gli header C: 15.638 (inclusi i motori di
  validazione olmoe/inkling ≈2.550). Un solo modello target (GLM-5.2).
- **Non scritto grazie al narrow focus**: niente framework multi-architettura (nuova
  architettura = nuovo piccolo motore), niente BLAS/backend generico, niente graph
  IR/scheduler, niente training, niente quantizzatore runtime nel path caldo (container
  pre-quantizzato, conversione offline in Python), niente allocatore generico (slab riusati
  + arena per-layer), niente libnuma/liburing (syscall dirette). Kernel SOLO per i formati
  del container (int4-g64/int8/int3/E8) e per le shape GLM.
- Metodo visibile nel codice: ogni ottimizzazione ha la misura nel commento, e le feature di
  misura (LOOKA, DISK_SPLIT, ROUTE_TRACE) sono zero-overhead se spente e provabilmente
  isolate dallo stato di eviction.

## Cosa rubiamo per il paging OPFS nel browser / cosa non traduce

**Da rubare (portabile quasi 1:1):**
1. **Slab coalescente per expert**: gate/up/down adiacenti ⇒ 1 read da ~19 MB. In OPFS: un
   file/offset-range per expert, un solo `FileSystemSyncAccessHandle.read()` nel worker —
   vale ancora di più visto l'overhead per-chiamata JS.
2. **Batch-union**: puro algoritmo (FASE A/B di `moe()`), identico in WGSL/JS. Per prefill e
   verify MTP è il moltiplicatore di banda più economico che esista.
3. **PILOT**: il router di L+1 è un matmul minuscolo (D×E) sullo stato post-attention di L,
   quasi gratis in GPU/WASM; il 71,6% di recall è proprietà del MODELLO, non dell'engine.
   Prefetch = read OPFS anticipata nel worker I/O. Replicare prima la metodologia LOOKA
   (contatori-only) per misurare il recall sul NOSTRO modello.
4. **`.coli_usage` + AUTOPIN con confidenza**: tripla (l,e,count) salvata a ogni turno, pin
   proporzionale a `min(hist/200k,1)`. In browser: file OPFS/IndexedDB; "pin" = expert in
   GPUBuffer/ArrayBuffer. Chiave: pin da storia PERSISTENTE, LRU e heat che DECADE in sessione.
5. **tier.h intero** (60 righe): LFRU `heat<<8|recency` + isteresi 25%+4 — logica pura,
   copiabile alla lettera, già estratta e testabile.
6. **Budget onesto + guardia misurata**: `cap_for_ram` (proiezione con slack esplicito) +
   `rss_guard` (enforcement sul consumo MISURATO ogni ~16 token). In browser: stima da
   `navigator.storage.estimate()`/limiti device + guardia sulla pressione reale, con `ecap`
   che scende e non risale.
7. **Eviction by swap, non by copy**: la promozione LRU scambia puntatori slab
   (`colibri.c:3647-3653`). In JS: riusare ArrayBuffer/GPUBuffer, mai riallocare per token.

**Cosa NON traduce (e cosa diventa):**
- **pthread + mutex/condvar (pilota, PIPE)** → niente thread POSIX. Diventano un **Worker
  I/O dedicato** (OPFS sync access handle è comunque worker-only) con coda `postMessage` o
  ring su `SharedArrayBuffer` + `Atomics.wait/notify` (richiede COOP/COEP). L'invariante
  "il pilota scrive solo layer > corrente" resta, ma implementato con Atomics su SAB.
- **`fadvise(WILLNEED)` (PILOT hint-only)** → non esiste: OPFS non ha page cache consultiva.
  Il nostro PILOT è per forza "PILOT_REAL" (read vere in una cache nostra) ⇒ il **guard di
  eviction anti-speculazione è obbligatorio**, non opzionale.
- **O_DIRECT / io_uring / dual-SSD** → non esistono. Concorrenza = più read in volo su uno o
  più handle; il dual-SSD non ha analogo (una sola origin-storage). Da misurare quante read
  OPFS parallele saturano il disco: l'equivalente del loro `iobench` va scritto per OPFS,
  stessi blocchi ~dimensione-expert.
- **mmap, NUMA, mlock, RSS via /proc** → niente equivalente; la contropartita browser è
  quota storage e memoria del tab (OOM-kill del tab = il loro OOM-kill kernel: la lezione
  della guardia misurata resta).
- **Scala**: colibrì paga 11 GB disco/token e lo accetta (0,05-6,8 tok/s). Col nostro
  modello piccolo i rapporti expert-size / banda OPFS / compute per layer vanno rimisurati:
  loro assumono compute abbastanza LENTO da nascondere una read; su WebGPU veloce con
  expert piccoli potrebbe valere l'opposto.

## Dubbi aperti

1. Il 71,6%/75,8% è misurato su GLM-5.2: quanto recall ha il router-lookahead sui MoE
   piccoli (OLMoE-class)? `olmoe.c` ha PILOT 1-3 layer ma il recall OLMoE non è nel repo.
2. Nessuna curva sistematica "hit-rate vs ore di uso" per il learned pinning: solo datapoint
   per host; "gets faster the more you use it" non è quantificato nel tempo.
3. La causa del collasso 0-4% acceptance con testa MTP int4 è documentata come fatto (#8) ma
   non spiegata; per noi conta se quantizzeremo una draft head.
4. `ecap` è un tetto UNICO per tutti i layer (solo il pin è per-layer): possibile margine
   (layer con routing più concentrato meritano cache più piccola). Nota minore: README dice
   "single C file (c/glm.c)" ma il file è `c/colibri.c` (rinominato).
