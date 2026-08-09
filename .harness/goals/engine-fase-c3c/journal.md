# Journal — engine-fase-c3c (paging in scarsità)

## it.0 — 2026-08-08 — setup (goal-setup)

Apertura su richiesta PI in chat ("partiamo con c3c", 2026-08-08). Pre-avvio
(docket item 2) era già ESEGUITO alla chiusura C3b: numeri fissati (sync path
+ optimistic b12/tetto, hit-rate LRU b11 0.9575 / b12 0.9756, landmine i-v),
banda ±5% applicata (ruling c3b item 9). Contratto v1 RILETTO: le 4 modifiche
dello split (decode ottimistico uscito, floor 13.43 entrato con clausola
pre-negoziata, PILOT-real → prefetch in-forward, instant-on relativo
[ASSUMED 1.25×]) sono coerenti con lo stato post-C3b; nota di rilettura: il
"TTFT a caldo ~18 s" del commento è superato — i riferimenti a caldo chiusi
da C3b sono 16.79 s (b12) / 12.60 s (tetto); quale dei due ancori il target
relativo di instant-on è parte della proposta per docket item 1 (fase 2).
PHASES.md: 9 fasi sequenziali, authority delta = none ovunque, nessuna
docket-born. Plan-check: registrato a docket item 3 (pattern c3b item 4).
Tag goal-engine-fase-c3c-start al commit di setup.

## it.1 — 2026-08-08 — FASE 1 DONE: WP banda fredda browser

**Fatto.** Tool nuovo `tools/opfs-cold/opfs-cold-bench.mjs`: misura in Chrome
(SyncAccessHandle, worker) la lettura OPFS a page cache FREDDA. Eviction DA
FUORI il browser (fsync + posix_fadvise DONTNEED ×2, python3) sui file di
backing OPFS del profilo dedicato — stessa tecnica del tool OS di C1 ⇒
confronto appaiato. Freddezza DIMOSTRATA nel JSON, non asserita:
`evictionEvidence` con residenza fincore prima/dopo ogni drop (6.44 GB → 0 B)
+ controprova warm sugli stessi offset (delta 8×). Dati incomprimibili
(crypto.getRandomValues; profilo su btrfs zstd:1), file 6 GiB, expert-size
esatto 5 325 512 B, self-served su porta fissa 5327 (l'origin determina il
bucket OPFS — bug porta effimera trovato e fisso prima delle run canoniche).

**Numeri (2 run canoniche committate, `opfs-cold-4090-linux-2026-08-08T10-14-*`):**
- random expert-size COLD: **1.79 / 1.94 GB/s, p50 3.1 / 2.9 ms/expert, p95
  4.5 / 3.7** (OS C1: 1.63 / p50 3.74 / p95 4.34)
- seq 1 MiB COLD: 3.43 / 3.73 GB/s (OS: 3.22); seq expert-size streaming:
  3.51 / 3.66 GB/s, p50 1.4 / 1.3 ms/expert (= regime instant-on)
- warm control: rand 11.41 / 11.88 GB/s p50 0.4-0.5 ms (OS: 10.92 / 0.46) —
  lo stack di misura coincide con opfs-bench fase A (10.7-11.7)

**Il finding: PARITÀ col bound OS (ratio 1.07-1.19, p50 sotto).** L'ipotesi
C1 "il freddo browser è ≤ dei numeri OS" era prudente ma la tassa browser sul
path freddo è ZERO. Il modello di banda (fase 6) e la spec (fase 2) usano il
numero browser misurato: ~1.8 GB/s random / ~3.0 ms p50 per expert;
instant-on streaming a ~3.5 GB/s ⇒ lower bound di lettura per il non-routed
+ caldi (1.53 GB) ≈ 0.44 s: il budget TTFT a freddo non muore di banda disco.

**Done-when riga 1:** tool committato in tools/opfs-cold/ PASS; JSON in
results/opfs-bench/ con p50/p95 random e sequenziale PASS; confronto
esplicito coi numeri OS C1 nel JSON (`comparisonVsOsC1` con ratio) PASS;
ledger §A e direction §8.3 aggiornati col numero browser PASS; freddezza
dimostrata dal protocollo nel JSON (`evictionEvidence` fincore + delta
cold/warm, drop dichiarato) PASS.

**Igiene:** 2 run pre-fix scartate (bug origin/porta effimera: bucket OPFS
duplicato nel profilo — numeri identici, ma il tool committato è quello
fisso e le run canoniche sono le sue). Profilo bench: ~6.4 GiB in
`~/.cache/blab-opfs-cold-profile` (riusabile con KEEP_FILE=1, cancellabile).

## it.2 — 2026-08-08 — FASE 2 DONE: spec C3c coi numeri del WP dentro

**Fatto.** Spec depositata:
`docs/superpowers/specs/2026-08-08-engine-fase-c3c-design.md` (10 §, pattern
c3b). Fissa: (§2) budget slab ctx-aware = allocCeiling − nonExpert(1 354 078
720 B) − KV(ctx) − work(ctx) − reserve [ASSUMED 256 MiB, si tara in fase 3],
con MIN_SLOTS 184 (pin-for-replay c3b) e throw esplicito sotto; (§3) prefetch
IN-FORWARD (tap hidden L → router L+1 dentro il token, K=4 [ASSUMED], flag
default off, recall in-engine vs 92.0% @K=8 oracolo, scostamento spiegato non
gateato) con esclusione ESPLICITA del predictor al confine (falsificato 3×);
(§4) tier.h+AUTOPIN da colibri §2 (eusage persistente OPFS, eheat con decay,
LFRU heat<<8|recency, isteresi 25%+4, pin cap 12.5% HARD con assert) misurata
vs LRU a 1472/736 slot vs ceiling Belady; (§5) modello di banda con formula e
tolleranza ±15% su 3 budget; (§6) instant-on con definizione operativa
RIPRODUCIBILE (OPFS popolata + VRAM vuota + page cache FREDDA col protocollo
eviction del WP fase 1) e strumento (glmbench cold-start, evictionEvidence nel
JSON); (§9) rimisura formale syncLogits nel primo report di fase 3 (landmine
iv chiusa lì).

**Numeri d'ingresso verificati dal codice/artefatti** (spec §1): parco 2944
slot (2688×5 308 416 + 256×5 505 024 = 15 678 308 352 B), non-expert
1 354 078 720 B, KV 108 288 B/token (glmmodel.ts:725 keyLen 576 × f32 × 47
layer; probe kvBytes 56 851 200 / 525 = conferma esatta). **Scoperta
collaterale: la landmine HANDOFF §5 "KV 54 KB/token ⇒ 361 MB @6k" era STALE
di 2×** (conto f16): vero = 665 MB @6k. Corretta (ruling
docs-update-when-stale), registrata a docket item 4.

**Proposta docket item 1 DEPOSITATA** (item 1-bis): (a) RACCOMANDATA relativo
1.25× AUTO-ANCORATO alla config della run instant-on (scioglie 16.79-vs-12.60:
l'ancora è la stessa run, entrambe le config riportate); aritmetica: senza
overlap 1.28-1.61× (I/O freddo extra 3.1-7.3 s su ~15.7 GB unici a 1.8-3.5
GB/s + 0.4 s non-expert), 1.25× richiede overlap ≥ ~60% dell'I/O dietro il
compute del prefill (12.6 s GPU — c'è spazio); (b) 1.4× conservativa; (c) 4 s
= fase D. Il ruling blocca SOLO fase 7.

**Done-when riga 2:** spec esiste e fissa budget slab/tier+AUTOPIN≤12.5%/
prefetch in-forward con esclusione predictor/definizione operativa instant-on
+ strumento/budget TTFT a freddo PASS; registrazione a docket (item 4) +
proposta item 1 depositata coi numeri (item 1-bis) PASS; la spec NON tocca
gate/soglie ⇒ nessuno STOP di ruling PASS.

## it.3 — 2026-08-08 — FASE 3 DONE: slab ctx-aware

**Fatto.** Formula della spec §2 in codice: `slabBudgetCtxAware` +
`slabWorkBytes` + costanti misurate (`NON_EXPERT_BYTES`, `KV_PER_TOKEN_BYTES`
108 288, `MIN_SLOTS` 168/16 per classe = pin-for-replay c3b I3, throw sotto)
in residency.ts; glmbench worker/page con `allocCeilingBytes` (budget
CALCOLATO, breakdown nel report) e `ctxMaxOverride`; runner `--budget-gib
auto` (ceiling misurato con nvidia-smi A CHROME LANCIATO: il footprint del
browser sta in used, stessa contabilità del probe it.19) + `--ctx-max` +
picco VRAM campionato ogni 5 s nel report. Unit
`tests/engine-slab-budget.test.ts` (6): identità della sottrazione, monotonia
in ctx, clamp al parco, MIN_SLOTS throw, riproduzione del regime dai numeri
del probe (12 GiB ≤ formula ≤ 12.88 a sessione minima).

**Due OOM istruttivi prima della run buona (cause distinte, diagnosticate dal
codice, non retry ciechi):** (1) workBytes dimenticava `attnPartialsM` del
prefill chunked = partials ×GLM_PREFILL_M(16) ≈ 253 MiB @ctx6k
(glmmodel.ts:1125 — lo stesso buffer di un OOM storico documentato a riga
641); (2) riserva 256 MiB < slack di sessione viva (>247 MiB: staging ring
Dawn durante il preload da ~13 GB + compositor) ⇒ TARATA 512 MiB coi due
punti OOM come evidenza (procedura prevista dalla spec §2). Spec aggiornata.

**Run committate (3, tutte user-session-light dichiarato, optimistic +
prefill chunked):**
- `bench-glm-4090-ctx6k-autobudget-2026-08-08.json`: ctxMax 6144, **NON va
  OOM** — budget CALCOLATO 12.146 GiB (breakdown nel JSON: ceiling 15 867 −
  nonExp 1354 − kv 665.3 − work 268.8 − riserva 536.9 MB), slot 2191+256
  (83.1%), decode 12.97, TTFT 14.53, **vramPeak osservato 15 853 MiB**.
- `bench-glm-4090-b12-optimistic-nonreg-c3c-2026-08-08.json`: b12 esplicito,
  codice nuovo — decode **12.92 vs 11.60 rif (+11%)**, TTFT **14.94 vs 16.79
  (−11%)**: migliorati, nessuna regressione; strutturale 2.188 = il punto
  della tassa del riferimento, P(dirty) 0.938 identico.
- `bench-glm-4090-ctx525-autobudget-2026-08-08.json`: **il budget calcolato
  a ctx ~500 riproduce il regime attuale** — 2348+256 slot (88.5%), decode
  **16.55 vs tetto 16.64 (−0.5%)**, TTFT **12.53 vs 12.60**, strutturale
  **1.875 ≤ 2 PASS**, floor 13.43 PASS; in sessione VIVA, senza config a
  mano (il tetto c3b richiedeva sessione minima e budget trovato a mano).
**Rimisura formale syncLogits (landmine iv, spec §9): syncFloorProbe
mapRoundTripMs median 0.09-0.11 ms nelle 3 run** — conferma il probe c3b
it.5 (0.08), il 7.6 era dell'era C3a. Landmine CHIUSA.

**Done-when riga 3:** ctx6k senza OOM con budget calcolato + VRAM di picco
nel JSON PASS; ctx ~500 riproduce il regime (b12 migliorato, tetto in banda
−0.5%) PASS; suite 344+7 ≥ 338+7 e tsc puliti PASS. Exit 4 delle run = gate
floor prefill 56.58 FAIL preesistente (materia fase 8 con clausola, non di
questa fase).

## it.4 — 2026-08-08 — FASE 4 DONE: prefetch in-forward + recall in-engine

**Meccanismo (spec §3, dietro flag `prefetch:"inforward"`, default off = path
bit-identico).** Al router del layer MoE L il pass accoda ANCHE il GEMV del
router di L+1 sullo stesso fnB (tap: +1 dispatch, pesi router già in VRAM);
i logits del tap viaggiano nella STESSA copy di staging e nella STESSA
mapAsync (staging 256→512 B): zero sync aggiunti. Le predizioni (top-4 via
routerSelect coi bias di L+1) si consumano al submit successivo PRIMA
dell'await: il fetch OPFS+writeBuffer cade nel tempo in cui la CPU aspetta
il router e la GPU lavora — writeBuffer dopo il submit = eseguita dopo i
suoi dispatch (ordine di coda), nessuno slab in volo si corrompe. Lo stato
del prefetch vive DENTRO la funzione di forward: una predizione non può
STRUTTURALMENTE attraversare il confine di token (esclusione WP-0 —
l'ultimo MoE non ha tap). Path: sezione sync per-posizione (select
"cpu"/"shadow"); oneSubmit ⇒ throw (I1-I5 c3b intatti). Telemetria nuova
(null quando spento): preds/fetches/resident/prefetchMs + recallPreds/
recallHits4/recallHits8. Flag su glmroute/glmconf (glmbench NON toccato: non serviva alla riga 4) + runner
--prefetch.

**Recall in-engine (done-when) — l'oracolo è REPLICATO nel motore:**
`routing-prefetch-inforward-2026-08-08.json`, full-corpus C1 (1 407 330
predizioni consumate): **recall@8 91.917% vs oracolo 92.0% (−0.08pp),
recall@4 77.045% vs 77.5% (−0.46pp)**. Spiegazione dello scostamento
(non gateato): il tap applica il router L+1 a ffn_norm_L(x) in f32 su GPU
(hidden PRIMA del contributo FFN/MoE di L, con la norm di L); l'oracolo C1
usava il suo hidden f64 al punto equivalente, su base campionaria diversa
(qui prefill+decode teacher-forced). < 0.5pp = il potere predittivo
dell'oracolo arriva in-engine intero.

**Identità del forward (done-when):** firma 14b ai conteggi ESATTI con
prefetch ACCESO — decode 208 441/235 520 (88.5025%), prefill
1 047 485/1 203 084 (87.0667%), router GPU 1 438 591/1 438 604 (99.9991%),
Sel produzione 0/5 754 416 difformi = 1.44M selezioni identiche al
riferimento; argmax ≡ cpuref-f64 **256/256 PASS** col prefetch acceso
(`conf-glm-prefetch-sample-2026-08-08.json`, golden 254/256 = le stesse 2
divergenze di C2); ktest 69/69 PASS; suite 344+7; tsc pulito.

**Osservazioni per la fase 5 (registrate, non decise):** (i) hit-rate
residency della run prefetch 98.16% vs 97.56% LRU pura b12 — MA il numero
non è direttamente confrontabile: gli ensure di prefetch alimentano gli
stessi contatori (resident 5 495 182 / fetches 134 138), il confronto
pulito di fase 5 va fatto sui miss AL MOMENTO D'USO agli stessi budget;
(ii) prefetchMs 222.6 s / 31 274 pos ≈ 7.1 ms/posizione dentro la finestra
d'attesa (0.04 ms/pred × 184 pred/pos): a b12 è quasi tutto touch LRU —
da tenere d'occhio come costo CPU della finestra di overlap ai budget
stretti. (iii) Il prefill CHUNKED (produzione) non ha il prefetch: il
meccanismo vive nel forward per-posizione (decode sync + prefill legacy);
l'estensione al chunked è materia dell'overlap instant-on (fase 7), non
di questa riga.

**Igiene:** primo tentativo glmconf fallito per lock OPFS transitorio
("Access Handles cannot be created…", stesso profilo; la glmroute
successiva e il retry sono passati puliti) — nessun fix necessario,
registrato come firma nota.

**Done-when riga 4:** prefetch dietro flag PASS; recall in-engine su corpus
C1 confrontato nel JSON con 92.0% @K=8 e scostamento spiegato PASS;
identità forward invariata (ktest 69/69, firma 14b esatta, cpuref 256/256)
PASS; suite+tsc puliti PASS.

## it.5 — 2026-08-08 — fase 5 (1/2): meccanica tier.h + AUTOPIN in codice

**Fatto (meccanica, spec §4 — la MISURA è it.6).** `ExpertCache` con
`policy:"lru"|"tier"` (default lru = zero overhead, stats.policy null):
eusage u32[3008] (storia, snapshot/load ADDITIVO per la persistenza OPFS del
chiamante via createWritable = commit atomico), eheat u16 con decay >>1 al
repin, erec (clock recency), `noteSelection(layer, experts)` chiamata dal
forward dopo routerSelect (no-op in lru); AUTOPIN con storia ≥5000 e budget
= min(cap, max(1, floor(cap×conf))) — il max(1) serve alle classi piccole
dove floor(cap×conf) resterebbe 0 fino a conf 0.5; **cap HARD 12.5% per
classe con throw** (ruling C1); REPIN ogni 2944 selezioni (LFRU score
heat×256+recency, isteresi ×1.25+4, max 4 swap, SOLO metadata — deviazione
dichiarata da colibri che nel repin carica ~20 MB/swap: qui la protezione
dall'eviction plasma il set residente nel tempo); eviction = LRU fra i non
pinnati (caller + policy), messaggio esplicito se nessuna vittima.
Harness glmroute: `--policy tier` + `--park-frac 0.5|0.25` (override slot
proporzionale per classe: 1344+128 / 672+64), config nel report. Fix in
corsa: cadenza repin scalata per SELEZIONI (non per chiamate).

**Test:** 5 unit nuovi (`engine-tier-policy.test.ts`): no-op in lru, cap
12.5% mai superato, pin sopravvive al churn completo della classe, load
additivo (2 load = doppia storia) + taglie/policy sbagliate = throw,
tutti-pinnati = rifiuto esplicito. Suite **349+7**, tsc pulito.

**Prossimo (it.6):** batch notturno delle 4 run full-corpus {lru, tier} ×
{1472, 736 slot} (tier CON prefetch in-forward, lru pura senza — la
formulazione del done-when), poi confronto sui miss al momento d'uso, delta
vs Belady WP-0, JSON committati, chiusura riga 5.

## it.6 — 2026-08-09 — FASE 5 DONE: la policy batte la LRU ai budget stretti

**Le 4 run full-corpus** ({lru, tier+AUTOPIN+prefetch} × {1472, 736 slot},
protocollo identico, firma 14b ai conteggi ESATTI in tutte — le policy
toccano la residenza, mai la selezione):
- `routing-policy-lru-b1472-2026-08-08.json` / `routing-policy-tier-b1472-2026-08-08.json`
- `routing-policy-lru-b736-2026-08-09.json` / `routing-policy-tier-b736-2026-08-09.json`

**Hit-rate AL MOMENTO D'USO** (depurato dagli ensure di prefetch, nota it.4:
useMiss = misses − prefetch.fetches, useReq = requests − prefetch.preds;
denominatore identico 5 754 416 selezioni):
- @1472 (50% parco): LRU 84.33% → tier **93.75% (+9.43pp)** — il ceiling
  Belady WP-0 a 1472 è 94.37% (decode): il gap replacement è colmato al ~94%.
- @736 (25% parco): LRU 62.34% → tier **87.37% (+25.04pp)** — SOPRA il
  Belady 82.01%. Non è un paradosso: Belady limita la sola replacement a
  demand-fetch; il prefetch in-forward anticipa i fetch prima dell'uso e
  esce dal perimetro del bound. Dichiarate le due basi diverse (Belady =
  trace decode-only del sim; use-hit = corpus pieno in-engine).
**Pin al cap esatto** (184 @1472, 92 @736 = 12.5% HARD, assert mai
scattato), repin swaps 13 929 / 10 859. **Costo dichiarato**: il tier legge
PIÙ byte totali (6.7 vs 4.8 TB @1472; 15.2 vs 11.6 TB @736 — le
mispredizioni si pagano in banda, +31-40%) in cambio di −60/−67% di miss
bloccanti; wall più veloce comunque (2.9 vs 2.4 / 2.1 vs 1.6 pos/s).
Input per la fase 6: il modello di banda deve usare i BYTE TOTALI letti nel
regime disk-bound, non i soli miss d'uso.

**Igiene di percorso:** run 3 (LRU@736) in TIMEOUT al primo colpo (300 min
insufficienti: hit 62% ⇒ 1.6 pos/s ⇒ ~5.5 h); causa banale, retry identico
con 420 min, PASS. **Regola di metodo adottata (domanda PI 2026-08-08 in
chat, registrata qui): full-corpus SOLO per firma, non-regressione e
riferimenti nuovi (questo batch: crea i riferimenti 1472/736); sim CPU su
traccia (pattern WP-0) o subset di prompt INTERI (mai --cap: taglia la
testa e perde il decode) per esplorazione e tuning parametri.**

**Done-when riga 5:** stessi budget stretti, tier+AUTOPIN+prefetch > LRU
pura a ENTRAMBI (+9.43pp / +25.04pp) PASS; JSON committati per policy ×
budget (4) PASS; delta confrontato col ceiling Belady (94% del gap a 1472;
sopra il bound a 736, spiegato) PASS; pin mai > 12.5% (assert nel path +
misurato al cap) PASS; suite 349+7 + tsc puliti PASS.

## it.7 — 2026-08-09 — FASE 6 DONE: modello di banda, ±1% sui 3 punti

**Punti misurati** (3 bench sync+tier+prefetch, prompt canonico, reps 3,
user-session-light dichiarato): b12 5.411 tok/s (hit aggregato 98.24%),
1472 2.837 (89.80%), 736 1.934 (71.17%). Exit 4 = floor FAIL dichiarato
(regime di scarsità: materia fase 8 con clausola). Plumbing nuovo: glmbench
--prefetch/--policy/--park-frac (parkFrac+optimistic = throw esplicito).

**Il modello** (`src/engine/bandmodel.ts`, committato CON TEST):
`wall = BASE + F(h)·(bytes/banda + FISSO) + STEP·[F>20]`, con F(h) =
(1−h)·368 (le richieste raddoppiano col prefetch — verificato ≤1.1% sui
contatori dei bench). Coefficienti DICHIARATI: BASE 167.43 dal bench sync
b12 di c3b (artefatto INDIPENDENTE), banda dal WP fase 1 (parametrica
warm/cold), C_fetch 2.426 e GPU_STEP 95 fittati sui punti {1472, 736} —
**b12 FUORI dal fit = predizione vera: +0.9%**; 1472 −0.3%, 736 −0.5%.
Tutti entro ±15% (done-when) con margine 15×. Il gradino ha evidenza
misurata: gpuBusy 48.2 → 83.7 → 85.0 ms/token (salto oltre soglia di
fetch, interleave copy/compute — non lineare, dichiarato nel JSON).
**Scoperta di forma: il wall in scarsità NON è BASE+stallo** — l'I/O del
prefetch e l'overhead CPU/GC dei fetch (5.3 MB l'uno) sbordano nel wall;
la formula semplice della spec §5 sottopredice del 30% a 736: la forma a
gradino è quella onesta coi meccanismi C3c accesi.

**Proiezioni fredde** (banda 1.79 GB/s dal WP, per fase 7): 5.02 / 2.24 /
1.27 tok/s ai tre budget. TTFT freddo: `coldTtftMs()` nel modulo
(warm + nonExpert/banda_seq + premio freddo sui byte unici − overlap) —
validazione in fase 7, BLOCCATA dal ruling docket item 1.

**Test:** 10 nuovi (`engine-band-model.test.ts`): il ±15% sui 3 punti è un
TEST in npm test (vincolo meccanico permanente, legge i JSON committati),
+ F(h) vs contatori, monotonia in banda e h, h=1 ⇒ BASE, throw su input
invalidi, coldTtft con e senza overlap. Suite **359+7**, tsc pulito.
Artefatti: 3 bench JSON + `band-model-vs-measured-2026-08-09.json`
(coefficienti, fit dichiarato, evidenza del gradino, proiezioni fredde).

**Done-when riga 6:** formula committata con test PASS; predice i 3 punti
misurati entro ±15% (di fatto ±1%) PASS; la stessa formula predice il TTFT
a freddo (coldTtftMs, verificabile in fase 7) PASS; JSON
predetto-vs-misurato committato PASS.

## it.8 — 2026-08-09 — FASE 7 DONE: instant-on nel budget del ruling (1.247 ≤ 1.25)

**Ruling recepito** (docket item 1 = opzione (a), PI in chat "ok (a)"):
ttftCold ≤ 1.25 × ttftWarm della STESSA config, auto-ancorato.

**Meccanismo costruito (l'overlap che mancava):** prefetch in-forward esteso
al PREFILL CHUNKED (il buco dichiarato in it.4) — tap batched (router L+1 su
fnBM, +1 dispatch/layer/chunk, stessa mapAsync con staging M raddoppiato a
offset fisso MPF), predizioni = UNIONE dei top-4 sulle m righe, consumate al
submit successivo nella finestra d'attesa; stato dentro il chunk (l'ultimo
MoE non ha tap: niente attraversa chunk né token). Effetto misurato: cold
30.55 → 24.57 s (−6.0 s), warm 22.87 → 19.65 (−3.2: l'overlap aiuta anche
il caldo).

**Strumento e protocollo** (`scripts/glm-instanton-run.mjs`, spec §6):
OPFS popolata + VRAM vuota (sessione fresca) + page cache FREDDA provata
(fadvise sui backing file + fincore before/after PER OGNI sessione fredda,
16.7 GB → 0). Composizione dichiarata: ttft = buildMs + warmup.prefill.ms +
primo token; import escluso (precondizione). Config di produzione: sync +
tier + prefetch, prefill chunked, budget ctx-aware auto.

**Percorso del verdetto, in trasparenza:** la v1 (1 sessione per ramo) ha
dato ratio 1.2505 — FAIL per 10.5 ms su 24.5 s (0.04%, rumore di sessione
singola). PRIMA di rimisurare è stato PRE-DICHIARATO il protocollo v2
(pattern B2 della casa: mediane di 3 sessioni per ramo, run fresche, il
verdetto è delle mediane qualunque sia). Esito v2: fredde 24 479/24 727/
24 890, calde 19 824/19 174/20 027 → **mediane 24 727 / 19 824, ratio
1.247 ≤ 1.25 PASS** — margine sottile (0.3%) e dichiarato: la distribuzione
del ratio lambisce il gate su questo host; su M4 (banda unificata) e in
fase D il margine è la leva, non il decimale.

**Coda di fase 6 chiusa:** coldTtftMs predice 23 303 vs misurato 24 727 =
**−5.8%, entro ±15%** (il done-when "predice il TTFT a freddo, verificabile
in fase 7" è ora VERIFICATO). Gap UX 4 s: 6.18× (riportato, fase D).

**Verifiche:** suite 359+7, tsc pulito, ktest 69/69 (flag off = path
identico); artefatti: `instanton-glm-4090-2026-08-09.json` (+6 session
JSON), evidenza eviction per sessione nel JSON.

**Osservazione a docket (non decisa qui):** le selezioni del PREFILL non
alimentano eusage/noteSelection (solo il decode) — colibri le conta;
candidata estensione in chiusura o fase D, non tocca i numeri di questa fase.

**Done-when riga 7:** TTFT a freddo ≤ budget del ruling (1.247 ≤ 1.25,
protocollo pre-dichiarato) PASS; JSON committato con gap UX 4 s esplicito
PASS; predizione del modello di banda confrontata nel JSON (−5.8%) PASS.

## it.9 — 2026-08-09 — FASE 8 DONE: floor PASS in produzione + non-regressione piena

**IL GATE: decode 15.641 ≥ 13.43 PASS** alla config di budget migliore
(ctx-aware auto = 12.737 GiB calcolato, optimistic, strutturale 2.000 ≤ 2
PASS, TTFT 12.90 s) — **in sessione utente viva** (user-session-light
dichiarato): il floor C1, che C3a dichiarò irraggiungibile per hardware e
C3b batté solo a sessione minima con budget a mano, ora passa in produzione
con budget CALCOLATO. La clausola pre-negoziata non serve. Gap UX nel
report (30 tok/s: 1.92×; TTFT 4 s: 3.22×).

**Blocco non-regressione (run fresche, albero congelato, 60 s fra run):**
- sync b12: 5.390 / 29.75 / 15.50 vs rif 5.299 / 25.78 / 17.88 — banda ±5%
  PASS su tutte (migliorati);
- optimistic b12: **13.172 / 31.26 / 14.74** vs rif c3b 11.60/16.79 e vs
  candidato item 5 (12.92/14.94): PASS — **docket item 5 RISOLTO: il nuovo
  riferimento b12 è 13.172 / 31.26 / 14.74** (ratchet, host state
  dichiarato); strutturale 2.188 = il punto della tassa, invariato;
- Qwen: 323.1 ± 0.4 vs 326.2 (−0.95%, banda ±5% PASS);
- golden full-corpus b11: **98.828125% = il PIN alla cifra** (1012/1024);
  cpuref 256/256 PASS;
- fase A: cpuref **512/512 (100%)**, golden 98.047% ≥ 97 PASS;
- routing full-corpus: **firma 14b ai conteggi ESATTI** (208 441/235 520,
  1 047 485/1 203 084, router GPU 1 438 591/1 438 604, Sel 0/5 754 416);
- ktest 69/69; suite 359+7; tsc pulito.

**Nota di protocollo:** le run girano coi meccanismi C3c in codice (formula
ctx-aware, tap, policy) tutti DIETRO flag off nei path di riferimento — le
bande confermano che il default è rimasto il comportamento storico.

**Done-when riga 8:** floor ≥13.43 PASS secco (niente clausola); gap UX nel
report PASS; blocco nonreg completo con run fresche committate PASS su ogni
riga; albero congelato durante le run PASS.

## it.10 — 2026-08-09 — FASE 9: GOAL CHIUSO — checklist DONE WHEN del contratto

1. **WP banda fredda browser** — PASS. tools/opfs-cold committato, freddezza
   provata (fincore + delta warm 8×), rand expert 1.79-1.94 GB/s p50 2.9-3.1
   ms vs OS 1.63/3.74 (parità, ratio 1.07-1.19), ledger §A + direction §8.3
   aggiornati. Artefatti: opfs-cold-4090-linux-2026-08-08*.json (it.1).
2. **Spec scritta e registrata DOPO il WP, coi suoi numeri** — PASS.
   docs/superpowers/specs/2026-08-08-engine-fase-c3c-design.md; docket item
   4; budget slab f(VRAM, nonExpert, KV(ctx)); tier.h+AUTOPIN ≤12.5%;
   instant-on operativo + strumento; budget TTFT freddo (it.2).
3. **Prefetch, non predictor** — PASS. Distinzione in spec §3/§8; predictor
   al confine MAI costruito (stato del prefetch DENTRO il forward:
   strutturalmente impossibile); recall IN-ENGINE 91.917% @K=8 vs 92.0
   oracolo (−0.08pp, spiegato non gateato), 77.045 @K=4.
   routing-prefetch-inforward-2026-08-08.json (it.4).
4. **Slab ctx-aware: ctx 6k senza OOM** — PASS. Budget CALCOLATO 12.146 GiB
   con breakdown + vramPeak 15 853 MiB nel JSON; a ctx 525 riproduce il
   regime (tetto −0.5%). bench-glm-4090-ctx6k-autobudget-2026-08-08.json
   (it.3).
5. **Modello di banda ±15% su ≥3 budget + TTFT freddo** — PASS. bandmodel.ts
   con test permanente in npm test; 3 punti a ±1% (b12 FUORI dal fit +0.9%);
   TTFT freddo predetto −5.8%. band-model-vs-measured-2026-08-09.json (it.7)
   + instanton JSON (it.8).
6. **Policy > LRU a 50% e 25%** — PASS. Use-hit +9.43pp @1472 (94% del gap
   Belady 94.37) e +25.04pp @736 (sopra Belady 82.01, spiegato: il prefetch
   esce dal bound demand-fetch); pin ≤12.5% HARD con assert; 4 JSON
   routing-policy-* (it.5-6).
7. **Floor C1 ≥ 13.43 con clausola pre-negoziata** — **PASS SECCO, clausola
   NON SERVITA**: decode 15.641 alla config di budget migliore (ctx-aware
   auto 12.737 GiB, optimistic, strutturale 2.000), in sessione utente viva;
   gap UX in ogni report (30 tok/s: 1.92×; thinking: 3.84×).
   bench-glm-4090-best-autobudget-nonreg-2026-08-09.json (it.9).
8. **Instant-on ≤ [ruling item 1]** — PASS. Ruling (a) 1.25× auto-ancorato;
   ratio 1.247 al protocollo v2 pre-dichiarato (mediane 3+3, eviction
   provata per sessione); overlap costruito (prefetch nel prefill chunked);
   gap UX 4 s: 6.18× riportato. instanton-glm-4090-2026-08-09.json (it.8).
9. **Non-regressione permanente** — PASS. Sync b12 5.390/29.75/15.50 e Qwen
   323.1 in banda ±5%; optimistic b12 13.172/31.26/14.74 (nuovo riferimento,
   item 5-bis); golden 98.828125% = pin; cpuref 256/256 + 512/512; routing
   = firma 14b esatta; ktest 69/69; suite 359+7; tsc pulito (it.9).
10. **Chiusura** — PASS: direction §7 fase C CHIUSA coi numeri (C completa:
    decode 4.64→15.64 = 3.4×, TTFT 88→12.9 s = 6.8×, 47→2 sync/token, bit-
    invariata); ledger §A riga paging CHIUSA; docket aggiornato (item 8 =
    input hero-demo M4, PI-gated); HANDOFF refresh; digest; push a verifica.

**Docket a fine goal:** item 1 risolto (a); 2 eseguito; 3 pre-autorizzato;
4/5/5-bis/6 registrazioni chiuse; 7 igiene fuori-goal (hostState power);
8 input M4 PI-gated. Nessuna decisione pendente DENTRO il goal.
