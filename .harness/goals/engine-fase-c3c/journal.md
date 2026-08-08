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
