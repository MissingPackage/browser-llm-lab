# Journal — engine-fase-c1

(append-only; una entry per iterazione, con evidenza e verdetto verifier)

## it.1 — fase 1: risorse oracolo + recon (2026-07-30, IN CORSO — gated dal download)

Plan-check RISOLTO (docket 1): ruling PI in chat, GGUF = layout del motore ⇒ Q4_0.

**Scelta GGUF**: `unsloth/GLM-4.7-Flash-GGUF` → `GLM-4.7-Flash-Q4_0.gguf` (17.22 GB).
Motivazione: Q4_0 puro = il layout quant che i kernel dequant-fusi del motore leggono
(criterio del ruling; stesso formato di qwen2.5-0.5b-instruct-q4_0.gguf dell'oracolo
fase A); unsloth già citato in direction §3; bartowski ha lo stesso Q4_0 (17.39 GB)
come alternativa. Download in corso in `~/.cache/blab-models/` (~2 MB/s ⇒ ~2h);
SHA-256 da registrare a download finito.

**Build oracolo**: checkout dedicato `~/Projects/llama.cpp-oracle` (FUORI repo),
`ggml-org/llama.cpp` commit `5f55650a78f92aff4d48d671423e888fac0469ff` (2026-07-30,
shallow). Build CPU-only Release (`-DGGML_CUDA=OFF -DLLAMA_CURL=OFF`, nvcc assente),
32 core: `llama-cli` e `llama-quantize` compilati, exit 0. CPU-only è coerente col
metodo colibri (thread pinnati, determinismo; niente varianza numerica GPU).

**Recon punti di tap (file:riga al commit sopra) — FINDING: zero patch necessarie.**
llama.cpp espone un eval-callback di scheduling (`cb_eval`, `include/llama.h:375`,
`common/common.h:488`) con pattern ask/observe già dimostrato in
`common/debug.cpp:143-185` (filtro regex sul nome tensore + `ggml_backend_tensor_get`)
e `examples/eval-callback/`. I tensori che servono sono GIÀ nominati per layer via
`cb(...)`:
- `ffn_moe_logits` — logits del router (`src/llama-graph.cpp:1846`);
- `ffn_moe_probs` / `ffn_moe_probs_biased` — probs sigmoid + bias di selezione
  (gating GLM = SIGMOID di default, `src/models/glm4-moe.cpp:17-20`; exp_probs_b
  stile DeepSeek-V3, `llama-graph.cpp:1883-1887`);
- `ffn_moe_topk` — expert selezionati `[n_expert_used, n_tokens]`
  (`src/llama-graph.cpp:1929`, via `ggml_argsort_top_k`);
- `post_attn_norm` — hidden post-attention normato, INPUT del router
  (`src/models/glm4-moe.cpp:219`) = lo stato su cui LOOKA applica il router di L+1;
- `l_out` — output di layer (`glm4-moe.cpp:262`), alternativa per la definizione
  esatta del predittore (decisione di spec, fase 2).
⇒ Il tool di traccia = eseguibile C++ NOSTRO in `tools/oracle-moe/` che linka la
build (pattern eval-callback), contatori/dump nel tool. Il constraint "llama.cpp
SOLO oracolo, mai vendored" resta intatto: nessuna modifica upstream. Il rischio
tecnico della fase 4 (tap hidden) si è ridotto: il tap è un filtro sul nome.

**Conteggio layer routed (per sanity-gate fase 3)**: da confermare allo smoke coi
metadati GGUF — `n_layer_dense_lead` e `n_layer_nextn` sono hparams
(`glm4-moe.cpp:12,23`); il builder SALTA il layer NextN nel forward
(`glm4-moe.cpp:52-53`). Il "46 MoE attesi" del GOAL va verificato (47 layer di
config upstream − dense lead − NextN: la spec fissa il numero esatto dai metadati).

**Completamento (stessa iterazione, dopo il download):**

- **GGUF verificato**: 17.22 GB in `~/.cache/blab-models/GLM-4.7-Flash-Q4_0.gguf`,
  SHA-256 `d0bbdfcde6e323ebf90a8b9e95da57100e972be1ec6f0bfa0fad0feaa426557e`.
- **CORREZIONE recon**: il GGUF unsloth è arch **`deepseek2`**, NON `glm4moe`
  (gguf-dump). Il builder effettivo è `src/models/deepseek2.cpp`; `build_moe_ffn`
  è condiviso ⇒ `ffn_moe_logits`/`ffn_moe_probs`/`ffn_moe_topk` restano validi
  (llama-graph.cpp:1847/1874/1929); il tap hidden post-attention in deepseek2 si
  chiama **`ffn_norm`** (`deepseek2.cpp:376`; alternative: `attn_out`:231,
  `l_out`:419). MLA confermata dai metadati (q_lora 768, kv_lora 512, rope 64 —
  direction §3 ok). **Niente NextN nel GGUF** (testa MTP droppata dalla
  conversione) → FINDING fuori-scope a docket item 2 (implicazioni fase D e C2).
- **Conteggio per sanity-gate fase 3 CONFERMATO**: block_count 47, dense lead 1,
  niente layer NextN nel forward ⇒ **46 layer MoE routed** (64 expert top-4 +
  1 shared, gating sigmoid, exp_probs_b stile DeepSeek-V3).
- **Smoke greedy exit 0**: `llama-cli -st --simple-io --temp 0 -t 16 -n 64` —
  output coerente (Rayleigh), log committato. NOTA STRUMENTO: la nuova UI chat
  di llama-cli IGNORA `-no-cnv` e loop-a all'infinito su stdin chiuso (676 MB di
  "> " nel primo tentativo, processo killato) ⇒ per run scriptate SEMPRE
  `-st --simple-io`, o llama-bench. Landmine da riportare in HANDOFF.
- **Throughput oracolo (llama-bench, 16 thread, r=2, cache calda)**:
  **pp512 56.58 ± 3.74 t/s; tg64 13.43 ± 0.10 t/s** (stddev decode minima ⇒
  niente thrashing nonostante host RAM 31 GB — il working set MoE per token è
  piccolo, effetto-colibri osservato di passaggio). Fattibilità corpus: 16k
  posizioni ≈ 9 min/run ⇒ la spec può permettersi corpus 2-3× senza problemi.
- Artefatti: `results/engine/moe-oracle/{oracle-smoke-2026-07-30.json,
  smoke-glm47flash-q4_0-2026-07-30.log, llama-bench-glm47flash-q4_0-2026-07-30.json}`.

**Done-when fase 1**: GGUF+SHA ✓, build+commit registrati ✓, smoke exit 0 con
log/JSON e tok/s ✓, recon file:riga nel journal ✓ (corretto deepseek2).
Verifier: **PASS** (agent loop-verifier, 2026-07-30 — tutti i punti con evidenza;
drift none; violazioni none; note minori applicate: sizeBytes esatto 17216676192,
ffn_moe_logits riga 1846). FASE 1 DONE.

## it.2 — fase 2: spec C1 (2026-07-30)

Spec scritta: `docs/superpowers/specs/2026-07-30-engine-fase-c1-design.md`,
ancorata ai fatti misurati di it.1 (arch deepseek2, 46 MoE, gating
sigmoid+exp_probs_b, pp/tg misurati). Sezioni: Strumentazione (tool C++
standalone via cb_eval, zero patch; LOOKA online con estrazione pesi router e
replica esatta dell'ordine di selezione di build_moe_ffn), Corpus (8 prompt,
~16k posizioni, prefill/decode separati), Metriche (recall@K decode-only,
baseline_prev, skew, working-set), Policy simulate (TS+vitest; LRU, LFRU
tier.h, +pin split-temporale, +prefetch con replay e guard), Sanity-gate
(la_tot 45/46 da metadati asseriti a runtime; autotest predittore ≥0.999),
Timebox e fallback (3 it.), Rischi (5, incl. fedeltà predittore e osservatore).
Richiesta ruling appesa: docket item 3, decisioni (a)-(f). Done-when fase 2:
spec esiste con sezioni grep-abili ✓, entry docket ✓.
STOP BY DESIGN dopo verifier: fasi 3-6 gated dal ruling PI.
Verifier: **PASS** (loop-verifier, 2026-07-30 — 7 sezioni presenti; coerenza
spec↔journal e spec↔GOAL verificate; claim tecnico sigmoid→+bias→top-k
verificato NEL CODICE a commit 5f55650a e nei metadati GGUF, incl. esistenza di
blk.N.ffn_gate_inp.weight/exp_probs_b.bias e ASSENZA di ffn_gate_inp.bias;
src/engine intatto; drift none). FASE 2 DONE, ruling pendente (docket 3).

## it.3 — fase 3: ROUTE_TRACE (2026-07-31)

Tool `tools/oracle-moe/trace.cpp` (C++ puro su llama.h, zero patch upstream,
linka la build via build.sh) + corpus 8 prompt generati da gen-corpus.py
(estratti CONGELATI da src/engine e docs/engine/study + 3 task hand-written;
corpus hash 2882e4ab…) + run-trace.sh (envelope: sha GGUF, commit oracolo,
threads, corpus hash).

**FINDING metodologico (primo run FAILed by design, sanity in azione)**:
llama.cpp POTA l'ultimo layer alle sole righe di output (`inp_out_ids`) — con
`llama_batch_get_one` solo l'ultima posizione ha logits ⇒ il routing del layer
46 spariva per 511/512 colonne del chunk ("colonne 1 != 512 su il=46", exit 2).
Fix: batch custom con `logits=true` su OGNI posizione di prefill (il routing
non cambia; si paga l'lm_head su tutte le righe). Lezione per chiunque
strumenti llama.cpp via cb_eval: l'ultimo layer non è come gli altri.

**Run canonica (exit 0, ~21 min, 16 thread)**: 31.274 posizioni routed
(26.154 prefill + 5.120 decode = 640×8, ZERO eos anticipati) — gate ≥16k/≥4k
PASS con margine ~2×. Sanity nel tool: colonne per layer per batch PASS su
tutti i batch, 46 layer MoE osservati esatti, slot tutti scritti, expert id
∈[0,64), top-4. Verifica indipendente post-run: 31.275 righe jsonl (=tot+
header), ultima riga parsata e valida (184 id). Artefatti:
`results/engine/moe-oracle/trace-2026-07-31.jsonl.gz` (input della fase 5;
SOSTITUISCE la 2026-07-30 di questa iterazione — routing bit-identico, superset
con le predizioni) + `trace-2026-07-31-summary.json` (envelope+sanity). Nota naming: la
data nel filename è UTC (run partita 23:xx UTC del 30) — coerente con
l'envelope, non un errore.

**Done-when fase 3**: tool committato ✓ (niente patch: il "patch-file" del
contratto si è rivelato non necessario, deviazione già registrata in spec
approvata), run exit 0 documentata ✓, traccia con envelope completo ✓,
sanity righe = posizioni × 46 ✓ (verificata anche fuori dal tool).
Verifier: **PASS** (loop-verifier, 2026-07-31 — checkout oracolo PULITO (zero
patch confermato), summary e traccia riverificati indipendentemente (31.275
righe, campioni validi, corpusHash ricalcolato identico), sanity nel codice
alle righe giuste, src/engine intatto; blocker segnalato: commit it.3 da fare
= questo commit; binario trace ignorato via .gitignore). FASE 3 DONE.

## it.4 — fase 4: LOOKA recall (2026-07-31) — 1 iterazione su 3 di timebox

Tool esteso (stesso `trace.cpp`, decisioni (a)/(b) del ruling): tap `ffn_norm-<L>`
via cb_eval; pesi router letti DIRETTAMENTE dal GGUF con `gguf.h`
(`blk.L.ffn_gate_inp.weight` [2048×64] f32 + `blk.L.exp_probs_b.bias` [64] f32,
layer 1..46, ~24 MB) — nessuna API interna di llama.cpp, checkout ancora intatto.
Predizione = `top-K(sigmoid(W[L+1]·h_L) + b[L+1])`, tie-break su indice minore
(argsort stabile). Un solo passaggio: al tap di L si predice sia L (autotest) sia
L+1 (lookahead); al `ffn_moe_topk` di L si confronta.

**AUTOTEST (gate hard dello strumento): self-recall 0.999997** su 5.754.416 slot
(= 46 layer × 31.274 posizioni × 4, atteso ESATTO), minimo per layer 0.999984
(il=1). La replica della selezione è fedele; il residuo ~3e-6 è aritmetica f32 su
pareggi, non un errore di ordine. `laTotPass` true: 5.629.320 = 45 layer-target ×
31.274 × 4, come da sanity-gate di spec.

**RISULTATO — lookahead di un layer, decode (921.600 slot su 5.120 token)**:

| metrica | valore | riferimento colibri (GLM-5.2) |
|---|---|---|
| recall@4 | **0.7746** | — |
| recall@6 | **0.8769** | — |
| recall@8 | **0.9197** | 0.716 (K=8 hint-only) |
| baseline "stessi expert del token t−1" | **0.3228** | 0.413 |

Prefill quasi identico (R@4 0.7696, R@8 0.9189 su 4.707.720 slot): la
predicibilità NON è un artefatto del regime autoregressivo.

**Lettura**: su GLM-4.7-Flash il routing è **più predicibile** che su GLM-5.2
(R@8 92.0% vs 71.6%) e la baseline è **più debole** (32.3% vs 41.3%) ⇒ il
segnale del router batte la persistenza temporale di 2.8×, non di 1.7×. Il
prefetch predittivo qui ha molto più margine di quanto assunto in direction §4.3.
Per-layer (45 target): R@4 media 0.7746 ± 0.054, min 0.638 (il=5), max 0.846
(il=41); R@8 min 0.803, max 0.969. I 6 layer sotto 0.70 R@4 sono TUTTI iniziali
(4-9) — coda bassa localizzata, non rumore diffuso: un budget di prefetch
K-adattivo per layer è un'idea DA ANNOTARE nel ledger in fase 6 (lo sweep dei
doc è lì per contratto; qui non è stata scritta).
Extra fuori-spec registrato nel JSON: predire il PRIMO layer MoE dallo stato del
layer denso dà R@8 0.558 — il salto denso→MoE è più difficile, coerente col
profilo dei layer bassi.

**Artefatti** (rigenerati nella stessa run canonica, exit 0, ~24 min):
`results/engine/moe-oracle/trace-2026-07-31-recall.json` (aggregati + 46 righe
per-layer + baseline + extra), `-summary.json` (ora con `autotest: PASS`),
`trace-2026-07-31.jsonl.gz` (identica per costruzione: 31.274 posizioni,
26.154p + 5.120d, zero EOS anticipati).

**Done-when fase 4**: report JSON con recall aggregato e per-layer per K∈{4,6,8} ✓,
baseline token-precedente sulla STESSA traccia ✓, sanity-gate di spec verdi
(autotest ≥0.999 ✓, la_tot atteso ✓). Timebox: chiusa in 1 iterazione su 3,
clausola di fallback NON esercitata.

Note di precisione sugli artefatti (dal verifier, da sanare in fase 6):
- `baselinePrev.decodeTransitions` = 235.152 è il conteggio di (layer × transizione
  di token), non di transizioni: 46 × (5120 − 8 prompt) = 46 × 8 × 639. Numero
  giusto, etichetta fuorviante.
- Spec §Sanity-gate scrive `la_tot == posizioni_decode × 45`; l'implementazione
  asserisce il superset `(n_moe−1) × posizioni_TOTALI × 4` (prefill incluso), la
  cui componente decode è esattamente 45 × 5120 × 4. La spec va allineata
  all'invariante più forte.

Verifier: **PASS** (loop-verifier, 2026-07-31 — aritmetica dei sanity ricalcolata
da zero (autotest.tot, laTot, decode.tot, extra.tot tutti esatti); baseline
RIPRODOTTA indipendentemente dalla traccia gz (235.152 transizioni, overlap
0.322790, 46/46 layer combacianti a <1e-5); fedeltà del predittore verificata
contro build_moe_ffn upstream + metadati (gating=2 sigmoid, group_count=1) e
non-tautologicità dell'autotest confermata; tool RICOMPILATO dal sorgente in
/tmp ed eseguito su 2 prompt (autotest 1.000000 PASS, stesso regime di recall);
exit-code 3 provato sul campo; checkout oracolo pulito; nessun gate sul recall
lookahead (decisione (f) rispettata); drift none). FASE 4 DONE.

## it.5 — fase 5: residenza + simulatore (2026-07-31)

Traccia rigenerata (run canonica, exit 0, numeri LOOKA IDENTICI — determinismo
verificato: digest del routing bit-identico alla run precedente) ora con le
predizioni top-8 per layer dumpate SULLE SOLE righe di decode (`"pr"`), che sono
l'input del replay di prefetch preteso dalla spec. La traccia 2026-07-30 è stata
SOSTITUITA da `trace-2026-07-31.jsonl.gz` (7.4 MB): stesso routing, superset di
informazione; i riferimenti di it.3/it.4 vanno letti sul nuovo file.

Simulatore in TypeScript (`tools/oracle-moe/sim/{policies,simulate,run-sim}.ts`,
logica pura riusabile in C3) + **15 unit vitest verdi** (`tests/oracle-moe-sim.test.ts`):
LRU, LFRU tier.h (heat<<8|recency, decay, isteresi), pin, guard anti-eviction,
invariante "budget pieno ⇒ hit-rate satura", determinismo, statistiche.

**RISULTATO 1 — la residenza non è skewed come nei riferimenti.** Expert toccati:
**2944/2944 (100%)** già sul solo decode. Skew decode: top-4 per layer coprono
**21.8%** delle selezioni, top-8 34.0%, top-16 52.4%, top-32 77.5% (uniforme
darebbe 6.25/12.5/25/50: siamo 3.5× più concentrati dell'uniforme, ma lontanissimi
da un 80/20). Working-set decode: **1.663 expert unici in 32 token**, 2.421 in 128,
2.815 in 512. Il working set è quasi l'intero parco.

**RISULTATO 2 — curve hit-rate vs budget (decode, config canonica di spec:
pin 50% del budget, K=8)**, valutate sulla seconda metà della traccia col pin
appreso sulla prima (split anti-leakage):

| budget (slot / GB) | LRU | LFRU | LFRU+pin | +prefetch |
|---|---|---|---|---|
| 184 / 1.0 | 24.9% | 18.0% | 17.2% | 17.7% |
| 368 / 2.0 | 42.7% | 33.0% | 29.3% | 29.2% |
| 736 / 3.9 | 61.4% | 59.6% | 51.0% | 51.8% |
| 1472 / 7.8 | 84.7% | 85.6% | 80.5% | 83.1% |
| 2208 / 11.7 | 96.4% | 95.2% | 95.2% | 98.2% |
| 2944 / 15.6 | 100% | 100% | 100% | 100% |

Con i parametri di spec **LRU batte le policy sofisticate** quasi ovunque: con
skew debole, dedicare metà cache a un pin statico è costo puro.

**RISULTATO 3 — le sensibilità ribaltano la conclusione (e il colpevole era il
parametro, non la policy)**. Riducendo la quota di pin e adattando K:

| budget | LRU | LFRU+pin(12.5%)+prefetch, K migliore |
|---|---|---|
| 736 | 61.4% | **67.5%** (K=2) |
| 1472 | 84.7% | **90.8%** (K=4) |
| 2208 | 96.4% | **98.9%** (K=8) |

Il pin monotonicamente peggiora al crescere della quota a budget stretti
(736: 58.3 → 55.9 → 51.0% per 12.5/25/50%); il K ottimo del prefetch **cresce col
budget** (a 736 K=8 fa thrashing: 58.9% vs 67.5% con K=2). Sensibilità al decay
dell'heat registrata anch'essa nel JSON (l'ottimo si sposta con la capienza:
512 a budget stretti, 4096 a 1472).

**Verdetto "modello ~2× la memoria" (ledger §A)**: a 1472 slot (cache = 50% del
parco routed, cioè modello 2× la cache) l'hit-rate di decode è **90.8%** nella
configurazione migliore, 84.7% con LRU nudo. Regge, ma non è gratis: ~9% di miss
× 5.3 MB/expert è il traffico che la banda OPFS deve sostenere — il costo va
chiuso in C3 col modello di banda (results/opfs-bench: warm ~gratis, cold
disco-bound), non qui.

**Caveat metodologico dichiarato**: l'hit-rate NON misura il beneficio di latenza
del prefetch (un layer di anticipo), solo l'occupazione della cache. Il prefetch
qui appare utile solo dove non spreca slot; il suo valore vero (nascondere la
read OPFS) richiede un modello di costo temporale — fase C3.

Artefatti: `results/engine/moe-oracle/residency-sim-2026-07-31.json` (residenza,
skew per-layer, working-set, 24 curve + 12 sensibilità decay + 18 knob).
**Done-when fase 5**: JSON statistiche ✓, simulatore committato con test verdi ✓
(npm test), curve per 4 policy × 6 budget ✓, punto di lavoro evidenziato ✓ (aggiunto DOPO
il primo verdetto: vedi completamento sotto), verdetto 2× esplicito ✓.
Verifier: **FAIL** al primo giro (loop-verifier, 2026-07-31) — done-when "punto
di lavoro del device evidenziato" ASSENTE dagli artefatti ma spuntato nel
journal: violazione sostanziale, corretta sotto. Tutto il resto PASS con
riproduzione indipendente (working-set e skew ricalcolati in Python identici;
LRU riscritto da zero → decodeHitRate identici a 4 decimali; sostituzione della
traccia verificata riga per riga su tutte le 31.274, routing identico, `pr` su
5120/5120 righe decode; npm test 181/181; src/engine intatto; oracolo pulito).

**Completamento it.5 (dopo verifier FAIL n.1 — done-when "punto di lavoro"
mancante, spuntato per errore nel primo giro).** Il punto di lavoro ora è nel
report, calcolato da numeri MISURATI (non stimati): VRAM da `nvidia-smi`
(**RTX 4090 Laptop, 16 GiB** — non la desktop 24 GB: dato nuovo per il progetto),
byte non-expert e per-expert dai tensori del GGUF
(`tools/oracle-moe/gguf-residency.py`: routed 15.68 GB, non-expert **1.53 GB**,
5.33 MB/expert). Con slack browser 10% e KV MLA a ctx 4k (0.22 GB):
**2.573 slot disponibili = 87.4% del parco routed, modello/cache = 1.14×** ⇒ sul
device dev il paging è quasi inoperante (hit-rate 96.4% LRU / 98.9% best config).
Conseguenza per il progetto: il regime di paging vero NON è la 4090 di sviluppo —
è mobile/M4-condiviso o contesti lunghi che mangiano la VRAM. Da riportare in
fase 6 (direction §5 dice "paging su 4090-16GB": vero, ma il fattore è 1.14×,
non un paging aggressivo).

Correzioni applicate dopo il FAIL: nota esplicita nel report sul working-set
calcolato attraverso i confini dei prompt (numero conservativo); rimozione dello
stato morto `speculative` in Lfru (non influenzava alcuna eviction — sarebbe
sembrato un meccanismo attivo in C3); rimosso `llama-bench-*.err` vuoto dal repo.

## it.6 — fase 6: sintesi e chiusura (2026-07-31)

**Checklist DONE WHEN del contratto — 6/6:**

1. **Spec C1 scritta e ruling registrato** ✓ —
   `docs/superpowers/specs/2026-07-30-engine-fase-c1-design.md` (7 sezioni),
   ruling PI "ok (a)-(f)" registrato nel docket item 3 (RISOLTO).
2. **Traccia di routing riproducibile** ✓ —
   `results/engine/moe-oracle/trace-2026-07-31.jsonl.gz` (31.274 posizioni ×
   46 layer), envelope con sha256 GGUF `d0bbdf…`, commit oracolo `5f55650a`,
   16 thread, greedy, corpus hash `2882e4ab…`. Tool `tools/oracle-moe/`
   committato, ZERO patch a llama.cpp (checkout verificato pulito a ogni
   iterazione).
3. **Recall lookahead misurato** ✓ — `trace-2026-07-31-recall.json`: decode
   R@4 0.7746 / R@6 0.8769 / R@8 0.9197 su 921.600 slot, per-layer (46 righe),
   baseline token-precedente 0.3228 sulla STESSA traccia; sanity `laTotPass`
   e autotest 0.999997 (gate 0.999).
4. **Statistiche di residenza** ✓ — `residency-sim-2026-07-31.json`: usage per
   (layer,expert), skew cumulativo top-{4,8,16,32,64} aggregato e per-layer,
   working-set W∈{32,128,512}. Co-attivazione COUPLE: NON inclusa (non era
   gratis dal tracciato — la spec la condizionava, nessun costo di scope).
5. **Simulatore trace-driven con test verdi** ✓ — `tools/oracle-moe/sim/`,
   15 unit vitest (suite intera 181/181), 24 curve (4 policy × 6 budget) +
   12 sensibilità decay + 18 knob; punto di lavoro del device nel report;
   verdetto "modello ~2× la memoria" esplicito (regge: 90.8% tarato).
6. **Decisione PREPARATA, non presa** ✓ — docket item 4 con proposta go/no-go
   PILOT + aritmetica del costo-miss dal bench OPFS; ledger §A (2 righe
   riscritte) e §I (2 rimandi nuovi) e direction §3/§5/§7 aggiornati coi
   numeri misurati.

**Sanate le note ereditate**: costante byte/expert unificata al valore misurato
(5.325.512 B; il 5.3 MB di direction §3 era analitico), `verdict2x` ora include
la configurazione tarata (90.8%), `nearestCurvePoint` seleziona LRU
esplicitamente come baseline invece di dipendere dall'ordine di generazione.
Resta come nota (non sanabile senza rerun): l'etichetta
`baselinePrev.decodeTransitions` conta (layer × transizione), non transizioni.

**Cosa lascia questo goal a C2/C3**: la traccia e il tool (rieseguibili su
qualunque GGUF supportato dall'oracolo), la logica delle policy in TS pronta
per il worker, tre numeri che cambiano il progetto (recall 92%, skew debole,
device dev a 1.14×) e una domanda aperta esplicita (banda OPFS a freddo).
Verifier finale: [da compilare]
