# Direction — il motore (2026-07-29)

Documento di direzione del nuovo motore di inferenza browser. Scritto nella sessione di
piano mandata dal PI (HANDOFF 2026-07-29), ancorato a: `estimates.md` (modello di budget
misurato, M1 chiusa in §8), `ideas-ledger.md` (inventario idee + rulings §F/§H),
`verification-sweep-2026-07-29.md`, e i quattro study report (`study/README.md`).
Questo doc decide *direzione e ordine*; non è una spec — ogni fase avrà la sua.

## 1. Cosa costruiamo, in una frase

Un motore di inferenza LLM per browser, **from scratch, narrow, GGUF-compatibile**
(ruling #14), il cui differenziale non è "eseguire un LLM su WebGPU" (esiste già:
WebLLM, LlamaWeb/wllama, LFM2) ma il **sistema di memoria** — esperti paginati con
prefetch predittivo, KV tiering/checkpoint su OPFS, adapter LoRA hot-swap, speculative
decoding — con **telemetria ed eval integrate by design**.

## 2. Funzione obiettivo (ruling PI 2026-07-28/29)

Massima **intelligenza** sotto vincolo di rate "sufficiente" (~30 tok/s morbidi, più
alto per reasoning) e memoria ≤ host. Non max tok/s: la velocità oltre soglia si spende
in capienza (modello più grande) o qualità. **Precisazione PI 2026-08-01** (ruling sul
gate prefill di C3, docket engine-fase-c3a item 1): il vincolo di rate è a due termini,
entrambi user-facing — **decode ≥30 tok/s** (≈60 in regime thinking: i token di
ragionamento sono latenza pura prima della risposta visibile) e **TTFT ≤4 s**. I floor
derivati dall'oracolo CPU sono gate d'ingresso intermedi, mai obiettivi. Due budget
misurabili:

```
touch/token ≤ BW_eff × (1/rate − T_fisso)        (velocità)
residenza   ≤ M_host (GPU+RAM+OPFS, con tassa browser)   (capienza)
```

L'oggetto pubblico è la **curva di Pareto qualità-vs-rate per device**. Terzo asse:
intelligenza per byte (specializzazione via LoRA). È ricerca/tecnologia/benchmark, non
prodotto (§F ledger): i moonshot sono capitoli legittimi, le evals di perdita
d'intelligenza sono parte del progetto.

**§2-bis — Razionale di sviluppo (ruling PI 2026-08-06, in chat, dopo il probe
VRAM di C3a it.19).** Quattro principi che ordinano tutte le scelte:
1. **Best engine sulla dev machine, poi si allarga.** Data una macchina di
   sviluppo (oggi: 4090 Laptop 16 GiB; poi M4 48 GB; S22 dichiarato-e-dopo),
   il motore si ottimizza al massimo LÌ, su 1-2 modelli scelti — non per tutti
   i modelli a priori. Kernel per altri modelli e altri hardware vengono dopo,
   quando il motore giusto esiste.
2. **La scarsità di VRAM è la condizione di progetto, non un incidente.** Sul
   browser degli smartphone la VRAM utile è una frazione di 12-16 GB; il
   valore del motore sta nei meccanismi che rendono VELOCE la residenza
   parziale: streaming da disco, quantizzazione asimmetrica, expert caricati
   a necessità (lineage ds4/colibri). Cambiare modello per farlo stare in
   VRAM è ammesso ma NON è il trucco principale.
3. **"Usabile" batte "massimo teorico"**: non 1-3 tok/s (né tok/minuto) — la
   soglia UX di §2 resta il criterio. Gli esperimenti estremi (es. il repo
   GitHub 2026 che esegue Kimi K3, 2.8T parametri, su CPU con 8 GB di RAM —
   da analizzare come catalogo di strategie [VERIFY]) si studiano e si
   ADATTANO, non si imitano.
4. Conseguenza sulla scala dei modelli (§6): il modello-tesi resta un modello
   che NON sta comodo nella memoria del dev-box — il regime "quasi ci sta"
   (GLM su 4090M) e quello "non ci sta per 2-4×" (slab budget ridotto ad
   arte) sono entrambi emulabili qui; l'abbondanza (M4 48 GB) no. È la
   macchina giusta per la tesi, e la residenza parziale veloce È la tesi.

## 3. Rulings vincolanti (non ridiscutere senza PI)

| # | Ruling | Data |
|---|---|---|
| 14 | **Scratch**, narrow, GGUF-compatibile; llama.cpp = **oracolo**, non substrato | 2026-07-29 |
| 13 | Modello-tesi **GLM-4.7-Flash** (30B-A3B, MLA, MIT); dev rungs Qwen3.5-0.8B/2B; first-light Qwen2.5-0.5B; v2 = ibridi | 2026-07-29 |
| — | First-light: **battere WebLLM (107 tok/s, Qwen2.5-0.5B q4, 4090) con le sole L1-L3** | 2026-07-29 |
| — | Desktop first (4090 = dev-loop, M4 = hero-demo MoE); mobile = dichiarato, dopo | 2026-07-28 |

**[VERIFY] del ruling #13 chiusi (2026-07-29, questa sessione):**
- **MTP nativa: sì.** `config.json` upstream: `num_nextn_predict_layers: 1` (testa
  NextN stile DeepSeek-V3 nel checkpoint) → lo spec-dec di prima battuta usa la testa
  del modello, senza dipendere dai checkpoint DeepSpec (che restano senza licenza
  dichiarata). Lezione colibri vincolante: testa draft **mai sotto int8**.
- **Benchmark indipendenti: esistono.** Artificial Analysis Intelligence Index **23**,
  #11/130 nella classe open-weights 4B-40B (mediana 9; misure AA, non vendor). Nota:
  Z.AI ha già rilasciato GLM-5 (API deprecata) — irrilevante per i pesi MIT, ma il
  panorama §H va rivisitato quando decideremo v2.
- Config verificata: 47 layer (primo denso), hidden 2048, 64 routed top-4 + 1 shared,
  MLA `kv_lora 512 + rope 64` → **1 152 B f16/layer/token ≈ 54 KB/token** (stessa
  aritmetica KV di colibri su GLM-5.2: il loro codice KV è direttamente rilevante);
  expert ≈ **5.33 MB q4** misurati dai tensori del GGUF (stima analitica 3×1536×2048 confermata; non-expert totale 1.53 GB — C1) — taglia amichevole per read OPFS, ~2 944
  expert totali ≈ 15-16 GB routed q4. GGUF unsloth/bartowski esistono (llama.cpp
  esegue l'architettura → l'oracolo funziona per questo modello).

## 4. La tesi tecnica (dal codice, non da opinioni)

Lo studio dei 4 codebase (`study/`) ha confermato la diagnosi a livello di sorgente:
ggml-webgpu *non può* fare ciò che il nostro motore farà, per contratto di interfaccia
(il backend vede i nodi ggml uno alla volta: bind group e param ricreati a ogni op di
ogni token, una sola fusione peephole, telemetria che cambia l'esecuzione quando la
accendi). Possedere il grafo compra:

1. **Piano di esecuzione compilato al load** — shape del modello note → bind group
   precomputati (L1), scratch preallocato e un submit/token (L2), fusione di catene
   dequant+matmul+bias+act+norm e sampling in un kernel (L3). Valore misurato su
   WebLLM-baseline: **1.44× con le sole L1-L3 su 4090** (estimates §3), di più su
   mobile (il floor per-dispatch domina). Riferimento incrociato: LlamaWeb ottiene
   +45-69% decode con "static allocation + arena" — due strade indipendenti, stessa
   conclusione.
2. **Telemetria nativa always-on** — ring di timestamp-query + contatori per-fase che
   NON alterano il batching (anti-pattern LlamaWeb documentato). Ogni decisione di
   budget del sistema di memoria è giustificata da numeri, stile ds4/colibri.
3. **Sistema di memoria come cuore** (gap statement verificato dallo sweep):
   - *Esperti*: slab size-class unica + 1 read coalescente per expert (ds4/colibri),
     LFRU `heat<<8|recency` con decadimento (tier.h ≈ 60 righe, copiabile), pinning
     appreso da routing heat persistito (AUTOPIN a confidenza), prefetch predittivo
     dal router di L+1 (PILOT: 71.6% recall su GLM-5.2 — **proprietà del modello, va
     rimisurata sul nostro**). In browser il PILOT è per forza "real load" (niente
     fadvise) ⇒ guard anti-eviction obbligatorio.
   - *KV*: MLA nativa del modello-tesi (54 KB/token); prefix-cache OPFS col design
     ds4 (chiave = hash del testo renderizzato, envelope+payload, logits salvati per
     resume senza decode, boundary allineati al prefill chunk; SHA-256 via WebCrypto).
   - *Budget onesto*: proiezione con slack esplicito + enforcement su consumo misurato
     (colibri `cap_for_ram`+`rss_guard`); failure mode browser = device lost/tab kill.
4. **Tap hidden-states progettati in v0** — tutti i draft DeepSpec (e ogni futura
   testa) consumano gli hidden di layer intermedi: il forward del motore espone N tap
   configurabili da subito, anche se lo spec-dec si costruisce dopo. Retrofittarli
   costerebbe una riscrittura.
5. **LoRA primitiva di prima classe** — `y = Q(W)x + B(Ax)` fusa nel kernel, adapter
   hot-swap su base cachata (gap browser confermato dallo sweep; lineage ardesia-gguf).

## 5. Cosa NON costruiamo (il narrow, per iscritto)

Anatomia dai riferimenti: colibri = 7k righe (un modello, zero deps); ds4 = 253k (ma
multi-backend + server + agent). Noi stiamo dal lato colibri: **un backend (WebGPU),
una famiglia di modelli per volta, zero astrazione**.

- Niente runner GGUF generico: leggiamo i container GGUF **dei nostri modelli target**,
  con validazione hard e exit su mismatch (postura ds4). "GGUF-compatibile" = formato
  file e quant layout, non "zoo di architetture".
- Niente graph IR/scheduler generico: il grafo è codice che incoda kernel, per modello.
- Niente multi-backend (WASM solo per tokenizer/CPU-side, mai fallback di compute:
  tensore oversize ⇒ split, non ripiego — anti-pattern LlamaWeb).
- Niente sampler zoo, niente server, niente training (il training LoRA in-browser è un
  moonshot separato, non v0/v1).
- Tokenizer e componenti non-differenzianti: in prestito (ruling #14).

## 6. La scala dei modelli

| Gradino | Modello | Serve a | Stato |
|---|---|---|---|
| First-light | Qwen2.5-0.5B q4 | battere 107 tok/s con L1-L3; grafo già noto al dispatch-level (269 disp/token misurati, invarianti di device — estimates §8) | gate di v0 |
| Dev rungs | Qwen3.5-0.8B / 2B (Apache) | sviluppo quotidiano, CI, evals | v0.5+ |
| **Tesi** | **GLM-4.7-Flash 30B-A3B** | MoE+MLA: hero-demo M4 (residenza piena, ~17 GB q4 in 48 GB); sul box dev (4090 Laptop 16 GiB) il paging è marginale — misurato C1: 2.573 slot su 2.944, fattore 1.14× | v1 |
| v2 | Qwen3.5-35B-A3B / Nemotron (ibridi) | kernel linear-attention fuori dal critical path di v0 | **FATTO (goal q1, 2026-08-10): famiglia Qwen 3.5/3.6 conforme — v. §7-bis** |

## 7. Ordine di costruzione (dal WP studio, §6 della sintesi)

**Fase A — execution core (la tesi).** Load GGUF (subset validato) + piano statico +
kernel minimi (GEMV/GEMM dequant-fusa q4, RMSNorm, RoPE, attention, catena sampling
fusa) + L1-L2-L3 + telemetria nativa + oracolo llama.cpp (golden logits desktop, KL sui
logit dal giorno zero — il motore espone i logits by design, WebLLM non lo fa).
**Gate: first-light.** Include da subito: negoziazione `requiredLimits` dall'adapter,
f32-first sul dev-loop (Chrome/Linux non espone shader-f16; percorso f16 dietro
feature-detect per M4/S22), tap hidden-states nel design del forward (§4.4).

**Fase B — memoria I: KV e persistenza.** Eredita dalla fase A il target
**≤100 dispatch/token** (ruling docket 4: la fase A ha chiuso a 123, floor
architetturale a 5 dispatch/layer; sotto 100 servono fusioni cross-layer o
megakernel parziali). Prefix-cache OPFS (design ds4), forward
multi-token (prefill chunk + futura verifica spec-dec), rollback KV (`crop` con length
pointer). Prima misura: banda OPFS in lettura (tool ~20 righe, ancora mancante).

**Fase C — memoria II: MoE e paging.** *Metodo prima del meccanismo*. Splittata in
C1/C2/C3 (ruling PI 2026-07-30). **C1 CHIUSA (2026-07-31)**: LOOKA replicato
sull'oracolo, recall **92.0% @K=8** (vs 71.6% di colibri su GLM-5.2), baseline 32.3%;
simulazione trace-driven: "modello ~2× la memoria" regge (90.8% hit-rate decode a
cache = 50% del parco, config tarata). Due correzioni alle assunzioni: (a) la
residenza NON è skewed (top-4/layer = 21.8% delle selezioni) ⇒ il learned pinning
vale poco e va tenuto ≤12.5% degli slot, il valore sta nel PREFETCH; (b) sul box dev
(4090 **Laptop**, 16 GiB) il modello sta all'87% in VRAM (fattore 1.14×) ⇒ il regime
di paging vero è mobile/M4-condiviso o contesti lunghi, non il dev-loop.
**C2 CHIUSA (2026-08-01)**: GLM-4.7-Flash gira end-to-end nel motore (47 layer,
MLA absorbed, MoE con residenza minima OPFS→VRAM LRU) con correttezza dimostrata —
logits argmax ≡ cpuref-f64 100% (campione ratificato), top-1 vs golden 98.83%
full-corpus; hit residenza 97.56% @12 GiB (sopra il 96.4% del simulatore C1).
Prestazione SOTTO il floor CPU con deroga PI (docket C2 item 6a): decode 4.64
tok/s vs floor 13.43, prefill senza percorso batched. Attribuzione misurata
(docket C2 item 8 = input C3): 215.5 ms/token = 56.1 stallo residenza (pack CPU
9.5 ms/miss, leva: repack all'import) + 158.9 struttura (1.816 dispatch + **47
sync router/token** — il collo dominante). Leve C3 in ordine: repack import,
eliminazione sync router (senza: tetto ~7 tok/s), prefill batched M>1, prefetch
LOOKA (valore residuo sul dev-box, decisivo su mobile/ctx lunghi).
**C3 SPLITTATA in C3a/C3b (ruling PI 2026-08-01**, docket engine-fase-c3a item 1):
l'attribuzione C2 separa 158.9 ms/token di *struttura* da 56.1 di *residenza* — due
assi con due condizioni di chiusura, un goal unico ne perdeva una.
- **C3a — struttura (il floor)**: repack all'import, eliminazione/batching dei 46
  sync router, prefill batched M>1. **Gate di chiusura = floor C1 (13.43/56.58)**;
  ogni bench riporta anche il gap dalla soglia UX (30 tok/s, TTFT 4 s), che è
  l'obiettivo vero — il floor prefill 56.58 su 461 token vale 8.15 s di TTFT.
  **CHIUSO 2026-08-06 (it.35) sotto clausola item 17a**: meccanismo completo
  (1 submit/0 sync a residenza totale in ktest; arena+Sel; prefillChunk M=16
  bit-identico), gate tok/s dichiarati irraggiungibili su questo device per
  hardware (−415 MiB fisici, probe it.19), non per struttura. Numeri di
  chiusura: decode 5.211 (da 4.64), prefill **25.78** (da 5.24, 4.9×), TTFT
  **17.88 s** (da 88, gap UX 4.47×), gpuBusy 54.2 ≤ 54.5, pack 0.0, dispatch
  1405 (da 1816), correttezza: cpuref 256/256 e 512/512, golden 98.828%
  full-corpus, routing = firma item 14b. Su M4 48 GB il gate è aritmetica.
- **C3b — decode ottimistico (il drain, regime near-total)**: 1 submit/token nel
  path pulito, flag di miss piggyback sul readback logits, replay esatto dal
  primo layer sporco (checkpoint hidden 376 KB/token), inserimenti slotTable al
  confine di token. Dimensionato da WP-0 (journal c3a it.20): è il meccanismo
  del regime ≥ ~88% di residenza (dev-box: 95.7% osservata); in scarsità
  collassa ⇒ la Pareto ha due segmenti. **Gate STRUTTURALE** (sync/token ≤ 2
  nel bench di produzione, tassa di replay entro la proiezione WP-0 ±25%);
  niente gate tok/s (WP-0 proietta 11.3 al tetto misurato coi kernel di oggi) —
  il floor 13.43 è ereditato da C3c con clausola pre-negoziata. NIENTE predictor
  GPU al confine di token (falsificato 3×) né repair batched.
  **CHIUSO 2026-08-08 (it.8)**: il drain e' eliminato — **47 → 2.188
  sync/token a b12 (−95%) e 1.891 ≤ 2 PASS al tetto** (2595 slot, sessione
  minima; ruling docket c3b item 8, opzione a). Decode **5.211 → 11.60 a b12
  (+123%) e 16.64 al tetto** (sopra il floor 13.43 ereditato da C3c), TTFT
  **17.88 → 16.79 / 12.60 s** (gap UX 3.15×), qualita' BIT-invariata (ktest
  identita' 131072/131072 hidden+logits; produzione 64/64 token = sync).
  Tassa di replay ×1.14 vs WP-0 (entro ±25%); fuori solo i 2 termini della
  cascata del replay (repair iterativo per prefisso, item 7) che il sim non
  modella, spiegati dal moltiplicatore dei round 1.39×. Non-regressione a
  perimetro pieno tutta PASS (banda ±5%, ruling item 9). Finding per C3c:
  P(dirty) sensibile allo stato cache post-prefill; leva P(dirty) = policy
  > LRU (Belady ~dimezza), intera in C3c.
- **C3c — paging (la residenza in scarsità vera)**: budget slab ctx-aware,
  tier.h, AUTOPIN, prefetch in-forward (non il predictor di confine), modello
  di banda, WP banda fredda browser (primo pezzo, blocca la spec), instant-on
  come TTFT a freddo (budget assoluto da ri-negoziare, docket c3c item 1), e il
  **floor 13.43 con clausola pre-negoziata**. Chartered 2026-08-07
  (.harness/goals/engine-fase-c3c/), parte a C3b chiuso.
  **CHIUSO 2026-08-09 (it.10, 9 fasi in 10 iterazioni)**: **il floor 13.43
  passa SECCO in produzione — decode 15.641 alla config di budget migliore,
  CALCOLATA dalla formula ctx-aware (12.737 GiB), in sessione utente viva**
  (niente clausola); ctx 6144 senza OOM. Banda fredda browser = parità col
  bound OS (1.79-1.94 GB/s random, tassa API zero). Prefetch in-forward:
  recall in-engine **91.92% @K=8 vs 92.0 dell'oracolo** (−0.08pp), zero sync
  aggiunti, firma routing invariata ai conteggi. Policy tier+AUTOPIN+prefetch
  vs LRU pura: use-hit **+9.43pp @1472 slot (94% del gap Belady), +25.04pp
  @736 (oltre Belady: il prefetch esce dal bound demand-fetch)**, pin ≤12.5%
  HARD. Modello di banda: 3 punti predetti a **±1%** (b12 fuori dal fit), il
  ±15% è un test permanente; TTFT freddo predetto a −5.8%. Instant-on:
  **1.247 ≤ 1.25× auto-ancorato** (ruling item 1a) con page cache fredda
  PROVATA; gap UX 6.18× → fase D. Qualità invariata ovunque (golden 98.828%
  AL PIN, cpuref 256/256+512/512, ktest 69/69).
**FASE C COMPLETA (C3a+C3b+C3c): decode 4.64 → 15.64 tok/s (3.4×), TTFT
88 → 12.9 s (6.8×), 47 → 2 sync/token, a qualità bit-invariata.**
Hero-demo M4 resta PI-gated per hardware (input nel docket c3c, item 8).

**Fase D — moltiplicatori.** Spec-dec: prima la MTP nativa del modello (verificata,
§3), poi eventualmente DSpark-style; verifica = rejection sampling esatto (lossless by
construction). LoRA hot-swap. Evals harness completo (sez. D ledger: KL/top-k vs
riferimento, perplexity in-browser, micro-suite; accept-rate speculativo come proxy
live di divergenza — idea DeepSpec).

Le fasi B/C/D si sequenziano dopo il gate di A; dentro ciascuna, spec dedicata prima
del codice (convenzione repo). Il benchmark pubblico e la roadmap generale del progetto
(consolidamento → ceiling+hero-demo → motore) restano fuori da questo doc.

## 7-bis. Generalizzazione — goal engine-fase-q1 CHIUSO (2026-08-10, coi numeri)

Il motore esegue il path testo della famiglia Qwen 3.5/3.6 (ibrida 3:1
Gated-DeltaNet : GQA) con la fedeltà del metodo GLM. Tre modelli, tre
soglie ratchet fissate su golden full-corpus (1024 posizioni, oracolo
llama.cpp b10333, provenance piena):

| Modello | top-1 ratchet | near-tie dei miss | riferimenti (correttezza-prima) |
|---|---|---|---|
| 4B denso (tied) | **1012/1024 = 98.828%** | sparsi, max 3/prompt | decode 22.93 tok/s · prefill seq 26.0 · TTFT 25.8 s (full-resident) |
| 9B denso | **1000/1024 = 97.656%** | 24/24 near-tie (23 top-2, mediana 0.066 logit) | 14.55 · 15.4 · 40.3 s |
| 35B-A3B MoE | **1013/1024 = 98.926%** | 11/11 near-tie (mediana 0.204, zero >1 logit) | col paging: v. tier |

I riferimenti sono DICHIARATI correttezza-prima (562-782 dispatch/token,
zero fusioni, readback per token nel decode; sul MoE 1 sync router/layer):
sono il FRAME su cui i moltiplicatori (fase D) lavorano, non numeri
competitivi. Pezzi nuovi permanenti: tokenizer in-engine (BPE byte-level,
id==oracolo, protocollo `--no-escape`+USER_DEFINED), reader parametrico
(shape dai metadata), kernel DeltaNet WGSL (ktest vs cpuref-f64 dalla
fonte b10333), dequant/gemv Q4_K, paging MoE parametrico (arena a chunk
≤2 GiB per classe down, LRU on-miss; residency.ts GLM INTATTO — rifit
della meccanica C3c piena rimandato alla fase D sul modello target).

**Tier 35B misurati** (bench p512-style, host dichiarato): 8/12/16 GB →
decode 0.79/2.00/3.40 tok/s (residenza 23/47/64%, hit 77/91/94%) — il
collasso in scarsità è 4.3×, attribuito al hit-rate (~73 miss/token al
tier 8; costo miss 17.64 ms fit su 3 punti, LIMITI dichiarati nel JSON:
il costo include il repack JS on-miss del regime correttezza-prima).
Full-corpus a budget 12 GiB: hit 98.55% con 121 421 eviction LRU reali,
zero OOM. Tier mobile 4B: cap 3 GiB rispettato (footprint 2.48), TTFT
mobile = proiezione PARAMETRICA (banda storage e compute factor liberi):
prefill-bound ⇒ il batching è la porta del mobile.

**Gap nativo a parità di regime** (stesso GGUF, stessa GPU, llama.cpp
Vulkan): decode **5.18× (4B) / 4.62× (9B)** — scomposto con misura:
readback+sync 5.1/3.6 ms/token (12%/5% — eliminabile col pattern GLM
decodeBatch), compute residuo 4.6×/4.4× (dispatch serializzato 15-29%
stimato, kernel non tuned, check WebGPU). Prefill 183×/171× = ASSENZA di
batching, non gap: leva n.1. Leve per ROI (docket q1 item 10-11):
prefill batched (pattern GLM) → decode multi-step no-readback → fusione
→ int8 (dot4I8Packed CONVERGE con subgroup-matrix: il probe ha trovato la
feature ESPOSTA sul nostro harness ma INT8-ONLY 16×16×32). Per il
writeup: pubblicare il 4.6-5.2×; il prefill solo post-batching.

**Recall lookahead 256-wide** (oracolo C1 su cpuref, subset p4+p7):
**82.67% @8** vs 91.92% GLM — spiegato: softmax-256 senza bias vs
sigmoid+bias-64, bersaglio 3.1% vs 6.25% del parco, profilo per-layer
(L0 44%: dall'embedding l'attn costruisce il segnale; max 90%). NOTA:
il recall@4 = 47.74% ha DENOMINATORE 8 (tetto teorico 50%), non
confrontabile con @8. Implicazione: il prefetch in fase D renderà meno
che su GLM — registrato, non promesso.

Artefatti: journal/docket `.harness/goals/engine-fase-q1/`; JSON in
`results/engine/` (conf, bench tier, looka, bandmodel-fit, gap); doc
`docs/engine/study/2026-08-10-q35-gap-decomposition.md`; spec
`docs/superpowers/specs/2026-08-10-engine-fase-q1-design.md`. Tutti i
numeri destinati al writeup si RIMISURANO al tag di release
(paper-contract-draft, ruling 2026-08-10).

## 8. Rischi dichiarati

1. **Prefill WebGPU è lento per tutti** (LlamaWeb −21-51% vs competitor). La nostra
   fusione aiuta ma non è dimostrato che basti: il first-light è dichiarato sul
   *decode*; il prefill ha come mitigazioni la prefix-cache OPFS (fase B) e il
   chunking. Rischio accettato, monitorato dal gate A.
   **CORREZIONE 2026-08-02 (probe C3a it.3, `results/engine/webgpu-limits-4090laptop-2026-08-02.json`)**:
   la clausola «subgroup-matrix assente nei browser» è **STALE**. L'adapter di
   questa macchina espone `chromium-experimental-subgroup-matrix`, `subgroups` e
   `subgroup-size-control`. **Perimetro**: le vediamo perché lanciamo Chrome con
   `--enable-unsafe-webgpu`, quindi valgono per esplorare il *ceiling* del motore,
   **non** per un confronto pubblico su browser stock — nei benchmark pubblici la
   clausola originale resta valida e va dichiarata (§8.4). Conseguenza sulle leve:
   il GEMV oggi usa `workgroup_size(64)` con riduzione in shared memory, un
   pattern che le subgroup ops rendono obsoleto; e il device negozia 256
   invocazioni/workgroup su **1024** disponibili. Materia della guerra ai
   dispatch (goal C3a fase 4b).
2. **Residuo 33% del budget 4090 non attribuito** (estimates §2): se fosse
   proporzionale a N_disp, L3 vale il doppio; se fosse fisso, L4 pesa di più. Si
   scioglie solo con timestamp-query nel runtime nostro (telemetria nativa, fase A).
3. **Banda OPFS** — misurata (fase 2 del goal, 2026-07-29, 4090/NVMe): write 2.2 GB/s,
   read via SyncAccessHandle 7.5-11.7 GB/s **a page-cache calda** ⇒ l'API non è il
   bottleneck. **Bound freddo misurato lato OS (2026-07-31, fadvise, 990 PRO)**:
   random expert-size **1.63 GB/s (3.74 ms/expert p50, 8× il warm)**, seq 3.22 GB/s.
   **WP browser ESEGUITO (2026-08-08, goal c3c fase 1, `tools/opfs-cold/`)**: il
   freddo IN CHROME è alla pari col bound OS — random expert-size **1.79-1.94
   GB/s (p50 2.9-3.1 ms/expert)**, seq 3.43-3.73 GB/s, streaming sequenziale
   expert-size p50 1.3-1.4 ms/expert; nessuna tassa browser sul path freddo
   (freddezza provata: fincore 0 B post-drop, delta warm 8×). I numeri del
   modello di banda C3c partono da qui, non dall'estrapolazione OS. Conseguenza sul target: "modello ~2× la memoria" vale come
   **2× la VRAM con spillover RAM-backed** (24% del budget a 30 tok/s); in regime
   disk-bound il tetto è ~18 tok/s o serve hit ≥94.5% con prefetch a overlap
   perfetto — il paging vive di hit-rate (prefetch prima leva, ruling item 4).
4. **Equità benchmark**: Chrome/Linux/NVIDIA non espone shader-f16 (M4/S22 sì) — da
   dichiarare in ogni confronto pubblico.
5. **S22**: T_fisso misurato ~18 ms/token di solo encode CPU (estimates §8) > budget a
   30 tok/s. Il mobile resta dichiarato-e-rimandato: la guerra al floor (fusione
   aggressiva, L3) è la precondizione, non un'opzione.
6. **Osservatore**: il profiling via patch prototype perturba i device CPU-bound (S22:
   −40% sotto patch). La telemetria del motore deve essere a costo ~zero quando spenta
   (lezione colibri: feature di misura zero-overhead se spente, provabilmente isolate).

## 9. Prime azioni della fase A (input per il goal)

1. Spec della fase A (spec-first, convenzione repo): IR/piano statico, formato interno
   pesi (da GGUF q4 → layout nostro), contratto dei tap, telemetria.
2. Tool banda OPFS (20 righe, sblocca stime fase B in anticipo).
3. M2 di estimates (contatore call-site su flushCommands del bundle WebLLM): dice quali
   dei 7 submit/token sono eliminabili — raffina il valore atteso di L2 prima di
   scrivere il motore. Costo ~10 righe.
4. Scheletro repo del motore (dentro questo repo, `src/engine/` — decisione di
   struttura da prendere nella spec A).
