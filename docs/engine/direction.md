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
in capienza (modello più grande) o qualità. Due budget misurabili:

```
touch/token ≤ BW_eff × (1/rate − T_fisso)        (velocità)
residenza   ≤ M_host (GPU+RAM+OPFS, con tassa browser)   (capienza)
```

L'oggetto pubblico è la **curva di Pareto qualità-vs-rate per device**. Terzo asse:
intelligenza per byte (specializzazione via LoRA). È ricerca/tecnologia/benchmark, non
prodotto (§F ledger): i moonshot sono capitoli legittimi, le evals di perdita
d'intelligenza sono parte del progetto.

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
  expert ≈ **5.3 MB q4** (3×1536×2048) — taglia amichevole per read OPFS, ~2 944
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
| **Tesi** | **GLM-4.7-Flash 30B-A3B** | MoE+MLA: hero-demo M4 (residenza piena, ~17 GB q4 in 48 GB) e paging su 4090-16GB (il sistema di memoria *serve* anche su desktop) | v1 |
| v2 | Qwen3.5-35B-A3B / Nemotron (ibridi) | kernel linear-attention fuori dal critical path di v0 | dichiarato |

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

**Fase C — memoria II: MoE e paging.** *Metodo prima del meccanismo*: replicare la
metodologia LOOKA di colibri (contatori-only) **sull'oracolo desktop** per misurare il
recall del router lookahead su GLM-4.7-Flash; solo poi slab+tier+AUTOPIN+PILOT-real in
browser. Gate intermedio: instant-on (router+shared+esperti caldi subito, resto
on-demand). Hero-demo M4.

**Fase D — moltiplicatori.** Spec-dec: prima la MTP nativa del modello (verificata,
§3), poi eventualmente DSpark-style; verifica = rejection sampling esatto (lossless by
construction). LoRA hot-swap. Evals harness completo (sez. D ledger: KL/top-k vs
riferimento, perplexity in-browser, micro-suite; accept-rate speculativo come proxy
live di divergenza — idea DeepSpec).

Le fasi B/C/D si sequenziano dopo il gate di A; dentro ciascuna, spec dedicata prima
del codice (convenzione repo). Il benchmark pubblico e la roadmap generale del progetto
(consolidamento → ceiling+hero-demo → motore) restano fuori da questo doc.

## 8. Rischi dichiarati

1. **Prefill WebGPU è lento per tutti** (LlamaWeb −21-51% vs competitor; subgroup-matrix
   assente nei browser). La nostra fusione aiuta ma non è dimostrato che basti: il
   first-light è dichiarato sul *decode*; il prefill ha come mitigazioni la
   prefix-cache OPFS (fase B) e il chunking. Rischio accettato, monitorato dal gate A.
2. **Residuo 33% del budget 4090 non attribuito** (estimates §2): se fosse
   proporzionale a N_disp, L3 vale il doppio; se fosse fisso, L4 pesa di più. Si
   scioglie solo con timestamp-query nel runtime nostro (telemetria nativa, fase A).
3. **Banda OPFS** — misurata (fase 2 del goal, 2026-07-29, 4090/NVMe): write 2.2 GB/s,
   read via SyncAccessHandle 7.5-11.7 GB/s **a page-cache calda** ⇒ l'API non è il
   bottleneck; il regime freddo è disco-bound e va ancora caratterizzato (drop_caches
   impossibile dal browser). Il dimensionamento del paging (fase C) usa: warm re-read
   ~gratis, cold-load ≈ banda NVMe del device. Il paging vive comunque di hit-rate
   (pinning+prefetch), target realistico "modello ~2× la memoria".
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
