# WP studio motori — sintesi incrociata (2026-07-29)

Quattro studi paralleli sui codebase di riferimento, eseguiti dopo il ruling **scratch**
(docket #14). Report: `ds4.md`, `colibri.md`, `deepspec.md`, `llamaweb.md` — ogni claim
cita file:funzione del codice clonato. Nessun codebase è stato eseguito/compilato: solo
lettura (caveat dichiarato in ogni report). Questa sintesi tiene le convergenze
cross-report che il piano di lavoro dovrà usare.

## 1. La tesi dello scratch è confermata a livello di codice (llamaweb)

ggml-webgpu fa un dispatch per nodo, ricrea bind group e scrive i param a ogni op di
ogni token, senza cache tra token: è il contratto dell'interfaccia ggml (il backend vede
i nodi uno alla volta), non una pigrizia. Una sola fusione in tutto il backend; la
telemetria esiste ma disattiva il batching (misura un'esecuzione diversa). Le nostre
L1-L3 + telemetria di produzione sono quindi il differenziatore reale — inesprimibili
in un fork. Da imparare comunque: l'arena uniform a slot rotanti e i 43 kernel a
dequant-fusa (23 formati) come letteratura WGSL.

## 2. Il sistema di memoria è già stato progettato due volte in nativo — convergenze

| Problema | ds4 | colibri | Traduzione browser |
|---|---|---|---|
| Read di un esperto | slab size-class unica | **1 pread coalescente ~19 MB** (gate/up/down adiacenti), viste zero-copy | OPFS sync access handle: identico |
| Eviction | min(hotness) con decadimento, LRU tie-break, entry correnti protette | LFRU `heat<<8\|recency`, decadimento, guard anti-eviction per la speculazione | tier.h di colibri ≈ 60 righe copiabili |
| Prefetch | **router su CPU** prima dell'encode GPU del layer | **PILOT**: router di L+1 eseguito sull'hidden di L — **71.6% recall misurato** (vs 41.3% baseline) | router in WASM o doppio submit; su OPFS niente fadvise → modalità "load reali" e guard obbligatori |
| Pinning | hotlist statica compilata | `.coli_usage` appreso, AUTOPIN a budget proporzionale (46.7 GB pin → hit 66%) | profilo di routing in OPFS: "più lo usi più va veloce" |
| Budget memoria | 80% working set − riserva 2 layer per prefill overlap | proiezione con slack esplicito + enforcement su RSS misurato ogni ~16 token | doppio meccanismo da copiare per la quota tab |

**Metodo prima del meccanismo**: la metodologia di misura del 71.6% (contatori-only nel
forward, LOOKA=1) va replicata sul NOSTRO modello di riferimento prima di costruire il
prefetcher.

## 3. KV persistente: due implementazioni di riferimento

ds4: chiave SHA1 del testo renderizzato (lookup per troncamenti), envelope con token +
**logits completi** (resume senza ri-decode), salvataggi allineati al prefill chunk,
read/write espliciti (no mmap) → traduzione OPFS diretta. colibri: append-only
crash-safe (`nrec` scritto per ultimo), chat riaperta senza re-prefill. Il design della
nostra prefix-cache OPFS parte da qui, non da zero.

## 4. Speculative: fattibile, ma ri-prioritizzato

DeepSpec è portabile (ops draft ⊂ ops target; DSpark ~634 MB incrementali q4, Eagle3
~170; rejection sampling **esatto**; 1-2 settimane incrementali su motore funzionante).
MA: (a) tutti i draft consumano gli hidden di 5 layer intermedi → **i "tap" vanno
progettati in v0 anche se lo spec-dec si costruisce dopo**; (b) ds4 ammette nel README
"at most a slight speedup" nel suo regime disk-bound; (c) colibri: draft int4 →
accettazione 0-4%, int8 ok → mai quantizzare la testa draft sotto int8. Conclusione:
leva desktop per quando il costo fisso domina, non fase-1. Caveat licenza: checkpoint
HF DeepSpec senza licenza dichiarata; i draft Gemma ereditano i Gemma Terms.

## 5. Reality check sull'effort del narrow

Spread enorme: colibri = **7 125 righe** di C per un 744B streaming (CPU-only, un
modello, zero deps); ds4 = ~253k righe (ma: 2 backend GPU, 435 kernel, server, agent,
CLI). Narrow non significa poco codice, significa zero astrazione — però un solo
backend (WebGPU), un modello piccolo e niente server ci mettono molto più vicini al
polo colibri che al polo ds4. Il piano di lavoro deve stimare da questi due ancoraggi.

## 6. Cosa alimenta quale decisione

- **Docket #13 (modello target)**: DeepSpec copre Qwen3-4/8/14B; ds4/colibri dimostrano
  il valore di MLA/attention compressa (KV 57× più piccola) — pesa verso architetture
  con KV compressa nativa. Non decide: input per il ruling.
- **Piano di lavoro (sessione nuova)**: ordine suggerito dai fatti — modello di
  esecuzione (L1-L3, la tesi) → KV/prefix-cache OPFS (design pronto) → paging esperti
  (metodo 71.6% prima, meccanismo poi) → spec-dec (tap progettati in v0, build dopo).
