# Fase 2 — Deep-dive MLC/WebGPU: Design

**Data**: 2026-07-27 · **Stato**: approvato (brainstorming con Cristiano)
**Spec madre**: `2026-07-25-browser-llm-serving-design.md`, sezione Fasatura, punto 2.

## Obiettivo

Smontare il percorso WebGPU di WebLLM/MLC fino ai kernel, con doppio scopo:

1. **Materiale didattico** per la futura pagina benchmark pubblica: orientare chi legge i
   numeri ("perché sono quelli").
2. **Comprensione progettuale per Cristiano**, in vista di un suo futuro motore di inferenza
   per browser.

Vincolo qualificante (richiesta esplicita del PI, 2026-07-27): NON un esercizio di sola
lettura del codice. Ogni sotto-sistema studiato deve produrre anche i suoi **bottleneck
reali** e **idee concrete di aggiramento**, cercate con creatività e prior art cross-engine
(es. `antirez/ds4` — SSD streaming per sopperire alla VRAM —, llama.cpp, ONNX Runtime,
WebNN, letteratura). "Capire com'è fatto" e "capire come lo si supererebbe" viaggiano
insieme.

## Deliverable

Nuova cartella `docs/deep-dive/` (i doc di prodotto vivono qui; `docs/superpowers/specs/`
resta per gli spec di processo):

| File | Contenuto |
|---|---|
| `compute-shader-dispatch.md` | Come i kernel TVM-compilati arrivano a girare via WebGPU (`WebGPUContext`, pipeline creation, dispatch) |
| `buffer-limit-2gb.md` | Storia del muro `maxStorageBufferBindingSize` ~2 GB, spezzettamento pesi sui buffer |
| `dequant-kernels.md` | Kernel dequant `q4f16_1`: layout, costo, fusione col matmul |
| `kv-cache-layout.md` | Layout KV cache: allocazione, paging, impatto su prefill/decode |
| `micro-bench-matmul.md` | Harness micro-bench + risultati (vedi sotto) |
| `engine-design-notes.md` | Sintesi finale: implicazioni per un motore custom. Cross-referenzia tutti gli altri. Doc personale, separato dai doc pubblicabili |

**Scheletro comune** dei quattro doc di sotto-sistema:

1. *Cosa fa* — meccanica del sotto-sistema.
2. *Perché i numeri sono quelli* — ancorato ai dati reali in `results/*.json` e al codice.
   Fonte primaria del codice: bundle non-minificato
   `node_modules/@mlc-ai/web-llm/lib/index.js` (contiene il runtime tvmjs; es.
   `WebGPUContext` alla riga 4359 nella versione attuale) + sorgenti upstream MLC/TVM dove
   il bundle non basta.
3. *Bottleneck & vie d'uscita* — output del passaggio creativo (skill dedicata, sotto):
   limiti identificati + idee di aggiramento valutate su fattibilità/costo, con prior art.
4. *Esperimenti* — presente solo se un aggiramento è stato effettivamente provato con uno
   studio di fattibilità (terminologia scelta dal PI: "esperimento di fattibilità", non
   "PoC" — non è un prodotto; vivono in `experiments/`).

## Metodologia — ciclo per sotto-sistema (ripetuto 4 volte, in sequenza)

Ordine: shader dispatch → buffer/2GB → dequant → KV cache. Ogni doc si appoggia sul
precedente (approccio A, sequenziale per fondamenta, scelto dal PI).

1. **Lettura**: bundle + doc upstream TVM/MLC pertinenti (context7/repo, mai a memoria).
2. **Spiegazione**: scrittura ancorata ai run già in `results/`.
3. **Passaggio creativo**: invocazione della skill `bottleneck-brainstorm` (nuova, sotto)
   sul sotto-sistema appena capito.
4. **Esperimento di fattibilità dove sensato**: se dal passaggio creativo emerge un'idea
   promettente e a basso costo, piccolo esperimento/script sotto `experiments/` per
   verificare che regga. Non produzione, non integrato. Criterio: 1–2 esperimenti in tutta
   la fase, sulle idee migliori — non uno per doc.
5. **Scrittura del doc** e commit.

## Nuova skill: `bottleneck-brainstorm`

Decisione PI (2026-07-27): meccanismo creativo come skill dedicata, non riuso ad-hoc di
scout/peripheral-vision.

- **Input**: un bottleneck tecnico descritto (es. "storage buffer WebGPU cap a 2 GB").
- **Processo**: giro di prior-art cross-engine (llama.cpp, ds4/SSD-streaming, ONNX Runtime,
  WebNN, paper) + generazione idee di aggiramento; non "chi fa cosa" ma "cosa si può rubare
  e adattare al caso browser-specifico".
- **Output**: idee valutate su fattibilità/costo/rischio, pronte per la sezione
  *Bottleneck & vie d'uscita* del doc.
- **Costruzione**: con `writing-skills`, come primo lavoro della fase; il primo
  sotto-sistema la usa e la raffina (dogfooding). Riusabile per fasi future e altri
  progetti.

## Micro-bench matmul

- **Dove**: nuova pagina/route nella SPA esistente — riusa worker, probe e pattern di
  export già in piedi dalla fase 1b.
- **Cosa misura**: matmul dequant q4 vs f16, a dimensioni di buffer/matrice diverse.
  Timing via `timestamp-query` WebGPU dove la feature è disponibile (presente sulla 4090,
  vedi probe nei run), fallback a timing CPU-side altrimenti.
- **Output**: JSON per device-run, stesso pattern di `results/` — schema versionato, device
  label manuale, niente fingerprinting.
- **Device**: 4090 subito; predisposto per M4 Pro e S22 Ultra (stessa pagina, stesso
  schema) quando Cristiano li testa manualmente. I numeri cross-device entrano nel doc man
  mano che arrivano; i buchi si loggano, mai silenziosi.

## Testing e validazione

- **Nessun claim senza citazione**: ogni affermazione nei doc punta a una riga di codice,
  un run in `results/`, o una fonte upstream. Dove non verificabile: marcato `[VERIFY]`.
- **Micro-bench**: sanity check su range plausibili + unit test sulla matematica delle
  metriche, coerente con l'approccio di fase 1b. I run reali restano manuali.
- **Esperimenti**: ognuno dichiara esplicitamente cosa dimostra e cosa no (README per
  esperimento in `experiments/<nome>/`).

## Fuori scope (esplicito)

- Nessun rework del motore WebLLM reale; `src/adapters/webllm.ts` non si tocca.
- Gli esperimenti di fattibilità restano isolati, mai integrati nella SPA di bench (eccetto il
  micro-bench matmul, che è deliverable a sé).
- Nessun nuovo backend implementato (WebNN resta nei Deferred dello spec madre).
- Il motore di inferenza custom di Cristiano: `engine-design-notes.md` ne prepara il
  terreno, ma la progettazione vera è fuori da questa fase.

## Transizione

Lavoro project-sized (6 documenti, skill nuova, micro-bench cross-device, esperimenti,
multi-sessione): dopo l'approvazione di questo spec → `goal-brief` → `/goal` con
`PHASES.md` come roadmap. Non `writing-plans`.

## Rischi

- Il bundle di web-llm potrebbe non bastare per i dettagli kernel (WGSL generato a runtime
  da TVM): potrebbe servire dump degli shader via `createShaderModule` intercettato o i
  sorgenti mlc-llm upstream. Trattato come parte del lavoro, non blocco.
- `timestamp-query` non disponibile ovunque (S22 da verificare): il fallback CPU-side è
  meno preciso — dichiarato nei risultati, come per la memoria stimata in fase 1b.
- Scope creep sugli esperimenti: il tetto (1–2 in tutta la fase) è nel contratto; oltre quel tetto
  serve un ruling PI.
