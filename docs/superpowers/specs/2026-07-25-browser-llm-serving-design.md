# Browser LLM Lab — Design

**Data**: 2026-07-25 · **Stato**: approvato (brainstorming con Cristiano)
**Domanda guida**: fino a dove ci si può spingere per servire un LLM interamente nel browser?

## Obiettivo e arco

Tre fasi, ognuna alimenta la successiva:

1. **Mappa + benchmark** — harness riproducibile che misura backend × modello × quant × device.
2. **Deep-dive tech** — smontare il percorso WebGPU di MLC fino ai kernel per spiegare *perché* i numeri sono quelli.
3. **Ceiling run + hero-demo** — il massimo ottenibile (tok/s e taglia modello) con una chat-demo come vetrina.

**Deliverable**: hero-demo come flagship; il benchmark pubblico condivisibile è un
interruttore da accendere dopo (vedi §9 Deferred). Espansione a beneficio della
community pianificata più avanti, dopo che il banco privato funziona.

## Device target (tre regimi di calcolo)

| Device | Regime | Cosa racconta |
|---|---|---|
| RTX 4090 mobile, Linux/Fedora (~16 GB VRAM) | dGPU discreta, memory-capped | il fastest tok/s |
| MacBook M4 Pro, 48 GB unificati (Metal) | grande capienza, banda minore | il modello più grande caricabile |
| Samsung S22 Ultra, 12 GB (Adreno, Android) | mobile, thermal/power-bound | il pavimento reale lato utenti |

Il "ceiling" di Fase 3 si biforca: tok/s max sulla 4090, taglia max sull'M4.

## Stack sotto test

- **WebLLM / MLC** (WebGPU, kernel TVM-compilati) — anche target del deep-dive Fase 2.
- **Transformers.js v3** (ONNX Runtime Web, WebGPU) — secondo percorso WebGPU indipendente.
- **wllama** (llama.cpp → WASM) — pavimento CPU/WASM (muro 4 GB di memoria WASM).

Confronto chiave: stesso base model, tre implementazioni di kernel diverse.

## Architettura

- **SPA statica** (Vite + TypeScript), tutto client-side, zero backend → hostabile su GitHub Pages.
- **`InferenceAdapter`** — interfaccia unica `load() / generate() / getLogprobs?() / capabilities() / dispose()`; tre implementazioni (`WebLLM`, `TransformersJs`, `Wllama`). Metriche, UI e demo parlano solo con questa interfaccia.
- `capabilities()` gestisce la non-uniformità (logits sì/no, streaming, worker): il modulo qualità la interroga e fa fallback da solo.
- **Tutto il compute in Web Worker** fin dallo scheletro → main-thread libero e UI-jank misurabile (deciso: non si bolt-on-a dopo).
- **Probe all'avvio**: limiti reali dell'adapter WebGPU (`maxStorageBufferBindingSize`, `maxBufferSize`, …) + device memory = riga-dati #0 di ogni run.
- Caching pesi: OPFS/Cache API; cold vs warm misurati separatamente.

## Set di modelli

Vincolo apples-to-apples: base model disponibile in tutti e tre i formati (MLC prebuilt · ONNX · GGUF). Famiglie: **Qwen2.5** e **Llama-3.2** (approvate).

| Fascia | Modelli | Dove gira |
|---|---|---|
| Tiny | Qwen2.5-0.5B | ovunque, incl. wllama sull'S22 |
| Small | Llama-3.2-1B · Qwen2.5-1.5B | tutti i device |
| Mid | Llama-3.2-3B · Phi-3.5-mini 3.8B | 4090 · M4 · (S22 al limite) |
| Large | Qwen2.5-7B · Llama-3.1-8B | 4090 · M4 (solo WebGPU) |
| Ceiling (F3) | Qwen2.5-14B → stretch 32B q4 | solo M4 48 GB |

Quant: il ~4-bit di default per stack (MLC `q4f16_1` · ONNX `q4` · GGUF `Q4_K_M`) + dove possibile un punto q8/f16 per la curva quant→qualità.

**[VERIFY] prima di congelare lo sweep**: disponibilità reale formato×quant sui repo attuali (HF Hub / context7). Buchi nella matrice ammessi ma loggati, mai silenziosi.

## Workload e definizioni metriche

- **TTFT/prefill**: prompt fisso 512 tok → tempo al primo token.
- **Decode tok/s**: 256 tok greedy, steady-state (escluso il primo).
- **Load+compile**: cold (OPFS vuota, include download) vs warm (cached), separati.
- **Peak memory**: dichiarata *stimata* — `performance.measureUserAgentSpecificMemory()` (richiede crossOriginIsolated) + `deviceMemory` + somma buffer allocati da noi. [VERIFY] copertura reale delle API per browser.
- **Qualità (leggera)**: perplexity su passaggio fisso *se* i logits sono esposti; fallback ~12 prompt deterministici (aritmetica, factual breve, format-following, task JSON) greedy + exact-match/regex. Stessi prompt su tutti i backend → delta di degradazione-da-quant per stack. [VERIFY] esposizione logits nei tre stack.
- **UX**: jank main-thread via long-task observer durante il decode.
- **Determinismo**: temp 0, seed fisso dove supportato, prompt-set congelato e versionato.

## Schema risultati e aggregazione

- Un JSON per device-run: `{ schemaVersion, deviceProbe, cells: [{stack, model, quant, metrics…}] }`.
- Aggregazione: notebook/pagina statica che legge `results/*.json` e produce i grafici comparativi. Stesso schema del futuro leaderboard pubblico → zero rework.
- Device identity = etichetta manuale (`4090-linux`, `m4pro`, `s22`), zero fingerprinting.

## Fasatura

- **1a — Scheletro**: SPA + worker + adapter WebLLM + probe + un modello end-to-end sulla 4090. Prova la pipeline.
- **1b — Matrice piena**: adapter Transformers.js + wllama; modulo metriche; sweep sui 3 device; export JSON.
- **2 — Deep-dive MLC**: compute shader, spezzettamento pesi sui buffer (storia del limite ~2 GB), kernel dequant, layout KV cache. Output: scritto "perché i numeri sono quelli" + eventuale micro-bench matmul.
- **3 — Ceiling + hero-demo**: i due ceiling + chat-skin sulla config vincente.

## Deferred (esplicitamente fuori scope ora)

- **Interruttore pubblico / community**: host su GitHub Pages + flusso "submit risultati via PR" verso `results/`, leaderboard statica. Nessun server, nessuna superficie d'abuso. Si attiva senza rework grazie a schema versionato + adapter interface. Espansione community da ripensare insieme quando il banco privato è maturo (richiesta esplicita di Cristiano, 2026-07-25).
- WebNN / MediaPipe come backend aggiuntivi.
- Eval qualità pesante (subset MMLU/GSM8K in-browser).

## Rischi e incognite

- WebGPU su Linux+NVIDIA (Fedora, Chrome): possibile richiesta di backend Vulkan/flag — rischio di setup in 1a.
- Logits non uniformi tra stack → fallback task-scoring già previsto.
- Memoria in-browser = stima best-effort, dichiarata come tale.
- Buchi matrice formato×quant → loggati.
- `maxStorageBufferBindingSize` (~2 GB su molti device) può cappare il vantaggio-capienza dell'M4 → trattato come finding, non solo rischio.
- wllama: tetto 4 GB WASM → limite duro sulla taglia modello.

## Testing

- **Conformance test per adapter**: stesso contratto per tutti (carica modello tiny, genera output deterministico, dichiara capabilities).
- Sanity sulle metriche (range plausibili) + unit test sulla matematica delle metriche in CI.
- I run di benchmark reali restano manuali (servono le GPU fisiche).
