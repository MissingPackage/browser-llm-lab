# Fase 1b — Matrice piena — Design

**Data**: 2026-07-26 · **Stato**: approvato (brainstorming con Cristiano)
**Contesto**: segue Fase 1b — fondamenta (mergiata in `main`, schema v2). Risolve il docket #7
("avvio 1b — matrice", HANDOFF.md). Deriva da
`docs/superpowers/specs/2026-07-25-browser-llm-serving-design.md` §Fasatura, riga "1b — Matrice
piena: adapter Transformers.js + wllama; modulo metriche; sweep sui 3 device; export JSON."

## Obiettivo

Aggiungere due nuovi stack di inferenza (Transformers.js, wllama) dietro l'`InferenceAdapter`
esistente, un modulo qualità-leggera che li confronta, ed eseguire lo sweep manuale sui 3 device
target (4090 già cablata, M4 Pro, Samsung S22 Ultra) sulle fasce modello Tiny/Small/Mid/Large.
La fascia Ceiling (14B→32B) resta assegnata a Fase 3, come da Fasatura originale — non in scope qui.

## Scope di questo round

- Adapter `TransformersJsAdapter` (Transformers.js v3 / ONNX Runtime Web).
- Adapter `WllamaAdapter` (wllama / llama.cpp → WASM).
- Modulo qualità-leggera (`quality.ts`), costruito dopo entrambi i nuovi adapter.
- Sweep manuale sui 3 device (stessa procedura già usata per la 4090 in Fase 1a/1b-fondamenta).
- Bump schema a v3 (`stack` union estesa, nuovo campo `qualityScore`).
- UI: selezione stack oltre a modello.

Fuori scope (deferred, non di questa fase):
- Automazione remota dell'esecuzione su M4/S22 (CI, device farm). Restano run manuali per ora;
  l'automazione è un passo successivo esplicito, da riaffrontare quando serve.
- Fascia Ceiling (Qwen2.5-14B→32B, solo M4): resta Fase 3 come da Fasatura originale.
- Soglie di pass/fail sul punteggio qualità: si riporta solo il numero grezzo.

## Architettura

- Due nuovi file, `src/adapters/transformersjs.ts` e `src/adapters/wllama.ts`, implementano
  l'interfaccia `InferenceAdapter` già esistente (`load() / generate() / capabilities() /
  dispose()`, più `getLogprobs?()` dove disponibile) — nessun cambio all'interfaccia stessa.
- `InferenceAdapter.id` e `BenchCell.stack` (schema.ts): union estesa da `"webllm"` a
  `"webllm" | "transformersjs" | "wllama"` — già anticipata come commento nel codice esistente
  (`schema.ts`, `adapters/types.ts`).
- Nuove dipendenze: `@huggingface/transformers` (Transformers.js v3) e `wllama`. Nessuna delle
  due è installata al momento di scrivere questo documento.
- Tutto il compute resta in Web Worker (`bench.worker.ts`), come già per WebLLM — nessun cambio
  al modello di esecuzione main-thread/worker.
- `capabilities().logprobs` decide quale percorso usa il modulo qualità (perplexity vs fallback
  a 12 prompt) — dispatch generico nel chiamante, non if/else per-stack sparsi nel codice.

## Matrice modello × stack × device

Vincolo ereditato dalla spec originale: stesso base model disponibile in più formati, famiglie
Qwen2.5 e Llama-3.2 (+ Phi-3.5-mini come modello Mid aggiuntivo, già nel set esistente).

Verificata la disponibilità reale dei formati su Hugging Face Hub (ricerca 2026-07-26):

| Fascia | Modello | WebLLM | Transformers.js | wllama |
|---|---|---|---|---|
| Tiny | Qwen2.5-0.5B-Instruct | ✓ | ✓ (`onnx-community`, fp32/fp16/int8/q4/q4f16/bnb4) | ✓ (Q4_K_M ≈0.37GB) |
| Small | Llama-3.2-1B-Instruct | ✓ | ✓ (`onnx-community`, gamma quant completa) | ✓ (Q4_K_M ≈0.75GB) |
| Small | Qwen2.5-1.5B-Instruct | ✓ | ✓ (`onnx-community`, gamma quant completa) | ✓ (Q4_K_M ≈0.92GB) |
| Mid | Llama-3.2-3B-Instruct | ✓ | ✓ (`onnx-community`, gamma quant completa) | ✓ (Q4_K_M ≈1.88GB, margine ampio) |
| Mid | Phi-3.5-mini-instruct (3.8B) | ✓ | ✓ (`onnx-community/Phi-3.5-mini-instruct-onnx-web`, **solo q4f16**) | ✓ (Q4_K_M ≈2.23GB, margine più risicato) |
| Large | Qwen2.5-7B-Instruct | ✓ | **✗** — nessun repo ONNX web-runnable (solo varianti vendor-locked AMD/NVIDIA/DirectML) | **✗** — Q4_K_M ≈4.36GB, supera il tetto WASM di 4GB sui soli pesi |
| Large | Llama-3.1-8B-Instruct | ✓ | **✗** — stesso motivo | **✗** — Q4_K_M ≈4.58GB, stesso motivo, margine peggiore |

**Gap noti, documentati non nascosti** (coerente con la spec originale, "buchi matrice
formato×quant → loggati"):
- **Fascia Large: solo WebLLM.** Gap strutturale (nessun formato disponibile per gli altri due
  stack), non un problema di tuning — non forzare quant più aggressivi per farcela entrare,
  degraderebbe la qualità sotto soglia utile.
- **Phi-3.5-mini su Transformers.js**: un solo livello di quant disponibile in ONNX web
  (`q4f16`), quindi nessun confronto quant→qualità possibile per questo modello su questo stack
  specifico (per gli altri modelli Mid/Small/Tiny la gamma quant completa resta disponibile).
- **wllama, fascia Mid**: margine di memoria più risicato di Tiny/Small (specialmente
  Phi-3.5-mini, ≈1.7GB di margine sotto il tetto WASM) — da tenere d'occhio con context lunghi
  o repliche multiple in sequenza.

Device: 4090 mobile (Linux/Fedora, già cablata — riuso del driver e2e esistente
`scripts/e2e-bench.mjs`), MacBook M4 Pro (Metal), Samsung S22 Ultra (Adreno/Android). Esecuzione
manuale su tutti e tre, stessa procedura già in uso per la 4090: build statica, run a mano nel
browser, export JSON, commit in `results/`.

## Modulo qualità-leggera

Nuovo modulo puro `src/quality.ts` (stesso stile di `metrics.ts`, nessuna dipendenza da
DOM/worker):

- Se `capabilities().logprobs === true`: perplexity su un passaggio di testo fisso.
- Altrimenti (fallback): nuovo set `src/qualityPrompts.ts`, ~12 prompt deterministici coprendo
  aritmetica, factual breve, format-following, task JSON — greedy, valutati via
  exact-match/regex. Stessi prompt su tutti gli stack/device per cui il fallback si applica.
- Punteggio riportato in `BenchCell.qualityScore`, grezzo (n/12 o valore perplexity) — **nessuna
  soglia di pass/fail**: la lettura comparativa resta a chi guarda i risultati aggregati, non
  al modulo stesso.
- Costruito **dopo** entrambi i nuovi adapter (serve avere ≥2 stack nuovi oltre a WebLLM per
  rendere il confronto significativo).

## Schema — bump a v3

Non retro-compatibile con v2 (stesso pattern già seguito per il bump v1→v2 in Fase
1b-fondamenta; i file v2 in `results/` restano storici).

- `BenchCell.stack`: union estesa `"webllm" | "transformersjs" | "wllama"`.
- `BenchCell.qualityScore`: nuovo campo,
  `{ kind: "perplexity", value: number } | { kind: "exact-match", value: number, total: 12 }`.
- Nessun altro cambio: `DeviceProbe`, `GenMetrics`/`GenMetricsAgg`, repliche restano come in v2.

## UI

Dropdown stack accanto al dropdown modello già esistente (`main.ts`/`render.ts`). Il dropdown
modello si filtra in base allo stack scelto, cosi da rendere impossibile selezionare a UI una
combinazione che ricade nel gap noto della fascia Large. Un run produce una cella (stack,
modello, quant, device-label inserito dall'utente) — stesso flusso attuale, non cambia il
modello "un bottone Run bench per combinazione".

## Testing

- **Conformance test per adapter** (già previsto dalla spec originale): stesso contratto per
  tutti e tre gli stack — carica un modello tiny, genera un output deterministico, dichiara
  `capabilities()` coerenti con l'implementazione reale (non hardcoded a priori).
- **Unit test puri su `quality.ts`**: matematica perplexity + scoring exact-match, in CI, nessuna
  GPU richiesta — stesso principio già seguito per `metrics.ts`.
- I run di benchmark reali restano manuali (servono le GPU fisiche) — invariato rispetto a oggi.

## Ordine di implementazione

1. `TransformersJsAdapter` (+ conformance test, + estensione schema/UI minima per farlo apparire).
2. `WllamaAdapter` (+ conformance test).
3. Modulo qualità (`quality.ts` + `qualityPrompts.ts` + bump schema `qualityScore` + unit test).
4. Sweep manuale sui 3 device sulle fasce Tiny/Small/Mid/Large (con il gap Large-only-WebLLM
   documentato, non forzato).

## Rischi e incognite riportate dalla spec originale (ancora aperte)

- Esposizione logits nei tre stack: da verificare in implementazione per ciascun adapter — se
  assente, il modulo qualità usa comunque il fallback a 12 prompt (già previsto, non blocca).
- Copertura reale delle API di stima memoria per browser (`measureUserAgentSpecificMemory`,
  `deviceMemory`): resta stima best-effort dichiarata come tale, invariato da Fase 1a.
- Automazione dell'esecuzione remota su M4/S22: deferred, da ripensare quando serve (non blocca
  questo round, che resta manuale).
