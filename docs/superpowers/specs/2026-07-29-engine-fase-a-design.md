# Engine fase A — execution core: Design

**Data**: 2026-07-29 · **Stato**: **approvato** (ruling PI 2026-07-29, docket goal item 2; gate emendati da item 3-4)
**Direction**: `docs/engine/direction.md` · **Contratto**: `.harness/goals/engine-fase-a/GOAL.md`

## Obiettivo

Il motore decodifica Qwen2.5-0.5B-Instruct **Q4_0 da GGUF** nel browser, con parità
numerica verificata contro llama.cpp (oracolo) e — con le sole L1-L3 — decode più veloce
di WebLLM sulla 4090 (first-light). Tutto ciò che non serve a questo è fuori fase.

## Struttura

```
src/engine/
  gguf.ts          parser GGUF (header, metadata KV, tensor table) — puro, testabile CI
  quant.ts         layout Q4_0 + dequant CPU di riferimento (bit-exact) — puro
  shape.ts         shape Qwen2.5-0.5B hardcoded (24 layer, d=896, 14Q/2KV head_dim 64,
                   ffn 4864, vocab 151936) + validazione hard del GGUF: nome/tipo/shape
                   per tensore, mismatch ⇒ throw (postura ds4)
  plan.ts          piano statico: compila la lista di step al load — puro, testabile
  kernels/         WGSL come template string + specializzazione (pattern shader-lib
                   LlamaWeb: la funzione restituisce pipeline + metadata delle decisioni)
  runtime.ts       device init (requiredLimits ESPLICITI dall'adapter), esecuzione del
                   piano, submit, readback, tap
  telemetry.ts     contatori + timestamp ring — puro nella parte di aggregazione
  engine.worker.ts orchestrazione nel worker (OPFS/fetch del GGUF, load, loop decode)
tests/engine-*.test.ts   unit CI senza GPU (gguf, quant, plan, telemetry)
engine.html + src/engine/page.ts   pagina bench, STESSI id della SPA (#device-label,
                   #run, #status, #results, #probe-box) ⇒ driver e profiler riusabili
scripts/gen-golden.py    (uv, llama-cpp-python pinnato) → results/engine/golden/
scripts/conformance-engine.mjs   confronto motore-vs-golden via Playwright
```

Il motore NON tocca `src/` esistente (adapters/schema/bench intatti); condivide solo
`src/prof/profiler.ts` (riuso del patch contatori sulla pagina engine).

## Formato pesi

- **Fase A: Q4_0 + Q8_0 + F32** — corretto dopo il parse del file reale (2026-07-29):
  il GGUF ufficiale "q4_0" di Qwen tiene `output.weight` **separato e in Q8_0** (niente
  tied embeddings nel file, contrariamente alla config HF) e ha **bias F32 su q/k/v**
  (architettura qwen2); norme F32. K-quants/IQ arrivano in fase C col modello-tesi.
  Pochi formati = percorsi di test bit-exact per ciascuno.
- **Repack al load** (CPU, una volta): da blocchi Q4_0 a layout GPU-friendly — nibbles
  in u32 row-major + buffer scale separato (f16 lette come u16→unpack in WGSL: niente
  dipendenza da shader-f16), offset allineati a 256 B. La **dequant è fusa nel matmul**
  (mai materializzata), lezione unanime LlamaWeb/ds4/colibri.
- Attivazioni f32 (dev-loop 4090 senza shader-f16); percorso f16 = feature-detect,
  fuori fase A. `lm_head` = `output.weight` Q8_0 (kernel GEMV dedicato, stesso schema
  fuso del Q4_0 con loader di blocco diverso — pattern `mul_mat_decls` di LlamaWeb).
- Upload: streaming a chunk dal fetch/OPFS direttamente in `writeBuffer` (mai l'intero
  modello nel heap WASM/JS).

## Piano statico

Il cuore della tesi (L1-L3 by design, non by optimization):

- `plan.ts` produce al load una **lista piatta di step** `{pipeline, bindGroup,
  workgroups, uniformOffset}`: bind group creati UNA volta (L1 — indirizzi dei tensori
  statici), parametri per-step in un **uniform ring con dynamic offset** (un solo bind
  group layout per i param, zero `createBindGroup` e zero `writeBuffer` per-dispatch
  a regime; i param variabili per token — seq_len, posizione RoPE — vivono in un
  singolo `writeBuffer` per token).
- **Due piani compilati**: decode (M=1, GEMV) e prefill (M=chunk, GEMM naive in fase A
  — il first-light è sul decode; il prefill deve solo funzionare per PROMPT_512).
- **Un token = un encoder, un pass, un submit** (L2 — M2 ha dimostrato che i 7 submit
  di WebLLM sono 5 evitabili + 2 fondibili): la copy del token campionato verso lo
  staging buffer sta nello stesso encoder; `mapAsync` 1/token.
- **Budget dispatch (L3): ≤100/token.** Fusioni dichiarate, per layer: ①
  rmsnorm+QKV-proj (norm fusa nel matmul) ② RoPE+attention+concat (FlashDecoding-style,
  1 kernel) ③ O-proj+residual+rmsnorm-successiva ④ gate/up+SiLU·mul (dual-matmul fuso)
  ⑤ down-proj+residual ⇒ 5×24=120… −24 dalla fusione ③ che assorbe la norm del blocco
  FFN ⇒ **~96 + final-norm+lm_head+argmax (2-3) ≤ 100**. Il conteggio esatto è output
  di `plan.ts` e finisce nel JSON di telemetria (il verifier legge quello).
- **Sampling fase A = greedy**: argmax su GPU (riduzione a 2 dispatch inclusa nel
  budget), il solo id torna via staging. Niente sampler zoo (contratto).
- Rollback/multi-token (per fase B/D): il piano prefill accetta M≤8 — è lo stesso
  percorso che servirà alla verifica speculativa; nessun lavoro extra ora, solo il
  parametro non hardcodato a 1.

## Tap

Contratto DeepSpec recepito in v0: `createPlan({taps: number[]})` — per ogni layer in
`taps`, uno step `copyBufferToBuffer` (hidden post-layer → buffer dedicato per-tap)
inserito nell'encoder. Costo zero se `taps=[]` (nessuno step emesso — verifica: il
conteggio dispatch nel JSON non cambia). Decode: 896 f32/tap; prefill: M×896. Test
strutturale: forward con `taps=[11]`, buffer letto, shape corretta e non-zero.

## Telemetria

Nativa, always-on, **zero-overhead da spenta** (soglia contrattuale <1%, misurata da un
test A/B sul bench: 3 run on vs off, delta medio):

- Livello 0 (statico, gratis): dispatch/submit/bind-group count dal piano — nel JSON.
- Livello 1 (default): encode CPU per token (`performance.now` attorno al loop di
  encoding, 2 letture/token), ms per fase (load/prefill/decode), tok/s.
- Livello 2 (feature-detect `timestamp-query`): begin/end del pass per token → GPU ms
  per token (quantizzazione Chrome ~100 µs dichiarata nel JSON). Ring di QuerySet,
  resolve asincrona ogni N token, mai bloccante.
- Livello 3 (diagnostica, opt-in esplicito): un pass per step con timestamp per-op —
  **cambia la struttura di esecuzione, mai default, mai nei numeri pubblicati**
  (anti-pattern LlamaWeb documentato).

## Soglie di conformance

- **Corpus in token-id** (niente tokenizer in fase A — decisione: il tokenizer è
  rimandato a quando servirà testo libero; il confronto con l'oracolo è a token-id
  identici in ingresso): 4 prompt × 128 token generati = **512 token greedy**,
  committati in `tests/fixtures/engine-corpus.json`.
- **Golden**: `scripts/gen-golden.py` (llama-cpp-python pinnato, stesso file GGUF
  Q4_0, logits f32) salva per posizione: argmax id + top-32 (id, logit). File in
  `results/engine/golden/` (~qualche MB).
- **Gate (DECISO, ruling PI 2026-07-29 docket item 3, opzione A)**: **gate doppio** —
  **top-1 vs cpuref-f64 ≥ 99%** (parità vera, oracolo-indipendente; misurato 100%)
  **E top-1 vs golden llama.cpp ≥ 97%** (sanity; misurato 98.05%). Razionale: la
  calibrazione ha mostrato che anche la matematica esatta concorda col golden solo al
  98.05% (l'oracolo CPU quantizza le attivazioni a Q8 nel vec_dot: 10 near-tie
  identici, maxΔlogit 1.12) ⇒ il 99% secco era sopra il noise floor dell'oracolo.
  Riportati non gated: agreement@8 e max|Δlogit| sui top-32. Secondo golden:
  `results/engine/golden/cpuref-argmax-*.json`.
- **Unit bit-exact**: `quant.ts` dequant CPU vs blocchi di riferimento generati dal
  golden script (la dequant Q4_0 in f32 è esatta: qualunque diff = bug di layout).
- First-light (fase 6): protocollo bench del repo (warmup + 3 repliche, PROMPT_512,
  `results/engine/`), baseline WebLLM ri-misurata stesso giorno; confronto cross-quant
  (Q4_0 GGUF vs q4f32_1 MLC) **dichiarato nel JSON** — stessa classe 4-bit, byte
  toccati/token entro ~5%.

## Non-goals di fase A

Tokenizer proprio · OPFS store/paging (fase B/C) · MoE/MLA (fase C) · spec-dec oltre i
tap (fase D) · f16 compute · prefill veloce · UI oltre la pagina bench · multi-modello.

## Ordine di implementazione (= PHASES 3→4→5)

1. `gguf.ts` + `quant.ts` + `shape.ts` + fixture/unit (CI, zero GPU) → golden script.
2. `runtime.ts` + kernel naive (un dispatch/op, bind group per-dispatch, come WebLLM
   oggi) → **parità prima della velocità** (conformance exit 0).
3. `plan.ts` + uniform ring + fusioni + telemetria + tap → contatori L1-L3 verdi
   (profiler: bindGroup=0/decode, submit=1/token, dispatch≤100) e conformance ANCORA
   verde → first-light.

## Rischi specifici della fase

- Fusione attention in 1 kernel (②) è il pezzo WGSL più difficile: fallback dichiarato
  = attention a 2-3 dispatch (budget sale a ~120-130, resta sotto 270/2) — il gate ≤100
  andrebbe rinegoziato via docket, non silenziosamente. [ESITO: floor architetturale
  misurato 5 dispatch/layer = 123; ruling docket 4 (2026-07-29): gate fase A ≤130,
  ≤100 = target fase B (megakernel/fusioni cross-layer).]
- llama-cpp-python su Fedora: build locale con BLAS off (CPU pura basta per 512 token
  su 0.5B) — pinnata nel golden script con `uv run --with`.
- Il confronto cross-quant del first-light è il punto debole dichiarato del gate: la
  mitigazione (byte/token entro ~5%) è nel JSON, la narrativa nel report finale.
