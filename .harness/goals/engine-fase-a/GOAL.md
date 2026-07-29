GOAL: Il motore (fase A, execution core) decodifica Qwen2.5-0.5B q4 da GGUF nel browser
più veloce di WebLLM sulla 4090, con piano statico L1-L3, telemetria nativa e parità
numerica verificata contro l'oracolo llama.cpp.

<!-- Contract approved by PI 2026-07-29 (chat, goal-brief; tutti gli [ASSUMED] approvati
     così com'erano: OPFS+M2 dentro il goal; soglie numeriche marcate "da spec A" vanno
     fissate nella spec prima del codice; merge su main resta ruling PI).
     Direction: docs/engine/direction.md. Goal start tag: goal-engine-fase-a-start. -->

DONE WHEN (all measurable):
- Spec di fase A scritta (docs/superpowers/specs/<data>-engine-fase-a-design.md) e
  ruling PI di approvazione registrato nel docket di HANDOFF.md.
- `npm test` verde con i test del motore inclusi (unit su IR/piano/quant-decode CPU-side,
  in CI senza GPU — convenzione metrics.ts/quality.ts).
- Parità numerica: script di conformance contro llama.cpp (STESSO file GGUF, greedy,
  corpus fisso committato) esce 0 con top-1 agreement ≥ 99% (soglia esatta e metrica KL
  fissate nella spec A) su ≥ 512 token generati.
- First-light: run sulla 4090 (protocollo bench del repo: warmup + 3 repliche,
  PROMPT_512, driver HEADED) con decodeToksPerSec.mean del motore > della cella WebLLM
  baseline ri-misurata lo stesso giorno stesso device (q4f32_1; confronto cross-quant
  GGUF-vs-MLC dichiarato nel JSON) — entrambi i JSON committati in results/engine/.
- Struttura L1-L3 provata dai contatori, non a giudizio: run del profiler
  (src/prof/profiler.ts riusato sulla pagina bench del motore) con, in finestra decode:
  createBindGroup = 0, submit/token = 1, dispatch/token ≤ 100 (target L3, estimates §3).
- Telemetria nativa: il JSON di bench del motore contiene sezione telemetry per-fase
  (encode CPU, GPU ms via timestamp-query dove disponibile) prodotta SENZA cambiare il
  batching, e un test dimostra overhead ~zero da spenta (soglia <1%, da confermare in
  spec A).
- Tap hidden-states: forward con N tap configurabili; test strutturale che legge i
  buffer tap (shape corretta, non-zero) su un forward reale.
- Tool banda OPFS scritto + un run committato in results/.
- M2 di estimates (contatore call-site flushCommands sul bundle WebLLM) eseguito e
  numeri annotati in estimates.md.

EVIDENCE OF DONE: npm test; node scripts/conformance-engine.mjs (exit 0 + report);
JSON in results/engine/ (motore + baseline WebLLM stesso giorno); JSON profiler con
contatori; diff estimates.md per M2; ls results/opfs-bench/.

AUTHORITY GRANTED:
- may do autonomously: branch engine/fase-a e commit/push sul branch; creare
  src/engine/**, tests/**, tools harness, prof/bench pages dedicate; run locali sulla
  4090; download GGUF/tokenizer da HF (spazio disco, zero spesa); aggiornare
  docs/engine/* quando stale (ruling 2026-07-29); docket/HANDOFF refresh.
- must docket (never do): merge di engine/fase-a su main (il gate first-light passato =
  candidatura al merge, il merge resta ruling PI); modifiche a direction.md §3 (rulings)
  o alla scala modelli; modifiche a src/adapters/** o allo schema results v3 della SPA
  esistente; run che richiedono M4/S22 (mani PI); ogni spesa; delete di codice
  committato >30 giorni.

CONSTRAINTS: zero attribution AI nei commit; llama.cpp SOLO oracolo (mai linkato/
vendored nel motore); erasableSyntaxOnly; landmine HANDOFF §4 (HEADED=1, timestamp
quantizzati, requiredLimits espliciti, shader-f16 assente su Chrome/Linux → f32-first);
spec-first per ogni sotto-fase; telemetria zero-overhead da spenta (lezione colibri);
tap progettati in v0 anche se consumati dopo (vincolo DeepSpec).

WORKING PROTOCOL: follow skills loop-iteration + done; verifier gate per cycle; digest
every cycle; stop-by-design when the remaining work is docket-gated.

CONTEXT ANCHORS: HANDOFF.md; docs/engine/direction.md (la direzione, §7/§9);
docs/engine/estimates.md (§3 leve, §8 misure cross-device); docs/engine/study/README.md
→ llamaweb.md (pattern da copiare: arena uniform, scratch oltre-dst, mul_mat_decls;
anti-pattern da evitare); results/dispatch-profile/ (baseline contatori).
