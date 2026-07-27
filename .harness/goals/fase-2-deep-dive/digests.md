# Digests — fase-2-deep-dive

## Iterazione 0 (2026-07-27) — setup

- Goal spine scaffoldato; contratto in GOAL.md, spec in
  `docs/superpowers/specs/2026-07-27-fase-2-deep-dive-design.md`.
- 7 fasi sequenziali: skill → 4 doc sotto-sistema (dogfood skill dalla fase 2) →
  micro-bench matmul (unica fase con codice SPA) → engine-design-notes + closure sweep.
- Primo target: fase 1, skill `bottleneck-brainstorm` v1 via writing-skills.
- Docket-born: plan-check (#1, pre-autorizzato condizionalmente in chat); gli esperimenti
  di fattibilità (max 2) nasceranno docket-born dentro le fasi 2–6.
- Stop-by-design dopo fase 7: merge, run M4/S22, promozione skill, terzo esperimento —
  tutto PI-gated.

## Iterazione 1 (2026-07-27) — fase 1 done

- Skill `bottleneck-brainstorm` v1 creata via TDD (2 RED + 1 GREEN, artefatti in baseline/),
  commit d79bc32 su feat/fase-2-deep-dive. Done-when fase 1 verde.
- Scoperta metodologica: la baseline non fallisce in capacità ma in forma → skill = contratto
  di output (tabella 5 colonne + instradamento esperimento/engine-notes/scartata).
- 3 findings load-bearing gratis dai run di test (1 GiB hardcoded, shader-f16 inutilizzato
  su S22, KV sized sul max) — salvati in baseline/, alimentano fasi 3-5 e docket #2.
- Verifier: FAIL su bookkeeping alla prima passata, sanato in-iterazione; re-check ok.
- Prossimo: fase 2, doc compute-shader-dispatch.md (primo dogfood ufficiale della skill).

## Iterazione 2 (2026-07-27) — fase 2 done

- `docs/deep-dive/compute-shader-dispatch.md` completo: kernel WGSL nel wasm TVM, encoder
  condiviso (1 submit per molti kernel, 1 sync per token), zero timestamp-query → tempo
  GPU per-kernel invisibile, load warm ≈ compile pipeline (fetch/compile sequenziali).
- Primo dogfood in-fase della skill: contratto rispettato, nessun raffinamento necessario.
- Docket #3 nuovo: 3 candidati esperimento per 2 slot (q4f16 S22, multi-step decode,
  overlap fetch/compile) — scelta PI.
- Done-when fase 2 tutto verde (heading, citazioni, journal); 6 [VERIFY] ammessi → fase 7.
- Prossimo: fase 3, doc buffer-limit-2gb.md (baseline run-A già pronta come materiale).

## Iterazione 3 (2026-07-27) — fase 3 done

- `docs/deep-dive/buffer-limit-2gb.md` completo: il muro è a tre piani (adapter ~2 GiB /
  richiesta WebLLM hardcoded 1 GiB / fallback 128-256 MiB) e il runtime vive al piano di
  mezzo; il cap limita il singolo tensore, non la taglia modello; nessun run l'ha mai
  toccato. Bottleneck reale osservato: upload pesi serializzato con sync per-tensore.
- Ri-verifica del materiale baseline run-A: confermato quasi tutto, trovato in più il
  dettaglio sync-per-tensore (non era in run-A).
- Docket #4: quarto candidato esperimento per due slot — la scelta PI si fa più ricca.
- Prossimo: fase 4, dequant-kernels.md (materiale: run-C GREEN del dogfood).

## Iterazione 4 (2026-07-27) — fase 4 parte 1: dump WGSL

- 34 kernel WGSL REALI catturati dal run live 4090 (tool nuovo `tools/wgsl-dump.mjs`,
  patch nel worker via Playwright, zero modifiche SPA). Fonte primaria per fasi 4-5.
- 2 tentativi falliti prima del successo (dev server sbagliato; SwiftShader in headless
  → hang eterno nel warm-up) — entrambi diagnosticati e fixati, guardia isReal aggiunta.
- Schema q4 confermato dal vivo: nibble packing u32, offset-7, scale/32, FMA fusa.
- Prossimo (iterazione 5): scrivere dequant-kernels.md col materiale primario.

## Iterazione 5 (2026-07-27) — fase 4 done

- `dequant-kernels.md` completo, primo doc su fonte primaria (34 WGSL reali): schema q4
  confermato (nibble/u32, offset −7, scale/32, FMA fusa), due forme di kernel (GEMM tiled
  prefill / GEMV registri decode).
- FINDING CHIAVE: entrambi i device al 4-6% del roofline di banda pesi → sui modelli
  piccoli i tok/s misurano l'overhead pipeline, non il device. Swap q4f16 declassato a
  secondario (docket #5); il micro-bench fase 6 discrimina occupancy vs launch overhead.
- Prossimo: fase 5, kv-cache-layout.md (kernel 008/009/028 già nel dump + run-B baseline).

## Iterazione 6 (2026-07-27) — fase 5 done

- `kv-cache-layout.md` completo: layout pagina decodificato dall'indicizzazione del
  kernel reale (16 KiB/pagina, [K|V][head][pos][dim]); attention decode FlashAttention-
  style; KV f32 al tetto contesto (32k) = 768 MiB = 2.9× i pesi; run reali ~11 MB.
- Fatti inchiodati da fonte primaria HF (dogfood): chunk 2048 → prompt 469 = un unico
  mega-dispatch, chunking mai esercitato; sliding window spenta.
- Docket #6: QUINTO candidato esperimento (warm-up pre-ramp, attacca il #12 ereditato).
- Prossimo: fase 6, micro-bench matmul (unica fase con codice SPA) — 2-4 iterazioni.

## Iterazione 7 (2026-07-27) — fase 6 parte 1: micro-bench costruito + run 4090

- src/microbench/ completo (kernel GEMV q4/f32/f16 modellati sul dump, timestamp-query
  + fallback, schema v1 con skipped[], 12 test nuovi → 102/102, build ok).
- 2 bug veri trovati dal primo run e fixati: quantizzazione timestamp Chrome (fix: batch
  16 dispatch/campione) e device senza requiredLimits (celle garbage silenziose → ora
  error scope + skipped[] espliciti).
- Run valido: banda VRAM 4090 misurata ~435 GB/s oltre-L2 (vs ~576 datasheet); q4 vince
  in pesi/s 1.26× oltre-L2 (137G vs 109G — corretto dal verifier: il mio primo conto
  diceva 1.6×). Curva L2→VRAM visibile.
- Prossimo (iterazione 8): doc micro-bench-matmul.md con l'analisi + chiusura fase 6.
