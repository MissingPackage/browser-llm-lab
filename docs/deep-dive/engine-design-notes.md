# Engine design notes — appunti per un motore di inferenza browser custom

**Doc personale** (unico dei sei non pensato per pubblicazione as-is — ruling PI
2026-07-27). Sintetizza cosa il deep-dive di fase 2 implica per la progettazione di un
motore di inferenza browser da zero. Fonti: `compute-shader-dispatch.md`,
`buffer-limit-2gb.md`, `dequant-kernels.md`, `kv-cache-layout.md`,
`micro-bench-matmul.md` — più i kernel reali in
`.harness/goals/fase-2-deep-dive/wgsl-dump/` e i run in `results/` e
`results/microbench/`.

> **Continua in `docs/engine/estimates.md`** (sessione di stima 2026-07-28): il backlog
> qui sotto è qualitativo e le sue righe erano instradate sotto il vincolo di fase
> "niente rework del motore". Lì le stesse idee sono quantificate contro un modello di
> budget misurato — e il ranking cambia. Nota: il "~34 dispatch per token" usato dai doc
> di sotto-sistema è misurato **270** (vedi `results/dispatch-profile/`).

## I cinque fatti che vincolano il design

1. **A taglia piccola comanda l'orchestrazione, non i kernel.** Sulla 4090 il lavoro GPU
   utile per token è ~1-2 ms contro 8.6-9.9 misurati: 75-85% del budget vive *tra* i
   kernel (270 dispatch + 7 submit + 1 sync per token, misurati —
   `results/dispatch-profile/`). Un motore custom compra più throughput
   riducendo dispatch e sync che ottimizzando WGSL.
2. **La banda di targa non è la banda.** 4090 laptop: 576 GB/s dichiarati, ~435 misurati
   in streaming puro (−25%); e qualsiasi working set sotto la L2 (64 MB su AD103) misura
   la cache, non la memoria. Ogni roofline interno al motore va calibrato su banda
   *misurata*, con il micro-bench integrato, non su datasheet.
3. **I limiti WebGPU vanno negoziati, mai assunti.** Tre lezioni convergenti: WebLLM
   chiede hardcoded 1 GiB (metà dello spazio adapter sprecato); il nostro stesso
   micro-bench, alla prima stesura, ha creato il device coi default (128 MiB) producendo
   celle garbage silenziose; il fallback LLVMPIPE esiste davvero. Regola: `requiredLimits`
   dai limiti dell'adapter + error scope su ogni percorso di allocazione + validazione
   con checksum.
4. **Il 4-bit vince, ma il kernel dequant è lontano dal suo tetto.** q4 batte f32 in
   pesi/s (1.26× oltre-L2) però viaggia al ~20% della banda misurata contro il ~100%
   dell'f32: il costo ALU della dequant e gli accessi alle scale mangiano il vantaggio
   teorico del packing. C'è margine di kernel engineering vero (vec4 loads, scale in
   shared memory, layout stile Marlin/`mul_mat_q`).
5. **La telemetria va progettata dentro, non aggiunta dopo.** WebLLM non richiede mai
   `timestamp-query`: il tempo per-kernel è invisibile per costruzione. Entrambi i
   device del progetto la espongono. Un motore custom la richiede dal giorno zero, con
   fallback CPU e gestione della quantizzazione dei timestamp (batch di dispatch).

## Backlog di design (le idee instradate "engine-notes" dai quattro doc)

| Area | Idea | Da | Note |
|---|---|---|---|
| Dispatch | Sync coalescing adattivo (multi-step + telemetria) | `compute-shader-dispatch.md` | prerequisiti: multi-step decode + timestamp-query |
| Dispatch | Demand-paging dei kernel (compilare solo i toccati dal primo forward) | `compute-shader-dispatch.md` | quota kernel realmente usati: da misurare |
| Dispatch | Fork/patch del runtime per timestamp permanente | `compute-shader-dispatch.md` | nel motore custom è il punto 5 sopra, nativo |
| Dispatch | Persistent kernel / megakernel (un dispatch per forward pass) | `dequant-kernels.md` | prior art TensorRT-LLM (~14.6% launch overhead) e MPK; WebGPU non ha le primitive — da riesaminare quando/se arrivano |
| Buffer | Richiesta limiti = adapter limits (mai hardcoded) | `buffer-limit-2gb.md` | già regola del micro-bench |
| Buffer | Split tensori oversize + kernel di indirezione | `buffer-limit-2gb.md` | serve solo per embedding/lm_head giganti o MoE |
| Buffer | Staging-ring / double buffering per l'upload pesi | `buffer-limit-2gb.md` | forma ambiziosa del "diradare la sync" (docket #4) |
| Kernel | Split-K multi-workgroup per riga (occupancy GEMV) | `dequant-kernels.md` | CUTLASS/Flash-Decoding; overhead del secondo passo da misurare |
| Kernel | Layout coalesced stile `mul_mat_q` / Marlin | `dequant-kernels.md` | attacca il fatto 4 |
| KV | Contesto/sliding-window device-aware dal probe | `kv-cache-layout.md` | ABI già esistente in MLC; regressione funzionale da comunicare |
| KV | KV cache quantizzata (int8/int4) | `kv-cache-layout.md` | la leva teorica più grossa sul mobile (768 MiB → 192-384) |
| KV | Allocazione pagine lazy vs upfront sul max | `kv-cache-layout.md` | dipende dalla questione aperta eager/lazy in TVM |

## Questioni aperte (i marker di verifica dei doc, consolidati qui — da chiudere prima di progettare sul serio)

- Allocazione KV: eager sull'intero `max_total_sequence_length` o lazy per pagina?
  (sorgenti TVM/relax upstream.)
- Quota shader-compile del load warm e ruolo della cache pipeline del browser
  (strumentare `asyncLoadWebGPUPipelines`).
- Attribuzione della varianza TTFT mobile: DVFS vs contesa OS vs thermal (candidato
  esperimento docket #6; l'S22 espone timestamp-query, quindi è misurabile).
- ALU vs coalescing nel kernel dequant: distinguibile solo variando il kernel
  (vec4 loads, scale in shared) — naturale estensione del micro-bench.
- Breakdown per-kernel dentro il runtime WebLLM (fork strumentato del bundle).
- Bimodalità osservata nella cella q4 16384² del micro-bench (transizione di stato
  cache/clock): capire il meccanismo prima di fidarsi dei valori puntuali a quella taglia.

## La forma del motore, in tre frasi

Un motore browser custom che valga la pena di esistere non è "WebLLM ma più veloce sui
kernel": è un runtime che **minimizza i punti di sincronizzazione** (multi-step decode,
upload pesi pipelined, prefetch compile∥fetch), **si misura da solo** (timestamp-query
nativa, roofline su banda misurata, metriche in pesi/s), e **negozia ogni limite col
device reale** (limits dall'adapter, contesto dal probe, quant dal feature set — f16
dove c'è, KV quantizzata dove la memoria è unificata e stretta). I tre punti insieme
attaccano il fatto 1, che è dove vivono i 7 ms su 9 che oggi si perdono.
