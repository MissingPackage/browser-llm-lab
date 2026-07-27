# Micro-bench matmul — misurare i kernel, non l'aneddoto

Quinto doc di deep-dive (spec: `docs/superpowers/specs/2026-07-27-fase-2-deep-dive-design.md`).
Qui si descrive l'harness costruito in fase 6 (`src/microbench/`, pagina `microbench.html`)
e si leggono i primi numeri reali (4090). Domanda a monte, ereditata da
`dequant-kernels.md`: i tok/s misurati stanno al 4-6% del roofline di banda pesi — quanto
del gap è nei kernel e quanto nell'orchestrazione?

## Cosa misura e come (metodologia)

- **Kernel**: GEMV decode-shape `y[N] = W[N,K]·x[K]` in tre varianti — `gemv-q4f32`
  (pesi 4-bit packed, stesso schema dei kernel TVM dumpati: 8 nibble per u32, dequant
  `(w>>k & 15) − 7`, scale f32 per gruppo di 32, FMA fusa), `gemv-f32`, `gemv-f16` (se il
  device espone `shader-f16`). Un workgroup da 64 thread per riga di output, riduzione ad
  albero in shared memory — la *forma* del kernel di decode di TVM (`wgsl-dump/027`), non
  la sua copia byte-per-byte.
- **Taglie**: quadrate N=K ∈ {896, 2048, 4096, 8192, 16384} — da hidden di Qwen2.5-0.5B
  a taglie da 7-8B. La progressione attraversa deliberatamente il confine della L2.
- **Timing**: `timestamp-query` WebGPU (delta GPU-side, ns) con fallback CPU
  (submit→`onSubmittedWorkDone`). 10 campioni per cella dopo 3 dispatch di warm-up.
- **Dati**: deterministici (PRNG mulberry32 con seed fissi), packing q4 con scale per
  gruppo calcolate dal max assoluto — niente `Math.random`, run riproducibili.
- **Validità**: error scope WebGPU (validation + out-of-memory) e checksum dell'output
  per ogni cella; una cella non valida finisce in `skipped[]` con la ragione — buchi
  loggati, mai numeri garbage.
- **Export**: JSON versionato (`schemaVersion: 1`, `kind: "microbench-matmul"`) in
  `results/microbench/`, label device manuale, niente fingerprinting — stesse convenzioni
  di `results/`.

### Due lezioni metodologiche pagate col primo run (e ora nel design)

1. **Chrome quantizza i timestamp GPU** (mitigazione di timing, ~100 µs): un singolo
   dispatch GEMV piccolo sta sotto il quanto e il delta viene 0. Fix: **16 dispatch
   identici per campione** nello stesso compute pass (serializzati dal WAW hazard sul
   buffer di output), tempo = delta/16.
2. **Un device senza `requiredLimits` nasce coi default WebGPU** (binding 128 MiB): la
   cella f32 16384² falliva la validazione *in silenzio* producendo numeri impossibili.
   È lo stesso errore concettuale del cap hardcoded di WebLLM documentato in
   `buffer-limit-2gb.md` — riprodotto in casa nostra al primo tentativo. Fix:
   `requiredLimits` al massimo dell'adapter + error scope + checksum.

## Numeri 4090 (run committato)

Run: `results/microbench/microbench-4090-linux-2026-07-27T04-28-42-421Z.json`
(Chrome branded, adapter nvidia/lovelace, 10 celle valide, 0 skipped, tutte
`timestamp-query`). `shader-f16` **non esposto** da Chrome su questo stack Linux/NVIDIA:
variante f16 assente per skip loggato — slot da riempire dove la feature c'è (l'S22 la
espone nel probe).

| Kernel | N=K | ms/dispatch | GB/s effettivi | Regime |
|---|---|---|---|---|
| gemv-q4f32 | 896 | 0.0054 | 94 | L2-resident (0.5 MB) |
| gemv-f32 | 896 | 0.0066 | 483 | L2-resident (3.2 MB) |
| gemv-q4f32 | 4096 | 0.0501 | 210 | L2-resident (10 MB) |
| gemv-f32 | 4096 | 0.0674 | 996 | L2-resident (67 MB, borderline) |
| gemv-q4f32 | 8192 | 0.1741 | 241 | L2-resident (40 MB) |
| gemv-f32 | 8192 | 0.6207 | **432** | **oltre L2 (268 MB)** |
| gemv-q4f32 | 16384 | 1.9537 | 86 | **oltre L2 (168 MB)** |
| gemv-f32 | 16384 | 2.4639 | **436** | **oltre L2 (1.07 GB)** |

Tre letture:

1. **La banda VRAM reale della 4090 laptop è ~435 GB/s** (misurata, f32 streaming oltre
   L2) — il ~75% del dato di targa ~576 GB/s usato come tetto in `dequant-kernels.md`.
   Il roofline "onesto" va quindi già abbassato del 25%.
2. **La curva si piega esattamente sul confine della L2** (AD103: 64 MB): l'f32 sta a
   480-1000 GB/s finché il working set entra in cache e crolla a ~435 quando la sfora.
   Qualsiasi micro-bench che non attraversi quel confine misura la cache, non la memoria.
3. **Il q4 in "GB/s effettivi" sembra 5× peggio dell'f32 (86 vs 436), ma è la metrica a
   ingannare**: a parità di *pesi processati al secondo* — la grandezza che genera tok/s —
   il q4 oltre-L2 fa 137 G pesi/s contro i 109 G dell'f32: **1.26× a favore del q4**.
   Il confronto onesto è solo la riga 16384² (entrambi oltre L2): a 8192² il q4 (42 MB)
   è ancora in cache mentre l'f32 (268 MB) no, e il rapporto apparente (3.6×) è sleale.
   Caveat sulla cella q4 16384²: i 10 campioni sono **bimodali** (7 a ~2.36 ms, 3 a
   ~0.7 ms; stdev 44% della media) — probabile transizione di stato cache/clock tra
   campioni; l'1.26× è la media dei due regimi e la direzione della lettura regge in
   entrambi, ma il valore puntuale va preso con quella larghezza.
   Resta vero che il kernel q4 è molto più lontano dal tetto di banda del suo omologo
   f32 (86/435 = 20% vs 436/435 = 100%): il costo ALU della dequant (shift+mask+convert
   per nibble) e l'accesso a scale non coalescato mangiano banda — è la conferma
   sperimentale che il kernel dequant di forma-TVM è ALU-bound sul 4-bit, non
   memory-bound. `[VERIFY: attribuzione ALU vs coalescing distinguibile solo variando il
   kernel (es. vec4 loads, scale in shared) — fuori scope della fase, materiale per
   engine-design-notes.md]`

### Cosa dice sul gap del 4-6% (la domanda di `dequant-kernels.md`)

Stima per ordini di grandezza, tutta da numeri misurati qui e in `results/`: alle taglie
del modello 0.5B (matrici 896-4864), un GEMV isolato costa 5-50 µs. Un token di decode ne
lancia ~34 (proiezioni + attention + norm + sampling, `compute-shader-dispatch.md`): anche
contando generosamente ~100-200 µs di puro tempo-kernel per l'intero forward pass più il
lm_head (85 MB q4 su 896×151936 → ~0.9 ms a 94 GB/s), il **lavoro GPU utile per token è
nell'ordine di 1-2 ms contro i 8.6-9.9 ms misurati end-to-end** sulla 4090. Il grosso del budget per
token — indicativamente il 75-85% — non è nei kernel: è tra i kernel (launch overhead,
orchestrazione, sync per token). Coerente col prior art TensorRT-LLM citato in
`dequant-kernels.md` (~14.6% di launch overhead su GPU datacenter con kernel molto più
grossi: qui i kernel sono minuscoli e l'overhead relativo esplode). La conferma
kernel-per-kernel richiede il breakdown con timestamp-query dentro il runtime WebLLM —
instradato a `engine-design-notes.md` (fork del bundle), non a un esperimento di questa
fase.

## Slot cross-device

| Device | Stato | Note |
|---|---|---|
| 4090 laptop (Linux, Chrome) | ✅ run committato | vedi sopra |
| MacBook M4 Pro | **pending** (run manuale di Cristiano) | aprire `microbench.html` dal dev server, label `m4pro`, Export → `results/microbench/` |
| Samsung S22 Ultra | **pending** (run manuale di Cristiano) | il probe espone `timestamp-query` E `shader-f16`: attesi sia il timing GPU-side sia la variante f16 che sulla 4090 manca |

I buchi restano dichiarati finché i run non arrivano; il doc si aggiorna appendendo, senza
riscrivere la metodologia.
