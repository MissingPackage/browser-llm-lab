GOAL: engine-fase-q1 — Il motore esegue il path testo della famiglia Qwen
3.5/3.6 (denso 2-4B mobile-target + denso 9B + MoE 35B-A3B) con la stessa
fedelta' bit-verificata del metodo GLM (oracoli cpuref + golden llama.cpp +
firma routing), con riferimenti decode/prefill/TTFT misurati ai budget
consumer 8/12/16 GB piu' il tier mobile emulato, e il gap nativo e'
DECOMPOSTO kernel-vs-paging tramite confronto full-residency a parita' di
modello, con le leve kernel ordinate per ROI misurato.

<!-- CONTRATTO v1 (chartered 2026-08-10, goal-brief in chat). Perimetro dal
     ruling PI docket c3c item 9/10 ("vai" 2026-08-09) + conferma WP gap
     2026-08-10 + emendamento PI 2026-08-10: terza taglia densa 2-4B
     aggiunta (target dispositivi mobili). Tutti gli [ASSUMED] della bozza
     APPROVATI dal PI in chat ("per il resto tutto ok") — restano marcati
     nel testo come tracce della negoziazione; ratifica formale in spec,
     pattern c3a item 3 / c3b item 4.
     Vincolo di valle registrato: i numeri del writeup si rimisurano al tag
     di release (paper-contract-draft, ruling 2026-08-10) — q1 produce
     riferimenti datati con hostState, non i numeri del paper.
     Goal start tag da creare all'avvio: goal-engine-fase-q1-start. -->

DONE WHEN (all measurable):
- Spec q1 scritta (docs/superpowers/specs/<data>-engine-fase-q1-design.md) e
  registrata a docket PRIMA del codice (regime di approvazione C3b/C3c:
  ruling PI bloccante solo se tocca gate o soglie). La spec fissa: taglie
  definitive e GGUF pinnati via SHA [ASSUMED APPROVATO: denso 2-4B
  mobile-target (scelta puntuale 2B vs 4B in spec, criterio = budget
  memoria mobile) + Qwen3.5-9B denso + Qwen3.6-35B-A3B, quant Q4 famiglia
  unsloth — 3.6 sul MoE perche' refresh del tier consumer, recon §1/§5];
  piano numerico DeltaNet (f32 obbligato da config, strategia di verifica
  della ricorrenza); protocollo di conformance per-modello; stato ATTUALE
  di subgroup-matrix nei browser (il [VERIFY] dello studio llamaweb si
  chiude qui, in spec, non dopo).
- Reader GGUF `qwen3_5`/`qwen3_5_moe` + tokenizer famiglia: token id
  IDENTICI all'oracolo llama.cpp su un corpus di conformance committato
  (text-only; mrope_section gestita anche in text-only, recon §7.2) —
  ktest/test dedicato PASS, exit 0.
- Kernel DeltaNet WGSL + path GQA variante (head_dim 256, partial rope
  0.25, attn output gate): conformance NUMERICA per-layer vs cpuref-f64 sul
  campione ratificato (argmax ==), sui TRE modelli. Il kernel e' il rischio
  dominante dichiarato: la spec ne fissa i gate intermedi.
- Golden per-modello: top-1 vs llama.cpp su corpus full >= soglia FISSATA
  alla prima run full verificata e mai piu' abbassata (pattern ratchet c3b
  item 1) [ASSUMED APPROVATO: nessun import del 98.828% — quello e' il PIN
  di GLM, ogni modello fissa il suo].
- WP DECOMPOSIZIONE GAP: il 9B denso FULL-RESIDENT (Q4 ~5.5 GB in 16 GB) vs
  llama.cpp Vulkan STESSA macchina/driver/GGUF, stesso protocollo p512/n64:
  JSON committato in results/engine/ con entrambi i lati + ratio
  decode/prefill; il gap KERNEL cosi' isolato dal gap PAGING e' riportato in
  un doc di studio con la scomposizione esplicita (kernel / dispatch /
  safety-check / paging) e le leve ordinate per ROI misurato.
- Leve kernel BOUNDED [ASSUMED APPROVATO il confine]: dot4I8Packed e tuning
  tile per-device implementati DIETRO FLAG con delta misurato SE il WP li
  conferma in cima al ROI (default = path storico, non-regressione
  intatta); subgroup-matrix = SPIKE dietro flag Chromium, misura-only, MAI
  path di default in questo goal; WASM-SIMD compute-at-data NON in questo
  goal (resta registrata, direction).
- MoE 35B-A3B nel regime C3c: meccanica parametrizzata (256 expert top-8,
  slot ~1.7 MB, slotTable/classi non piu' GLM-shaped), recall del prefetch
  RIMISURATO sul router 256-wide (il 91.92% di GLM non si assume — recon
  §7.3), bandmodel rifittato coi punti nuovi; run ai budget consumer
  8/12/16 GB [ASSUMED APPROVATO: emulati con cap di budget slab sulla 4090,
  come i b12/1472/736 di C3c — hardware reale 8-12 GB e' PI-gated] con JSON
  committati: decode/prefill/TTFT per tier + gap dalla soglia UX riportato.
  [ASSUMED APPROVATO: NESSUN floor assoluto per la famiglia nuova in q1 —
  i floor si negoziano coi numeri della prima misura, docket alla spec; il
  floor 13.43 resta di GLM.]
- Tier MOBILE (emendamento PI 2026-08-10): il denso 2-4B misurato a un
  budget stretto emulato definito in spec (proxy del regime mobile:
  VRAM/banda da dichiarare, non da assumere) con JSON committato
  decode/prefill/TTFT; l'esecuzione su DISPOSITIVO mobile reale (S22 o
  altro) resta PI-gated a docket — qui si produce il riferimento e la
  proiezione, non la demo.
- Prefill/TTFT dentro il goal: ogni bench JSON riporta prefill e TTFT; il
  collasso in scarsita' (30→4 tok/s su GLM) e' RIMISURATO sulla famiglia
  nuova ai budget stretti, con attribuzione nel doc di studio (il TTFT
  prefill-bound e' il primo numero dei device piccoli — landmine C3c).
- Non-regressione GLM permanente: riferimenti 2026-08-09 invariati in banda
  ±5% (b12 optimistic 13.172/31.26/14.74; sync in banda), golden GLM
  98.828% AL PIN, cpuref 256+512, firma routing esatta, Qwen2.5 esistente in
  banda, ktest tutti PASS, npm test verde, npx tsc --noEmit pulito — a OGNI
  merge, non solo in chiusura.
- Chiusura: docket q1; direction aggiornata (sezione generalizzazione coi
  numeri); ledger; HANDOFF refresh; riferimenti datati con hostState
  dichiarato pronti come input del goal writeup (che li RIMISURERA' al tag
  di release — non sono i numeri del paper).

EVIDENCE OF DONE: file spec + entry docket; ktest/test tokenizer e DeltaNet
PASS in npm test; JSON conformance per-modello (cpuref + golden + firma) in
results/engine/; JSON confronto full-residency noi-vs-llama.cpp + doc di
studio della decomposizione; JSON bench per tier mobile+8/12/16 con
hostState; JSON recall prefetch 256-wide + fit bandmodel; run
non-regressione GLM fresche ad albero congelato; diff di direction + ledger
+ HANDOFF.

AUTHORITY GRANTED:
- may do autonomously: come C3c (src/engine/**, tests/**, tools/**,
  scripts/**, docs/engine/** quando stale; commit/push su main a iterazione
  VERIFICATA; merge su main a goal chiuso e verificato; run locali su 4090;
  artefatti in ~/.cache) PIU': nuovi file modello/kernel per la famiglia
  Qwen (qui sono l'oggetto del goal); download GGUF da HF (repo ufficiali o
  unsloth, SHA pinnata in spec) fino a ~60 GB di disco [ASSUMED APPROVATO];
  build/run llama.cpp locale SOLO come oracolo e baseline (regola
  esistente).
- must docket (never do): promuovere una leva kernel a path di DEFAULT;
  implementare WASM-SIMD compute-at-data; fase D / MTP / spec-dec (il draft
  MTP resta in canna, non si tocca); fissare floor per la famiglia nuova
  senza ruling; toccare il modello-tesi GLM oltre la non-regressione;
  quant nuove; benchmark (fase-1b/fase-2 in STANDBY deliberato, non
  riaprire); M4/S22/hardware nuovo — incluso l'ANDARE sul dispositivo
  mobile reale per il tier 2-4B; spese; azioni di publishing (split,
  upload, licenza pesi — sequenza a valle, split-plan); delete di codice
  committato <30 giorni.

CONSTRAINTS: le standing di C3c (spec-first; non-regressione permanente;
bench quiescenti o hostState DICHIARATO; albero congelato + 60 s fra run
GPU; parita' di protocollo; llama.cpp SOLO oracolo/baseline; f32-first —
e `mamba_ssm_dtype: float32` e' obbligo di config, non scelta; near-tie mai
gateati; determinismo; no pipe sui runner; full-corpus solo per
firma/nonreg/riferimenti; zero attribution AI) PIU': recall/hit-rate GLM
NON si assumono sulla famiglia nuova — si rimisurano; le leve kernel nel
writeup si REGISTRANO, non si promettono; ogni numero destinato al paper
nasce con hostState + data (verra' rimisurato al tag di release).

WORKING PROTOCOL: follow skills loop-iteration + done; verifier gate per
cycle; digest every cycle; stop-by-design when the remaining work is
docket-gated.

CONTEXT ANCHORS: HANDOFF.md; .harness/goals/engine-fase-c3c/{GOAL,docket}.md
(item 9/10 = il ruling di perimetro; item 6 = eusage/prefill, candidata a
entrare qui; item 8 = M4 PI-gated);
docs/engine/study/2026-08-09-qwen35-family-recon.md (la mappa e i prezzi);
results/engine/native-baseline-llamacpp-vulkan-2026-08-09.json;
docs/engine/study/llamaweb.md (kernel literature + il [VERIFY]
subgroup-matrix da chiudere in spec); docs/engine/direction.md §7/§8;
docs/publishing/{split-plan,paper-contract-draft}.md (vincoli di valle);
src/engine/{residency,bandmodel}.ts + glmmodel.ts (la meccanica da
parametrizzare); prior art DeltaNet: ggml (CPU/CUDA qwen3_5) + kernel
Triton ufficiali.
