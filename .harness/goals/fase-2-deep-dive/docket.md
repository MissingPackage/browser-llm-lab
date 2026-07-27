# Docket — fase-2-deep-dive

Decisioni PI-gated e questioni aperte. Append-only; le decisioni le prende Cristiano.

1. **plan-check** (aperto 2026-07-27, iterazione 0). Goal di prodotto → PHASES.md va
   approvato dal PI prima dell'iterazione 1. **Nota**: il PI ha pre-autorizzato in chat
   l'avvio del product-loop subito dopo il setup ("facciamo goal per fare il setup e una
   volta ottenuto tutto parti con il product-loop", 2026-07-27) — la tabella fasi è stata
   mostrata in chat nel digest di setup. Trattato come approvazione condizionale: il loop
   parte, ma qualunque obiezione del PI alla tabella riapre questa voce e la decomposizione
   si corregge via ruling prima di proseguire.

2. **Candidato esperimento di fattibilità: swap `q4f32_1` → `q4f16_1` su S22** (aperto
   2026-07-27, iterazione 1, emerso dal test GREEN della skill). Il probe S22 espone
   `shader-f16` ma la config bench fissa `q4f32_1` ovunque (`scripts/seq-bench.mjs:20`,
   `src/conformance/page.ts:26`). Costo quasi zero (un modelId), nessun tocco a
   `webllm.ts`; chiuderebbe il buco "quanto del gap 4090→S22 è banda intrinseca e quanto
   percorso f32 non necessario". Consuma 1 dei 2 slot esperimento del goal e il run S22 è
   manuale (mani di Cristiano) → decisione PI: approvare questo come esperimento #1?
   (La sezione completa con l'analisi è in `baseline/run-C-green-dequant.md`.)

3. **Candidati esperimento dal dogfood fase 2** (aperto 2026-07-27, iterazione 2). La
   sezione bottleneck di `compute-shader-dispatch.md` instrada come `esperimento`:
   (a) **multi-step decode** (accumulare 4-8 forward pass prima del readback del token —
   prior art vLLM +28%; rompe lo streaming percepito, guadagno da misurare);
   (b) **overlap fetch pesi ↔ compile pipeline** al load (oggi sequenziali per struttura,
   bundle 12549-12564; alto rapporto guadagno/costo se confermato, cronometrabile subito).
   La raccomandazione principale della sezione (timestamp-query) NON consuma uno slot:
   è già infrastruttura pianificata del micro-bench di fase 6 per spec. Con docket #2
   (swap q4f16_1 su S22) i candidati agli slot esperimento sono ora TRE per DUE posti
   → decisione PI: quali due approvare (o nessuno per ora).

4. **Candidato esperimento dal dogfood fase 3: diradare il `device.sync()` per-tensore
   nell'upload pesi** (aperto 2026-07-27, iterazione 3). Il loader fa una sync GPU
   completa per OGNI tensore (bundle 7086-7106) — centinaia di round-trip dentro la
   finestra di load warm misurata (1.5-1.9 s 4090, 6.1 s S22). Esperimento: build locale
   patchata del bundle vendored (nessun tocco a webllm.ts), criterio binario loadMs
   prima/dopo sullo stesso harness. Rischio noto da verificare: la sync potrebbe limitare
   il picco di memoria staging. **Con #2 e #3 i candidati sono ora QUATTRO per DUE slot**
   (q4f16 S22 · multi-step decode · overlap fetch/compile · sync-diradata upload) →
   decisione PI su quali due approvare.

5. **Re-ranking analitico dei candidati esperimento** (2026-07-27, iterazione 5, dal
   dogfood fase 4 coi dati del dump WGSL). Finding: entrambi i device stanno al 4-6% del
   roofline di banda pesi (misurato vs ~576/51 GB/s dichiarati) → sui modelli piccoli i
   tok/s misurano l'overhead della pipeline, non la banda. Conseguenze per la scelta slot:
   (a) il "micro-bench a taglie crescenti" raccomandato dal dogfood È il design della
   fase 6 (nessuno slot consumato); (b) **docket #2 (swap q4f16) declassato a secondario**
   — spiega al più 2× di un gap di 20×; (c) le due ipotesi da discriminare col micro-bench
   sono occupancy del GEMV (1 workgroup da 64 thread per riga output) vs kernel-launch
   overhead (~34 dispatch/token; prior art TensorRT-LLM: 14.6% su Qwen2.5-1.5B).
   Nessuna decisione presa: è input analitico per il ruling PI sugli slot (#2, #3, #4).

6. **Candidato esperimento dal dogfood fase 5: warm-up pre-ramp clock per il TTFT mobile**
   (aperto 2026-07-27, iterazione 6). Burst di compute scartato prima del timer TTFT,
   harness-level (zero tocco a webllm.ts), per falsificare/confermare l'ipotesi DVFS
   dietro la varianza 104% del TTFT S22 (docket #12 ereditato: high-variance non guarda
   il TTFT). QUINTO candidato per DUE slot; nota: attacca direttamente il tuo #12.
   Fatto rilevante emerso: prefill_chunk_size=2048 e prompt bench=469 tok → il chunking
   non è MAI stato esercitato da nessun run committato; tunarlo richiede prima un corpus
   prompt più lungo (dipendenza propedeutica registrata, nessuna azione).

--- AGGIORNAMENTI POST-CHIUSURA (2026-07-27, sera) ---

7. **Docket #2 ESEGUITO dal PI** (run manuale S22, `results/s22-ultra-2026-07-27T18-09-45-362Z.json`,
   committato su main 340ab91). Esito: decode 6.99 → **11.6 tok/s (+66%)**, TTFT
   8195±2792 ms → **2600±10 ms (−68%, varianza da 34% a 0.4%)**. Consuma lo slot
   esperimento #1 (salvo ruling contrario). Due conseguenze analitiche:
   (a) il +66% sta dentro la busta "≤2×" prevista dal re-ranking (#5) — il gap col
   roofline resta (S22 ora a ~6.4% del tetto);
   (b) **la varianza TTFT è COLLASSATA col percorso f16** → l'ipotesi DVFS pura del
   candidato #6 (warm-up pre-ramp) esce ridimensionata: la varianza era legata al
   percorso f32 (prefill 3.15× più lento → finestra termica/DVFS più esposta), non a un
   ramp-up intrinseco. **#6 declassato**; il ruling sul secondo slot va rifatto alla luce
   di questo.

8. **Primo datapoint M4 Pro** (`results/m4-pro-2026-07-27T16-36-08-710Z.json`, main).
   Fatti load-bearing:
   (a) decode webllm **98.3 tok/s ≈ la 4090** (101-116) con metà banda (~273 vs 576 GB/s)
   → conferma indipendente della tesi "a 0.5B comanda l'orchestrazione";
   (b) TTFT 214 ms < 4090 (290-344) e warm load **0.6 s** vs 1.5-1.9 s → su memoria
   unificata upload+sync per-tensore costano molto meno (ridimensiona il candidato #4
   sul device Apple, lo lascia intatto su 4090);
   (c) **il binding cap NON è una costante dell'API**: M4 espone maxStorageBufferBindingSize
   = 4294967292 (2³²−4, come maxBufferSize), non 2³¹−4 → falsifica la frase di
   `buffer-limit-2gb.md` "architetture che convergono sullo stesso valore = vincolo
   dell'API". Correzione del doc necessaria sul branch (fatta, vedi journal).
