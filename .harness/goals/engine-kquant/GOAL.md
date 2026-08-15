GOAL: engine-kquant — il GEMM di prefill smette di rileggere M volte i pesi
non-q4_0: la famiglia K-quant (Q5_K, Q4_K, Q6_K) e Q8_0 ricevono la forma
multi-riga con riuso vero, misurata al banco; sul Qwen3.5-4B `ssm_out` Q5_K e
`ffn_down` Q4_1 entrano in produzione, e il tempo al primo token a caldo scende
sotto la barra che `engine-ttft` ha mancato.

<!-- CONTRATTO v1 (chartered 2026-08-14, PI in chat: «Confermo il focus sui K
     quant. Q4_1 dentro. Per il ttft proporrei una doppia barra: 22500 e 18000.
     La seconda come nice to have. Se non si raggiunge va bene anche la prima.
     è un extra effort. Ok sulla barra del segmento.»)

     PROVENIENZA. Il goal non nasce da un'intuizione: nasce da una misura del
     goal precedente. `gemm:deltanet-out` e' il PRIMO termine del prefill —
     12.169 ms su 32.101, il 37,9% — con soli 24 dispatch (uno per layer
     DeltaNet) perche' `ssm_out` e' Q5_K e `gemvB` (q35gpumodel.ts:778) lo manda
     sul fallback legacy: M gemv replicate su `wid.z`, riuso pesi ZERO, 16
     riletture per chunk, 89,9 GB/s su un motore che ne ha dimostrati ~300.
     L'item 19 del docket `engine-ttft` aveva registrato questo residuo come una
     CODA («l'11,54% dei byte resta sul percorso vecchio»): la quota di BYTE
     sottostimava il peso proprio perche' la forma legacy rilegge i pesi M
     volte. E' la leva piu' grande rimasta sul TTFT.

     ARITMETICA DELLA LEVA — tutta da artefatti in albero, ricalcolabile senza
     GPU (inventario pinnato in tests/engine-prefillgemmplan.test.ts:313,
     M = PREFILL_M = 16):
       one-pass totale     2.046.935.040 B
       coperti oggi (F)    1.810.759.680 B = 88,463%   → 172/248 siti
       legacy oggi  (L)      236.175.360 B = 11,537%   →  76/248 siti
         · 24 × ssm_out  Q5_K  K=4096 N=2560  173.015.040 B = 8,453%
         ·  4 × ffn_down Q4_1  K=9216 N=2560   58.982.400 B = 2,881%
         · 48 × ssm_alpha/beta Q8_0 K=2560 N=32  4.177.920 B = 0,204%
       rapporto = M·(F+L)/(F+M·L):
         oggi                5,8593×   (pinnato, test [6c])
         + Q5_K             10,938×
         + Q5_K e Q4_1      15,524×    ← lo scope di questo goal
         + tutti e tre      16,000×    (copertura 100%)

     E IN TEMPO, dai due segmenti misurati nel checkpoint del 2026-08-14:
       gemm:deltanet-out  12.169 ms — 1.093 GB (395 chunk × 173.015.040 × 16)
         395 e non 396: ceil(6333/16) darebbe 396, ma il checkpoint dichiara
         "chunks": 395 (6320 token). Il numero si legge dall'artefatto, non si
         ricalcola — correzione del verificatore, it.1.
         a multirow: 68,5 GB. ALLA STESSA banda misurata oggi (89,9 GB/s, cioe'
         senza assumere NESSUN miglioramento per byte) ~760 ms ⇒ −11,4 s
       gemm:ffn-down       4.971 ms — 520,9 GB, di cui 373,7 (il 71%) sono le 4
         Q4_1 rilette 16 volte e 147,2 le 28 q4_0 gia' multirow
         a multirow: 170,6 GB ⇒ ~1,6 s alla banda aggregata di oggi ⇒ −3,3 s
       TOTALE proiettato −14,7 s ⇒ prefill 32,1 → ~17,4 s.
     Il conto torna anche dall'altro lato: 9.350 ms fuori dai pass GPU (invarianti
     per questa leva) + 8,05 s di pass residui = 17,4 s. E' anche il PAVIMENTO di
     questo goal: sotto i ~9,4 s non si scende togliendo byte ai pesi.

     IL 35B, e perche' NON e' qui dentro. Il PI: «il vero target è il 35B (e in
     futuro modelli anche più grandi). Se non lo facciamo in questo goal, allora
     sarà la prossima priorità.» Il 35B non ha UN BYTE di q4_0 (header dump
     2026-08-10: expert Q4_K 17,67 GB · Q6_K 0,66 · attn Q8_0 1,09 · linear_attn
     Q8_0 0,27 · head Q8_0 0,54): la via veloce oggi ne copre lo 0%. Ma il suo
     collo non e' il kernel — 17,7 GB di expert non stanno in 16 GiB di scheda
     (residency-bound), il suo prefill NON passa dal piano del 4B e non e' mai
     stato rimisurato dopo i kernel nuovi, quindi non esiste nemmeno la baseline
     da cui partire.
     [REFUSO CORRETTO in it.8, come il "K=2560" di it.2 — non e' una modifica di
     scope.] Il contratto diceva che il prefill del 35B «gira su un piano DIVERSO
     (`moeprefillplan.ts`)». E' FALSO, e verificato: `planMoeChunk` ha un solo
     consumatore di produzione, `glmmodel.ts:1368` — il GLM, non il 35B. Il 35B
     ripete per riga la catena del DECODE (`q35gpumodel.ts:2743-2778`, ramo
     `moe`): readback CPU dei router logits per ogni riga, `pinUnion` che calcola
     gia' l'unione ma solo per pinnare gli slot, poi un `for m2` con
     `prepLayer` + `encodeExperts` per riga. Chi charter-a il goal 35B parta da
     qui: `moeprefillplan.ts` e' gia' parametrico su `{nExpert, nExpertUsed}` e
     `{256, 8}` lo soddisfa per struttura, quindi il piano CPU-side NON e' il
     lavoro — il lavoro e' il ramo `moe` di `q35gpumodel.ts`.
     DECISIONE MIA (meccanismo e ordine, non funzione obiettivo): la FAMIGLIA DI
     KERNEL si fa qui e per intero — Q5_K, Q4_K, Q6_K e Q8_0, tutte misurate al
     banco e verificate col ktest; il CABLAGGIO e la misura end-to-end del 35B
     sono il goal successivo, che sara' wiring + residency e non lavoro di
     kernel. Cosi' il goal 35B parte da forme MISURATE, che e' la regola che
     questo motore applica da tre goal, e questo goal conserva UNA condizione di
     completamento invece di due. -->

DONE WHEN (all measurable):

- FASE 0 SUPERATA PRIMA DI QUALUNQUE CABLAGGIO: micro-bench isolato delle forme
  candidate multi-riga (banco `src/microbench/ttGemm.ts` esteso, zero run di
  modello), JSON committato in `results/microbench/` con G pesi/s e GB/s per
  variante, a M = 1, 8, 16, su TUTTE E QUATTRO le famiglie e sulle shape REALI:
    Q5_K  K=4096 N=2560   (4B `ssm_out`)
    Q4_1  K=9216 N=2560   (4B `ffn_down` dei layer 0-3)
    Q4_K  [2048, 512] e [512, 2048] — expert gate/up e down del 35B
    Q6_K  [512, 2048] — expert down di 3 layer del 35B
    Q8_0  [2048, 4096] — attn q-proj del 35B (1,09 GB: la seconda famiglia
          per byte la' dentro)
    (Le shape del 35B vengono dall'header dump 2026-08-10: dModel 2048,
     dFfnExpert 512, nExpert 256. Il contratto scriveva "K=2560", che e' il
     dModel del 4B — refuso corretto in it.2, non una modifica di scope.)
  REGOLA DI STOP, ereditata testuale dai due goal precedenti: se per una
  famiglia nessuna variante supera la forma legacy attuale di >= 1,5×, quella
  famiglia si chiude col numero e non si cabla. Una famiglia sotto la regola di
  stop NON fa fallire il goal: fa fallire il suo cablaggio, ed e' registrata.

- COPERTURA DEL 4B, verificabile in CI SENZA GPU: il rapporto pinnato del test
  [6c] (`tests/engine-prefillgemmplan.test.ts`) sale da **5,8593× a >= 15,5×**
  sull'inventario per-layer INTERO a M=16, con le 24 `ssm_out` Q5_K e le 4
  `ffn_down` Q4_1 fra i siti `multirow` e non piu' fra le `exceptions`.
  Il test [6d] — che esiste come promemoria e dice di se' «se un giorno la
  copertura sale, questo test fallisce e va cancellato con la sua ragione» — va
  cancellato con quella ragione scritta, non aggirato.

- I 48 SITI Q8_0 `ssm_alpha`/`ssm_beta` DEL 4B SONO ESCLUSI COI NUMERI, non
  dimenticati: 0,204% dei byte e **N=32 righe di uscita** contro le 64 per
  workgroup della forma split-K (mezzo workgroup per dispatch). L'esclusione
  vale per QUESTE shape, non per la famiglia: sul 35B il Q8_0 e' 1,09 GB con N
  grande, ed e' per questo che la sua forma si misura comunque in fase 0. Se la
  misura smentisce l'esclusione, entrano.

- SEGMENTI, sullo stesso JSON `kind: "q35-ttft-kernel-checkpoint"` prompt-idx 0,
  con byte del segmento e GB/s effettivi pubblicati accanto:
    `gemm:deltanet-out` **<= 2.000 ms**  (oggi 12.169; proiezione ~760)
    `gemm:ffn-down`     **<= 2.000 ms**  (oggi 4.971; proiezione ~1.600)
  Le barre stanno 2,6× e 1,25× sopra la proiezione a banda INVARIATA: lasciano
  margine a un unpack K-quant piu' costoso per byte senza regalare il risultato.

- TTFT A CALDO sul prompt-idx 0 (6333 token), `prefill.ms + decode.firstMs`,
  **DOPPIA BARRA (ruling PI 2026-08-14)**:
    (a) **< 22.500 ms — LA BARRA DEL GOAL.** Mancarla e' mancare il goal.
    (b) **< 18.000 ms — NICE TO HAVE, extra effort.** Non mancarla non fa
        fallire il goal; raggiungerla si dichiara nel consuntivo con la misura.
  La proiezione e' ~17,4 s, cioe' la (b) sta ESATTAMENTE sulla proiezione: e' un
  bersaglio di ottimo, non di sicurezza. La (a) ha ~5 s di margine, molto sopra
  il rumore misurato fra run (±2,4%).

- PREFILL: `prefill.tokS > 282` sul prompt-idx 0 — la barra (a) vista dal lato
  del rate (6333 token in < 22,5 s; oggi 197,25). Cade e sale con lei.

- IL PREFILL RESTA PIU' VELOCE DEL DECODE (`prefill.tokS > decode.tokS`, stesso
  JSON): guardia contro un instradamento ricaduto sul path del decode.

- CORRETTEZZA DEI KERNEL NUOVI, con tolleranza DERIVATA e non scelta: un caso
  ktest per OGNI forma prodotta (Q5_K, Q4_1, Q4_K, Q6_K, Q8_0 — anche quelle non
  cablate) contro il riferimento CPU, e un floor test che ri-deriva la tolleranza
  dal PAVIMENTO ARITMETICO f32 del testo WGSL generato, con e senza contrazione
  FMA, sul modello di `src/engine/kquantfast.ts` +
  `tests/engine-kquant-f32floor.test.ts`. Una tolleranza tarata sul device di
  sviluppo e' un gate che passa per fortuna.
  UNA FORMA VERIFICATA MA NON CABLATA NON E' UNA FORMA INVENTATA: il predicato di
  ammissibilita' del piano continua a instradare in produzione SOLO i kind
  misurati E cablati. E' il guardiano, e non si allarga per comodita'.

- GATE SECCHI A OGNI MERGE, nessuno piu' largo di come sta oggi:
  `node .harness/tools/engine-ktest.mjs` **tutti PASS** (oggi 101/0);
  top-1 contro l'oracolo llama.cpp **>= 1012/1024 = 98,828%** su ENTRAMBI i
  bracci; sequenze generate **identiche 8/8**; `npx vitest run` verde;
  `npx tsc --noEmit` pulito.

- NON-REGRESSIONE, banda ±5%: decode 4B **>= 45,5 tok/s** a ctx 6333; GLM b12
  optimistic entro ±5% di 13.172 / 31,26 / 14,74, misurata fresca su host
  dichiarato.

- PORTABILITA', ed e' la landmine di it.24 applicata a questo goal: il fabbisogno
  di workgroup storage di OGNI kernel nuovo e' una FORMULA esportata accanto al
  kernel ed entra nel `Math.max` CALCOLATO di `QWEN_WORKGROUP_STORAGE_BYTES`.
  `npx vitest run tests/gpulimits.test.ts` verde. Un kernel nuovo che chiede piu'
  del massimo dichiarato rompe `createComputePipeline` su OGNI device, non su
  qualcuno.

- UNA SOLA SEDE PER LA ROTTA: l'ammissibilita' del kind resta in
  `prefillgemmplan.ts`, che la CHIEDE al kernel (`prefillGemmCheck`); nessun
  secondo predicato in `q35gpumodel.ts`. E' il bug di it.7 (due posti che
  decidevano le righe-per-workgroup, e le decidevano diverse), e il gate che lo
  rileva esiste gia'.

- DISCIPLINA DEL PORT: il testo WGSL che va in produzione e' quello misurato al
  banco, riga per riga; ogni divergenza sta in `PREFILL_GEMM_PORT_DIFFS` con la
  sua ragione (il test fallisce nelle due direzioni, quindi niente chiavi morte).

- I BYTE DEL SEGMENTO SI CHIEDONO AL METER, non si ricopiano:
  `scripts/build-ttft-checkpoint.mjs:108` oggi ricopia a mano `173_015_040`
  invece di derivarlo da `prefillbytes.ts`. E' un secondo posto che decide lo
  stesso numero — la forma esatta del bug di it.7 — e quel segmento in questo
  goal cambia forma. Va risolto qui.

- CONSUNTIVO: `docs/engine/kquant-consuntivo-<data>.md`, voce per voce, con
  l'artefatto accanto a ogni clausola, la nuova ripartizione del tempo per
  segmento, e il termine che diventa PRIMO dopo questa leva nominato con la sua
  misura fresca — non ipotizzato. Piu' la scheda di consegna al goal 35B: quali
  forme sono misurate, con che numeri, e cosa manca per cablarle.

EVIDENCE OF DONE:
- fase 0: `BASE_URL=http://localhost:5199 node scripts/tt-microbench-run.mjs …`
  → un JSON per variante in `results/microbench/`
- copertura: `npx vitest run tests/engine-prefillgemmplan.test.ts` → [6c] col
  rapporto >= 15,5 stampato e pinnato
- segmenti + TTFT: `node scripts/q35-bench-run.mjs --prompt-idx 0 --n-decode 64
  --vram-gib 8 --declared quiescent` e `node scripts/build-ttft-checkpoint.mjs`
  → JSON `kind: "q35-ttft-kernel-checkpoint"` con `gemm:deltanet-out` e
  `gemm:ffn-down` in ms, byte e GB/s, e `loadMs` / `prefill.{ms,tokS}` /
  `decode.firstMs` SCOMPOSTI accanto al `ttftMs` aggregato.
  ATTENZIONE: `--prompt-idx` ha default 4 = 388 token. E' il flag da cui e'
  venuto il numero sbagliato corretto in it.0 del goal precedente. Leggerlo
  PRIMA di spendere GPU.
- correttezza: `node .harness/tools/engine-ktest.mjs` con vite acceso → tutti
  PASS; runner dell'oracolo top-1; `npx vitest run`; `npx tsc --noEmit`
- non-reg: runner decode 4B a ctx 6333 e runner GLM, entrambi con host dichiarato
- portabilita': `npx vitest run tests/gpulimits.test.ts`

AUTHORITY GRANTED:
- may do autonomously: scrivere i kernel WGSL nuovi e i loro gemelli al banco;
  estendere `prefillgemmplan.ts`, `prefillbytes.ts`, `kquantfast.ts` e il wiring
  di `q35gpumodel.ts`; cancellare il test [6d] scrivendone la ragione; aggiornare
  golden e fixture quando la riscrittura lo richiede, dichiarandolo nel journal;
  creare runner e fixture nuovi; eseguire bench GPU e ktest; committare e pushare
  su origin/main a fine task; mergiare a goal chiuso e verificato; aprire e
  chiudere item di docket che sono lavoro mio; escludere una forma coi numeri e
  dichiararlo; usare i workflow dell'harness (sdd-conductor, second-opinion,
  research-campaign, pattern-coverage, pattern-migration) e i subagent.
- must docket (never do): cambiare i gate di correttezza, le bande di
  non-regressione o le barre di questo contratto; cambiare `PREFILL_M`; CABLARE
  il 35B o GLM (il kernel si', il cablaggio e' il goal successivo); toccare il
  path 0.5B (e' l'altro goal deciso dal PI); cancellare o riscrivere codice
  committato da piu' di 30 giorni fuori dal brief; pubblicare numeri fuori dal
  tag di release; spendere denaro; toccare i goal in standby (fase-1b, fase-2).

CONSTRAINTS:
- WebGPU reale: `shader-f16` NON e' disponibile su questo stack Chrome/Linux.
  `packed_4x8_integer_dot_product` e' una LANGUAGE FEATURE
  (`navigator.gpu.wgslLanguageFeatures`), non un'estensione: nessun `enable` nel
  WGSL, o la compilazione fallisce con «expected extension» (costo' una run in
  it.5). Ogni via intera nuova va accompagnata dal suo fallback f32 DICHIARATO,
  come la q4_0.
- La forma multi-riga q4_0 esistente e' q4_0-only PER COSTRUZIONE
  (`wgsl.ts:3880 prefillGemmCheck`): questo goal la estende o le affianca forme
  nuove — non la aggira con un ramo che salta il predicato.
- Nessuna attribuzione AI in commit, PR o merge.
- Le metriche misurate non peggiorano mai: banda ±5% su tok/s e TTFT, gate di
  correttezza secchi.
- I bench costosi si eseguono su codice finale, non su stati intermedi.
- Ogni misura dichiara host e contesto; nessun tok/s senza `decodeContext`,
  nessun `ttftMs` senza i tre termini scomposti.
- Full-corpus solo per firma/non-reg/riferimenti nuovi; l'esplorazione va su
  simulazione o su prompt interi scelti, mai su `--cap`.
- Il server di sviluppo va avviato staccato e verificato con `curl`, mai con
  `pgrep` (fa match sulla propria riga di comando e dichiara vivo un server
  morto).

FUORI SCOPE (registrati, non aperti):
- **Il CABLAGGIO del 35B e di GLM** — e' il goal successivo, gia' deciso dal PI
  come priorita' dopo questo. Sara' wiring + residency: le forme di kernel
  arrivano da qui gia' misurate. Cio' che gli manca e che questo goal NON gli
  da': una baseline fresca del 35B (mai rimisurato dopo i kernel nuovi), il
  cablaggio dentro `moeprefillplan.ts` (unione expert + gather + readback CPU per
  layer) e la residency dei 17,67 GB di expert su 16 GiB di scheda.
- **Il 29,1% fuori dai pass GPU** (9.350 ms: encode CPU, submit, buchi fra
  submit). Non scende con questa leva, e DOPO questa leva diventa oltre il 50%
  del TTFT: e' il candidato naturale al goal dopo il 35B, e il consuntivo deve
  nominarlo con la sua misura fresca.
- Il LOAD del modello (10,89 s): ha una soglia sua, e' famiglia residency/IO.
- Il path 0.5B: e' l'altro goal deciso dal PI.
- I benchmark comparativi fra stack: un goal chiuso, l'altro in pausa dichiarata.

WORKING PROTOCOL: skills loop-iteration + done; verifier gate per ciclo; digest
ogni ciclo; stop-by-design quando il resto e' docket-gated.
VEICOLI, per indicazione del PI («utilizza i soliti workflow, che hanno
funzionato bene negli ultimi goal»): `sdd-conductor` per le ondate di
implementazione a spec, `second-opinion` sui rami che meritano sfiducia verso un
revisore solo, `pattern-coverage` come gate pre-merge di una convenzione nuova,
`research-campaign` per le righe con una tesi da pre-registrare.
VINCOLO CHE CI SI INTRECCIA, e va rispettato: finche' il watchdog dell'harness
non e' corretto, **niente `/loop` con un workflow in volo** — una sessione che
aspetta un workflow e' silenziosa per costruzione e il watchdog la duplica
(misurato tre volte il 2026-08-14, anche con il risveglio allungato a 60 min).
Il lavoro lungo va fatto con un umano presente, oppure con
`ScheduleWakeup{stop:true}` prima di lanciare il workflow.

CONTEXT ANCHORS:
- `HANDOFF.md` — mappa e landmine (`--prompt-idx`, il server che muore, «un
  artefatto si legge dal suo `kind`, non dal nome del file», il watchdog)
- `docs/engine/ttft-consuntivo-2026-08-14.md` §3 e §4 — la ripartizione del tempo
  per segmento e il perche' la barra e' caduta
- `.harness/goals/engine-ttft/docket.md` item 19 (il ruling che ha assegnato
  questo residuo), item 26, item 27 (il triage di chiusura)
- `src/engine/q35gpumodel.ts:778` `gemvB` — il bivio via veloce / legacy;
  `:464` `loadW` ramo K-quant — dove `ssm_out` prende oggi il gemello batch
- `src/engine/kernels/wgsl.ts:3820-4100` — split-K q4_0 idot + f32,
  `prefillGemmCheck`, `PREFILL_GEMM_PORT_DIFFS`
- `src/engine/prefillgemmplan.ts` — la sede UNICA della rotta
- `src/engine/prefillbytes.ts` — il meter dei byte (i kind K-quant ci sono gia')
- `src/engine/kquantfast.ts` + `tests/engine-kquant-f32floor.test.ts` — la
  famiglia FAST Q5_K/Q6_K a M=1 gia' in produzione: prior art dell'unpack e
  modello per la derivazione della tolleranza
- `tests/engine-prefillgemmplan.test.ts:250-440` — inventario pinnato, [6c], [6d]
- `src/engine/moe.ts` + `src/engine/moeprefillplan.ts` — il piano del 35B, per
  scrivere la scheda di consegna al goal successivo (non per cablarlo qui)
- `results/engine/q35-header-dump-2026-08-10.json` — l'istogramma dei tipi delle
  tre famiglie, da cui vengono le shape della fase 0
- `src/microbench/ttGemm.ts` + `scripts/tt-microbench-run.mjs` — il banco
