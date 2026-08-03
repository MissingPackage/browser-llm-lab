# Digests — engine-fase-c3a

## it.1 (2026-08-01) — fase 1 DONE, fase 2 bloccata da ruling

- **Fatto**: telemetria di attribuzione nel motore GLM (opt-in, zero-overhead da
  spenta; timestamp-query su ogni pass, `mapAsync` dopo il submit come da
  root-cause di fase A) + TTFT e gap dalla funzione obiettivo nel report di
  bench + probe indipendente del floor di sync. Bench ri-eseguito a macchina
  quiescente: `results/engine/bench-glm-4090-b12-attrib-2026-08-01.json`.
- **Difetto trovato e corretto dentro l'iterazione**: il check di identità che
  ho aggiunto ha scoperto un double-count (`encodeCpuMs` includeva `ensureMs`).
  Corretto e ri-misurato; ora l'identità chiude allo 0.3%.
- **Il numero che cambia il goal**: wall 215.0 ms/token = `gpuBusy` 78.2 +
  stallo 53.8 + sync/CPU 83.0 ⇒ 63.6% fuori dalla GPU, ma i readback veri
  valgono solo 7.6 ms/token (probe). Con le leve 1 e 2 del contratto al 100%:
  **10.18 tok/s** contro un gate 13.43 che richiede ≤74.46 ms/token — e
  `gpuBusy` da solo ne occupa 78.2. **Le leve del contratto non bastano.**
- **Ma la GPU è sotto-clockata dalle bolle** che la leva 2 rimuove: 1746 MHz
  medi su 3105 di cap, utilizzo 34.6%. Al limite ottimistico ⇒ 15.63 tok/s,
  gate PASS. **Forbice 10.2-15.6, col gate dentro.**
- **Cross-check indipendente**: 2.22 GB di pesi/token su 576 GB/s reali ⇒ floor
  memory-bound 3.85 ms/token; `gpuBusy` è **20× sopra** (43 µs per dispatch su
  1816 dispatch/token). Quarta leva disponibile — granularità dei dispatch —
  ma è kernel engineering, ultima per direction §2.
- **Non regredito**: decode headline 4.678 (C2: 4.640), prefill 5.235 (5.221),
  `npm test` 220 passed + 2 skipped, `tsc --noEmit` pulito.
- **TTFT misurato per la prima volta: 88.06 s** contro il budget UX di 4 s
  (gap 22×). Il decode è a 6.41× dai 30 tok/s.
- **Prossimo**: STOP by design. La fase 2 (spec) non è decidibile senza il
  ruling di docket item 4 (quale ramo assumere) — e la risposta cambia il
  meccanismo che la spec deve scegliere. Aperti anche item 2 (clausola di
  fallback, ora quantificata) e item 5 (correzione di forma della riga PHASES).
- **Gate di protocollo scoperto**: verifier indipendente non lanciato (la policy
  di sessione vieta subagent non richiesti). Verifica fatta in proprio e
  meccanica; segnalato, non aggirato.

## it.2 (2026-08-01) — fase 2 DONE, ruling di spec pendente

- **Ruling recepiti**: quarta leva nel perimetro (GOAL emendamento 1), fase 4b
  in PHASES, clausola di fallback rimandata a fine fase 4, riga fase 1 corretta.
  Della (a) scartata ho tenuto il pezzo utile: la fase 4 chiude con una
  **ri-misura obbligatoria di clock e `gpuBusy`**, che dimensiona la 4b con un
  fatto invece che con l'assunzione ottimistica.
- **Spec scritta**: `docs/superpowers/specs/2026-08-01-engine-fase-c3a-design.md`,
  8 sezioni, budget per leva che somma esattamente al gate (74.4 ms/token).
- **La spec corregge il contratto su un punto di sostanza**: il contratto dice
  "eliminare i 46 readback", ma la misura dice che i readback valgono 7.6 degli
  83 ms/token — il resto è **frammentazione dei submit**. Quindi il criterio
  diventa **minimizzare i submit, non i readback**.
- **Pipelining scartato esplicitamente** (era una delle tre vie del contratto):
  nel decode il token t+1 dipende da t e il layer l+1 da l. Resta per il
  prefill e per la fase D.
- **[VERIFY] aperto**: `maxStorageBufferBindingSize` e `maxBufferSize` reali non
  sono mai stati letti (il codice negozia `min(limite, 2 GiB)` senza guardarlo).
  La via raccomandata per la leva 2 dipende da quel numero ⇒ probe da ~20 righe
  come primo task della fase 4.
- **Conto che cambia le aspettative sul prefill**: con M=16 il TTFT scenderebbe
  verso ~5-6 s, ancora sopra i 4 s. Il batching da solo non basta.
- **Prossimo**: STOP by design, ruling di spec a docket item 6 (5 decisioni).

## it.3 (2026-08-02) — assunti corretti, meccanismo del readback capito

- **Gate tautologico rimosso**: ktest verificava `dispatchesPerToken === 61`, ma
  quel valore È la formula — confrontava la formula con se stessa. Ora due
  asserzioni: piano 61 **e** runtime 63 (= piano + testa). **La formula era
  anche sbagliata**: non contava la testa, quindi tutto C2 ha riportato 1816
  dispatch invece di 1818.
- **Il gate nuovo ha trovato un bug alla prima esecuzione**, nella
  strumentazione appena scritta (`dispatches` contava sempre, `forwards` no).
  Corretto. ktest 30/30 PASS.
- **Probe WebGPU**: il device prende i default di spec su tre limiti che
  l'adapter offre più alti — 8 storage buffer su 16, 256 invocazioni/workgroup
  su 1024, 2 GiB di buffer su 4. Ruling PI: negoziarli subito.
  Bonus: `subgroup-matrix` è ESPOSTA ⇒ direction §8 rischio 1 è stale.
- **Mia attribuzione sbagliata, corretta**: gli ~75 ms non erano "latenza dei
  submit" (che vale 0.6 ms/token). `mapAsync` è una **barriera**: ogni layer
  drena la coda. Confermato aritmeticamente (gpuBusy/wall 36.4% ≈ utilizzo
  34.6%).
- **Il pattern che risolve è provato da tre implementazioni** (ORT PR #27998,
  ggml-webgpu, MLC): binding FISSO, expert = offset aritmetico nello shader.
  Le API che sembravano promettenti (binding_array, bindless) non coprono i
  buffer.
- **L'ostacolo vero**: residenza totale richiede 15.91 GiB contro 15.25
  usabili ⇒ **deficit 0.67 GiB = 4.6% del parco**. Zero-drain costa qualità:
  docket item 8, decisione sulla funzione obiettivo.
- **DVFS dimostrato nel meccanismo**: `gpu_idle` attivo in 34/40 campioni,
  power cap e thermal 1/40. I clock bassi vengono dall'inattività, non dai
  limiti. Resta non misurato *di quanto* scenderebbe `gpuBusy` a clock pieni.
- **Ruling recepiti**: prefetch LOOKA dentro C3a (emendamento 2a) — ribalta la
  valutazione del docket C2 item 8; limiti negoziati in fase 3 (2b).

## it.4-7 (2026-08-02) — FASE 3 CHIUSA: limiti derivati + repack all'import

- **Limiti derivati dai consumatori** (ruling PI "capire caso per caso"): non
  più costanti inventate né massimo dell'adapter, ma `min(adapter, requisito)`
  con il consumatore scritto accanto. Requisiti reali: 256 invocazioni (non
  1024), 7 storage buffer (non 16), 250,5 MiB di binding (la testa Q6_K, che
  sfonda il default di spec), `max(30 848, 4·ctxMax+256)` di workgroup storage.
  Un test **scansiona il WGSL vero** e fa cadere la derivazione se un kernel la
  supera: è ciò che tiene insieme limite e consumatore.
- **Due cose che nessuno aveva scritto**: il vecchio cap 32768 limitava il
  contesto a **8128 token**; e il consumatore massimo di workgroup storage del
  path Qwen non è quello citato nel commento (30 848 B, non 19,7 KB — margine
  reale 1 920 B).
- **Repack all'import**: pack CPU **42,5 → 0,0 ms/token**, decode 4,640 →
  **4,912** (+5,9%), stallo 56,1 → 34,5.
- **Ma il budget di spec era sbagliato**: la leva vale −21,6 ms/token, non
  −41,4. Read+upload crescono di 20,7 perché le letture expert passano da
  "prevalentemente in page cache" (4,08 GB/s) a "quasi tutte fuori" (1,29 GB/s):
  due file da 33 GB contro 14 GB di cache. **Il repack scambia CPU con I/O.**
  Tre ipotesi testate, due smentite (cache fredda transitoria; frammentazione —
  lo slab ha 2 extent contro i 263 del GGUF).
- **Conseguenza**: i 18,4 ms/token di lettura sono ora materia della leva 2 —
  è tempo in cui la GPU è ferma, quindi lo nasconde il prefetch.
- **Conformance full-corpus NON eseguita** (~5 h): sostituita da due verifiche
  byte-identiche (VRAM e file su disco, 7 campioni). Resta da fare in fase 6.
- Verifica: `tsc` pulito, **252 test verdi** (da 220 a inizio goal), ktest
  30/30, due bench a macchina quiescente.

## it.9 (2026-08-02) — fase 4 slice 1: router top-4 su GPU

- **Ruling recepiti**: item 11 (sostituzione della conformance accettata, gate
  full-corpus alla fase 6) e item 8 (**si paga la residenza totale** ⇒ la leva 2
  si progetta senza miss). Conseguenza registrata, non decisa: il requant è
  fuori dagli `owns` della fase 4 ⇒ emendamento PHASES per una **fase 4c**.
- **`routerTopKWgsl`**: la selezione top-4 esce dalla CPU. È la precondizione
  dello strato 1 — il binding fisso da solo non toglie il readback, serve che
  gli id vivano su GPU.
- **Fedeltà misurata, non dichiarata**: f32 vs f64 dà insiemi identici su 64
  estrazioni (pesi maxRel 1.6e-7) e, col pareggio **costruito**, tiene fino a
  1e-6 di separazione con primo flip a 1e-7 ⇒ 10× di margine sul gate 1e-5.
  Il caso random da solo era un gate finto: non scende mai sotto 3.43e-5.
- Verifica: `tsc` pulito, `npm test` 252 passed + 2 skipped, **ktest 32/32**.
- Non fatto: il kernel non è nel path caldo (cablarlo ora non toglierebbe sync —
  il readback serve anche a sapere cosa caricare). Prossimo: bind group layout
  esplicito, base-offset nei GEMV expert, `wExp[k]` collassati, tabella slot
  su GPU.

## it.10 (2026-08-02) — fase 4b: famiglia fusa sulla catena expert

- **Ruling PI** (docket item 12): la 4b esce da `blocked-by-4` e parte subito.
- **La causa del 5.3× non è la fusione, è la struttura del gemv**: load `vec4`
  invece di 4 scalari, `x` in shared invece che riletta per riga, `dot()`
  invece di estrazione byte a byte, 4 righe per workgroup invece di 1.
- **Portati e cablati**: `pairGemvSiluFastWgsl` (gate+up+silu, senza rms perché
  nel MoE la norm è a monte del router) e `gemvAccumFastWgsl` (down q4_0/q4_1).
  Catena expert **4 → 2 dispatch**; dispatch/token **1818 → 1450**.
- **Correttezza**: ktest **35/35**, incluse le due prove end-to-end contro f64
  (L2rel 2.4e-7, argmax 6/6). Path Qwen non toccato.
- **Il numero, e la smentita**: `gpuBusy` 75.86 → **69.11** (−8.9%), decode
  4.912 → **4.967** (+1.1%). Il modello di it.8 prevedeva `gpuBusy` → ~14.
  La catena expert è il 41% dei byte e ha reso l'8.9% del tempo GPU: o pesa
  poco su `gpuBusy`, o i kernel nuovi non sono 5× — **indistinguibile senza
  l'attribuzione per categoria**, che è il primo task che spec §4 prescrive
  alla 4b e che ho saltato credendolo assolto da it.8.
- **Il fatto strutturale**: −6.75 ms di GPU → −4.27 di wall → +1.1% di decode,
  perché sync/CPU SALE a 99.7 (50% del wall). Le proiezioni del report danno
  **9.77 tok/s anche batchando tutti e 46 i sync**. Finché il drain c'è, la 4b
  non si converte in tok/s. Ordine delle fasi: da riportare al PI.

## it.11 (2026-08-02) — attribuzione per categoria: il bersaglio era l'attention

- `gpuBusy` scomposto per categoria di kernel (replica bycat dedicata):
  **attn MLA 51.2 ms/token = 74.5%**; la catena expert, 41% dei byte, vale il
  5.8%. Il modello "byte letti ⇒ tempo" è morto; la 4b si riorganizza
  sull'attention. Report `bench-glm-4090-b12-bycat-2026-08-02.json`.

## it.12-14 (2026-08-03) — FASE 4b CHIUSA: gpuBusy 78.2 → 54.2, sotto la soglia derivata

- **it.12 flash-decoding sulla MLA**: chunk di cache in shared riusato da
  tutte le 20 head (via il fattore 20 di riletture del monolitico). attn
  51.2 → 27.5 ms/token (bycat); decode 4.982.
- **it.13 famiglia fast sui K-quant**: 8 thread/superblocco e word in registri
  al posto di 56 thread fermi e load byte-a-byte. shexp 14.6 → 5.5, head
  9.6 → 3.8; decode **5.054** (nuovo massimo); dispatch/token 1405.
- **it.14 conformance reale**: argmax ≡ cpuref-f64 **256/256** sul campione
  ratificato; top-1 golden 99.22% (le 2 divergenze = le stesse di C2, stesso
  token di cpuref). Full-corpus a fase 6 (item 11).
- **Gate 4b: gpuBusy 54.2 ≤ 54.5 PASS** con attribuzione kernel-vs-clock:
  clock SM medi 863 MHz (vs 1746 di it.1, CSV in results/engine/) — la
  riduzione è tutta kernel, i clock remavano contro.
- **Processo nuovo**: implementazione delegata a Opus su design chiuso, doppia
  review avversaria (Opus + Codex) per iterazione, verifier indipendente alla
  chiusura di fase. Tre round di fix da finding reali, zero difetti numerici
  sopravvissuti.
- **Il vincolo confermato**: coi 46 drain i guadagni GPU non diventano tok/s
  (decode +1.7% a fronte di −15 di gpuBusy). Proiezione: batchare tutto dà
  12.47 tok/s < gate 13.43 ⇒ la fase 4 (eliminazione) è il prossimo pezzo.
