# PHASES — engine-ttft (decomposizione it.0, 2026-08-13)

Sequenziale. Le righe 2 e 3 toccano lo STESSO file
(`src/engine/kernels/wgsl.ts`) e lo stesso assemblatore
(`src/engine/q35gpumodel.ts`): nessun `parallel-group`, la proprietà esclusiva
dei path non c'è. Stessa ragione del goal precedente.

**Metrica obiettivo del goal**: `prefill.ms + decode.firstMs` sul prompt-idx 0
(6333 token), a modello caldo (ruling A). **Bersaglio: il piu' in basso
possibile**, barra meccanica **< 21.905 ms** = un quarto della baseline
(ruling PI 2026-08-13, che toglie i 4.000 dopo la proiezione della riga 1). Ogni riga
dichiara quanto muove QUESTA metrica, o si dichiara gate.

---

## Punto di partenza — e tre correzioni trovate verificando la fattibilità (C7)

Il contratto nasceva da `results/engine/q35-bench-4b-tier8-fase-d-it43.json`.
Test-fit dei done-when prima di scriverli in tabella: tre cose non stavano in
piedi come scritte.

**(C7-1) Il bench prefilla UNA POSIZIONE ALLA VOLTA.**
`q35conf.worker.ts:207` — «Riferimenti full-resident (fase 4, it.10): prefill
sequenziale». I 32,91 tok/s NON sono il prefill a chunk che rende poco: sono il
prefill a chunk **che non viene eseguito**. `prefillChunk` esiste
(`q35gpumodel.ts:1281`), è in albero, ha un gate di conformità **bit per bit**
(`q35conf.worker.ts:301-367`) e non è sul percorso del bench. La prima misura del
goal non è una riscrittura: è instradare il bench su ciò che c'è già.

**(C7-2) M=8 è degenere per l'obiettivo — la riga del contratto va a M ≥ 16.**
Conto alla banda GIÀ DIMOSTRATA da questo motore (300 GB/s effettivi sulla
scansione KV, goal precedente), con riuso dei pesi PERFETTO, pesi Q4_0 2,25 GB:

| M | passate sui pesi | byte letti | tempo a 300 GB/s |
|---|---|---|---|
| 1 | 6333 | 14.249 GB | 47,50 s |
| 8 | 792 | 1.781 GB | **5,94 s** |
| 16 | 396 | 891 GB | 2,97 s |
| 32 | 198 | 445 GB | 1,48 s |

A M=8, anche con riuso perfetto e alla banda migliore mai misurata qui, la sola
lettura dei pesi sfonda il budget di 3,96 s. Il done-when «riuso ≥ 4x a M=8» del
contratto chiede una cosa che non basterebbe comunque: in tabella va **M ≥ 16**.
`q35gpumodel.ts:559` cappa oggi `M_MAX = min(16, prefillM)` — il tetto è
esattamente al minimo utile, e va alzato o giustificato.

**(C7-3) La clausola (e2a) non è sul percorso di prodotto.**
`rmsPairGemmSiluChunkFast` — il consumatore dei 30.848 B — è importato da
`gpuforward.ts`, che è l'assemblatore di **Qwen2.5-0.5B** (riga 1 del file), non
il path q35 del 4B. La clausola resta valida come portabilità del path di
conformità, ma **non tocca la metrica di questo goal**. La (e2b) invece sì: il
ramo `batch` legacy dell'attenzione gira sul 4B e chiede `4·ctxMax + 256` B non
dichiarati.

**Fabbisogno di calcolo, per sapere in anticipo se il bersaglio esiste.**
Prefill di 6333 token su 4e9 parametri ≈ 2·P·T = **50,7 TFLOP**. In 3,96 s sono
**12,8 TFLOP/s sostenuti**. Il picco fp32 di questo device non è mai stato
misurato in WebGPU: è una sonda della riga 1, ed è ciò che decide se la riga 5
chiude col bersaglio o con l'esclusione. [proiezione, non promessa]

**Scomposizione del bersaglio** (da riconfermare in riga 0):
`prefill.ms` 11.760 su 387 token → sul prompt-idx 0, a parità di rate, ~192 s.
`decode.firstMs` 46 ms — irrilevante, ma va nel JSON per il ruling A.
`loadMs` 10.890 — **fuori scope**, ruling A: goal suo.

Gate PERMANENTI a ogni riga (impliciti in ogni done-when): ktest tutti PASS,
`npx vitest run` verde, `npx tsc --noEmit` pulito, argmax del golden 4B IDENTICO
al pin, GLM non-regredito, decode 4B ≥ 45,5 tok/s a ctx 6333.

---

| # | phase | done-when (mechanical) | veicolo | authority delta | owns | stima | status |
|---|-------|------------------------|---------|-----------------|------|-------|--------|
| 0 | **Baseline onesta** — il numero da cui parte il goal oggi non esiste | JSON committato in `results/engine/`, prompt-idx 0 (6333 token, verificato nel golden), `hostState.declared = "quiescent"`, con `loadMs` / `prefill.ms` / `decode.firstMs` **distinti**, per DUE bracci sullo stesso host e nello stesso file: **(a)** prefill sequenziale — il path del bench di oggi (`q35conf.worker.ts:207`); **(b)** prefill instradato su `prefillChunk` a M=16, che è già in albero e già inchiodato bit per bit dal gate `q35-prefillchunk-4b`. Il braccio (b) richiede di instradare il bench su `prefillChunk` — modifica del worker, non del kernel. Entrambi i bracci riportano `prefill.tokS` e il TTFT a caldo `prefill.ms + decode.firstMs`. **Nessuna riscrittura di kernel in questa riga** | diretto | none | `src/engine/q35conf/**`, `scripts/**`, `results/engine/**` | 1 it | **done (it.1)**: TTFT a caldo **87.618 ms** (prefill 87.582 + firstMs 36) contro un bersaglio di 4.000 ⇒ **serve 21,9x**. Braccio (b) REFUTATO dalla misura: il chunking a M=16 e' **2,10x piu' LENTO** (34,36 contro 72,30 tok/s), perche' instrada sul GEMV lento mentre `step()` usa quello veloce. Artefatto `results/engine/ttft-baseline-4b-prompt0-2026-08-13.json`. Gate: ktest 100 PASS/0 FAIL, vitest 531\|10, tsc pulito, non-reg decode 47,79 (>= 45,53) |
| 1 | **Sonde e varianti** — decidere SE le leve esistono, e se il bersaglio esiste, prima di riscrivere | JSON committato con, per OGNI variante isolata, G pesi/s, GB/s (`bytesEmitted`/`bytesUnique`, già emessi da `kdRunner.ts:580-605`) e **byte di peso emessi per token prefillato**, zero run di modello: **(a)** sonda del PICCO di calcolo del device in WebGPU (TFLOP/s fp32 su GEMM densa, shape reale) — è il numero che dice se 12,8 TFLOP/s sostenuti sono al 40% o al 200% del picco, e quindi se la riga 5 può chiudere col bersaglio; **(b)** GEMM multi-riga: forma attuale (`gemvQuantWgsl` con `batch`, che è M GEMV replicate su `wid.z` — riuso ZERO per costruzione) · tile con pesi in registri · tile con pesi in shared · a M = 1, 8, 16, 32; **(c)** attenzione a chunk del prefill: legacy attuale · softmax in streaming · KV letta una volta per gruppo GQA · letture vec4, a ctx 388 e ≥ 6000; **(d)** workgroup storage richiesto da ogni variante, misurato non dedotto — e, poiché il tetto è NEGOZIABILE (`requiredLimits` di `requestDevice`, docket item 6), la **curva throughput-vs-M a limite concesso variabile**: chiedere il massimo costa occupancy, quindi serve il ginocchio, non il massimo. Include `adapter.limits` di questo device accanto ai numeri. Predizioni PRE-REGISTRATE prima di eseguire, graduate dopo da un verificatore indipendente. **REGOLA DI STOP: se nessuna variante supera la forma attuale di ≥ 1,5x sulla propria metrica, la riga chiude il goal e il piano si riscrive invece di procedere** | `research-campaign` (prediction-gated: pre-registra → esegui → grade indipendente → memo) | none | `src/microbench/**`, `scripts/**`, `results/engine/**`, `docs/deep-dive/**` | 2 it | **done (it.2)**: REGOLA DI STOP **non scattata**. Vince `splitk` (pesi in registri + attivazioni in shared + K su 4 workgroup): **43,1x** sulla forma attuale a M=16 su K2560xN9216 (262.881 contro 6.101 tok/s), riuso dei pesi **16,0x** (13,27 -> 0,83 MB per token), workgroup storage **4.096 B** (sotto il pavimento di spec). Attenzione a chunk: **6,76x** a ctx 6333 e **5,38x** a ctx 388 dalla sola softmax in streaming + vec4 — la fusione GQA **peggiora** (occupancy). Picco fp32 misurato **9,26 TFLOP/s**: i 12,8 richiesti sono al **138%**. Proiezione del pavimento **~8.700 ms** (10,1x sulla baseline, 2,2x dal bersaglio) => la riga 5 chiudera' per esclusione. Tetto negoziabile **non e' una leva** (spread 0,1-2,3% su {16.384..49.152}); il legacy dell'attenzione **non crea la pipeline** a 16.384 ne' a 24.576. Artefatto `results/microbench/ttft-riga1-4090-linux-2026-08-13T13-06-23-120Z.json` (34 celle, 0 skipped), memo `docs/deep-dive/ttft-riga1-memo-2026-08-13.md`. Gate: ktest 100 PASS/0 FAIL, vitest 545|10, tsc pulito |
| 2 | **GEMM multi-riga con riuso vero** — la leva misurata a 43,1x sul kernel | **(a0) DUE CELLE PRIMA DI COSTRUIRE** (fase 0 residua, docket item 12c + item 11, autorizzato dal PI): `splitk-coldw` — la vincitrice a pesi freddi, che discrimina se il suo vantaggio è traffico o occupancy, e che la riga 1 non ha girato — e `splitk-idot`, la stessa forma con `packed_4x8_integer_dot_product` al posto del dequant in virgola mobile. Se `idot` vince, è LEI la forma che va in produzione; (a) il prefill del 4B smette di usare `gemvQuantWgsl` con `batch: true` (= M GEMV replicate su `wid.z`) e adotta la variante vincente di riga 1; (b) **RIUSO, misurato non dedotto**: byte di peso emessi per token prefillato scende di **≥ 5,5x a M ≥ 16 sull'INVENTARIO PER-LAYER INTERO**, nello stesso JSON PRIMA/DOPO. **Il ≥ 8x precedente era irraggiungibile a qualunque M praticabile** (tetto 8,67x, servirebbe M ≥ 92): la forma nuova è q4_0-only e l'11,54% dei byte del 4B — 24 `ssm_out` Q5_K + 4 `ffn_down` Q4_1 — resta sul percorso vecchio. Misurato 5,86x. Ruling PI 2026-08-13, docket item 19. **Il residuo è scope del goal K-quant, e va NOMINATO nel consuntivo, non lasciato implicito**; (c) `prefill.tokS` sul prompt-idx 0 PRIMA e DOPO nello stesso JSON; (d) **copertura della convenzione dichiarata**: quanti dei siti di prefill la adottano su quanti esistono, con worklist ed eccezioni motivate — una convenzione applicata al 30% è il difetto, non il progresso; (e) conformità: il gate `q35-prefillchunk-4b` resta **bit per bit** contro `step()` sequenziale, oppure la tolleranza si dichiara PRIMA con la ragione numerica; (f) argmax del golden 4B IDENTICO al pin | `sdd-conductor` (build multi-task: kernel + assemblatore + gate + fixture) → poi `second-opinion` sul kernel numerico | none | `src/engine/kernels/**`, `src/engine/q35gpumodel.ts`, `src/engine/prefillplan.ts`, `tests/**`, `results/engine/**` | 3 it | **in corso (it.14)**: cablaggio VERIFICATO e mergiato (`7bc6b55`) — il ktest dopo il riavvio fa 100 PASS/0 FAIL sullo stesso ramo che ieri moriva, quindi la causa era l'infrastruttura. Misurato sul prompt-idx 0: prefill **34,36 → 111,16 tok/s = 3,24×** sul braccio chunked M=16, TTFT a caldo **87.618 → 56.984 ms = 1,54×**, decode 47,17 (−1,30%, in banda). (b) riuso **5,8593×** sull'inventario per-layer INTERO ≥ 5,5× del ruling ✓ · (c) prima/dopo ✓ (baseline it.1 + artefatto it.14) · (e) `q35-prefillchunk-4b` bit per bit ✓ · (f) argmax golden 4B ✓. **NON chiusa: (a0) la forma in produzione è la PERDENTE.** `q35gpumodel.ts:746` chiama `prefillGemmQ4SplitKWgsl` (via f32, che `wgsl.ts:3795` dichiara «FALLBACK ... non come alternativa preferibile»); `prefillGemmQ4SplitKIdotWgsl` (`wgsl.ts:3718`, 1,745× sopra la f32) è in albero, testata, **e nessun call-site la raggiunge**. Manca anche (d), la copertura dei siti con worklist |
| 3 | **Attenzione a chunk del prefill** — gli stessi tre difetti già chiusi sul decode | (a) `attnDecodeWgsl` con `batch: true` smette di instradare al legacy (`wgsl.ts:530`): softmax in streaming (niente `scores: array<f32, ctxMax>`), KV letta una volta per gruppo GQA, letture vec4; (b) ms per chunk a ctx 388 **e** a ctx ≥ 6000 PRIMA e DOPO nello stesso JSON; (c) il golden `tests/fixtures/attn-decode-legacy.golden.json` aggiornato DELIBERATAMENTE, con la ragione nel journal — non per sbaglio; (d) ktest del prefill PASS contro cpuref con la tolleranza dichiarata PRIMA (cambia l'ordine delle somme: **non bit-identico per costruzione**, come la riga 1 del goal precedente); (e) argmax del golden 4B IDENTICO al pin | `sdd-conductor` → `second-opinion` sul kernel numerico | none | `src/engine/kernels/**`, `src/engine/q35gpumodel.ts`, `tests/**`, `results/engine/**` | 2 it | ready |
| 4 | **Portabilità (e2)** — con lo scoping corretto da (C7-3) | (e2b, percorso di prodotto) `engineNeeds` CONTA il ramo `batch` dell'attenzione: dopo la riga 3 il suo fabbisogno è costante in ctxMax e dichiarato, e `tests/gpulimits.test.ts` lo asserisce come garanzia invece che come debito. (e2a, percorso di conformità 0.5B) `rmsPairGemmSiluChunkFast` (`4·K·mMax + 256·mMax + 16·mMax`, K=896: 30.848 B a mMax=8, **61.696 a mMax=16**) — o scende sotto 16.384 B, o la clausola si dichiara **debito del path 0.5B** con la ragione, e il done-when del contratto va a docket per un ruling. **NOTA: alzare mMax PEGGIORA (e2a) — le due leve tirano in direzioni opposte, ed è la riga 1 a dover trovare una forma il cui shared non scali con M** | diretto | none | `src/engine/gpulimits.ts`, `tests/gpulimits.test.ts`, `src/engine/kernels/**` | 1 it | ready |
| 5 | **CHECKPOINT — la discesa massima, contabilizzata** | `node scripts/q35-bench-run.mjs --prompt-idx 0 --n-decode 64 --vram-gib 8 --declared quiescent` → JSON committato con `prompt.tokens ≥ 6000` e `prefill.ms + decode.firstMs` **< 21.905 ms** (un quarto della baseline); PIÙ, obbligatorio e non alternativo, un JSON `kind: "q35-ttft-kernel-checkpoint"` con banda efficace **per segmento**, **workgroup in volo per dispatch** (il termine che la riga 1 ha scoperto essere il collo, e che il motore non misura da nessuna parte), TFLOP/s sostenuti contro il picco misurato in riga 1, e il conto di dove finisce il tempo che resta. Il ruling del PI toglie il bersaglio dei 4 s: il checkpoint non sceglie più fra «raggiunto» ed «escluso», **contabilizza la discesa e il tetto**. Ratchet di correttezza intatti, decode 4B ≥ 45,5 tok/s a ctx 6333, non-reg GLM fresca | diretto | none | `results/engine/**`, `docs/engine/**` | 1 it | ready |
| 6 | **Chiusura** | checklist del DONE WHEN del contratto voce per voce con evidenza; HANDOFF §1 aggiornato; triage del docket; il goal successivo eredita la lista di ciò che è stato escluso coi numeri | diretto | none | `docs/**`, `HANDOFF.md`, `.harness/goals/**` | 1 it | ready |

**Conteggio dei gate** (ruling C9): righe 0, 1, 5 sono misura/decisione; righe 2,
3, 4 muovono la metrica o rimuovono un vincolo che la blocca; riga 6 è chiusura.
Tre righe su sette muovono il numero — ma la riga 1 è l'unica che può chiudere
il goal, ed è quella che decide se le altre due esistono. Stessa forma che ha
funzionato nel goal precedente.

**Legame con l'obiettivo, riga per riga** (ruling C9):

- riga 0 — **GATE**, e potrebbe muovere la metrica da sola senza scrivere un
  kernel: instradare il prefill su `prefillChunk` a M=16 è codice già in albero
  e già bit-esatto. Quanto renda è ignoto: è la misura che lo dice.
- riga 1 — **GATE che può chiudere il goal.** Non muove la metrica: decide se le
  leve esistono E se il bersaglio esiste (sonda del picco di calcolo). È l'unica
  riga con potere di chiusura.
- riga 2 — attacca il termine dominante. Con riuso 16x a M=16 e alla banda già
  dimostrata, la lettura dei pesi passa da 47,5 s a ~3 s [proiezione]. Resta il
  calcolo: 50,7 TFLOP, che la riga 1 avrà quotato.
- riga 3 — attacca l'attenzione del prefill, che a ctx 6333 è il termine che sul
  decode valeva il 65%. Sul prefill il peso è diverso (la KV si legge una volta
  per M righe): quanto valga è la misura di riga 1(c), non un'assunzione.
- riga 4 — **non muove la metrica**: rimuove un vincolo di portabilità e chiude
  una sotto-dichiarazione già viva. È gate.
- riga 5 — checkpoint, non muove niente: constata.
