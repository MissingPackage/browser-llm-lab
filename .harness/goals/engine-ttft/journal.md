# JOURNAL — engine-ttft

## it.0 (2026-08-13) — decomposizione, e tre correzioni trovate verificando la fattibilità

Goal aperto dopo la chiusura di `engine-kernel-decode` (47,93 tok/s a ctx 6333,
4,82x). Ruling A del PI in chat: i 4 secondi valgono sul TTFT **a modello
caldo** (`prefill.ms + decode.firstMs`); il load ha una soglia sua, separata.

**Il contratto nasce già da una correzione.** Il numero «TTFT 22,7 s su un
prompt da 6k» che HANDOFF portava è sbagliato in due punti, entrambi verificati
sull'artefatto `results/engine/q35-bench-4b-tier8-fase-d-it43.json`:

1. il prompt è il **prompt-idx 4 = 388 token**, non 6k. I 6333 sono il
   prompt-idx 0, usato per il DECODE del goal precedente. Due run su prompt
   diversi, appaiate per errore nel consuntivo.
2. `ttftMs = loadMs + prefillMs + firstMs` (`q35conf.worker.ts:246`): dei
   22.695 ms, **10.890 sono il LOAD** e 11.760 il prefill di 387 token. Il primo
   token vero costa 46 ms.

Il TTFT su un prompt da 6k **non è mai stato misurato**. Da qui il primo
done-when del contratto e la riga 0 di PHASES.

**Poi il test-fit dei done-when (ruling C7) ne ha cambiati altri tre.** Sono in
`PHASES.md` per esteso e a docket item 1 per il ruling; qui il sunto:

- **(C7-1)** il bench prefilla **una posizione alla volta**
  (`q35conf.worker.ts:207`, «prefill sequenziale»). I 32,91 tok/s non misurano
  un chunking che rende poco: ne misurano l'assenza. `prefillChunk` esiste
  (`q35gpumodel.ts:1281`), è gated bit per bit, e non è sul percorso del bench.
- **(C7-2)** M=8 è **degenere per l'obiettivo**: con riuso perfetto e a 300 GB/s
  (la banda migliore mai misurata qui) sono 5,94 s di sola lettura pesi, sopra
  il budget di 3,96. Il done-when del contratto dice «≥ 4x a M=8»; PHASES scrive
  **≥ 8x a M ≥ 16**. Scostamento dichiarato, ruling richiesto.
- **(C7-3)** la clausola (e2a) è sul path di conformità **0.5B**:
  `rmsPairGemmSiluChunkFast` è importato da `gpuforward.ts`, che assembla
  Qwen2.5-0.5B. Non tocca la metrica di questo goal. La (e2b) sì.

**E un conflitto strutturale fra due done-when del contratto**, che nessuno dei
due nominava: alzare `mMax` per il riuso peggiora il workgroup storage
(`mMax·3856` B — 30.848 a M=8, 61.696 a M=16), che (e2a) vuole sotto 16.384. Le
due leve tirano in direzioni opposte. È la riga 1 a dover trovare una forma il
cui shared non scali con M.

**Perché la forma attuale non riusa niente, per costruzione**: il prefill del 4B
non usa una GEMM. Usa `gemvQuantWgsl` con `batch: true` dispatchata su
`[gx, gy, M_MAX]` (`q35gpumodel.ts:720`) — cioè **M GEMV replicate sull'asse z**,
ognuna che rilegge la riga di pesi per intero. E il GEMV veloce nuovo del goal
precedente **rifiuta esplicitamente `batch`** (`wgsl.ts:216-249`): il prefill non
ha ricevuto nemmeno l'1,77x dei moltiplicatori. Due fatti che spiegano perché
oggi il prefill (32,91 tok/s) è più LENTO del decode (47,93).

**Fabbisogno di calcolo, per sapere se il bersaglio esiste**: 2·P·T con P=4e9 e
T=6333 ≈ **50,7 TFLOP**, cioè **12,8 TFLOP/s sostenuti** in 3,96 s. Il picco
fp32 di questo device in WebGPU non è mai stato misurato: è la sonda (a) della
riga 1, ed è ciò che decide se la riga 5 chiude col bersaglio o con
l'esclusione. [proiezione, non promessa]

**STOP by design**: goal di prodotto ⇒ gate `plan-check`. PHASES.md aspetta
l'approvazione del PI prima dell'iterazione 1.

## it.1 (2026-08-13, riga 0) — BASELINE MISURATA, e l'ipotesi della riga è refutata

Due bracci sul prompt-idx 0 (6333 token), host `quiescent` verificato a mano
prima di ogni run (GPU 0%, 572/581 MiB), tre termini scomposti in entrambi.
Artefatto unico: `results/engine/ttft-baseline-4b-prompt0-2026-08-13.json`.

| | load | prefill | tok/s prefill | firstMs | **TTFT a caldo** | decode |
|---|---|---|---|---|---|---|
| (a) sequenziale | 10.892 | 87.582 | **72,30** | 36 | **87.618 ms** | 47,79 |
| (b) chunked M=16 | 10.366 | 184.283 | 34,36 | 27 | 184.310 ms | 44,81 |

**IL CHUNKING È 2,10× PIÙ LENTO.** L'ipotesi che la riga 0 doveva testare —
«instradare il bench su `prefillChunk` potrebbe muovere la metrica senza
scrivere un kernel» — è **refutata dalla misura**. Va nella direzione opposta.

**La causa combacia col censimento di it.0, e questo è il valore del risultato.**
`prefillChunk` instrada su `gemvQuantWgsl` con `batch: true`, cioè M GEMV
replicate sull'asse z con riuso dei pesi ZERO; e il GEMV veloce del goal
precedente (vec4 + 2 righe/workgroup + `subgroupAdd`) **rifiuta `batch` per
costruzione** (`wgsl.ts:216-249`). Il path sequenziale usa `step()`, che quei
kernel veloci li ha. Quindi oggi il chunking paga M volte la lettura dei pesi
**col kernel lento**, e il sequenziale la paga una volta per token **col kernel
veloce** — e vince. Non è un difetto del chunking: è che il chunking non ha mai
ricevuto le ottimizzazioni del decode.

**Conseguenza per la riga 2**: non basta "usare `prefillChunk`". La leva 1 deve
portare il riuso vero AL kernel veloce, non instradare su quello lento.

**Correzione a un numero mio di it.0**: avevo proiettato ~192 s di prefill dai
32,91 tok/s del contratto. Quel 32,91 veniva da `q35-bench-4b-tier8-fase-d-it43`
— codice vecchio, prompt-idx 4 da 388 token. Sul codice di oggi il prefill
sequenziale rende **72,30 tok/s**: l'attenzione riscritta del goal precedente ha
già dato 2,20× anche al prefill, perché il prefill sequenziale *è* il path del
decode. Il prefill misurato è **87,6 s**, non 192.

**LA BASELINE DEL GOAL**: TTFT a caldo **87.618 ms** contro un bersaglio di
4.000. Serve **21,9×**, non i 48× che la proiezione sbagliata suggeriva.

**Primo bench end-to-end del 4B sul codice di oggi**: decode 47,79 tok/s a ctx
6333-6397, che conferma il 47,93 del checkpoint di micro-bench e soddisfa il
gate di non-regressione (≥ 45,53).

**Osservazione, non regressione**: nel braccio (b) il decode rende 44,81 tok/s,
−6,2% dal riferimento e fuori banda. Il codice committato non è toccato (senza
`--prefill-m` il modello non si costruisce con `prefillM`), ma costruire il
modello con `prefillM: 16` sembra costare ~6% al decode. n=1 per braccio:
registrato, non concluso.

**Codice toccato** (dentro l'authority della riga 0: `src/engine/q35conf/**`,
`scripts/**`): il braccio del prefill nel bench, con `prefillPath` DICHIARATO
nel report — è l'applicazione diretta della lezione di it.9 del goal precedente,
dove due JSON incompatibili furono riconciliati confrontando quel campo e non i
numeri. E il fix di `--out` assoluto/relativo (docket item 3) su
`q35-bench-run.mjs`, fatto qui perché la riga toccava comunque il runner e la
modifica viaggia con due run che la esercitano.

**Gate**: ktest 100 PASS / 0 FAIL · vitest 531 passed | 10 skipped ·
`tsc --noEmit` pulito · non-reg decode PASS.
