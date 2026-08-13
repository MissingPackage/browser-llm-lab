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
