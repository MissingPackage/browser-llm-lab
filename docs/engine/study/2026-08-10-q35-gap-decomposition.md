# WP decomposizione gap — qwen35 denso full-resident vs llama.cpp Vulkan

Data: 2026-08-10 · q1 fase 5 (spec §6) · Artefatti:
`results/engine/native-vs-browser-q35-2026-08-10.json` (protocollo e host),
`q35-bench-{4b,9b}-fullresident-2026-08-10.json` (lato browser),
prior: `docs/engine/study/llamaweb.md` (safety check 14-42%, tuning +41%).

## 1. Il confronto (STESSO GGUF, stessa GPU/driver, full residency)

| | nativo Vulkan (b10333) | browser (correttezza-prima) | gap |
|---|---|---|---|
| 4B decode | 118.7 tok/s (8.43 ms/tok) | 22.93 tok/s (43.6 ms/tok) | **5.18×** |
| 9B decode | 67.2 tok/s (14.9 ms/tok) | 14.55 tok/s (68.7 ms/tok) | **4.62×** |
| 4B prefill | 4 755 tok/s | 26.0 tok/s (SEQUENZIALE) | 183× |
| 9B prefill | 2 622 tok/s | 15.4 tok/s (SEQUENZIALE) | 171× |

Il lato browser è l'orchestratore **correttezza-prima** di fase 4: 562
dispatch/token, zero fusioni, readback logits per token nel decode,
prefill = decode sequenziale. Il gap è il PREZZO DELLA NUDITÀ, non un
limite del web: va letto con la decomposizione sotto.

## 2. Decomposizione del decode (il confronto onesto: entrambi token-per-token)

**Misurato** (non stimato): la differenza decode−prefillSeq isola il costo
readback+sync — stessi 562 dispatch, il prefill (read=false) salta solo
copy 1 MB + mapAsync:

| componente (4B) | ms/token | quota del gap |
|---|---|---|
| forward GPU (compute+dispatch) | 38.5 | — |
| readback+sync per token | **5.1 (misurato)** | 12% del totale |
| nativo | 8.4 | — |
| **gap di solo compute** | 38.5/8.4 = **4.6×** | |

(9B: readback 3.6 ms, compute 65.1 vs 14.9 nativo = 4.4× — coerente.)

Dentro il 4.6× di compute, per stima DICHIARATA (telemetria per-op assente
nell'orchestratore — arriverà coi moltiplicatori):
- **dispatch overhead**: 562 dispatch/token × 10-20 µs ≈ 5.6-11.2 ms/token
  (≈15-29% del forward): la sequenza è interamente serializzata, nessuna
  fusione; llama.cpp Vulkan ha kernel fusi e ~10× meno dispatch.
- **qualità kernel**: i nostri gemv sono riduzioni 64-thread-per-riga
  senza tuning per-device (prior LlamaWeb: +41% dal solo tuning) e senza
  dot4I8Packed; llama.cpp usa kernel maturi con subgroup ops.
- **safety check WebGPU**: 14-42% (prior LlamaWeb misurato sul prefill;
  sul decode gemv-bound l'effetto è nella fascia bassa).
- Nota di regime: il floor di banda pesi del 4B (2.4 GiB/token) su questa
  GPU è ~3.4 ms/token → il nativo è a 2.5× dal floor, noi a 12.8×.

## 3. Il prefill NON è un gap: è un'assenza

183×/171× misurano l'assenza del BATCHING (prefill = N decode sequenziali).
Il percorso esiste già in casa (GLM: prefill chunked con batch gemv,
pairGemvSilu batch, kv-append batch). Portarlo su qwen35 è la leva
singola più grande del goal per il TTFT (25.8 s → ordine 2-4 s a batching
GLM-class; il tier mobile di fase 8 è prefill-bound: landmine C3c).

## 4. Leve ordinate per ROI (metodo: misura dove c'è, aritmetica dichiarata altrove)

1. **Prefill batched/chunked** (pattern GLM esistente) — ROI: ~50-100× sul
   prefill, TTFT da 25.8 s a pochi secondi. Nessuna incognita di metodo:
   è portare un pattern già a regime su GLM. → fase 6/7 naturale.
2. **Decode multi-step senza readback** (pattern GLM decodeBatch B2:
   argmax on-GPU + embed gather feedback, K step/submit) — ROI MISURATO
   qui: −5.1 ms/token subito (+13% decode) + abilita l'ammortamento dei
   submit. Pattern esistente in gpuforward.
3. **Fusione + riduzione dispatch** (562 → ~10× meno: rmsnorm+gemv fusi,
   pairGemvSilu su ffn, epiloghi fusi) — attacca i 5.6-11.2 ms di
   overhead stimato; pattern pairGemvSilu esistente.
4. **Tuning kernel per-device + dot4I8Packed** (fase 6 da contratto,
   dietro flag) — prior +41% (LlamaWeb) sul matmul; dot4I8Packed è WGSL
   core, disponibile oggi.
5. **subgroup-matrix** (tensor core, prefill-only) — SPIKE dietro flag
   Chromium come da contratto: nel browser stabile non esiste (spec §7);
   il batched prefill (leva 1) è comunque prerequisito per sfruttarlo.

## 5. Cosa dice per il writeup (gate di onestà)

Il numero da pubblicare è il gap A PARITÀ di regime: **decode full-resident
4.6-5.2× (di cui 12% readback già eliminabile con pattern in casa)**; il
prefill si pubblica SOLO dopo il batching (oggi misura un'assenza, non un
confronto). Vale la regola della direction: le leve si registrano coi
numeri, non si promettono.
