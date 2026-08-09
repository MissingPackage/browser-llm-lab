# Spec — engine-fase-q1: generalizzazione alla famiglia Qwen 3.5/3.6 (4B + 9B + 35B-A3B)

Data: 2026-08-10 · Goal: `.harness/goals/engine-fase-q1/GOAL.md` (contratto v1, aperto al tag `goal-engine-fase-q1-start`)
Input: recon famiglia (`docs/engine/study/2026-08-09-qwen35-family-recon.md`),
baseline nativa (`results/engine/native-baseline-llamacpp-vulkan-2026-08-09.json`),
studio LlamaWeb (`docs/engine/study/llamaweb.md`), config HF verificati
2026-08-09 (9B, 35B-A3B) e 2026-08-10 (4B), ricognizione hub GGUF 2026-08-10
(questa spec, §2).

## 0. Obiettivo e ratifica del contratto

La fase C ha chiuso il motore sul modello-tesi (GLM-4.7-Flash: decode 15.64,
TTFT 12.9 s, qualità bit-invariata). q1 generalizza: tre taglie Qwen 3.5/3.6
col metodo di fedeltà intatto, riferimenti ai tier consumer, e la
decomposizione del gap nativo che la baseline 2026-08-09 ha aperto (4.3×/34×
— quanto è kernel, quanto è paging).

**Ratifica formale (pattern c3a item 3 / c3b item 4).** Gli [ASSUMED] del
contratto v1 sono stati approvati dal PI in chat 2026-08-10 ("per il resto
tutto ok") con un emendamento (taglia mobile = 4B, il 2B è eventuale futuro)
e qui si ratificano: (i) taglie 4B/9B/35B-A3B; (ii) nessun floor assoluto
per la famiglia nuova — si negozia coi numeri della prima misura (docket);
(iii) tier consumer emulati con cap di budget slab sulla 4090, hardware
reale PI-gated; (iv) confine delle leve kernel (§7); (v) soglie golden
per-modello a ratchet, nessun import del PIN GLM; (vi) ~60 GB disco per i
GGUF. Questa spec NON tocca gate né soglie del contratto ⇒ nessuno STOP di
ruling (regime C3b); un punto è DEVIAZIONE dichiarata dal testo letterale
del contratto e va a docket come registrazione: §2, quant del MoE.

## 1. Modelli pinnati (ricognizione hub 2026-08-10)

| Modello | Repo | File | Byte | sha256 |
|---|---|---|---|---|
| Qwen3.5-4B (denso, mobile-target) | `unsloth/Qwen3.5-4B-GGUF` | `Qwen3.5-4B-Q4_0.gguf` | 2 583 221 408 | `298fcb5fe7a77ccc79745ae24751560c5ac56874caff4bb39b1f2055bd72b8bb` |
| Qwen3.5-9B (denso) | `unsloth/Qwen3.5-9B-GGUF` | `Qwen3.5-9B-Q4_0.gguf` | 5 379 417 312 | `17670346b4260ddcb0173965145155885024f3c9a4a24389a3370751edbcde24` |
| Qwen3.6-35B-A3B (MoE) | `unsloth/Qwen3.6-35B-A3B-GGUF` | `Qwen3.6-35B-A3B-UD-Q4_K_S.gguf` | 20 893 015 008 | `a8138f183e3993f12cdc23afd2babb8cdb084e64088ce4a256d49101d47b949c` |

SHA = oid LFS dal pointer raw HF (verificate 2026-08-10); il download in
fase 2 ricontrolla sha256 su file scaricato (script, exit 0 = gate).
Licenza: Apache 2.0 su tutta la famiglia. Disco: ~29 GB totali, dentro i
60 dell'authority. I repo `*-MTP-GGUF` (draft MTP incluso) esistono e sono
il riferimento per la fase D: NON si scaricano in q1.

## 2. Decisione quant — deviazione dichiarata (docket)

**Il Q4_0 del 35B-A3B non esiste da fonte pulita**: verificati 2026-08-10
unsloth (solo UD-IQ4/UD-Q4_K), Qwen ufficiale (non pubblica GGUF per le
famiglie 3.5/3.6), e la ricerca hub (gli altri Q4 del base model sono IQ/
APEX; il resto sono finetune, inaccettabili come base). Il contratto dice
"quant Q4 famiglia unsloth": la lettera è salva, ma lo spirito "stessa
quant del metodo GLM (Q4_0)" no. Decisione (meccanismo, ruling
decide-dont-escalate):

- **Densi 4B/9B = Q4_0** — kernel esistenti, parità piena col metodo GLM.
- **MoE 35B = UD-Q4_K_S** — prezzo: dequant **Q4_K** per gli expert (la
  macchineria K-quant esiste già: `kquantfast.ts` serve la head Q6_K di
  GLM; letteratura kernel: `mul_mat_decls.tmpl` di llama.cpp, studio
  LlamaWeb §3). Il file UD è un MIX per-tensore deciso da unsloth: fase 2
  enumera i type effettivi dal header GGUF e il reader FALLISCE RUMOROSO su
  type non supportato (mai fallback silenzioso — lezione LlamaWeb, cliff
  supports_op).
- Alternativa SCARTATA: auto-quantizzare Q4_0 dai safetensors ufficiali —
  sfora l'authority disco (~70 GB BF16 + conversione > 100 GB), ricade nel
  must-docket "quant nuove", e perde la proprietà più forte del metodo:
  **stessi byte nel motore e nell'oracolo** (llama-bench gira sul file
  pinnato, byte per byte).

Conseguenza dichiarata sul confronto: il gap kernel-vs-paging (§6) si
misura sui DENSI in Q4_0 (parità di formato con GLM); i numeri del 35B
hanno formato expert diverso (≈4.5 bpw vs 4.5 Q4_0 — slot ≈1.77 MB, stima
recon confermata) e NON si confrontano testa-a-testa coi numeri GLM per
formato, solo per regime.

## 3. Architettura verificata (config HF, non assunta)

Scheletro COMUNE (config 4B letto 2026-08-10; 9B/35B recon 2026-08-09):
`full_attention_interval: 4` (3 linear : 1 full), linear attention famiglia
Gated DeltaNet (`linear_conv_kernel_dim: 4`, 16 K-head / 32 V-head @ 128,
`mamba_ssm_dtype: float32`), GQA sui layer full (head_dim 256,
`partial_rotary_factor: 0.25`, mrope interleaved `mrope_section [11,11,10]`,
rope_theta 1e7), vocab 248 320, ctx 262 144, MTP head 1 layer (FUORI q1),
`attn_output_gate: true`, visione in tower separata (path testo
autosufficiente).

| | 4B | 9B | 35B-A3B |
|---|---|---|---|
| layer (full-attn) | 32 (8) | 32 (8) [VERIFY fase 2 dal GGUF] | 40 (10) |
| hidden | 2560 | ~4096 [VERIFY fase 2] | 2048 |
| head Q/KV | 16/4 | [VERIFY fase 2] | [VERIFY fase 2] |
| FFN / MoE | inter 9216 | inter [VERIFY] | 256 expert top-8 + 1 shared, moe_inter 512 |
| tie embeddings | **true** | [VERIFY fase 2] | [VERIFY fase 2] |
| Q4 file | 2.58 GB | 5.38 GB | 20.9 GB |

I [VERIFY] si chiudono in fase 2 dal header GGUF (fonte più vicina ai byte
che eseguiamo); nessun gate dipende da essi prima di fase 2. Il conto KV:
KV vive SOLO sui layer full (8/32 o 10/40) ⇒ a parità di contesto ~4× meno
di un GQA pieno; lo stato lineare è O(1)/layer (16×128×128 f32 + conv
state). La formula slab ctx-aware si parametrizza su
`nLayerFull × kvPerLayer(headDim, nKvHead)` in fase 4.

## 4. Piano numerico DeltaNet (il rischio dominante)

Semantica di riferimento: implementazione CPU di ggml per `qwen3_5`
(oracolo eseguibile, stessa fonte dei golden) + kernel Triton ufficiali
come letteratura secondaria. Strategia in 4 passi, gate meccanici:

1. **cpuref-f64 prima del WGSL** (fase 3): porting TS a doppia precisione
   della catena linear-attn (conv k4 → proiezioni → delta rule gated →
   stato ricorrente → output gate), testato per plausibilità su input
   sintetici E su attivazioni reali estratte dal 4B via llama.cpp
   (`--dump`-style probe o eval intermedio; il campione si committa).
2. **Kernel WGSL contro cpuref** (fase 3, ktest): f32 con accumuli f32,
   stato ricorrente f32 OBBLIGATO (config), var-in-loop azzerate
   esplicitamente (landmine WGSL/Tint storica). Tolleranze: max-abs-err e
   cosine RIPORTATI per lunghezze {1, 16, 128, 512}; gate SECCO solo
   sull'argmax dei logits end-to-end (near-tie mai gateati, regola C2).
3. **Localizzazione della divergenza**: la ricorrenza accumula errore con
   la sequenza ⇒ il protocollo confronta a lunghezze crescenti e per-layer,
   così una deriva si attribuisce al layer/passo, non al modello intero.
4. **Chunked prefill della linear attention**: lo stato passa fra chunk
   (analogo del `pos === kvLen` hard di GLM); il test di equivalenza
   chunked-vs-unchunked (stessi logits) è parte dei ktest di fase 3.

## 5. Protocollo di conformance per-modello (metodo GLM trasferito)

Per CIASCUNO dei 3 modelli, nell'ordine:
1. **Tokenizer**: token id identici a llama.cpp sul corpus di conformance
   committato (testi: multilingua, code, emoji, special/chat template,
   edge whitespace). Test in `npm test`.
2. **cpuref**: argmax == cpuref-f64 sul campione ratificato (256 e 512
   token, come GLM).
3. **Golden**: top-1 vs llama.cpp su corpus full; soglia FISSATA alla
   prima run verificata (ratchet, mai abbassata). Full-corpus SOLO per
   fissare/verificare riferimenti (ruling 2026-08-09).
4. **Firma routing** (solo 35B): conteggi esatti expert×layer sul corpus,
   registrata e mai più mossa (metodo c3a item 14b).
Oracolo: llama.cpp locale b10333 (Vulkan) — il supporto `qwen3_5(_moe)` si
VERIFICA al primo run di fase 2 (recon lo dà per maturo; se il binario
locale fosse troppo vecchio per la famiglia: upgrade pinnato a build
dichiarata, stesso protocollo — registrazione a docket, non ruling).

## 6. WP decomposizione gap + tier (fasi 5 e 8)

**Full-residency** (fase 5): 9B Q4_0 interamente in VRAM (5.4 GB su 16) vs
llama.cpp Vulkan `-ngl 99`, STESSO file/driver/protocollo p512/n64, host
dichiarato. Senza paging nell'equazione, il ratio residuo È il gap
kernel+dispatch+safety-check; la scomposizione usa i numeri LlamaWeb come
prior (safety check 14-42% prefill; tuning per-device +41%) e la telemetria
nostra (gpuBusy per-op) come misura. Output: JSON + doc di studio con leve
ordinate per ROI misurato.

**Tier** (fase 8): budget slab cap sulla 4090 — 8 GiB / 12 GiB / 16 GiB
per il 35B (regimi ~45% / ~65% / ~85-90% di residenza, recon §5); recall
prefetch rimisurato sul router 256-wide top-8 (il 91.92% di GLM non si
assume); bandmodel rifittato (F(h), costi per-fetch con slot 1.77 MB — il
costo per-miss cala ~3×, la frequenza sale: il fit è da rifare, non da
scalare).

**Tier mobile** (fase 8, definizione operativa DICHIARATA): riferimento =
classe 8 GB RAM unificata (S22). Il proxy sulla 4090 emula la PRESSIONE DI
MEMORIA, non il silicio: 4B full-resident con budget slab cap 3 GiB
(modello 2.58 GB + KV + attivazioni ≤ cap; ctx di prova da fissare in fase
8 col conto KV di §3). Il JSON riporta decode/prefill/TTFT al cap + una
PROIEZIONE parametrica via bandmodel con banda storage come PARAMETRO
LIBERO dichiarato — i numeri di banda mobile NON si estrapolano (lezione
colibri): i numeri veri arrivano solo dal device, che è PI-gated. Il
deliverable mobile di q1 è riferimento emulato + proiezione parametrica,
NON una claim di performance mobile.

## 7. Stato subgroup-matrix e confine delle leve (chiusura [VERIFY] LlamaWeb)

Stato verificato 2026-08-10 (fonti: Chrome Developers blog "New in WebGPU"
125→146 e "What's next for WebGPU"; Khronos WebGL+WebGPU update SIGGRAPH
lug 2026; codice llama.cpp @ e9fa0781):
- **subgroups**: SHIPPED da Chrome 134 (stabile; subgroup_id/num_subgroups
  e subgroup_uniformity in Chrome 145).
- **subgroup-matrix** (tensor core in WGSL): IN STANDARDIZZAZIONE ATTIVA,
  **nessun origin trial in Chrome 146-148 (ago 2026)**; esiste solo come
  feature sperimentale Dawn (`chromium-experimental-subgroup-matrix`) — in
  llama.cpp la detection è sotto `#ifndef __EMSCRIPTEN__`, cioè solo
  nativa. ⇒ Il [VERIFY] dello studio LlamaWeb è CHIUSO: nel browser di
  oggi i tensor core NON sono raggiungibili da WGSL stabile.
- Conseguenza sulle fasi: la fase 6 spike prova EMPIRICAMENTE il flag sul
  nostro Chrome/Dawn (la ground truth è il device, non il blog); se la
  feature non è esposta, il done-when di fase 6 chiude con
  "impraticabilità documentata con fonte". Leve DENTRO il goal, dietro
  flag e solo se il WP le conferma in cima al ROI: `dot4I8Packed` (WGSL
  core, disponibile OGGI — path MMVQ-style per il decode) e tuning tile
  per-device (prior +41%, LlamaWeb). FUORI dal goal: WASM-SIMD
  compute-at-data (registrata in direction), promozione di qualunque leva
  a default (docket).

## 8. Cosa NON è in questa spec (confini)

Visione/multimodale (tower separata, path testo autosufficiente — ma il
tokenizer DEVE gestire mrope_section in text-only, rischio recon §7.2);
MTP/spec-dec (fase D); 2B (eventuale futuro, ruling PI); floor famiglia
nuova (docket alla prima misura); benchmark pubblico (standby); hardware
reale mobile/M4 (PI-gated); WASM-SIMD.

## 9. Registrazioni a docket q1 (con questa spec)

- item: spec depositata (questa) — registrazione, non richiede ruling.
- item: deviazione quant MoE (§2) — REGISTRAZIONE con motivazione; se il
  PI preferisce l'auto-quant Q4_0, è un ruling che riapre authority disco
  e must-docket.
- item: oracolo — eventuale upgrade build llama.cpp per supporto famiglia
  (§5), registrazione se serve.
