# Spec — engine-fase-c2: GLM-4.7-Flash (MoE+MLA) nel motore

Goal: `.harness/goals/engine-fase-c2/GOAL.md` (contratto approvato 2026-07-31,
emendamento non-regressione). Stato: **RULING RICHIESTO** (docket item 2 del
goal). Ancore misurate: dump metadata/tensori del GGUF (sotto, 2026-07-31),
traccia e simulazione C1 (`results/engine/moe-oracle/`), floor tok/s oracolo
CPU (llama-bench C1), motore fase A/B (`src/engine/`, qwen2 a 287.5 tok/s).

## 1. Fatti dal file (dump 2026-07-31, GGUF sha d0bbdfc…, 844 tensori)

Metadata deepseek2 rilevanti: block_count **47**, leading_dense **1**, n_embd
2048, n_head **20**, q_lora **768**, kv_lora **512**, rope_dims **64**,
key_length 576 (=512+64), value_length 512, key/value_length_mla **256**
(v_head 256, qk_nope 192+rope 64), expert_count 64, used 4, shared 1,
expert_ffn 1536, gating **sigmoid** (func=2) + `exp_probs_b.bias`,
**weights_norm=true**, **weights_scale=1.8**, ffn denso (blk.0) 10240,
rope_base 1e6, rms_eps 1e-5, vocab 154880, ctx_train 202752.

**FINDING (cambia il perimetro kernel): il file "Q4_0" è a quant MISTA.**

| Classe tensore | Tipo GGML | Note |
|---|---|---|
| attn_q_a/q_b, attn_output, ffn_gate/up (denso), ffn_gate_exps/up_exps | **Q4_0** (2) | già supportato |
| **ffn_down_exps blk.5-46** (2.688 expert) | **Q4_0** (2) | già supportato |
| attn_kv_a_mqa, attn_k_b, attn_v_b | **Q8_0** (8) | già supportato |
| **ffn_down_exps blk.1-4** (256 expert), ffn_down (denso blk.0) | **Q4_1** (3) | NUOVO |
| ffn_gate_shexp/up_shexp | **Q5_K** (13) | NUOVO |
| ffn_down_shexp, output.weight | **Q6_K** (14) | NUOVO |
| token_embd | Q4_0 | embedGather esistente |
| norms, ffn_gate_inp, exp_probs_b | F32 | — |

(Verificato per-layer sul file, 2026-07-31, post-verifier: il down degli
expert è Q4_1 SOLO sui primi 4 layer MoE — 256 expert da 5.505.024 B; i
restanti 2.688 sono interamente Q4_0 — 5.308.416 B. Media pesata 5.325.512 =
esattamente l'`expertBytesQ4` misurato da residency-sim C1.)

**Decisione (a) — kernel, non repack**: si implementano dequant-GEMV per
**Q4_1** (blocco 32: scala f16 + min f16 + nibbles) e **Q5_K/Q6_K**
(superblocco 256, layout k-quant). Il repack a load verso tipi già supportati
è ESCLUSO: requantizzare cambia la numerica e sfonda il confronto con
l'oracolo (il gate golden è sul MEDESIMO file). Costo: 3 varianti WGSL della
GEMV dequant-fusa (Q8_0 c'è già), pattern di wgsl.ts (shape baked, template).

## 2. Subset GGUF deepseek2 (fase 3)

`shape.ts` guadagna `GLM47_FLASH: ModelShape` (arch "deepseek2") con la
tabella §1 come `PER_LAYER`/`PER_MOE_LAYER`/`TOP_LEVEL` attesi (dims e tipo
ESATTI dal dump; blk.0 denso: ffn_gate/up/down; blk.1-46: exps+shexp+router).
Validazione hard invariata (postura ds4: throw su ogni mismatch, conteggio
tensori totale 844 asserito). `gguf.ts`: GGML_TYPE esteso {Q4_1:3, Q5_K:13,
Q6_K:14} con `tensorByteSize` esatti (Q4_1: 20 B/32; Q5_K: 176 B/256; Q6_K:
210 B/256). Il parser resta puro; fixture sintetica nei test come per qwen2.
Il file NON transita mai per intero in un ArrayBuffer: load streaming a
sezioni (header + tensori non-expert), gli expert restano nella sorgente
(§5). u64>2^53 resta throw (il file è 17.2 GB, sotto soglia).

## 3. MLA — formulazione ABSORBED nel motore, naive solo in cpuref

**Decisione (b)**: il motore esegue l'MLA nella formulazione **absorbed**
(quella per cui llama.cpp splitta attn_k_b/attn_v_b nel GGUF): cache per
token per layer = `[c_kv (512) | k_rope (64)]` f32 nel KV store (f32-first
dev-loop; f16 dietro feature-detect come oggi) ⇒ 576 valori/token/layer,
**54 KB/token f16 sui 47 layer** (aritmetica direction §3 confermata dai
metadati). Percorso decode per head h (20 head, per token):
- `q = q_b(h) · q_a_norm(q_a(x))` → split q_nope (192) + q_rope (64), RoPE
  su q_rope e k_rope (rope_dims 64, base 1e6, NEOX-style da verificare in
  deepseek2.cpp fase 4);
- score: `q_nope^T · k_b(h) · c_kv + q_rope^T · k_rope` — `k_b` [192×512]
  assorbe la decompressione K (per head, Q8_0);
- out: `attn · c_kv → v_b(h) [512→256] → attn_output` (5120=20×256 → 2048).
La variante naive (decomprimere K/V per head nel KV store) NON entra nel
motore: cache 20×(256+192+64) vs 576 = 18× più grande, fuori budget. Vive in
`cpuref.ts` f64 come riferimento di conformance layer-level (gate doppio,
landmine oracolo-q8). Attention split sul contesto (attnsplit, lezione B2)
si applica al path absorbed (score su 576, softmax log-sum-exp invariata).

## 4. MoE (fase 5)

- **Router** (per i 46 layer MoE): logits = `ffn_gate_inp · h` (f32, GEMV
  f32 nostro), `sel = sigmoid(logits) + exp_probs_b` per la SELEZIONE top-4
  (replica bit-fedele di C1 `predict()`, tie-break indice minore); pesi di
  mixing = `sigmoid(logits)` dei selezionati SENZA bias, normalizzati a
  somma 1 (`weights_norm=true`) × **1.8** (`weights_scale`) — da verificare
  riga-per-riga in `build_moe_ffn` (llama-graph.cpp, commit 5f55650) in
  fase 5, come C1 fece per la selezione.
- **Expert FFN**: per ciascuno dei 4 selezionati, catena
  `silu(gate_exps[e]·h) ⊙ up_exps[e]·h → down_exps[e]` con GEMV dequant-fusa
  (gate/up Q4_0, down Q4_1) sugli slot residenti (§5); accumulo pesato nel
  buffer di output. Decode = 1 token ⇒ 4 GEMV-chain/layer; il batch K>1 del
  decodeBatch esegue per-token (union expert), niente grouped-GEMM in C2
  (ottimizzazione da docket se serve).
- **Shared expert**: sempre attivo, path denso `shexp` (Q5_K/Q6_K), stessa
  catena silu-gate; somma con i routed PRIMA di attn_output? NO — somma al
  risultato MoE (ordine esatto da deepseek2.cpp, verifica in codice fase 5).
- **Layer 0 denso**: ffn classico 10240 (Q4_0/Q4_1), riuso kernel fase A.
- Il tap `ffn_norm` per il futuro LOOKA/prefetch di C3 è già un tap
  hidden-state del contratto (§4.4 direction): il forward lo espone.

## 5. Residenza minima (fase 5) — il MINIMO che fa girare 17 GB in 16 GiB

- **Sorgente byte**: il GGUF vive in **OPFS** (copiato una volta al primo
  load, SHA-256 verificato). Questo NON è il paging di C3: nessuna policy su
  OPFS, nessun tier, nessun prefetch — OPFS è semplicemente DOVE sta il file
  in un browser (un ArrayBuffer da 15.6 GB di staging in un tab su host
  31 GB non è credibile; il warm re-read misura 10-11 GB/s). ⚠ Deviazione
  dalla lettera del contratto ("staging in RAM"; "OPFS-backed experts" è
  listato must-docket) → **questo è uno dei punti del RULING**.
- **Residente fisso in VRAM**: tutto il non-expert (1.53 GB misurati C1:
  attn, norms, router, shexp, denso, embd, output) + KV + scratch.
- **Slab expert**: DUE size-class esatte (fatto §1: 5.308.416 B per i 2.688
  expert down-Q4_0, 5.505.024 B per i 256 down-Q4_1 di blk.1-4); N buffer
  GPU ≤ maxStorageBufferBindingSize negoziato per classe, slot indirizzato
  (classe, buffer, offset), bind group per-slot precomputati al load. Slot
  totali = riparto del budget residuo tra le classi in proporzione al parco
  (256/2688); atteso ~2.5k complessivi sul dev box. (Alternativa scartata:
  classe unica paddata a 5.505.024 = ~0.5 GB sprecati ≈ 100 slot.)
- **Policy**: **LRU pura** (il simulatore C1 dice che a budget 87% LRU fa
  96.4% decode hit — la differenza col tuned è materiale solo nel regime 2×,
  che è C3). Miss ⇒ read OPFS (SyncAccessHandle, worker) → staging (un buffer
  riusato) → `writeBuffer` allo slot LRU-vittima. SINCRONO nel forward
  (stallo dichiarato e misurato: è il numero che C3 deve battere).
- **Telemetria** (liv.1, zero-overhead spenta): hit/miss per token, byte
  letti da OPFS, ms di stallo upload, occupazione slot — nel report bench.

## 6. Piano statico e forward (fasi 4-6)

Riuso dell'impianto fase A/B: bind group precomputati al load, un submit per
token (decode), prefill chunked, attnsplit, decodeBatch K≤8, error scope
validation/oom come contratto, `pos === kvLen` hard. Differenze GLM: 47
layer, i bind group degli expert slot sono per-slot (il piano riferisce lo
slot, la residenza aggiorna la mappa expert→slot); dispatch/token attesi
~5×47 + 4×3×46 + router ≈ **~800-1000** (misura, NON gate — il ≤100 di B è
esplicitamente non ereditato dal contratto). Se il decode GLM regge il floor
(§8) con ~900 dispatch, la fusione è materia di C3+/D, non di C2.

## 7. Conformance (fase 4 layer-level, fase 6 full-model)

- **Gate doppio** (landmine C1: l'oracolo CPU quantizza le attivazioni q8):
  (i) vs `cpuref` f64 (naive MLA + MoE ref): argmax agreement ≥99% sulle
  posizioni golden; (ii) vs golden oracolo: top-1 agreement ≥97%.
  Metrica secondaria registrata (non gate): KL media sui top-32 e max |Δ|
  logit sull'intersezione. Corpus: golden fase 1 (8 prompt C1 × ≤128 gen,
  `results/engine/golden/glm47flash/`).
- **Routing** (fase 5) — **EMENDATO, ruling PI 2026-07-31 (docket item 4,
  opzione a)**: il set-match top-4 vs traccia oracolo è MISURA INFORMATIVA
  (report JSON in results/engine/, per-fase e per-layer), NON gate. Il gate
  di correttezza del percorso MoE è il gate doppio full-model qui sopra —
  in particolare il ramo (i) vs cpuref-f64, che il discriminatore di it.9
  ha mostrato essere il confronto corretto (esatto-vs-esatto). Motivo
  dell'emendamento: la soglia ≥99% originale era tarata sull'autotest C1
  (recall 0.999 usando l'hidden DELL'ORACOLO) e non regge sul replay
  engine-vs-oracolo — l'oracolo CPU quantizza le attivazioni q8 e ribalta i
  near-tie 4°/5° (misurato: decode 85.85% p4 / 94.11% p7; cpuref-f64 esatto
  IDENTICO al motore, stessi 28/28 mismatch — routing-cpuref-analysis).
  Il router resta verificato bit-fedele a build_moe_ffn (it.6) e fedele
  all'aritmetica esatta (it.9). CONTINGENZA (ruling): se la fase 6
  sollevasse dubbi che richiedono di nuovo un oracolo sul routing, si
  rigenera una build llama.cpp con attivazioni f32 pure e si ri-traccia
  (era l'opzione b). Testo originale del gate: match ≥99% degli insiemi
  su (posizione decode, layer), stop sotto soglia.
- Sanity permanenti: error scope su ogni submit; diag cold-start; niente
  `tail` sui pageerror (landmine).

## 8. Bench e gate (fase 6)

- **Gate non-regressione GLM** (emendamento contratto): decode ≥ **13.43**
  tok/s, prefill ≥ **56.58** tok/s (floor = oracolo CPU C1, r=2; il bench
  del motore usa lo stesso protocollo canonico di B2: mediana su run
  ripetute, finestre dichiarate, ctx del corpus golden).
- **Gate non-regressione Qwen**: conformance fase A verde e bench first-light
  ≥ ultimi valori committati (K=8 287.5 ±2.3 → soglia = 287.5−2σ; K=1 238.3
  −2σ; prefill 697.8 ms +2σ), stessa metodologia same-day di B2.
- Report: tok/s, dispatch/token, hit-rate residenza, stallo upload ms/token,
  occupazione — input diretti del goal C3.

## 9. Fasi, timebox, rischi

Fasi = PHASES.md (3: reader; 4: MLA; 5: MoE+residenza; 6: E2E+gate).
Timebox: fase 4 **4 iterazioni**, fase 5 **4 iterazioni** (le due incognite
vere: absorbed-MLA in WGSL e slab+LRU); sforamento ⇒ docket con analisi, non
scorciatoie (contratto). Rischi dichiarati:
1. **k-quant WGSL** (Q5_K/Q6_K): layout superblocco intricato; mitigazione:
   dequant scalare "correttezza prima" (le shexp sono GEMV 2048×1536, non il
   collo di bottiglia), ottimizzazione solo se il floor §8 fallisce.
2. **VRAM contesa dal compositor** (16 GiB condivisi): budget con slack 10%
   e slots calcolati a runtime dai limiti negoziati; failure = device lost
   (landmine nota).
3. **Fedeltà router f32-vs-oracolo**: C1 ha già dimostrato la replica a
   0.999997 in f32 CPU; la GPU f32 può divergere sui pareggi — il gate ≥99%
   assorbe; sotto, si confronta il percorso su cpuref.
4. **Stallo miss sincrono**: a hit 96.4% e miss warm ~0.5 ms ⇒ ~3.5 ms/token
   attesi; se il floor decode fallisse SOLO per gli stalli, il numero è
   comunque il deliverable (input C3) e la deroga va a docket.
5. **Osservatore**: telemetria residenza a contatori, niente timestamp per
   misura (tsq resta il protocollo B2, senza tap).

## 10. Decisioni chieste al ruling (sintesi)

(a) kernel Q4_1/Q5_K/Q6_K invece di repack (— §1);
(b) MLA absorbed nel motore, naive solo cpuref (— §3, supera l'[ASSUMED
    naive-first] del contratto: la naive non sta nel budget KV);
(c) OPFS come sorgente dei byte expert nella residenza minima (— §5,
    deviazione dichiarata dalla lettera del contratto);
(d) LRU pura come policy C2 (tuned/prefetch = C3);
(e) soglie di conformance §7 (99/97/99) e protocollo bench §8;
(f) timebox 4+4 per fasi 4-5.
