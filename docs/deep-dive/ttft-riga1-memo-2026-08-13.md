# MEMO — riga 1 di `engine-ttft`: sonde e varianti del prefill

Fase di sole misure, prediction-gated. Le predizioni sono in
`docs/deep-dive/ttft-riga1-prereg-2026-08-13.md`, **committate prima** di
eseguire (`79cc3bd`, poi il banco a `d29eb0d`, poi i risultati): l'ordine dei
commit è verificabile con `git log --follow` e non è stato rinegoziato a
risultato noto.

Artefatto: `results/microbench/ttft-riga1-4090-linux-2026-08-13T13-06-23-120Z.json`
(34 celle, **0 skipped**, `timingSource: timestamp-query` su tutte,
`hostState.declared = quiescent`, RTX 4090 Laptop / Lovelace / Linux).

---

## LE DUE RISPOSTE SECCHE

**Quale forma vince, e di quanto.** Vince **`splitk`** — pesi in registri,
attivazioni in workgroup memory, K spezzato su 4 workgroup — che a M = 16 sulla
forma K2560×N9216 fa **262.881 token/s contro i 6.101 della forma attuale:
43,1x**. Non era la forma predetta (la pre-registrazione dava per vincente
`regs`, che si ferma a 20,3x): P3 cade sull'identità della vincitrice e regge
sulla sua conseguenza, perché `splitk` e `regs` hanno lo **stesso** workgroup
storage (4.096 B a M=16). La REGOLA DI STOP non scatta, e non ci va nemmeno
vicino.

**Il bersaglio dei 4 s esiste?** **No.** Il picco fp32 sostenuto di questo device
in WebGPU è **9,3 TFLOP/s** (GEMM densa, shape reale del prefill): i 12,8
TFLOP/s che il bersaglio richiede sono il **138%** di quel picco. Il conto
completo, sui *tempi misurati* delle forme vincenti, proietta un pavimento di
**~8.700 ms** — 10,1x sulla baseline di 87.618 ms, e **più del doppio** dei
4.000. La riga 5 chiuderà con `decision: "excluded-by-numbers"`.

---

## COM'È ANDATA LA SCOMMESSA: 7 enunciati, 4 pieni, 2 parziali, 1 caduto

| | enunciato | esito | il numero |
|---|---|---|---|
| **P0** | la regola di stop NON scatta (≥ 1,5x; stima 6–20x) | **CONFERMATO**, banda sottostimata | **43,1x** a M=16 |
| **P1** | picco fp32 6–16 TFLOP/s (punto 10) | **CONFERMATO** | **9,26** TFLOP/s |
| **P2** | il bersaglio cade; pavimento 5.000–15.000 ms (punto 8.000) | **CONFERMATO** | proiezione **8.665 ms** |
| **P3** | vince `regs`; shared ≤ 8.192 B | **CADUTO** sulla vincitrice, **retto** sullo shared | `splitk` batte `regs` di **2,13x**; entrambe **4.096 B** |
| **P4** | riuso ≥ 6x/12x/20x a M=8/16/32; sublineare oltre 16 | **CONFERMATO** | **8,0x / 16,0x / 32,0x**; M32/M16 = **1,101** |
| **P5** | legacy 8–20 ms; ≥ 4x a ctx lungo; **< 1,5x a ctx 388**; non compila a 16.384 | **3 su 4** | 12,30 ms; 6,76x; **5,42x** a ctx 388 (refutato); non compila ✓ |
| **P6** | throughput piatto entro ±5% spazzando il tetto | **CONFERMATO** | spread **0,1–2,3%** |

Le due cose che la pre-registrazione ha sbagliato sono informative, non
cosmetiche, e sono descritte sotto: **non era il traffico dei pesi il collo, era
l'occupancy**, e **la leva dell'attenzione non è solo a contesto lungo**.

---

## (a) IL PICCO DI CALCOLO — e perché la sonda è un PAVIMENTO, non un tetto

GEMM densa fp32, tile 64×64×8, `workgroup_size` 256, 4×4 uscite per thread,
workgroup storage 4.096 B costante nelle shape. Le due shape misurate
**interleavate fra loro**, 40 campioni in 4 passate.

| shape | p50 | ultimi 20 campioni | miglior campione |
|---|---|---|---|
| 4096³ | 17,04 ms → **8,07** TFLOP/s | 15,99 ms → 8,59 | 14,80 ms → 9,29 |
| 6336×2560×9216 (prefill 4B) | 33,53 ms → **8,92** TFLOP/s | 32,28 ms → **9,26** | 30,84 ms → 9,70 |

**Correzione di metodo, dichiarata**: la prima esecuzione misurava una shape
alla volta con 3 warm-up e 10 campioni, e dava IQR **16,6%** con p50 5,86
TFLOP/s — cioè un numero che cadeva *sotto* il bordo della banda P1 per un
artefatto di misura. I campioni mostrano una **rampa di boost**: i primi 5-8
dispatch girano a 30-39 ms e poi il regime scende a 15-16. Su dispatch da decine
di ms, dopo un idle, 3 warm-up non bastano. La misura è stata rifatta con le due
shape interleavate e 4 passate; il codice pubblicato è quello che ha prodotto il
JSON pubblicato. **Non è il numero che è stato scelto, è il disegno che era
insufficiente, e lo si dice.**

**Il tetto vero è più alto della sonda.** La stessa aritmetica utile, misurata
sul kernel q4_0 vincente sulla shape reale, arriva a **13,65 TFLOP/s** (M=32) —
*sopra* la GEMM densa fp32. Non è un paradosso: il q4_0 muove pesi 8x più
piccoli e ha un blocking sui registri con intensità aritmetica molto migliore
per byte mosso. Quindi la sonda (a) va letta per quello che è: **un limite
inferiore del picco raggiungibile**, non il picco della macchina. Il picco
teorico fp32 di questa AD103 (76 SM × 128 lane × ~1,7 GHz) è ~33 TFLOP/s: la
GEMM densa in WGSL ne prende il **28%**, il kernel q4_0 il **41%**.

Questo non salva il bersaglio, come si vede sotto — ma cambia il verso della
frase: non «WebGPU non arriva ai 12,8 TFLOP/s», bensì **«12,8 TFLOP/s sostenuti
sono raggiungibili sul singolo matmul e non sopravvivono al modello intero»**.

## (b) IL MOLTIPLICATORE MULTI-RIGA — 43,1x, e il collo non era quello previsto

Forma attuale = `gemvQuantWgsl({ batch: true })` **importata** da
`src/engine/kernels/wgsl.ts`, non ricopiata: è M GEMV replicate su `wid.z`, con
riuso dei pesi zero per costruzione. Shape dominante K2560×N9216, checksum di
ogni variante entro **2·10⁻⁸** relativo dalla base (tolleranza di lavoro 10⁻³),
IQR ≤ 0,32%.

| M | base (attuale) | `regs` | `shared` | `splitk` | best/base | MB di peso per token: base → variante |
|---|---|---|---|---|---|---|
| 1 | 6.066 tok/s | 34.265 | 10.572 | **58.962** | 9,7x | 13,27 → 13,27 (1,0x) |
| 8 | 6.064 | 114.364 | 21.492 | **219.684** | 36,2x | 13,27 → 1,66 (**8,0x**) |
| 16 | 6.101 | 123.640 | 22.900 | **262.881** | **43,1x** | 13,27 → 0,83 (**16,0x**) |
| 32 | 6.083 | 169.090 | 23.548 | **289.352** | 47,6x | 13,27 → 0,41 (**32,0x**) |

**Il riuso è pieno**: 8x / 16x / 32x, cioè il massimo teorico. Il done-when della
riga 2 chiede ≥ 8x a M ≥ 16: è servito il doppio.

**Ma il riuso non è ciò che spiega il salto.** Tre misure lo dicono, e sono la
parte del lavoro che vale più della tabella:

1. **La forma attuale non è limitata dalla banda.** A M=16 emette 212 MB di pesi
   in 2,62 ms = 81 GB/s. Un device che ne fa 300+ non è a corto di banda: è a
   corto di ALU, perché il kernel dequantizza nibble per nibble con load
   scalari.
2. **Nemmeno `regs` lo è.** Legge i suoi 13,3 MB unici in 0,129 ms = **108
   GB/s**. Un ordine di grandezza sotto la banda del device. `regs` a N=9216
   lancia **144 workgroup** su 76 SM: è **occupancy-bound**, non memory-bound.
3. **Ecco perché `splitk` vince**: stesso identico corpo, stesso identico
   workgroup storage, ma 576 workgroup invece di 144. **2,13x, tutto di
   occupancy.** La pre-registrazione aveva modellato il problema come traffico
   di memoria e ha nominato la vincitrice sbagliata per quel motivo.

**La cella `coldw` (pre-registrata) chiude l'obiezione della cache.** Con 8 copie
dei pesi ruotate a ogni dispatch (106 MB, oltre la L2):

| | L2-resident | pesi freddi (106 MB) |
|---|---|---|
| forma attuale | 2,6225 ms | 2,6157 ms |
| `regs` | 0,1294 ms | 0,1400 ms (**+8,2%**) |
| **rapporto** | **20,3x** | **18,7x** |

Il vantaggio non è un artefatto della L2: sopravvive quasi intatto a pesi
freddi. E la forma attuale è *identica* fredda o calda, il che conferma che non
sta aspettando la memoria.

**`shared` — la forma che avrebbe creato il conflitto di docket item 1 — è anche
quella che perde.** 0,6987 ms a M=16: **11,5x più lenta** di `splitk`, e l'unica
il cui workgroup storage cresce coi pesi (8.320 B a M=16, 12.416 a M=32). La
riga 4 non deve negoziare niente.

**Sul down-proj (K9216×N2560) il quadro tiene ma è più povero**: `splitk` 0,1363
ms (5,54 TFLOP/s) contro base 2,3478 (17,2x). K lungo e N corto: meno righe da
distribuire, ed è il termine che tira giù la media del blocco FFN.

**Il ginocchio in M è già passato a 16.** `splitk` guadagna 1,197x da M=8 a
M=16 e solo **1,101x** da 16 a 32 (soglia di refutazione 1,6). Il costo in
workgroup storage invece raddoppia (4.096 → 8.192 B). **M = 16 è la scelta
giusta e M = 32 non paga**: il done-when C7-2 che chiedeva M ≥ 16 è confermato
dalla misura, e l'idea di spingere oltre è esclusa.

## (c) L'ATTENZIONE A CHUNK — 6,76x, e la leva NON è solo a contesto lungo

Legacy = `attnDecodeWgsl({ batch: true })` **importato**, che instrada su
`attnDecodeLegacyWgsl` (switch a `wgsl.ts:530`). M = 16.

| variante | ctx 388 | x su legacy | ctx 6333 | x su legacy | KV emessa a 6333 | wg storage |
|---|---|---|---|---|---|---|
| legacy | 0,6881 ms | 1,00 | **12,2993 ms** | 1,00 | 3,32 GB | **25.856 B** |
| `stream` (softmax online + vec4) | 0,1280 | 5,38 | **1,8207** | **6,76** | 3,32 GB | 1.536 B |
| `gqa-stream` (+ KV una volta per gruppo) | 0,1382 | 4,98 | 2,0879 | 5,89 | 0,83 GB | 6.144 B |
| `gqa-rows2` (+ 2 righe per workgroup) | **0,1270** | **5,42** | 1,9436 | 6,33 | **0,42 GB** | 12.288 B |

Il legacy misura **12,30 ms** per chunk a ctx 6333: la pre-registrazione diceva
8–20 con punto 12. Centrato.

**Due sorprese, entrambe contro il modello mentale del piano.**

**La prima: la fusione GQA non paga, e nemmeno la fusione delle righe.** Le due
varianti che tagliano il traffico KV di 4x e di 8x sono *più lente* di quella che
non lo taglia affatto. Il motivo è lo stesso di (b): a M=16 `stream` lancia
16×16 = 256 workgroup, `gqa-stream` ne lancia 64 e `gqa-rows2` ne lancia 32 — su
76 SM. **Tutto il guadagno di 6,76x viene dalla softmax in streaming e dalle
letture vec4**, cioè da due dei tre difetti chiusi sul decode; il terzo (la
ridondanza GQA) sul prefill è già pagato dalle righe del chunk e toglierlo
costa più di quanto renda. La riga 3 va riscritta di conseguenza: **portare la
softmax in streaming, lasciar stare la fusione GQA**, o fonderla solo insieme a
uno split del contesto che restituisca i workgroup persi.

**La seconda: la leva esiste anche a contesto corto.** A ctx 388 il rapporto
misurato è **5,42x**, non «< 1,5x». La P5 ipotizzava che a contesto corto il
regime fosse dominato dal lancio dei dispatch: non lo è — il legacy paga
`scores[ctxMax]` (25.856 B di workgroup memory, allocati **a prescindere dal
contesto vero**) e quello strozza l'occupancy anche quando il contesto è 388.
Conseguenza pratica: la riga 3 rende su **tutta** la lunghezza del prompt, non
solo sulla coda.

**Il sotto-enunciato di portabilità è confermato, misurando.** Chiedendo a
`requestDevice` esattamente `maxComputeWorkgroupStorageSize: 16384`, la pipeline
del legacy **fallisce in creazione**:

> `GPUPipelineError: The total use of workgroup storage (25856 bytes) is larger
> than the maximum allowed (16384 bytes). This adapter supports a higher
> maxComputeWorkgroupStorageSize of 49152, …`

E fallisce anche a 24.576. Passa solo da 32.768 in su. Oggi non rompe niente
solo perché il motore fa concedere 30.848 B — un debito non dichiarato in
`engineNeeds`, esattamente come la riga 4 sospettava. **Tutte e tre le varianti
streaming (1.536 / 6.144 / 12.288 B) creano la pipeline a 16.384**: il debito si
estingue con la riga 3, senza toccare i limiti.

## (d) IL TETTO NEGOZIABILE — non è una leva

`adapter.limits` di questo device, riportati accanto ai numeri come chiede il
done-when:

```
maxComputeWorkgroupStorageSize    49 152   (pavimento di spec: 16 384)
maxComputeInvocationsPerWorkgroup  1 024
maxComputeWorkgroupSizeX / Y       1 024      maxComputeWorkgroupSizeZ      64
maxComputeWorkgroupsPerDimension  65 535
maxStorageBufferBindingSize  2 147 483 644   maxBufferSize     4 294 967 292
maxStorageBuffersPerShaderStage       16     minStorageBufferOffsetAlignment 256
```

Concessi al banco: 30.848 B di workgroup storage (la negoziazione di
`gpulimits.ts`), 8 storage buffer, 256 invocazioni.

Spazzata con **device distinti** e `requiredLimits` espliciti, misura
interleavata fra i device, 64 dispatch per campione:

| forma (wg storage) | 16.384 | 24.576 | 32.768 | 49.152 | spread | 49.152 vs 16.384 |
|---|---|---|---|---|---|---|
| `regs` (4.096 B) | 0,1490 ms | 0,1495 | 0,1485 | 0,1477 | 1,2% | +0,8% |
| `shared` (8.320 B) | 0,7202 | 0,7209 | 0,7210 | 0,7202 | **0,1%** | +0,0% |
| `splitk` (4.096 B) | 0,0895 | 0,0895 | 0,0879 | 0,0875 | 2,3% | +2,2% |

**Piatto entro ±5% su tutte e tre le forme.** Chiedere il massimo non paga e non
costa: su forme il cui shared non scala coi pesi il tetto negoziabile
semplicemente **non è una leva**, e la soglia di refutazione (+15% a 49.152) non
è nemmeno sfiorata. Il ginocchio è a 16.384 o sotto — cioè il motore può stare
nel **pavimento di spec**, che è la cosa che serve alla portabilità.

## L'ESTRAPOLAZIONE — proiezione dichiarata, non promessa

Formula fissata *prima* di misurare (§DISEGNO punto 8 della pre-registrazione),
qui istanziata coi tempi misurati. Shape 4B: 32 layer, di cui **8 a full
attention** (`fullInterval = 4`) e 24 Gated DeltaNet.

**Termine matmul.** Il blocco FFN di un layer a M=16 con `splitk`: gate 0,0609 +
up 0,0609 + down 0,1363 = **0,2580 ms** per 16 token, cioè **8,78 TFLOP/s
effettivi** sul mix di shape reale (non i 12,40 della sola shape migliore: il
down-proj costa). Applicato ai 50,7 TFLOP del prefill (2·P·T, P = 4·10⁹,
T = 6333): **5.776 ms**. Al miglior tasso osservato in assoluto (13,65 TFLOP/s,
M=32, sola shape gate/up): 3.713 ms — cioè **anche il caso più ottimista
consuma da solo quasi tutto il budget di 3.960 ms**.

**Termine attenzione.** `stream` a ctx 6333 costa 1,8207 ms per chunk per layer;
il costo è lineare nel contesto e il contesto medio sul prefill è ~0,501×6333:
8 layer × 396 chunk × 1,8207 ms × 0,501 = **2.888 ms**.

**Somma: 8.665 ms.** Dentro la banda P2 pre-registrata di 5.000–15.000 ms, e
vicinissima al punto dichiarato (8.000). E *non contiene ancora*: i 24 layer
DeltaNet, rmsnorm, RoPE, silu, la testa, e i ~570 dispatch per token di
overhead. **8.665 ms è un pavimento ottimista, e sta a 2,2x dal bersaglio.**

Guadagno proiettato sulla baseline: **87.618 → ~8.700 ms, cioè 10,1x**. È molto,
e non basta.

## COSA DEVE FARE IL PIANO, ADESSO

1. **La riga 2 si fa, e cambia forma vincente**: non `regs` ma **`splitk`**
   (pesi in registri + attivazioni in shared + K spezzato su 4). 43,1x a M=16 in
   isolamento, riuso 16x, 4.096 B di workgroup storage.
2. **La riga 3 si fa, e cambia contenuto**: softmax in streaming e letture vec4
   **sì** (6,76x a ctx lungo, 5,38x a ctx corto); **fusione GQA no**, costa
   occupancy più di quanto tagli traffico. Il golden
   `tests/fixtures/attn-decode-legacy.golden.json` andrà aggiornato
   deliberatamente.
3. **La riga 4 non ha più il conflitto di docket item 1**: la forma vincente non
   spende shared sui pesi, e alzare mMax a 16 costa 4.096 B. La clausola (e2b)
   si chiude dalla riga 3 (25.856 → 1.536 B), non da una negoziazione di limiti.
4. **La riga 5 chiude con `decision: "excluded-by-numbers"`** e il conto qui
   sopra. **M = 32 non è la via d'uscita** (+10% di token/s per il doppio dello
   shared).
5. **Il termine che il piano non ha ancora guardato è l'occupancy**, ed è quello
   che ha deciso *entrambe* le sonde: la vincitrice di (b) e la sorpresa di (c).
   Il prossimo goal, se ce n'è uno, parte da lì e non dal traffico di memoria.

## LIMITI DICHIARATI DI QUESTA MISURA

- Tutte le celle sono **isolate**: nessuna include il costo di lancio nel piano
  reale, né la contesa fra dispatch consecutivi. Le proiezioni sono ottimiste
  per costruzione.
- L'attenzione è misurata su un working set di 51,9 MB che sta nella L2 di
  questa Lovelace. Nel motore ogni layer paga una lettura fredda e poi riusa:
  l'approssimazione è ragionevole ma **non è stata verificata** con una cella
  `coldkv` (la pre-registrazione la chiedeva solo per i pesi).
- Il down-proj è stato misurato **solo a M=16**: la sua scalabilità in M è
  interpolata, non misurata, e pesa 1/3 del termine FFN.
- I 24 layer Gated DeltaNet **non sono stati sondati affatto**: la proiezione li
  omette, quindi il pavimento vero è più alto di 8.665 ms, non più basso.
- `shader-f16` resta assente su questo stack (sonda ri-eseguita: `exposed:
  false`), `subgroups` presente e corretto con `subgroupSize` osservato **32**,
  `packed_4x8_integer_dot_product` presente e corretto. Nessuna variante di
  questa riga li usa: c'è una leva intera, il dot-product intero impacchettato,
  che nessuno ha ancora provato sul dequant q4_0.
