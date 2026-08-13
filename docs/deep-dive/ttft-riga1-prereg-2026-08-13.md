# PRE-REGISTRAZIONE — riga 1 di `engine-ttft` (sonde e varianti del prefill)

Scritta e **committata PRIMA** di eseguire qualunque misura del work package.
`git log --follow` di questo file è il timestamp che rende falsificabile tutto il
resto: il grade si fa contro QUESTO testo, senza rinegoziare le soglie a
risultato noto.

Anchor: `.harness/goals/engine-ttft/{GOAL.md,PHASES.md,docket.md,journal.md}`,
`results/engine/ttft-baseline-4b-prompt0-2026-08-13.json` (it.1),
`results/microbench/kernel-decode-fase0-4090-linux-2026-08-12T23-49-24-010Z.json`
(fase 0 del goal precedente — è la baseline strumentale di questo banco),
`docs/deep-dive/kernel-decode-fase0-prereg-2026-08-13.md` (la forma che ha
funzionato: si eredita il disegno, non solo il rito).

---

## TESI DA FALSIFICARE

Due enunciati distinti, che possono cadere separatamente:

**(T1)** Esiste una forma di moltiplicatore multi-riga che, a M ≥ 16, riduce di
≥ 8x i byte di peso letti per token prefillato rispetto alla forma attuale
(`gemvQuantWgsl` con `batch: true` = M GEMV replicate su `wid.z`, riuso ZERO per
costruzione), **senza sfondare il workgroup storage negoziabile del device**.

**(T2)** Il bersaglio dei 4 s è raggiungibile sul picco di calcolo reale di
questo device in WebGPU.

Sono separati apposta: T1 può reggere e T2 cadere. È esattamente lo scenario in
cui la riga 2 vale la pena di essere fatta (si passa da 87,6 s a un ordine di
grandezza diverso) e la riga 5 chiude comunque per esclusione coi numeri.

## REGOLA DI STOP (dal contratto, pre-accettata dal PI, non negoziabile)

Se **nessuna** variante supera la forma attuale di ≥ 1,5x sulla propria metrica,
la riga CHIUDE IL GOAL e il piano si riscrive invece di procedere. Un risultato
negativo vale quanto uno positivo: non si cerca di salvarlo.

---

## IL CONTO CHE RENDE LE PREDIZIONI FALSIFICABILI

Tutto da artefatti già in albero, nessun numero nuovo:

- Baseline (it.1): TTFT a caldo **87.618 ms** (prefill 87.582 + firstMs 36),
  prefill sequenziale **72,30 tok/s** = 13,83 ms/token. Bersaglio 4.000 ms ⇒
  **21,9x**.
- Bersaglio in rate: 6333 token in ≤ 3,96 s = **1.600 tok/s** = 0,625 ms/token.
- Pesi Q4_0 del 4B ≈ **2,25 GB** (4e9 × 0,5625 B). Il prefill sequenziale li
  rilegge una volta per token: 2,25 GB / 13,83 ms = **163 GB/s effettivi a
  livello di motore** — cioè poco più di metà dei 300 GB/s già dimostrati da
  questo motore sulla scansione KV, e ben sotto i ~390 GB/s del GEMV veloce
  isolato (`vec4-rows2-sg`, K9216×N2560).
- **Con riuso perfetto a M=16**: 396 passate × 2,25 GB = 891 GB in 3,96 s =
  **225 GB/s** ⇒ *sotto* la banda già dimostrata. **A M ≥ 16 la banda smette di
  essere il vincolo stringente.** Diventa il calcolo.
- Calcolo: 2·P·T con P = 4e9, T = 6333 ⇒ **50,7 TFLOP**, cioè **12,8 TFLOP/s
  sostenuti** in 3,96 s. Questo numero non è mai stato misurato qui: è la
  sonda (a), ed è l'unico dato che decide T2.

Vincoli di piattaforma già accertati e non ri-discussi: `shader-f16` **assente**
su questo stack Chrome/Linux (niente tensor core per questa via), `subgroups`
presente e corretto con `subgroupSize` osservato **32**,
`packed_4x8_integer_dot_product` presente. `adapter.limits
.maxComputeWorkgroupStorageSize` = **49.152** contro i 16.384 del pavimento di
spec; oggi il motore ne fa concedere 30.848.

**Baseline strumentale del banco** (fase 0 del goal precedente, stesso device,
stesso harness) — le celle contro cui si legge tutto il resto:

| cella | p50 | metrica |
|---|---|---|
| gemv q4_0 `base` (scalare, la forma che il prefill usa oggi) K2560×N9216 | 0,1642 ms | 143,7 G pesi/s |
| gemv q4_0 `vec4-rows2-sg` (il veloce del decode) K2560×N9216 | 0,0385 ms | 613,4 G pesi/s |
| gemv q4_0 `base` K2560×N248320 (lm_head, l'unica non L2-resident) | 5,8517 ms | 108,6 G pesi/s |
| attn-decode `base` n=6333 | 9,980 ms | 5,2 GB/s effettivi |
| attn-decode `split` n=6333 | 0,296 ms | 175,3 GB/s effettivi |

---

## PREDIZIONI (7 enunciati; tutti sulla stessa sessione browser, stesso device, host `quiescent`)

### P0 — REGOLA DI STOP: **NON scatterà**

Almeno una variante multi-riga supera la forma attuale (`gemvQuantWgsl` con
`batch: true`, M GEMV su `wid.z`) di **≥ 1,5x** sul throughput in token/s a
M = 16, sulla forma K2560×N9216.

**Stima puntuale: 6–20x**, non 1,5. La ragione per cui la stima è così alta e
non è ottimismo: la forma attuale non è "un moltiplicatore lento", è *il kernel
scalare del 2026-07 replicato M volte con riuso zero*. Batte già 4,3x per il
solo passaggio a vec4 a M=1 (143,7 → 613,4 G pesi/s), e il riuso a M=16 è un
fattore ulteriore e ortogonale.

**REFUTATA se**: il rapporto migliore su quella cella è < 1,5x ⇒ regola di stop,
la riga chiude il goal.

### P1 — SONDA DEL PICCO DI CALCOLO (done-when *a*)

Una GEMM densa fp32 in WGSL (tiling classico shared+registri, `workgroup_size`
256, tile 64×64×8, shape quadrata 4096³ **e** le shape reali del 4B) misura
**6–16 TFLOP/s sostenuti**, stima puntuale **10 TFLOP/s**.

Conseguenza aritmetica che si registra ORA: i 12,8 TFLOP/s richiesti dal
bersaglio sono quindi al **80–215% del tetto fp32 WebGPU di questo device**
(punto ~128%). Non al 40%.

**REFUTATA se**: la sonda misura < 6 o > 16 TFLOP/s.

### P2 — IL BERSAGLIO: **T2 CADE, la riga 5 chiuderà per esclusione coi numeri**

Dato P1, e dato che il prefill vero non è una GEMM fp32 nuda (dequantizza q4_0 a
runtime — unpack dei nibble e scala per peso — e paga attenzione, rmsnorm,
softmax, RoPE e ~570 dispatch per token di overhead), la frazione del picco
sostenibile end-to-end è **≤ 50%** della sonda.

Enunciato: **il pavimento raggiungibile del TTFT a caldo su questo device è
5.000–15.000 ms** (punto ~8.000), cioè un miglioramento di **6–17x** sulla
baseline di 87.618 ms, che **non raggiunge i 4.000 ms**. La riga 5 chiuderà con
`decision: "excluded-by-numbers"`.

**REFUTATA se**: la sonda P1 misura **≥ 26 TFLOP/s** (2x il requisito) — nel qual
caso il bersaglio è dentro il budget di calcolo e il collo è solo la forma del
kernel. Refutata anche se una variante di P3, estrapolata a modello intero col
conto dichiarato in §DISEGNO, proietta < 4.000 ms.

### P3 — QUALE FORMA VINCE, e la sua conseguenza sul conflitto di docket item 1

Fra le quattro forme misurate a M = 1, 8, 16, 32 —
(i) attuale `batch` su `wid.z`;
(ii) **tile con i pesi in registri** (ogni thread tiene il frammento di peso e
accumula su M colonne di X, X in shared);
(iii) tile con i pesi in shared;
(iv) split-K —
**vince (ii)** a M = 16 sulla forma K2560×N9216.

E la conseguenza che conta per il piano: **il workgroup storage MISURATO della
forma vincente a M=16 è ≤ 8.192 B**, cioè *sotto* il pavimento di spec di 16.384.
Il motivo è strutturale, non fortunato: in (ii) la shared tiene solo il blocco di
attivazioni M×blockK (16×32×4 = 2.048 B), non i pesi, quindi lo shared scala con
M ma con un coefficiente 128 B/riga invece dei 3.856 B/riga del kernel fuso.

Se regge, **il conflitto strutturale registrato a docket item 1** («alzare mMax
peggiora (e2a)») **non esiste per la forma vincente**, e la riga 4 può restare
sotto 16.384 senza negoziare nulla.

**REFUTATA se**: (iii) o (iv) vince di > 10% su (ii); **oppure** la forma
vincente richiede > 16.384 B di workgroup storage a M=16.

### P4 — RIUSO, che è il done-when della riga 2 verificato in isolamento (done-when *b*)

Per la forma vincente, `bytesEmitted` per token prefillato rispetto alla forma
attuale scende di:

| M | riduzione predetta | ideale |
|---|---|---|
| 8 | ≥ 6x | 8x |
| 16 | **≥ 12x** | 16x |
| 32 | ≥ 20x | 32x |

E il throughput in token/s è **sublineare oltre M=16**: `tokens/s(M=32) ≤ 1,6 ×
tokens/s(M=16)`. Cioè **il ginocchio della curva è fra M=16 e M=32**, non oltre.

**REFUTATA se**: a M=16 la riduzione misurata è < 12x (il done-when del contratto
chiede ≥ 8x: fra 8x e 12x il contratto è soddisfatto ma QUESTA predizione è
refutata, e va detto), **oppure** se `tokens/s(M=32) > 1,6 × tokens/s(M=16)` —
nel qual caso il ginocchio è più in là e M va spinto oltre 32.

### P5 — ATTENZIONE A CHUNK DEL PREFILL (done-when *c*)

Il legacy (`attnDecodeLegacyWgsl`, raggiunto da `attnDecodeWgsl` con
`batch: true`, switch a `wgsl.ts:530`) a **ctx 6333 con M=16** misura **8–20 ms
per chunk** (punto 12), cioè 0,5–1,3 ms per token. La variante migliore (softmax
in streaming + KV letta una volta per gruppo GQA + letture vec4 + split del
contesto) raggiunge:

- **≥ 4x** sul ms-per-token a ctx ≥ 6000;
- **< 1,5x** a ctx 388, dove il termine è piccolo e il regime è dominato dal
  lancio dei dispatch, non dalla banda.

Il secondo pezzo è la parte informativa: dice che la leva della riga 3 **esiste
solo a contesto lungo**, cioè esattamente dove il goal la vuole.

Sotto-enunciato di portabilità, misurato non dedotto: il legacy dichiara
`scores: array<f32, ctxMax>` ⇒ **4·6400 + 256 = 25.856 B**, che **NON entra** nel
pavimento di spec. Chiedendo a `requestDevice` esattamente
`maxComputeWorkgroupStorageSize: 16384`, la pipeline del legacy a ctxMax 6400
**fallisce in creazione**. Oggi non rompe solo perché il motore fa concedere
30.848.

**REFUTATA se**: il rapporto a ctx ≥ 6000 è < 4x; **oppure** se a ctx 388 è
≥ 1,5x (la leva varrebbe anche a contesto corto e il modello mentale è
sbagliato); **oppure** se il legacy compila a 16.384 B concessi.

### P6 — IL GINOCCHIO DEI LIMITI: chiedere il massimo NON paga (done-when *d*)

Il throughput della forma vincente a M=16, con
`maxComputeWorkgroupStorageSize` concesso spazzato su {16.384, 24.576, 32.768,
49.152}, è **piatto entro ±5%**. Il ginocchio è **a 16.384 o sotto**: su questa
forma il tetto negoziabile **non è una leva**, perché (P3) la forma vincente non
consuma shared proporzionale ai pesi.

`adapter.limits` completo va riportato accanto ai numeri, come richiesto dal
done-when.

**REFUTATA se**: 49.152 concessi rendono **≥ +15%** rispetto a 16.384 — nel qual
caso il tetto È una leva e la riga 4 va riformulata come chiede docket item 6.

---

## DISEGNO (fissato PRIMA di misurare — deviazioni si dichiarano come tali)

1. **Il banco esiste: si estende, non si riscrive.** `src/microbench/kdRunner.ts`
   (celle con `kernel/variant/shape/msPerOp/effectiveGBps/bytesUnique/
   bytesEmitted/weightsPerSecond`), `kdGemv.ts`, `kdAttn.ts`, entry
   `kdbench.html`, driver `scripts/kd-microbench-run.mjs`. **Nessun file di
   `src/engine/**` viene modificato.**
2. **La forma attuale non si ricopia: si IMPORTA** da
   `src/engine/kernels/wgsl.ts` (`gemvQuantWgsl` con `batch: true`,
   `attnDecodeWgsl`/`attnDecodeLegacyWgsl`). Una baseline riscritta a mano non è
   la baseline.
3. **Esecuzione INTERLEAVATA** (round-robin variante-per-ripetizione), mai a
   blocchi: la deriva DVFS non deve coincidere con l'ordine delle varianti.
   ≥ 3 warm-up scartati, ≥ 10 campioni misurati, si riporta p50 + dispersione.
4. **Gate di lavoro per cella**: checksum finito, ≠ 0, entro **1e-3 relativo**
   dal checksum della forma attuale sulla stessa cella. Fuori tolleranza ⇒ cella
   `skipped` con la ragione, **non** cella pubblicata.
5. **Residenza in cache, dichiarata ora.** K2560×N9216 in Q4_0 pesa 13,3 MB e sta
   nella L2 di questa Lovelace: la cella misura la L2, e la L2 **favorisce la
   forma attuale** (che rilegge tutto M volte). Quindi il vantaggio misurato
   della forma vincente su cella L2-resident è un **limite INFERIORE** del
   vantaggio nel motore, dove i pesi sono 2,25 GB e streammano dalla VRAM. Si
   aggiunge per questo un asse `coldw`: ≥ 8 copie dei pesi ruotate a ogni
   dispatch (8 × 13,3 = 106 MB, oltre la L2), stesso trucco del `coldkv` di
   fase 0. Le due letture si pubblicano entrambe.
6. **Tempo**: `timestamp-query` quando presente, altrimenti CPU con
   `queue.onSubmittedWorkDone()`; la sorgente finisce nella cella
   (`timingSource`).
7. **Il limite si spazza creando device distinti** con `requiredLimits`
   espliciti, non deducendo. `grantedLimits` di ciascuno finisce nel JSON.
8. **Estrapolazione a modello intero** (serve a P2, e si dichiara la formula
   PRIMA): `prefill_ms ≈ nLayer × Σ_matrici (ms_per_chunk / M) × T + attn`, con i
   ms per chunk misurati sulle shape reali. È una **proiezione**, non una
   promessa: nel memo va marcata come tale e la riga 5 la ri-misura end-to-end.
9. **Niente numero senza il suo `hostState` e la sua shape.** `adapter.limits` e
   `subgroupSizeObserved` nel probe di ogni run file.

## VINCOLI OPERATIVI (violarli invalida la misura)

- **Zero run di modello.** Solo micro-bench isolati.
- **I bench GPU sono seriali ed esclusivi**: nessun altro agente gira mentre una
  misura è in corso. Due runner playwright sullo stesso profilo si bloccano a
  vicenda.
- Il server vite è **già acceso su :5199**: `BASE_URL=http://localhost:5199` al
  driver (il suo default è 5173, e leggere i parametri di un runner PRIMA di
  spenderci GPU è la landmine più recidiva di questo progetto).
- Gate prima di dichiarare fatto: `npx vitest run` verde, `npx tsc --noEmit`
  pulito, `node .harness/tools/engine-ktest.mjs` tutti PASS.

## ESITO ATTESO

Un JSON per variante sotto `results/microbench/`, un memo in
`docs/deep-dive/`, e la risposta secca a due domande: **quale forma vince e di
quanto** (P0/P3/P4), e **se il bersaglio dei 4 s esiste o va escluso coi numeri**
(P1/P2).

## COSA MI ASPETTO DI SBAGLIARE

Dichiarato in anticipo perché il grade lo possa usare:

- **P1 è l'enunciato meno ancorato**: il picco fp32 WGSL di questo device non è
  mai stato misurato, e la banda 6–16 TFLOP/s viene da una regola del pollice
  (20–45% del picco teorico), non da un dato di questo progetto. È la predizione
  che ha la probabilità più alta di cadere, ed è anche quella da cui dipende P2.
- **P3 potrebbe vincere per la ragione sbagliata**: se (ii) batte (iii), potrebbe
  essere per l'occupancy e non per il traffico. La cella `coldw` è lì per
  discriminare — se il vantaggio di (ii) sparisce a pesi freddi, la causa
  dichiarata è falsa anche se il rapporto regge.
- **P5 assume che il legacy a M=16 non degradi linearmente in M**: con M=16 sono
  256 workgroup invece di 16, quindi l'occupancy migliora e il legacy potrebbe
  sembrare meno peggio di quanto sia per token. Se il legacy misura ≪ 8 ms per
  chunk, la leva della riga 3 vale meno di quanto il piano assume.

---

## ADDENDUM it.4 (2026-08-13) — la cella che mancava, pre-registrata prima di girarla

`splitk-coldw`. La riga 1 ha pre-registrato la cella `coldw` come **discriminante
della CAUSA**: se il vantaggio di una forma evapora coi pesi fuori dalla L2,
allora era traffico e non occupancy. L'ha girata su `regs` e sulla forma
attuale, **non su `splitk`, che è la forma che ha vinto** (grade indipendente,
drift (c); docket item 12c). La causa della vincitrice è quindi oggi
indiscriminata, e la riga 2 sta per costruirci sopra.

**P7 — enunciato**: `splitk` a pesi freddi (8 copie ruotate, working set 106 MiB,
oltre la L2 di questa Lovelace) resta **entro +15%** del suo p50 a caldo
(0,0609 ms a M=16, K2560×N9216), cioè `splitk-coldw` ≤ 0,0700 ms.

**Perché**: `splitk` legge ogni blocco di peso **una volta sola** e lo usa per
tutte le M righe — non ha niente da riguadagnare dalla cache, a differenza della
forma attuale che rilegge tutto M volte. Il precedente misurato nella stessa
sessione è `regs`: 0,1294 → 0,1400 a freddo, **+8,2%**. `splitk` ha lo stesso
corpo e lo stesso riuso, quindi mi aspetto lo stesso ordine di degrado.

**failCondition**: P7 è REFUTATA se `splitk-coldw` > 0,0700 ms. Se degrada oltre
**2×** (> 0,122 ms), la conclusione della riga 1 va riscritta: il vantaggio di
43,1× sarebbe in parte un artefatto di L2-residenza del banco, la proiezione di
8.665 ms sarebbe ottimistica, e la riga 2 andrebbe ripianificata prima di
scrivere codice di motore.

**Nota di onestà sulla direzione dell'errore**: la forma ATTUALE è quella che la
L2 favorisce (rilegge i pesi M volte), e infatti la sua cella fredda è
identica alla calda (2,6157 contro 2,6225 — è già limitata da altro). Quindi il
rapporto 43,1× misurato a caldo è, se mai, un **limite inferiore** di quello nel
motore, non superiore. P7 verifica che non ci sia una sorpresa nell'altro verso.
