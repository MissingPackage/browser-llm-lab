# PRE-REGISTRAZIONE — fase 0 di `engine-kernel-decode`

Scritta e committata **PRIMA** di eseguire qualunque misura (`git log` di questo
file = il timestamp che rende falsificabile tutto il resto). Il grade della fase
si fa contro QUESTO testo, senza rinegoziare le soglie.

Anchor: `.harness/goals/engine-kernel-decode/{GOAL.md,PHASES.md,docket.md}`,
`docs/deep-dive/headroom-2026-08-12.md` (+ ADDENDUM it.59).

## TESI

I due kernel caldi del decode (`attnDecodeWgsl`, `gemvQuantWgsl` kind `q4_0`)
lasciano sul tavolo un fattore ≥ 1,5× recuperabile con varianti note.

## REGOLA DI STOP (dal contratto)

Se **nessuna** variante batte la forma attuale di ≥ 1,5× sulla stessa cella, la
fase CHIUDE il goal e il piano si riscrive. Un risultato negativo vale quanto
uno positivo: non si cerca di salvarlo.

## PREDIZIONI (5 enunciati, tutti nella stessa sessione browser, stesso device)

Forme REALI: attenzione `nHead=16, nKvHead=4, headDim=256, n=6333`, un layer per
dispatch; GEMV q4_0 `K2560×N2560`, `K2560×N9216`, `K9216×N2560`, lm_head
`K2560×N248320`.

**P0 (REGOLA DI STOP — enunciato di testa).** La fase 0 NON farà scattare la
regola di stop: almeno una variante di almeno uno dei due kernel supera la forma
attuale di ≥ 1,5× sulla stessa cella.

**P1 (SONDA delle feature, presente/assente per misura).** `subgroups` presente e
uno shader con `subgroupAdd` compila e produce il risultato corretto;
`shader-f16` ASSENTE; `packed_4x8_integer_dot_product` (`dot4I8Packed`) PRESENTE
in `navigator.gpu.wgslLanguageFeatures` [confidenza bassa, è l'unica delle tre
chiamata al buio]; `chromium-experimental-subgroup-matrix` esposta ma non
istanziabile in configurazione u8/f32.

**P2 (FEDELTÀ della baseline, gate della fase prima delle varianti).** La forma
attuale dell'attenzione riprodotta isolata a n=6333 misura **8,23 ms/dispatch
± 40%** (banda 4,9–11,5 ms; equivalente 4,5–8,8 GB/s su 51,9 MB di byte unici).
Se cade in banda, l'attribuzione dei 10,4 µs/posizione ad `attnDecodeWgsl` — che
l'ADDENDUM it.59 lascia esplicitamente [DA VERIFICARE] — è confermata. Se misura
MOLTO più veloce (< 4,1 ms/dispatch, meno di metà del previsto), la pendenza del
motore non è quel kernel: risultato negativo di pari valore, e la fase 1 sta
puntando il bersaglio sbagliato.

**P3 (ATTENZIONE, leva dominante e sua causa).** La variante migliore raggiunge
**≥ 4,0×** la forma attuale sulla cella n=6333 (stima puntuale 6–8×, cioè ~1,0–1,4
ms/dispatch e ~37–50 GB/s effettivi su byte unici; NON mi aspetto di avvicinare i
435 GB/s di picco). Inoltre, sulla CAUSA: fra le tre varianti isolate misurate da
sole, **lo split del contesto su più workgroup con softmax in streaming è la più
veloce** — più della fusione GQA (un workgroup per gruppo) da sola e più delle
letture vec4 da sole.

**P4 (GEMV q4_0).** Sulla lm_head (`K=2560, N=248320` — l'unica delle quattro
forme il cui working set, 357,6 MB, esce dalla L2 da 72 MB e quindi l'unica
confrontabile con il motore) la forma attuale misura **133 G pesi/s ± 25%**
(100–166) e la variante migliore raggiunge **≥ 1,5×** (stima puntuale 2–2,5×,
cioè 270–330 G pesi/s), restando comunque SOTTO i ~500 G pesi/s di llama.cpp
Vulkan sullo stesso GGUF/host/driver. Sulle tre forme cache-resident (2560² =
3,7 MB; 2560×9216 = 13,3 MB; 9216×2560 = 13,3 MB) la forma attuale misura PIÙ di
133 G pesi/s, perché la cella misura la L2 e non la VRAM: è dichiarato ORA, prima
di misurare, e quelle tre celle **non valgono come baseline del motore**.

**P5 (ordine delle leve).** Il fattore di guadagno della migliore variante di
attenzione è **maggiore** del fattore della migliore variante di GEMV. Se cade,
l'ordine delle fasi 1 e 2 in `PHASES.md` va invertito.

## DISEGNO (fissato prima di misurare)

- Una cella per `(kernel, variante, forma)`. Baseline e varianti compilate dallo
  STESSO harness, con gli STESSI buffer e la stessa sessione.
- Le varianti si eseguono **INTERLEAVATE** (round-robin variante-per-ripetizione),
  non in blocchi, per non far coincidere la deriva DVFS con l'ordine.
- La baseline non è "una forma equivalente": i sorgenti WGSL della baseline sono
  **importati** da `src/engine/kernels/wgsl.ts` (`attnDecodeWgsl`,
  `gemvQuantWgsl`), non ricopiati. Nessun file di `src/engine/**` viene toccato.
- TEMPO: `timestamp-query` quando presente, altrimenti CPU con
  `queue.onSubmittedWorkDone()`; la sorgente è scritta nella cella
  (`timingSource`).
- GEMV: idioma dei **16 dispatch identici nello stesso pass** (serializzati
  dall'hazard WAW su `y`), tempo per dispatch = delta / 16 — necessario perché
  Chrome quantizza i timestamp a ~100 µs.
- Attenzione a n=6333: il dispatch atteso è ~8 ms, ben sopra il quanto ⇒ si
  misura **a dispatch singolo E in batch da 16** sullo STESSO buffer di output
  (l'hazard WAW deve restare). Si riportano entrambi; **il grade usa il singolo**.
- RIPETIZIONI: ≥ 3 warm-up scartati, ≥ 10 campioni misurati. Si riporta MEDIANA e
  dispersione (p50, min, max, IQR). Statistica di confronto:
  `speedup = p50(forma attuale) / p50(variante)` nella stessa sessione.
- DERIVATE: `G pesi/s = N·K / t` (GEMV); `GB/s effettivi` = byte UNICI / t, con
  byte unici GEMV q4_0 = `N·K·0,5625 + (N·K/32)·2 + K·4` e byte unici attenzione
  = `n · 4 kvhead · 256 · 2 · 4 = n · 8192` (51,9 MB a n=6333). Per le varianti
  di attenzione si riporta ANCHE il **traffico emesso** (×4 nella forma attuale
  per la ridondanza GQA), così che la dedup si legga separata dal parallelismo.

## PROVA CHE IL KERNEL HA LAVORATO (gate obbligatorio)

`checksum` = somma degli output, finito e ≠ 0. In più, gate NUOVO senza il quale
una variante "veloce" può semplicemente saltare lavoro: **il checksum di ogni
variante deve stare entro 1e-3 RELATIVO al checksum della forma attuale sulla
stessa cella** (tolleranza e non identità perché streaming-softmax e split
cambiano l'ordine delle somme). Fuori tolleranza ⇒ la cella è `skipped` con
motivo, non un numero.

## CONSEGNA

1. questo file, committato prima di misurare;
2. un JSON in `results/microbench/` (schema esteso, `kind` nuovo, `schemaVersion`
   bumpato) con una cella per variante, più il blocco della sonda P1
   (`features`, `wgslLanguageFeatures`, esito di compilazione di `subgroupAdd` e
   `dot4I8Packed`), `deviceLabel`, `hostState.declared`, e il contesto dichiarato
   di ogni cella;
3. grade indipendente = aritmetica sul JSON (`scripts/kd-grade.mjs`): p50 per
   cella, speedup contro la cella "forma attuale" della stessa forma, confronto
   con le soglie di P0–P5 **senza rinegoziarle**;
4. memo in `docs/deep-dive/` col verdetto sulla regola di stop e la variante
   vincente per ciascuno dei due kernel.

## VARIANTI DICHIARATE (l'elenco è chiuso qui)

Attenzione (cella unica n=6333):
- `base` — `attnDecodeWgsl` importato: 1 workgroup (64 thread) per head,
  `scores: array<f32, ctxMax>` in workgroup memory, ciclo scalare su headDim,
  `kvHead = h/4`;
- `vec4` — identica alla base, letture `vec4<f32>` su headDim;
- `stream` — softmax in **streaming** a tile di 64 posizioni, niente
  `scores[ctxMax]`, 1 workgroup per head, letture vec4;
- `gqa-stream` — come `stream` ma **un workgroup per gruppo GQA** (4 workgroup,
  KV letta 1 volta invece di 4);
- `split` — come `stream` + **split del contesto** su S=16 workgroup per head con
  kernel di combinazione dei parziali (log-sum-exp);
- `split-gqa` — combinazione di `gqa-stream` e `split`.

Le tre varianti "isolate" di P3 condividono streaming e vec4 (isolare la fusione
GQA senza streaming è impossibile: 4·ctxMax·4 B di `scores` sfondano la workgroup
memory). L'asse isolato è dunque **gqa vs split vs nessuno dei due**, con
`stream` come termine di paragone comune. Dichiarato qui, prima di misurare.

GEMV q4_0 (4 forme):
- `base` — `gemvQuantWgsl({kind:"q4_0"})` importato;
- `vec4` — una load `vec4<u32>` per blocco + `dot()` su `vec4<f32>` di x;
- `vec4-sg` — `vec4` con riduzione via `subgroupAdd` (se la feature esiste);
- `vec4-rows4` — `vec4` con 4 righe per workgroup (16 lane per riga);
- `vec4-rows2-sg` — `vec4` con 2 righe per workgroup, una riga per subgroup
  (richiede `subgroupSize == 32`, altrimenti cella `skipped` con motivo);
- `dot4I8Packed` — solo **sonda di compilazione/esecuzione**: usarlo per q4_0
  richiede attivazioni quantizzate a int8, cioè un cambio di algoritmo che
  violerebbe la tolleranza 1e-3 sul checksum. Dichiarato ora: non entra come
  cella di velocità.
