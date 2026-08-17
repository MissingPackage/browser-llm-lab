# SPEC — Q2_K e Q3_K nel motore, con parità di ottimizzazioni

*2026-08-17 · goal `engine-velocita-decode` · ruling del PI: «Kernel Q2_K + Q3_K»*

## 1. Perché, con i numeri che l'hanno deciso

Il 35B è **residency-bound**: parco expert 17,07 GiB contro un'arena da 11,17 —
il 65% — e in chat fa **11,5 tok/s** contro i **40,06** misurati al banco a zero
miss. Il collo non è il kernel, è la capienza.

Un quant più aggressivo la risolve. Misurato il 2026-08-17, appaiato e
teacher-forced (`results/eval/quant-compare-q4ks-vs-q2k-2026-08-17.json`):

    parco expert (slab impacchettati)   17,074 -> 10,391 GiB    = 100% residente
    byte per miss                    1.769.472 -> 1.032.192/1.146.880 B  (-42%)
    costo misurato                   +0,13 bit/token, -0,15 logprob in matematica

Il file scelto è `bartowski/Qwen_Qwen3.6-35B-A3B-GGUF → Q2_K` (13,5 GB, già in
`~/.cache/blab-models/q35-eval/`), i cui expert sono **Q2_K + Q3_K**: K-quant,
stessa famiglia dei tre che il motore già legge. *I quant Unsloth «UD» sono stati
scartati perché usano i-quant a codebook, che sono un'altra famiglia di kernel.*

Contesto: `docs/architettura/QUANTIZZAZIONE.md` spiega i formati per intero.

## 2. User story

> Come motore, voglio **leggere ed eseguire i tensori Q2_K e Q3_K con la stessa
> qualità di implementazione dei K-quant che già supporto**, così che il 35B
> possa girare su un quant il cui parco expert sta interamente nell'arena.

## 3. La decisione di forma, che NON è dell'implementatore

I gemv K-quant oggi sono **tre funzioni scritte a mano**: `gemvQ4KWgsl`
(`wgsl.ts:2278`), `gemvQ5KWgsl` (`:2412`), `gemvQ6KWgsl` (`:2489`), più
`gemvQ6KFastWgsl` (`:3752`). Aggiungerne due a mano sarebbe la **quarta e quinta
copia**.

Il ruling di riuso di questo progetto è esplicito: *«il trigger è la RIPETIZIONE,
non la copertura: la seconda copia scritta a mano è una domanda, la terza no»*. E
il ruling del PI su una famiglia nuova è *«stesse ottimizzazioni degli altri
K-quant»*.

**Quindi: si fattorizza un nucleo parametrico e i cinque formati ne derivano.**
Non è un rifacimento a piacere — è vincolato da un gate secco:

> **I tre formati esistenti devono uscire BIT-IDENTICI dal nucleo nuovo.** Non
> "entro tolleranza": identici. Se la migrazione di q4_K/q5_K/q6_K non è
> bit-identica, la fattorizzazione è sbagliata e il task è BLOCKED.

Ciò che il nucleo deve parametrizzare, letto dai tre esistenti: byte per
superblocco, forma dell'unpack (dove stanno i bit bassi e quelli alti), presenza
di `dmin` (affine contro simmetrico), posizione della scala (in `Q6_K` sta **in
coda**, offset 208), e la ripartizione del lavoro, che ha già un helper
condiviso: `kquantWorkSplit` (`wgsl.ts:2143`).

## 4. Contratto — task, `owns` disgiunti, done-when meccanici

### T1 · BF16, il tipo che blocca il caricamento

Il file di bartowski tiene il **router (`ffn_gate_inp`) in BF16** (ggml 30), e il
motore non lo legge: `validateQwen35` lancia. BF16→F32 è uno shift (i 16 bit
stanno nella metà alta del float32), il tensore è piccolo (2048×256) e sale in
VRAM già in f32 come oggi.

- **owns**: `src/engine/gguf.ts`, `src/engine/quant.ts`, `src/engine/q35shape.ts`,
  `tests/engine-bf16.test.ts` (nuovo)
- **done-when**:
  - `GGML_TYPE.BF16 = 30`, `tensorByteSize` = 2 byte/elemento (nessun blocco);
  - una conversione `bf16ToF32` in `quant.ts` con un caso che verifica **zeri,
    negativi, subnormali, inf e NaN** contro i bit attesi;
  - la allow-list del router accetta BF16 **solo lì**: non si allarga
    `W_QUANT`, perché un BF16 dove ci aspettiamo pesi quantizzati resta un
    errore da fermare;
  - `npx tsc --noEmit` pulito, `npx vitest run` verde.

### T2 · Il nucleo parametrico dei gemv K-quant, coi tre esistenti migrati

- **owns**: `src/engine/kernels/wgsl.ts`, `tests/engine-kquant-core.test.ts`
  (nuovo)
- **done-when**:
  - un descrittore per formato (byte/superblocco, unpack, `dmin` sì/no,
    posizione della scala) e **un** generatore che li consuma;
  - `gemvQ4KWgsl`, `gemvQ5KWgsl`, `gemvQ6KWgsl` restano esportati con la stessa
    firma (i chiamanti non cambiano) ma delegano al nucleo;
  - **il WGSL emesso per i tre formati è IDENTICO carattere per carattere a
    quello di prima**, su un campione di shape che includa quelle di produzione
    (`{K:2048,N:512}`, `{K:512,N:2048}`, `{K:2048,N:4096}`) e le varianti
    `batch` e `arena`. Il test pinna le stringhe con uno snapshot preso PRIMA
    della modifica;
  - nessun cambiamento di comportamento: `npx vitest run` verde senza toccare
    asserzioni esistenti.

### T3 · Q2_K e Q3_K derivati dal nucleo *(dipende da T2)*

- **owns**: `src/engine/kernels/wgsl.ts`, `src/engine/q35gpumodel.ts`
- **done-when**:
  - `gemvQ2KWgsl` e `gemvQ3KWgsl` esistono **come istanze del descrittore**, non
    come funzioni scritte a mano (il test di T2 verifica che il generatore sia
    uno solo);
  - l'aritmetica segue `dequantQ2_K`/`dequantQ3_K` di `quant.ts`, che sono già
    verificati byte-identici a `llama-quantize`: **Q2_K è affine** (`d`,`dmin`,
    scale e min a 4 bit per 16 sotto-blocchi), **Q3_K è simmetrico** con la
    `hmask` separata e le scale a 6 bit ricomposte come `q3kScale6`;
  - il ramo expert di `q35gpumodel` (`:1748-1754`, `:587-594`) instrada i due
    formati nuovi come fa per gli altri;
  - `npx tsc --noEmit` pulito, `npx vitest run` verde.

### T4 · La forma multi-riga del prefill *(dipende da T2)*

- **owns**: `src/engine/prefillgemmplan.ts`, `src/engine/prefillkquant.ts`,
  `tests/engine-prefill-q2k-q3k.test.ts` (nuovo)
- **done-when**:
  - `PREFILL_GEMM_KINDS` (`wgsl.ts:4150`) include `q2_K` e `q3_K`;
  - **il predicato sulla SHAPE esiste e ha un test che lo dimostra.** È la
    trappola registrata: *«il flag `wired` è per FORMATO, non per shape»*, e
    accendere un formato senza predicato farebbe entrare nello stesso istante i
    48 siti `ssm_alpha`/`ssm_beta` del 4B con N=32. Un caso deve mostrare che
    una shape con N piccolo **non** viene ammessa;
  - la guardia doppia del cablaggio (`route.via !== "legacy" && kk === "<fmt>"`)
    è rispettata: è il reperto che ha già intercettato un caso vero.

### T5 · Conformance per formato contro il riferimento CPU

- **owns**: `src/engine/ktest/**`, `tests/engine-kquant-cpuref.test.ts` (nuovo)
- **done-when**:
  - un caso ktest per `q2_K` e uno per `q3_K` che confrontano il gemv GPU col
    `dequant*` di `quant.ts` sulle shape degli expert del 35B
    (`K=2048,N=512` e `K=512,N=2048`), con la stessa soglia dei casi K-quant
    esistenti;
  - i casi sono **registrati** nell'elenco del ktest, non solo scritti.

### T6 · Le classi d'arena e il piano VRAM

- **owns**: `src/engine/residency.ts`, `src/engine/q35expertstore.ts`,
  `tests/engine-arena-q2k.test.ts` (nuovo)
- **done-when**:
  - un test SENZA GPU che, dato l'header del `Q2_K`, verifica **parco 10,391
    GiB, due classi (`q3k` 5120×1.146.880 B, `q2k` 5120×1.032.192 B) e 100% di
    residenza a budget 11,17 GiB** — sono i numeri già misurati il 2026-08-17,
    e il test li pinna perché una regressione di geometria non si vede a occhio;
  - nessun formato cablato a mano: la config resta **dedotta dall'header**.

## 5. Vincoli su TUTTI i task

- **Nessuna esecuzione su GPU dentro il workflow.** Niente ktest, niente bench,
  niente dev server: la GPU è una sola e la contesa fra agenti è un difetto già
  pagato. I gate GPU li esegue la sessione, in sequenza, dopo l'integrazione.
- **Non si tocca la numerica dei formati esistenti** se non attraverso T2, e lì
  con l'identità carattere per carattere del WGSL emesso.
- Ogni file nuovo di test sta in `tests/` e gira senza GPU e senza rete.
- Commenti in italiano, come il resto del repo, e spiegano il **perché** non il
  cosa.

## 6. Da leggere prima di scrivere

| file | cosa ci si trova |
|---|---|
| `src/engine/quant.ts:363-460` | `dequantQ2_K`/`dequantQ3_K` e `q3kScale6`: l'aritmetica di riferimento, già byte-identica a `llama-quantize` |
| `src/engine/quant.ts:145-232` | i layout di `Q4_K`/`Q5_K`/`Q6_K`, compreso che in Q6_K la scala sta in coda |
| `src/engine/kernels/wgsl.ts:2143` | `kquantWorkSplit`, la ripartizione già condivisa |
| `src/engine/kernels/wgsl.ts:2278,2412,2489` | i tre gemv da fattorizzare |
| `src/engine/moe.ts:37-60` | `QuantKind` e la geometria dei blocchi (Q2_K/Q3_K già inseriti) |
| `docs/architettura/QUANTIZZAZIONE.md` | le famiglie, i layout, la distinzione formato/ricetta |
| `results/eval/q35-header-dump-bartowski-2026-08-17.json` | la composizione vera del file bersaglio |

## 7. Fuori scope

- Gli **i-quant** (`IQ*`): altra famiglia, altro lavoro.
- La **sostituzione del modello di default**: questo lavoro rende il formato
  eseguibile; quale file la chat carichi è una decisione successiva, e va presa
  con la misura end-to-end in mano.
- Il **convertitore slab** per il nuovo quant: il formato slab è parametrico e
  lo assorbe, ma l'artefatto non si rigenera qui.
