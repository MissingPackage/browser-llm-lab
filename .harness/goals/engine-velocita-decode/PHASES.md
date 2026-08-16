# PHASES — engine-velocita-decode (it.0 2026-08-15, RI-SCOPATO in it.2)

**RI-SCOPO di it.2** (ruling PI): il goal non è più «la residency del 35B» ma
**la velocità di decode del motore**, con leve globali. La riga 2 vecchia (la
fetch fuori dal path seriale) è diventata **2b** e ha perso il primo posto: la
misura dice che il termine dominante è il pass, non la residency. La forma a
gather — che il contratto aveva messo fuori scope — è la nuova riga 2.

Sequenziale. Le righe 2, 2b e 3 toccano file che si intersecano
(`q35gpumodel.ts`, `residency.ts`, `wgsl.ts`): nessun `parallel-group`, la
proprietà esclusiva dei path non c'è. Stessa ragione dei tre goal precedenti.

**Metrica obiettivo**: `decode.tokS` a caldo, select `optimistic`, contesto
dichiarato, **su tutte e quattro le famiglie**. **Barra: 35B ≥ 30 tok/s**
(nice-to-have 45); 4B, 9B e GLM non regrediscono. Punti di partenza misurati:
35B **22,9 tok/s** come TETTO della sola residency (token pulito 43,585 ms,
it.2) — la barra richiede per costruzione la leva sul pass.

**STATO DELLA METRICA (it.18)**: 35B **30,74 tok/s**, barra 30 **superata** con
tutta la dispersione sopra. Due leve, entrambe sul pass: `kfan` (riga 2c,
22,58 → 28,90) e il router parallelo (riga 2e, 28,90 → 30,74). **Il goal non è
chiuso**: mancano la riga 4 (la barra su tutte e quattro le famiglie, con
`decodeContext` dichiarato) e la riga 6, che è un gate di merge oggi **rosso su
UNA voce**: il GLM b12 fuori banda (docket item 7). Il ktest è tornato verde in
it.19 — `111 PASS / 0 FAIL` — e il suo rosso non era una regressione ma un
fixture avvelenato da un runner di bench (docket item 6, chiuso).

**LA REGOLA CHE VALE SU OGNI RIGA — CORRETTA DAL PI in it.12.** Diceva «una
leva è misurata su ≥ 2 famiglie»: era una mia riformulazione, ed era **una
metrica di copertura al posto di un principio di riuso**. Mi ha mandato a
pianificare la terza scrittura a mano dello stesso kernel. La regola vera:

> **Riusa ciò che c'è già. Costruisci globale quando possibile, poi estendi.
> Propaga i fix sul vecchio quando serve.** E per l'ordine: *prima* le
> ottimizzazioni globali e riusabili — i modelli nuovi le ereditano da soli —
> *poi*, solo se non bastano o se l'architettura è diversa, quelle specifiche.

**Il trigger operativo è la RIPETIZIONE, non la copertura**: se sto per scrivere
a mano la seconda copia di una forma, la domanda è se va fattorizzata; alla
terza non è più una domanda. Il veicolo esiste già ed è `pattern-migration`
(`fix-dont-fence`, 2026-08-14).

---

## Punto di partenza — tre correzioni trovate nel test-fit dei done-when

**(C0-1) «8,34 tok/s» non è un punto di partenza legittimo, e il goal non può
usarlo come tale.** Viene da `chat-35b-2026-08-15T16-25-12.json`, che si dichiara
da sé NON-riferimento: nessun warm-up scartato, nessuna replica, prompt scelto a
mano, e il primo turno dopo il load (`cacheState: "cold"`) — cioè l'arena si sta
riempiendo *durante* la misura. I suoi **contatori** valgono (sono somme di
eventi: miss, replay, byte), i suoi **tempi medi** no. La riga 1 deve produrre il
punto di partenza vero, e il consuntivo deve confrontarsi con QUELLO. Scriverlo
qui evita che il goal si chiuda vantando un 3,6× contro un numero gonfiato dal
transitorio a freddo.

**(C0-2) La somma dei termini non chiude, e non chiuderà nemmeno strumentando
solo la fetch.** Oggi: `tailCpuMs` 121.691 = `repairMs` 76.160 + 45.531 di
residuo. Dentro `repairMs`: pack 7.590 + upload 6.272 + read 3,8 + **62.294 non
nominati**. Il residuo di 45.531 ms *fuori* da `repairMs` è il tempo dei
`runPass` di replay awaited a `:2005` più la contabilità (`account`,
`noteResidentHit`, `routing`), e non è mai stato separato. Strumentare la sola
fetch lascerebbe il 31% di `tailCpuMs` ancora anonimo: la clausola «≥ 95% di
`tailCpuMs` nominato» pretende **tre** contatori nuovi, non uno.

**(C0-4, trovata in it.1 e registrata subito) «Il token pulito costa 21,1 ms»
è FALSO, o almeno non dimostrato — e con esso cade la certezza che le righe 2 e
3 bastino.** Il contratto l'aveva derivato come `tokenMs − tailCpuMs`. Due
controlli lo rifiutano:

1. **L'aritmetica non chiude.** `readbackMs` / submit = 89.924 / 2.407 =
   **37,4 ms per pass**, mentre l'intera porzione non-tail del token vale
   21,8 ms per decode step (144.744 − 121.691, su 1.058 step: `tokenMs` si
   accumula solo nel ramo ottimistico, `q35gpumodel.ts:2621`, quindi il
   denominatore sono gli step di decode e non i 1.092 forward). Un pass non può
   contenere un'attesa più lunga della finestra che lo contiene. La sottrazione
   mescola pass iniziale e pass di replay, che stanno da parti opposte di
   `tTail`.
2. **Esiste una misura diretta che dice il contrario.**
   `results/engine/q35-vramplan-35b-it35.json`, pass `optimistic-warm`:
   **0 miss, 0 dirtyTokens, 0 replay, `repairMs` 0** — un token pulito vero, non
   dedotto. Costa **43,74 ms**: `readbackWait` **40,98**, `encodeMs` 1,13,
   `argmaxMs` 0,39, **`tailCpuMs` 0,194**. Il 2026-08-11, su codice precedente
   ai kernel di `engine-kquant`.

**Perché è una correzione di PIANO e non una nota.** Con un token pulito a
~43 ms, togliere il 100% della tassa di residency dà **~23 tok/s: sotto la barra
dei 30**. Le righe 2 e 3 non chiuderebbero il goal, e servirebbe una quarta leva
sul pass stesso — dove `readbackWait` era il **94%** del token pulito.

> **RISOLTA in it.2, ed è l'esito sfavorevole.** Misura diretta,
> `q35-optimistic-35b-cleantoken-2026-08-15.json`, `optimistic-warm` a 0 miss:
> **`tokenMs` = 43,585 ms = 22,9 tok/s**, `readbackWait` **40,753 = 93,5%**,
> `submitsPerToken` 1. Entro lo 0,7% dalla misura del 2026-08-11 su un albero
> diverso. **Le righe 2 e 3 non chiudono il goal.** La barra dei 30 tok/s
> richiede la quarta leva — il pass — e la sola leva nota su quel termine è la
> forma a gather K-quant della consegna §4.2-4.4, che questo contratto ha messo
> **fuori scope** sulla base della stima poi ritrattata. Non riapro lo scope da
> solo: è funzione obiettivo. **Docket item 3, in attesa di ruling.**

*Il reperto che NON cade: la lentezza del 35B non è un limite fisico. Il
pavimento di banda è ~2,9 ms/token contro i ~120 misurati, e l'89,7% di token
sporchi con miss al 2,97% è un fatto contato, non stimato.*

**(C0-3) Ci sono DUE siti di fetch, non uno, e hanno regimi diversi.**
`q35gpumodel.ts:1994` è il repair del decode ottimistico (miss scoperti a fine
pass); `:2050` è `prepLayer`, cioè il path sync e il prefill a chunk (miss noti
prima del layer). Contarli insieme renderebbe illeggibile quale dei due paga —
e il prefill del 35B è dichiarato fuori scope proprio perché non si mescoli.
Il contatore va **per sito**, con due nomi diversi.

---

## Tabella

| # | phase | done-when (mechanical) | authority delta | owns | status | stima |
|---|-------|------------------------|-----------------|------|--------|-------|
| 1 | **La misura prima della leva**: tre contatori nuovi che nominano il residuo (fetch per sito C0-3, tempo dei pass di replay, flush della slot table), il commento stale di `perf()` corretto, **il token pulito misurato** (C0-4: decide se il contratto è eseguibile come scritto) e **la baseline fresca di riferimento** del 35B. Nessuna ottimizzazione. | **il token pulito**: riproduzione della forma di `q35-vramplan-35b-it35.json` sull'albero di oggi (39 token, arena 12 GiB, passata calda a 0 miss), `tokenMs` letto e confrontato coi 43,74 ms del 2026-08-11 — e se è ~43, il contratto si riapre sulla quarta leva PRIMA della riga 2; `perf()` espone `fetchRepairMs`/`fetchRepairCalls`/`fetchRepairBytes`, `fetchPrepMs`/`fetchPrepCalls`, `replayPassMs`, `flushMs`, propagati ai tre worker che li serializzano; `npx vitest run` verde con un test SENZA GPU che pinna la forma dei contatori e verifica che i delta per turno li includano tutti; `npx tsc --noEmit` pulito; artefatto `kind: "q35-residency-baseline"` in `results/engine/` con host dichiarato quiescente, warm-up scartato, ≥ 3 repliche, `decodeContext` dichiarato, e **`nominati / tailCpuMs ≥ 0,95` NEL REGIME SPORCO** verificato nell'artefatto stesso (precisazione di it.2: su una passata a 0 miss `namedFrac` vale 0/0 — senza repair il 100% di `tailCpu` È contabilità, e la clausola letta lì dichiarerebbe fallita una riga riuscita). | none | `src/engine/q35gpumodel.ts`, `src/engine/chat/chat.worker.ts`, `src/engine/q35conf/**`, `src/engine/glmbench/**`, `tests/**`, `scripts/`, `results/engine/` | **done (it.1-it.2)** — `namedFrac` **0,9995**, token pulito **43,585 ms**; resta da produrre la baseline nel regime sporco a contesto lungo, che dipende dal ruling dell'item 3 | 1-2 it, **2 consumate** |
| 2 | **LA FORMA A GATHER DIVENTA UNIVERSALE — leva del PREFILL e del GLM.** ⚠️ **CORRETTA in it.5: NON muove la barra del decode.** Il gather raggruppa le RIGHE di un chunk per expert; nel decode le righe sono una, quindi l'unione è 8 expert con una riga ciascuno = gli stessi 8 dispatch di adesso. Il ~2,6× della consegna §4.4 è calcolato «per chunk da M=16»: è una leva di prefill. La leva del decode è la riga 2c. Non «gather per il 35B»: un solo path a gather che regge ogni famiglia MoE e ogni formato. I due kernel esistono, funzionano e sono in produzione **su GLM**: `pairGemvSiluGatherWgsl` (`wgsl.ts:3221`), `gemvDownSlotsWgsl` (`:3303`), `moeCombineWgsl` (`:3381`), cablati a `glmmodel.ts:1368-1412`. Sono fermi a **q4_0 e top-4** (`m * 4u` scritto a mano a `wgsl.ts:3296`, `:3351`, `:3375`); `moeCombineWgsl` è già parametrico su `nUsed`. L'aritmetica K-quant da riusare è quella verificata da `engine-kquant` riga 4 — unpack, offset, termine `Σx` — **non** la sua geometria multi-riga. Sotto-range su slab K-quant: `slotTensorRanges` (`residency.ts:189`) esiste già, la vista legacy `slotBindRanges` **lancia** sui K-quant. | i due kernel parametrici su `nUsed` e sui formati K-quant, con un caso ktest PASS per formato contro il cpuref; **una forma sola che regge GLM (top-4, q4_0) E 35B (top-8, Q4_K/Q6_K)** — non due scritture: se la seconda famiglia richiede di riscrivere il kernel a mano, la riga è sbagliata e va fattorizzata prima; GLM b12 non regredisce (±5%); **bit-identità col path sequenziale rimisurata** (gate secco: il down del 35B usa `accum: true` col peso dalla `Sel`, il contratto a slot pretende che il down scriva NON pesato e che sia la combine a sommare in ordine k — sono due contratti diversi sullo stesso tensore, consegna §4.4c). | none | `src/engine/kernels/wgsl.ts`, `src/engine/q35gpumodel.ts`, `src/engine/glmmodel.ts`, `src/engine/moeprefillplan.ts`, `src/engine/ktest/**`, `tests/**` | todo | 3-4 it |
| 2b | **IL RAGGRUPPAMENTO DELLE RICHIESTE DI I/O — leva globale, chiesta dal PI.** Misurato in it.2 e non assunto: **2,1×** fra 6,90 ms/fetch (24 richieste concorrenti, `prepLayer`) e 3,27 ms (qualche centinaio, il repair). Stessi byte, stesso server, stesso `readExpert`: cambia solo il raggruppamento. Va messo nel path di I/O **condiviso** (`range()`/`readRange`, `chat.worker.ts:62`, `q35conf.worker.ts:186`), non nei due call site del 35B — altrimenti è la «piccolezza specifica» che il ruling vieta. Candidati, da misurare non da assumere: coalescing delle Range adiacenti in una richiesta sola (gate+up+down di un expert sono tre tensori diversi, quindi tre offset: il coalescing vero è **fra expert dello stesso tensore**), finestra di concorrenza esplicita, sorgente non-HTTP (OPFS `FileSystemSyncAccessHandle`). | tempo per fetch **< 1,5 ms** (oggi 3,27 nel repair, 6,90 in prep) sul contatore della riga 1; **l'effetto misurato ANCHE sul LOAD di 4B, 9B e GLM**, che passano dallo stesso `range()` — se lì non paga si dichiara col numero invece di tacere; nessuna regressione dei gate di correttezza. | none | `src/engine/chat/chat.worker.ts`, `src/engine/q35conf/**`, `src/engine/q35expertstore.ts`, `src/engine/residency.ts` | todo | 1-2 it |
| 2c | **KFAN — il collasso dei k. ATTERRATA in it.9: 22,58 → 28,90 tok/s, gate argmax 39/39.** Resta aperto solo il riuso su altre famiglie MoE, che NON si fa ricopiando il kernel: si fa quando serve, fattorizzando (ruling PI it.12). | Trovata in it.5 con l'aritmetica del token pulito: `readbackWait` 40,753 ms per **1.320 dispatch** (=(8 expert×4)+1, ×40 layer) = **30,9 µs l'uno**, cioè **13,9 GB/s** effettivi sui 566,2 MB di pesi expert per token — contro gli **0,98 ms** che quegli stessi byte costerebbero a 576 GB/s. **41× sopra il pavimento di banda: dispatch/occupancy-bound**, coerente col reperto della consegna §3.3 (32-64 workgroup su 128 SM, 17,2 GB/s misurati). Nel decode l'asse con 8 elementi non è la riga, è **il top-K**: `encodeExperts` (`q35gpumodel.ts:2099-2108`) fa `for k2 = 0..8` e dispaccia 4 volte per ciascuno. Con `wid.z = k`: **1.320 → 200 dispatch/token** e occupazione ×8. Verificato sul codice che è piccolo: `selBuf` è già `array<Sel>` in storage (`wgsl.ts:3090`) e le `topK` entry di un (riga, layer) sono **contigue** (`selIdx = (row*nMoeLayer+m)*topK+k`, `q35gpumodel.ts:1377`), quindi il preambolo diventa `selBuf[moeIdx.selIdx + wid.z]` — **una riga**. La corsa sull'accumulo si risolve col **contratto a slot di it.4**: ogni k scrive il suo slot, `moeCombine` somma in ordine k, e a M=1 con `nUsed=8` è *esattamente* la catena di somme del decode di oggi. **Il divieto `batch && arena` (`wgsl.ts:2175`, `:2360`) va affrontato introducendo `kfan` come modo A SÉ, non allentando la guardia**: per `batch` (righe) + arena il divieto resta giusto. | `decode.tokS` del 35B su artefatto di riferimento **≥ 30** a caldo; dispatch/token del path expert **≤ 250** (oggi 1.320), letto da `submits` o da un contatore dedicato; **bit-identità col path sequenziale**, che qui è pretesa forte e non tollerata: a M=1 la combine somma gli stessi k nello stesso ordine, quindi **identità esatta**, non una tolleranza; **misurato su GLM E 35B** (il GLM fa `gemvAccumFast` k=0..3, stessa struttura); floor test FMA per la contrazione (il rischio isolato in it.3). | none | `src/engine/kernels/wgsl.ts`, `src/engine/q35gpumodel.ts`, `src/engine/glmmodel.ts`, `src/engine/ktest/**`, `tests/**` | todo | 2-3 it |
| 2d | **`ssmGemv` — il primo termine misurato, e globale PER COSTRUZIONE.** Misurato in it.10 sul braccio kfan-ON: **6,98 ms/token** contro i 5,15 dell'expert. Non è MoE: è la proiezione DeltaNet, e **ce l'hanno 4B, 9B e 35B** — quindi una forma sola serve tutti e tre, che è precisamente l'ordine che il PI ha dato (*prima le globali e riusabili, i modelli nuovi le ereditano da soli*). 30 pass per token, il termine più grosso del decode. *NB: `engine-kquant` ha ottimizzato `gemm:deltanet-out` nel PREFILL — questo è il GEMV del DECODE, path diverso, e la prima cosa da verificare è se la forma già in casa si riusa invece di scriverne una nuova.* | prima di scrivere: **verificato quale forma esistente si riusa** (il prefill ha già una via veloce per questi tensori) e dichiarato perché quella non basta al decode, se non basta; poi `ssmGemv` **≤ 4,0 ms/token** sul 35B a sonda accesa; **e la stessa forma misurata su 4B e 9B** — non riscritta per ciascuno, ereditata: se una famiglia costringe a ricopiare il kernel a mano, la riga è sbagliata e va fattorizzata prima (`pattern-migration`); `decode.tokS` del 35B **≥ 30**; nessuna regressione su 4B/9B/GLM; ktest verde. | none | `src/engine/kernels/wgsl.ts`, `src/engine/q35gpumodel.ts`, `tests/**` | **todo — MISURATA in it.21, si costruisce.** Il banco sulle shape vere a M=1 contro il kernel che il decode emette DAVVERO (braccio `base-decode`, non `base-batch-z`): **3,89×** su `K=2048 N=8192` (il 66,32% dei byte) e 3,30× su N=4096. Lo split-K arriva al **91,0% del picco di banda**, il GEMV del decode sta al 23,4%. Proiezione: `ssmGemv` 6,99 → 1,97 ms, meno 0,78 ms di dispatch aggiunti = **~4,24 ms/token netti**, token 32,5 → ~28,3 = **~35,4 tok/s**. Pre-registrazione e graduatoria: `docs/deep-dive/velocita-decode-2d-prereg-2026-08-16.md`. **Manca al piano**: i due buffer (parziali N×fette f32, x quantizzata) sono VRAM sottratta all'arena expert, che su questo modello è il vincolo. | 2-3 it, **6 consumate** (5 sul prefill + 1 di misura) |
| 2e | **IL ROUTER — la selezione top-k diventa parallela. ATTERRATA in it.18: 28,90 → 30,74 tok/s, BARRA SUPERATA.** Riga aperta in it.18 dalla chiusura del docket item 5 (uscita 3, decisa da me: è ordine di lavoro). Il reperto: `router` 2,883 ms/token su 40 dispatch = **72 µs a layer** per scegliere 8 expert su 256, contro i 16,6 µs del `routerGemv` che quei logit li produce con 256×2048 MAC. La causa non era il dispatch: `routerTopKWgsl` (`wgsl.ts:3859`) faceva la selezione **su un thread solo** — 8×256 confronti seriali su un `array<bool, 256>` in memoria PRIVATA con indice dinamico, cioè scratch — difesa da un commento che dichiarava la serializzazione «la specifica» perché una riduzione «cambierebbe il tie-break». **Non lo cambia**: lo scan con `>` stretto è il massimo sull'ordine TOTALE (punteggio, −indice), e il massimo su un ordine totale è associativo. Globale per costruzione: **un kernel solo**, montato da `glmmodel.ts:1068` e `q35gpumodel.ts:1758` — non riscritto due volte. | `router` **0,836 ms/token** (era 2,883) e 20,9 µs a dispatch, ormai il pavimento (`routerGemv` ne fa 16,6); **tutte le altre categorie GPU entro ±0,012 ms** — firma di una modifica isolata; `decode.tokS` 35B **30,74** con dispersione [30,55-31,11] interamente sopra la barra; **gate argmax 39/39 IDENTICI**, routingDiff 0; i quattro casi ktest router stampano **le stesse cifre** prima e dopo (incluso `near-tie`, che è il tie-break); GLM non regredisce (11,33 con / 11,35 senza, stesso host). Artefatti: `q35-router-par-ab-2026-08-16.json`, `q35-router-par-gputime-2026-08-16.json`, `bench-glm-4090-b12-routerpar-5199-2026-08-16.json` + `-BASELINE-nostash-`. | none | `src/engine/kernels/wgsl.ts`, `tests/engine-routertopk-parallel.test.ts` | **done (it.18)** | 1 it, **1 consumata** |
| 3 | **L'unità di riparazione smette di essere il prefisso del pass.** Il router di un layer è noto prima che i suoi expert servano: il candidato è sovrapporre l'`ensure` del layer L+1 al calcolo del layer L, così il miss costa la fetch (già scesa in riga 2) e **non** 34,8 layer rigiocati. Il path `sync` esistente è il termine di paragone, non il bersaglio: a caldo faceva 132,8 ms/token contro 43,6 dell'ottimistico (`q35gpumodel.ts:2121-2133`) — ma quel confronto è PRIMA della riga 2, e la riga 2 ne cambia il segno. **Rimisurare i due bracci prima di scegliere.** | `replayLayers / (tokens × nLayer)` **≤ 0,20** (oggi 0,87) sull'artefatto di riferimento, **oppure** il replay a prefisso è sostituito e i suoi contatori dichiarati obsoleti con la ragione scritta; `dirtyTokens` resta un contatore valido e il suo valore è dichiarato; bit-identità del prefill MoE **rimisurata** (gate secco: qualunque cosa tocchi l'ordine delle somme la rompe). | none | `src/engine/q35gpumodel.ts` (tail ottimistico), `src/engine/moe.ts` | todo | 3-4 it |
| 4 | **La barra, e la prova che le leve erano globali**: misura di chiusura su TUTTE E QUATTRO le famiglie. | artefatto `kind: "engine-decode-checkpoint"` con `decode.tokS` prima/dopo per **4B, 9B, 35B e GLM**, `decodeContext` dichiarato per ciascuno, host quiescente, warm-up scartato, ≥ 3 repliche; **35B ≥ 30** a caldo (e il valore contro il nice-to-have 45); nessuna delle altre tre sotto la sua misura di partenza − 5%; la ripartizione del token con tutti i termini nominati (≥ 0,95 nel regime sporco). **Se una leva ha pagato su una famiglia sola, si dichiara col numero invece di nasconderlo in una media** — e la domanda che segue è *perché non è stata ereditata*: se la risposta è «andava riscritta a mano», il difetto è nella forma, non nella copertura. | none | `scripts/`, `results/engine/` | todo | 1-2 it |
| 5 | **Il 35B riceve il suo default di ragionamento.** Gated sulla riga 4: si esegue solo a barra passata, per la ragione scritta nel contratto (col thinking acceso il 35B genera molti più token). | il prompt reso in-page rispetta la polarità del template per famiglia (Qwen3.5 default OFF, Qwen3.6 default ON — verificato sui `chatTemplateRaw` dei tre JSON del 2026-08-15); la scelta è **dichiarata** nel campo `params.chatTemplate` del JSON invece di essere implicita; un test SENZA GPU che pinna le due polarità e fallisce se un modello nuovo entra senza dichiarare la sua; un turno di chat 35B in artefatto che mostra il blocco `<think>` **non vuoto** e i tok/s con thinking acceso. | none | `src/engine/chat/**`, `tests/**`, `results/chat/` | todo | 1 it |
| 6 | **GATE DI MERGE** (riga di sola verifica, dichiarata tale). | `node .harness/tools/engine-ktest.mjs` tutti PASS; top-1 vs oracolo llama.cpp ≥ 1012/1024 su entrambi i bracci; sequenze generate identiche 8/8; **decode 4B ≥ 45,5 tok/s a ctx 6333**; **`prefill.ms + decode.firstMs` < 22.500 ms**; `gemm:deltanet-out` ≤ 2.000 ms e `gemm:ffn-down` ≤ 2.000 ms; GLM b12 optimistic entro ±5% di 13.172 / 31,26 / 14,74; bit-identità prefill MoE 35B; `npx vitest run` verde; `npx tsc --noEmit` pulito. Tutto su host dichiarato. | none | — (gate) | todo | 1 it |
| 7 | **Consuntivo e consegna.** | `docs/engine/velocita-decode-consuntivo-<data>.md` clausola per clausola con l'artefatto accanto, la nuova ripartizione del token, e **il termine che diventa primo dopo questa leva nominato con la sua misura fresca**; `HANDOFF.md` §1 aggiornato; `GLOSSARY.md` coi termini coniati. | none | `docs/engine/`, `HANDOFF.md`, `GLOSSARY.md` | todo | 1 it |

**Conto dei gate**: una riga su sette è gate puro (la 6), una è misura pura (la
1, che non tocca una riga di ottimizzazione), una è la barra (la 4). Tre righe
muovono la metrica: 2, 3 e — al contrario, di proposito — la 5.

**La riga che può chiudere il goal da sola**: la 2. Se i 62.294 ms della fetch
scendono a ~1,3 ms/miss (il costo pack+upload già misurato), il token passa da
~120 ms a ~72 ms = 13,9 tok/s: **non basta**. Serve anche la 3: senza replay,
~120 − 62 − 45 = 23 ms/token = ~43 tok/s. *Conto mio dai contatori del turno, con
la riserva che i due termini non sono perfettamente additivi — meno replay
significa anche meno miss.* Le due righe si tengono: nessuna delle due, da sola,
porta il 35B sopra la soglia.
