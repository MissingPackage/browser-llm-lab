# HANDOFF — browser-llm-lab   (aggiornato 2026-08-16, notte)

## 1. Next decidable

**GOAL `engine-velocita-decode` ATTIVO. LA BARRA E' SUPERATA, IL GOAL NO.**

**IL RISULTATO, MISURATO** (`results/engine/q35-router-par-ab-2026-08-16.json`):

    decode 35B   22,58  ->  28,90  ->  30,74 tok/s
                 (base)    (kfan)     (router parallelo, it.18)
    barra 30 tok/s = 33,33 ms/token · siamo a 32,531 ms
    dispersione [32,143-32,732] = [30,55-31,11] tok/s: TUTTA sopra la barra
    gate argmax 39/39 IDENTICI, routingDiff 0

**IL LAVORO E' FERMO SU UNA DECISIONE TUA — docket item 7, tre uscite.** Non ho
altro di decidibile su questo goal senza quel ruling: la riga 4 (la barra su
tutte e quattro le famiglie) e la riga 6 (gate di merge) dipendono entrambe da
cosa il gate GLM deve confrontare.

**La riga 6 oggi**: «ktest tutti PASS» **verde** (`111 PASS / 0 FAIL`, it.19),
«GLM b12 entro ±5%» **rosso — ma il rosso non è del motore.**

**IL RIFERIMENTO DEL GLM SI SMENTISCE DA SOLO** (it.20, diagnosi senza una run
nuova — sta tutta dentro l'artefatto del 2026-08-15):

    STESSO file, STESSA run
      warm-up  prefill   19,10 GiB / 6,414 s  =  2,98 GiB/s
      warm-up  decode     1,51 GiB / 0,460 s  =  3,29 GiB/s
      REPLICHE decode    24,21 MiB/token / 2,475 ms  =  9,55 GiB/s   <- 2,9x

Lo stesso file, a minuti di distanza, tre volte più veloce della passata che
l'aveva appena letto. **Nessun disco fa questo**: le repliche non hanno letto,
hanno ripreso dalla **page cache del sistema operativo**. Oggi, quattro run
consecutive, warm-up e repliche coincidono a 1,31–2,34 GiB/s e nessuna si è
scaldata. I 15,330 tok/s non misuravano il motore: misuravano la RAM libera di
quel pomeriggio.

**Il codice è escluso, verificato**: nessun commit fra `bb3d430` (il commit del
riferimento) e HEAD tocca `glmsource.ts`/`glmmodel.ts`/`residency.ts`/
`expertstore.ts`/`glmbench/`, e `bytesRead`/`misses`/`evictions` coincidono
**cifra per cifra** fra le due date. Stessi byte, stesso codice, sette volte il
tempo.

**Corretto ciò che era mio**: il regime di lettura ora è **dichiarato** —
`readGiBs`/`readRegime` (`disk` | `os-cache` | `non-misurato`) in ogni fase del
report GLM e **nella headline**, che è il livello che i gate leggono; il runner
lo stampa e su `os-cache` avvisa in chiaro. Soglia `OPFS_DEVICE_CEILING_GIBS = 4`
con la provenienza misurata accanto. Verificato su run vera:
`readGiBs 1,336 · readRegime "disk"`.

**LA DECISIONE CHE E' TUA**: la riga 6 confronta il GLM con numeri presi senza
sapere in quale regime. Riprendere il riferimento a cache fredda (perde la
storia) · rendere il gate condizionale al regime (la conserva, ma serve un
riferimento per regime) · spostare il gate su `gpuBusy`, che fra le due date si
muove dello 0,8% invece che del 26% (misura il motore, ma **il 57% del token GLM
sta fuori dalla GPU**, quindi dichiarerebbe sano un motore che l'utente vede
lento). La mia lettura: la seconda, poi la prima per il solo b12. Le tre uscite
per esteso stanno nell'item.

**IL ROSSO DEL KTEST NON ERA UNA REGRESSIONE — era un file con un nome che
mentiva** (it.19, chiuso senza il bisect che avevo stimato 1-2 iterazioni).
`public/models/q35/golden-full.json` aveva **un nome, due scrittori**
(`q35-conf-run.mjs` e `q35-bench-run.mjs`, che ci copiano il golden della LORO
run — qualunque modello, qualunque `--golden-kind`) **e due lettori**, di cui
uno solo verificava lo SHA. Ci stava il golden **smoke del 35B**: 39 token
invece di 6.461, quindi il caso MTP misurava l'accept-rate su 37 confronti
contro un riferimento preso su 62. **L'avevo avvelenato io in it.18** lanciando
l'A/B con `--golden-kind smoke`, e il `111 PASS` di `engine-kquant` era
riproducibile solo perché l'ultima run di bench era stata, per caso, una 4B
full. Corretto alla radice: scratch dei bench → `golden-run.json`, fixture del
ktest → `golden-q35-4b-full.json` (lo copia `engine-ktest.mjs`, nessun runner lo
scrive), e il caso **verifica la finestra** invece di assumerla. Ora
`q35-mtp-draft-4b` dà **31/62 = 50,0%**, il riferimento cifra per cifra.

**IL REPERTO DA RICORDARE**: il difetto produceva **un FAIL con un numero
plausibile in un gate di merge** — 32,4% contro 50%, esattamente la forma di una
regressione del modello. Escluderlo è costato mezz'ora di GPU (due parchi
kernel interi con `git stash`); la causa l'hanno data **quattro righe di test
lette**. `engine-fase-d` it.53 aveva già visto lo stesso sintomo e l'aveva
risolto puntando il file giusto **senza togliere la collisione**.

**COM'ERA LA LEVA, perché è il reperto riusabile.** `router` costava **2,883
ms/token su 40 dispatch = 72 µs a layer** per scegliere 8 expert su 256, contro
i **16,6 µs** del GEMV che quei logit li produce con 256×2048 MAC. Quattro volte
più lento per 250 volte meno lavoro. La causa: `routerTopKWgsl` faceva la
selezione **su un thread solo su 64**, con un `array<bool, 256>` in memoria
privata a indice dinamico (scratch), **difeso da un commento che dichiarava la
serializzazione "la specifica" perché una riduzione "cambierebbe il tie-break"**.
Non lo cambia: lo scan con `>` stretto è il massimo sull'ordine TOTALE
(punteggio, −indice), e il massimo su un ordine totale non dipende da come si
associa. Ora: **0,836 ms/token, 20,9 µs a dispatch — il pavimento.** Tutte le
altre categorie GPU entro ±0,012 ms.

**E' UN KERNEL SOLO**, montato da `glmmodel.ts:1068` e `q35gpumodel.ts:1758`:
globale per costruzione, non riscritto due volte. Sul GLM rende poco e va detto
col numero — `gpuBusy` 38,9 → 38,6 — perché il GLM fa top-4 su 64 (otto volte
meno da parallelizzare) e il 57% del suo token sta comunque fuori dalla GPU.

**IL PRIMO TERMINE ADESSO** (`q35-router-par-gputime-2026-08-16.json`, braccio
kfan-ON, sonda accesa che perturba):

    ssmGemv    6,992   <- il primo
    expert     5,152
    attn       2,539
    shexp      2,157
    ssmOut     2,046
    router     0,836   <- era 2,883
    TOT GPU   22,872   (era 24,902)

`ssmGemv` e' la proiezione DeltaNet: **non e' MoE, e ce l'hanno anche 4B e 9B**.
E' la riga 2d, che resta aperta.

**LA RIGA 2d E' MISURATA E SI COSTRUISCE (it.21), con la regola scritta prima
dei numeri.** Banco sulle shape vere a M=1, contro il kernel che il decode
emette DAVVERO (braccio `base-decode`, non la forma del prefill):

    K2048 N=4096   base-decode 0,1616 ms ( 9,6% del picco) -> splitk 0,0489  3,30x
    K2048 N=8192   base-decode 0,1324 ms (23,4% del picco) -> splitk 0,0340  3,89x
                                                     splitk = 91,0% del picco

N=8192 e' `attn_qkv`, il 66,32% dei byte di `ssmGemv`. **Non era un problema di
occupazione: era di banda** — il GEMV a un workgroup per riga non satura il bus
e aggiungere righe non lo cura, lo cura spezzare il K. (Avevo previsto il
contrario e l'ho registrato come previsione sbagliata.)

Proiezione, dichiarata come criterio di spesa e NON come risultato: `ssmGemv`
6,99 → 1,97 ms, meno 0,78 ms dei 90 dispatch aggiunti = **~4,24 ms/token netti**
⇒ token 32,5 → ~28,3 = **~35,4 tok/s**. Pre-registrazione e graduatoria in
`docs/deep-dive/velocita-decode-2d-prereg-2026-08-16.md`.

**IL CONTO DELLA VRAM E' FATTO (it.22) ED E' UN NON-PROBLEMA**: `part` a M=1 =
128 KiB (N=8192) + 64 (N=4096), `xq` 2 KiB, `xsc` 256 B ⇒ **194 KiB con due
`part` separati = 0,0015% dell'arena da 12 GiB**. Il termine che rendeva grosso
il conto del prefill e' `M_MAX=16`; nel decode M=1 e sparisce. I buffer del
prefill NON si riusano (`prefillOn = M_MAX > 0`, e gli A/B del decode non
passano `--prefill-m`): la rotta alloca i suoi, il che a 194 KiB e' comunque la
scelta giusta — legare una leva di decode all'aver acceso il prefill a chunk
sarebbe accoppiamento gratuito.

**LA ROTTA E' IN ALBERO E SPENTA (it.23, fette 1-4).** `gemv` — l'emettitore del
DECODE — interroga `planPrefillGemm` a `M: 1` e, sui tensori che il piano
ammette, emette quantX → GEMM a fette → combine; i buffer sono i suoi (~131 KiB,
nessuna condizione sulla famiglia: decide la shape, non il modello).

**Il piano statico si costruisce UNA volta, quindi il flag non poteva essere di
costruzione**: sarebbe stato immutabile per la vita del modello e l'A/B avrebbe
confrontato DUE PROCESSI, cioe' due stati di cache — la classe di errore che
it.20 ha appena pagato (15,3 contro 11,3 tok/s sullo stesso codice). Il piano
porta quindi **entrambi i bracci** (`Step.arm`) e l'encoder ne salta uno tramite
un predicato solo, `stepOn`.

**IL BRACCIO SPENTO E' PROVATO NO-OP SU GPU**: ktest **111 PASS / 0 FAIL** e
`q35-model-4b-argmax` resta a **570 dispatch/token** con argmax 6/6 identico. Se
il filtro sbagliasse girerebbero entrambi i bracci, il conteggio salirebbe e
l'argmax cadrebbe.

**LA PROSSIMA ITERAZIONE — il flag ACCESO non e' mai stato eseguito.** Serve
`--splitk` su `q35-conf-run.mjs` (sul modello di `--kfan`), e poi **in
quest'ordine**: leggere `splitkAvail()` (quanti tensori il piano ha davvero
instradato — senza, un A/B piatto e' ambiguo) · gate **argmax 39/39** fra OFF e
ON · `ssmGemv` dalla sonda · `decode.tokS` · infine 4B e 9B.

Le cinque iterazioni precedenti della riga hanno prodotto, tutto in albero e
verde: la verifica di riuso (il kernel veloce ESISTE ed e' ktest-ato), il
predicato su N in `kernelVerdict`, il cablaggio del q8_0 con 11 test aggiornati.
**Il loro valore e' atterrato sul PREFILL del 35B, fuori scope**, e i 1,27 ms che
inseguivano li ha presi il router.

**LE SHAPE, ESATTE** (`q35shape.ts:86-89` + meta dell'header dump, K=2048):

    attn_qkv   N=8192  AMMESSO   66,32% dei pesi dei quattro
    attn_gate  N=4096  AMMESSO   33,16%
    ssm_beta   N=32    ESCLUSO    0,26%
    ssm_alpha  N=32    ESCLUSO    0,26%      -> ammessi 99,48%

**Anche il 35B ha `ssm_alpha`/`ssm_beta` a N=32, come il 4B**: il predicato non
protegge la peculiarita' di un modello, protegge quella coppia di tensori
nell'architettura ovunque compaia.

**LE DUE TRAPPOLE DEL PASSO CHE RESTA, pagate in it.17:**
1. **`gemvB` NON e' il decode**, e' il prefill. `loadW` restituisce due
   emettitori: `push` -> `gemv` (decode, `q35gpumodel.ts:862`, sei righe) e
   `pushB` -> `gemvB` (prefill, dove sta la rotta). Non c'e' un ramo da aprire:
   c'e' una rotta da costruire altrove. `gemv` emette UN dispatch, la forma
   split-K ne vuole TRE piu' due buffer che nel decode non esistono.
2. **Il 3,26x e' misurato contro `base-batch-z`**, non contro cio' che `gemv`
   emette (non-batch, e senza il `vec4Rows2` che vale solo per q4_0). Il
   guadagno del decode va MISURATO, non ereditato dal banco.

**COSA E' IN ALBERO E FUNZIONA**: kfan su `gemvQ4KWgsl`/`gemvQ6KWgsl`
(`kfan: {nUsed, xPerK}`), `moeCombineWgsl({weightsFromSel})`, il cablaggio nel
ramo ottimistico di `q35gpumodel.ts`, `setKfan()` per l'A/B a caldo, `--kfan` in
`q35-conf-run.mjs` col `kfanGate` sull'argmax, `gpuCat` per passata, il
predicato `PREFILL_GEMM_ROWS_PER_WG` e il q8_0 cablato. Il kfan e' spento di
default.

**TRE DIFETTI TOLTI SULLA STRADA** (non erano nel brief):
1. **`setInFlight(true)` senza `finally`**: fra submit e readback c'e'
   `popErrorScope`, che lancia. Il flag I1 restava alzato e OGNI passata
   successiva moriva su «ensure con token in volo» — **e l'errore riportato era
   quello del braccio SANO**, il primo a inciamparci.
2. **`pCombine` con layout AUTO** e un offset dinamico passato a
   `setBindGroup`: errore di validazione WebGPU.
3. **`arenaSlotWgsl`** era una costante con l'indice di Sel cablato; ora e' una
   funzione col default invariato.

**LA LEZIONE DEL GOAL, SEI volte**: una divisione non e' una misura, e una
descrizione del codice non e' il codice. 21,1 ms (token pulito), «16%» (il
gather), «due contratti diversi» (accum/slot), 30,9 us per dispatch (era 8,65),
«gemvB e' il decode» (e' il prefill), e **«la serializzazione E' la specifica»**
(non lo era, e valeva 2,05 ms/token). **Il correttivo che ha funzionato ogni
volta: leggere il codice, o misurare.** La sesta ha una coda: il commento
sbagliato e' sopravvissuto a goal interi perche' **nessun gate poteva vederlo** —
la forma seriale da' gli stessi numeri, quindi i ktest restavano verdi cifra per
cifra. Ora c'e' `tests/engine-routertopk-parallel.test.ts`, che pinna la FORMA e
non i valori (13 casi; 12 falliscono sul kernel vecchio, verificato).

---

**GOAL `engine-kquant` CHIUSO — tutte e sette le righe, il 2026-08-15.**

**IL RISULTATO, MISURATO:**

    TTFT a caldo   32.127 -> 17.153 ms   = 1,873x
    barra 22.500   PASSATA · nice-to-have 18.000  PASSATO
    prefill        197,25 -> 369,72 tok/s
    gemm:deltanet-out  12.169 -> 572,8 ms      (barra 2.000)
    gemm:ffn-down       4.971 -> 1.413,4 ms    (barra 2.000)

La proiezione del micro-banco di fase 0 (~16,9 s) ha sbagliato dell'**1,5%**.

**IL GATE DI MERGE, rifatto sull'albero FINALE** (dopo le correzioni di it.11):

    ktest                      111 PASS / 0 FAIL
    top-1 vs oracolo           1012/1024 su ENTRAMBI i bracci
    sequenze generate          8/8 IDENTICHE, zero token di differenza
    decode 4B a ctx 6333       47,06 tok/s   (barra 45,5)
    vitest 1019 passed | 10 skipped · tsc exit 0

Consuntivo: `docs/engine/kquant-consuntivo-2026-08-15.md` (589 righe, clausola
per clausola). Consegna al goal successivo:
`docs/engine/kquant-consegna-35b-2026-08-15.md` (423 righe).

**IL TERMINE CHE E' DIVENTATO PRIMO**: `deltanet:recurrence`, **1.732,1 ms** =
10,11% del prefill. E il fatto piu' grande accanto: **GPU nei pass 46,5%, fuori
53,5%** — piu' della meta' del prefill non e' piu' calcolo. **Un goal che
volesse ancora TTFT non lo trova nei kernel.**

## 2. Cosa il goal consegna a chi viene dopo

**LE DUE TRAPPOLE, scritte perche' non si paghino due volte:**
1. **Il flag `wired` e' per FORMATO, non per shape.** Q4_K, Q6_K e Q8_0 sono
   portati e verificati su GPU vera ma NON cablati. Il giorno in cui il goal 35B
   accendera' `q8_0` per i suoi tensori attn (N=4096), i **48 siti
   `ssm_alpha`/`ssm_beta` del 4B con N=32** entrerebbero nello stesso istante,
   perche' `prefillGemmCheck` guarda kind, K e fette ma **non N**. Serve un
   predicato sulla shape PRIMA di girare quel flag.
2. **Il prefill del 35B NON gira su `moeprefillplan.ts`** (refuso del contratto
   corretto in it.8): `planMoeChunk` ha un solo consumatore di produzione,
   `glmmodel.ts:1368`, che e' il GLM. Il 35B ripete per riga la catena del
   DECODE (`q35gpumodel.ts:2743-2778`): 40 round-trip di readback per chunk, 512
   dispatch per layer. Il piano CPU-side e' gia' parametrico su
   `{nExpert, nExpertUsed}` e regge `{256, 8}`: **il lavoro e' quel ramo `moe`.**

**IL COLLO DEL 35B NON E' IL KERNEL, E' LA RESIDENCY**: 19,45 GiB di modello
contro ~16 GB di scheda, expert Q4_K 17.666.408.448 B.

**LA REGOLA CHE IL 15 AGOSTO HA COMPRATO A CARO PREZZO** (tre difetti, una sola
malattia — trattare un comando come documentazione invece che come codice):
**un comando lasciato per la ripresa si esegue come sta scritto prima di
lasciarlo, e i flag di un runner si leggono dal SORGENTE, mai eseguendolo.**
- `BASE_URL` mancante nel comando di ripresa dell'HANDOFF (il ktest di default
  parla alla 5173);
- `--prefill-m 16` mancante nel comando di evidenza del contratto (default
  `null` ⇒ prefill SEQUENZIALE: 91.230 ms invece di 17.126, una run buttata);
- un `--help` dato a un runner che non lo conosce, che ha eseguito il bench coi
  default **e ucciso una run di conformita' in corso**.
E `--conf-prefill-m` NON e' `--prefill-m` (quest'ultimo esce al gate dei due
bracci senza arrivare al replay golden); `--prefill-batch` del GLM e' un
BOOLEANO 0/1, non una M.

Comandi, tutti eseguiti come stanno scritti:

    setsid nohup npx vite --port 5199 > /tmp/vite-5199.log 2>&1 < /dev/null &
    BASE_URL=http://localhost:5199 node .harness/tools/engine-ktest.mjs   # 111 PASS / 0 FAIL
    node scripts/q35-bench-run.mjs --prompt-idx 0 --n-decode 64 --vram-gib 8 --prefill-m 16 --declared quiescent
    node scripts/q35-conf-run.mjs --out results/engine/<seq>.json
    node scripts/q35-conf-run.mjs --conf-prefill-m 16 --out results/engine/<chunk>.json
    node scripts/glm-bench-run.mjs --prompt 6 --ngen 64 --reps 3 --budget-gib 12 --select optimistic --prefill-batch 1 --host-state quiescent --out results/engine/<glm>.json
    node scripts/build-ttft-checkpoint.mjs <bench.json> <segmenti.json> <out.json> --ratchet <ratchet.json>

**POTATURA FATTA** (2026-08-15, chiesta dal PI a goal chiuso): i 35 worktree di
workflow in `.claude/worktrees` sono stati rimossi, **1,2 GB liberati**. Prima
di cancellare sono stati controllati uno per uno: due avevano commit non
raggiungibili da main, ed erano entrambi impalcatura dichiarata («base: patch
t1+t2+t7 — *non parte della mia patch*», «commit locale al worktree per isolare
la diff»), coi genitori gia' in main. **Quattro file esistevano solo li'** —
`engine-q35prefillwiring.test.ts`, `engine-prefillroute.test.ts`,
`engine-gemvquant-shape.test.ts`, `gemv-quant-5114160.golden.json` — e sono
stati messi al sicuro in `~/worktree-salvage-2026-08-15/` invece di sparire.
Restano 35 branch `worktree-wf_*` (soli riferimenti: `.git` pesa 23 MB in
tutto); si cancellano con `git branch -D` se danno fastidio all'elenco.

**IL REPERTO DA NON PERDERE**: la guardia doppia del cablaggio
(`route.via !== "legacy" && kk === "<formato>"`) **ha intercettato un caso
vero** il giorno dopo essere stata scritta. Quando il piano ha accettato il
q4_1 prima che `gemvB` emettesse i suoi kernel, quei tensori sono ricaduti
sulla legacy invece di essere letti col kernel del q4_0 — nibble senza offset e
scale col passo sbagliato, cioe' logit storti senza nessun errore WebGPU.
Chi tocca quel sito non tolga quella condizione.

**LA FASE 0 HA DETTO SI' A TUTTE E CINQUE LE FAMIGLIE.** Rapporto legacy/veloce
a M=16, `idot | f32`, barra 1,5x. Artefatto:
`results/microbench/kquant-fase0-4090-linux-2026-08-14T19-29-20-014Z.json`
(54 celle, **zero scartate**). Prereg:
`docs/deep-dive/kquant-fase0-prereg-2026-08-14.md`.

| famiglia | shape | M=16 |
|---|---|---|
| Q8_0 | `[2048, 4096]` | **35,20 \| 17,63** |
| Q5_K | `[4096, 2560]` | **28,10 \| 6,17** |
| Q4_1 | `[9216, 2560]` | **22,57 \| 16,63** |
| Q6_K | `[512, 2048]` | **6,13 \| 1,66** |
| Q4_K | `[512,2048]` / `[2048,512]` | **5,23 \| 1,85** e **4,16 \| 1,14** |

**Il banco riproduce il segmento vero in millisecondi**: 24 × 395 × 1,2700 =
12.039 contro i **12.169 ms** misurati su `gemm:deltanet-out`. Proiezione
**−15,2 s ⇒ TTFT ~16,9 s** (proiezione da microbench: la conferma e' la riga 5).

**TRE COSE DA SAPERE PRIMA DI USARE QUESTI NUMERI:**
1. **Il guadagno e' una proprieta' della SHAPE, non del formato.** Tre
   previsioni su cinque sono cadute perche' lo attribuivo all'unpack. Conta
   quanto costa rileggere la matrice M volte: tensori grandi 22-35x, tensori
   piccoli 4-6x. E il crollo ha DUE cause di peso simile — la cache (il legacy
   e' 2,2x piu' veloce per peso sulle shape piccole) e l'occupancy (la forma
   veloce e' 2,3x piu' lenta: 32-64 workgroup su 128 SM).
2. **Il 4-6x del 35B NON e' un limite inferiore** (affermazione ritrattata in
   it.3): per Q4_K e Q6_K il braccio legacy misurato **non e' il percorso di
   produzione** — gli expert girano in regime d'ARENA, dove `batch` e' vietato
   per costruzione (`wgsl.ts:2175`, `:2359`). Vale per una shape su un braccio
   ipotetico. Il goal 35B parta da qui, non dal numero nudo.
3. **Sul Q4_K la via f32 non passa la regola di stop** (1,14x su `[2048,512]`):
   senza `packed_4x8_integer_dot_product`, li', la forma multi-riga non paga.

**A M=1 la forma multi-riga PERDE** (0,91x sul Q4_K): il piano non deve mai
offrirla al decode.

**Da misurare, non dedurre**: la quota Q4_1 di `gemm:ffn-down` era stimata dal
banco. Risolto in it.5 — quei quattro siti hanno una categoria di misura propria
(`gemm:ffn-down-q41`), cosi' la riga 5 attribuisce quel tempo invece di dedurlo.

Toglie ai pesi non-q4_0 le M riletture per chunk: `ssm_out` **Q5_K** (37,9% del
prefill) e `ffn_down` **Q4_1** (il 71% dei byte del segmento `gemm:ffn-down`)
passano alla forma multi-riga. **Barra: TTFT a caldo < 22.500 ms** (oggi
32.127), **nice to have < 18.000**; copertura del piano da 5,86× a **≥ 15,5×**;
`gemm:deltanet-out` e `gemm:ffn-down` entrambi **≤ 2.000 ms**. Proiezione
−14,7 s ⇒ ~17,4 s, e quel numero e' anche il **pavimento**: sotto i ~9,4 s
(il tempo fuori dai pass GPU) non si scende togliendo byte ai pesi.

**Il 35B e' il goal SUCCESSIVO, deciso dal PI.** Le forme di kernel che gli
servono (Q4_K, Q6_K, Q8_0) nascono qui — misurate al banco e verificate col
ktest — ma **non vengono cablate**: il 35B non ha un byte di q4_0 (expert Q4_K
17,67 GB), pero' il suo collo e' la **residency**, non il kernel, e non ha una
baseline fresca. Sara' wiring + residency. **Il 9B ha la stessa identica
struttura del 4B**: questa leva vale li' senza una riga di codice in piu'.

---

**Goal `engine-ttft` CHIUSO** (2026-08-14, ruling del PI). Il tempo al primo
token a modello caldo sul prompt da 6333 token: **87.618 -> 32.127 ms = 2,727x**.
La barra meccanica del contratto (< 21.905 ms) **non e' stata raggiunta**: manca
1,467x. Dieci clausole su dodici soddisfatte; le due che cadono sono la stessa —
la barra e la sua gemella `prefill.tokS > 289` — e **la causa e' misurata e
stava fuori dalla portata del goal**.

Consuntivo voce per voce, con l'artefatto accanto a ogni clausola:
`docs/engine/ttft-consuntivo-2026-08-14.md`.

**I DUE GOAL SUCCESSIVI, decisi dal PI. L'ORDINE FRA LORO E' L'UNICA COSA
APERTA** — e i due assi sono diversi, quindi non e' una preferenza ma una
scelta:

1. **K-quant — vale il 37,9% del tempo del prefill.** `gemm:deltanet-out` e' il
   primo termine con soli 24 dispatch, uno per layer DeltaNet, perche' `ssm_out`
   e' **Q5_K** e cade sul fallback legacy: riuso dei pesi ZERO, riletti 16 volte
   per chunk, **89,9 GB/s** su un motore che ne ha dimostrati ~300. **Il ruling
   del riuso lo aveva registrato come una CODA** («l'11,54% dei byte resta sul
   percorso vecchio»): la quota di byte sottostimava il peso proprio perche' la
   forma legacy rilegge i pesi M volte. **E' la leva piu' grande rimasta sul
   tempo al primo token.**
2. **0.5B — non muove la metrica di prodotto, muove la RAGGIUNGIBILITA'.** E' il
   path di conformita', e il suo valore e' girare su device che concedono il
   minimo di spec WebGPU (16.384 B di memoria di gruppo). Scope fissato dal PI:
   **«migreremo il possibile»** — tre siti su quattro adottano la forma
   split-K esistente; il quarto (`gemvResidualFast`, down-proj del **decode** a
   M=1) e' **non-migrabile per costruzione** e va dichiarato tale, non contato
   come buco. Veicolo: workflow `pattern-migration`, che ora produce
   `nonMigrable` + una lettura in chiaro invece di una percentuale.

**PRIMA DI TOCCARE IL 0.5B, leggere questo**: i consumatori sopra la garanzia
WebGPU sono **quattro**, non uno, e alimentano **un solo valore condiviso**
(`QWEN_WORKGROUP_STORAGE_BYTES`, che il motore chiede come tetto del device).
Migrarne un sottoinsieme fa SCENDERE quel massimo mentre gli altri chiedono il
valore vecchio ⇒ `createComputePipeline` fallisce **su ogni device**. La
costante e' gia' stata resa un `Math.max` **calcolato** dalle formule accanto ai
kernel (it.24) proprio per questo: quella correzione e' il prerequisito, ed e'
gia' in albero.

**LE LEVE DI `engine-ttft`, tutte in produzione e misurate prima/dopo**:
moltiplicatore multi-riga `splitk` (prefill 34,36 -> 111,16 tok/s) · via intera
`dot4I8Packed` (-> 123,26) · attenzione del prefill in streaming (-> 196,41).

**ESCLUSE COI NUMERI — l'eredita' piu' utile per chi riprende:**
- **fusione delle teste GQA sul prefill: PIU' LENTA** (2,0879 contro 1,8207 ms).
  Sul decode aveva funzionato: il verso e' opposto, e chi riprova deve saperlo.
- **il tetto di memoria di gruppo negoziabile non e' una leva** (spread 0,1-2,3%).
- **la ricorrenza DeltaNet NON e' il collo**: 5,0% del tempo contro il 47,3% dei
  dispatch. **Contare i dispatch non e' misurare il tempo** — l'avevo concluso
  dal solo conteggio, ed era gia' finito in un HANDOFF prima che il cronometro
  lo smentisse.

**Dove finisce il tempo** (`results/engine/q35-ttft-kernel-checkpoint-4b-2026-08-14.json`):
70,9% dentro i pass GPU, **29,1% fuori** (encode CPU, submit, buchi fra submit —
non attribuito piu' finemente, e lo dichiaro). **1,578 TFLOP/s contro il picco
fp32 misurato di 9,26 = 17%**: non e' un'efficienza, e' la prova che **il collo
non e' l'ALU**.

**Gate alla chiusura**: tsc pulito · vitest **680 | 10** · ktest **101 PASS /
0 FAIL** · top-1 contro l'oracolo llama.cpp **1012/1024 = 98,828% su ENTRAMBI i
bracci** · sequenze generate **identiche 8/8** · decode **48,15** tok/s (soglia
45,5).

**Goal `engine-kernel-decode` CHIUSO** (2026-08-13): decode a contesto 6333 da
9,95 a 47,93 tok/s, 4,82x, sopra la soglia di prodotto di 30.

## 2. Mappa

**Destinazione.** Far girare in browser il modello più capace possibile
restando usabile: almeno **30 token/secondo** e **primo token entro 4 secondi**.
Il secondo termine è un'aspirazione di prodotto, non una promessa di questa
macchina: su un prompt da 6k il 4B ha un pavimento misurato di ~9,4 s, e il PI
ha riscritto l'obiettivo del goal in «il più in basso possibile» (2026-08-13).

**Distanza adesso**: Qwen 4B **48,15 tok/s a contesto 6333** (era 9,95) —
**sopra i 30 dell'obiettivo**. Il TTFT a caldo sul prompt da 6333 token è
**32,1 s MISURATO** (it.25, era 87,6 in it.1) contro una barra di 21,9 s: manca
**1,47×**. Il goal `engine-ttft` ha il consuntivo pronto e la barra NON
raggiunta; la causa è misurata e sta **fuori dalla sua portata** — il 37,9% del
prefill è un `ssm_out` **Q5_K** sul percorso vecchio, e le leve di questo goal
sono q4_0-only per costruzione. **La prossima leva sul tempo al primo token è il
goal K-quant**, che finora era registrato come una coda.
GLM-4.7-Flash resta residency-bound
(~13 tok/s, TTFT 14,7): nessuna delle leve di questo goal lo tocca. Il 35B non
è stato rimisurato dopo i kernel nuovi.

**Decisioni prese** (indice: il contenuto vive nel posto indicato, non qui)

- Il checkpoint di misura ridotto a vero merge gate — `PHASES.md` riga 6
- Il repack dei pesi resta dov'era, reso 6 volte più veloce — docket, "la fase 2 ha centrato l'obiettivo ma non la lettera"
- Il decode ottimistico è il percorso attivo; la soglia d'ingresso esclusa coi numeri — docket, "i tre numeri per tarare la soglia"
- Prefetch e politica a livelli esclusi perché misurati inutili — journal it.37
- I riferimenti di velocità Qwen di ieri misurano il percorso vecchio: da rifare — docket, "i riferimenti misurano il path sync"
- Il confronto col motore nativo non è a parità — docket, "il gap nativo e la page cache"

**Nebbia** (non ancora deciso né specificato)

- Quanto valga davvero il 74%: 23 posizioni danno 52-90%, e su un prompt solo
- Se il 35B abbia lo stesso accept-rate: mai misurato (in CPU costerebbe ore)
- Se GLM possa avere una testa equivalente: mai valutato
- Cosa debbano misurare i "riferimenti" — il percorso storico (confrontabile) o
  quello di oggi (vero). Nessuno dei due da solo basta

**Fuori scope**

- I benchmark comparativi fra stack: un goal chiuso, l'altro in pausa dichiarata
- Il raggruppamento degli expert e il pipelining del decode: registrati, non aperti
- **L'attenzione a chunk del prefill** (`attnDecodeWgsl` con `batch`) ha gli
  STESSI tre difetti di quella del decode — ridondanza GQA 4x, letture scalari,
  `scores` in memoria di gruppo — e la stessa riscrittura li toglie. È ora la
  riga 3 del goal `engine-ttft`: non più fuori scope
- Tutti i numeri destinati alla pubblicazione: si rimisurano al tag di release

## 3. Landmines

- **I JSON in `results/` possono mentire in silenzio.** I runner uscivano con
  successo anche quando la GPU falliva, scrivendo numeri plausibili. Ora c'è una
  sentinella su tutti e quattro, ma i file scritti PRIMA di oggi non sono
  protetti.
- **Gli 8 riferimenti Qwen del 2026-08-11 (`*fase-d-it43*`) misurano il percorso
  vecchio**: non usarli per la pubblicazione.
- **`ttftMs` INCLUDE il caricamento del modello** (`q35conf.worker.ts:246`:
  `loadMs + prefillMs + firstMs`). Un TTFT citato senza i tre termini scomposti
  non dice cosa sta misurando — ed è così che 10,89 s di load sono finiti in un
  obiettivo di prodotto sul prefill.
- **`--prompt-idx` ha default 4 = 388 token, non il prompt da 6k.** È il flag da
  cui veniva il numero sbagliato corretto in it.0. I prompt ≥ 6000 token sono
  l'idx 0 (6333) e l'idx 5 (6128).
- **Il server di sviluppo muore da solo: TRE volte in una sessione**
  (2026-08-13, tutte exit 144 = ucciso da segnale, non crash). Evidenza: il log
  si ferma su «ready» senza una riga di errore, nessun OOM in `dmesg`, 19 GB
  liberi. L'ipotesi — il supervisore dei comandi in background lo reap al confine
  di turno — è **CONFERMATA per i confini di turno**: i tre morti erano tutti lì, e
  avviato staccato ha superato il confine successivo (verificato con `curl`).
  **NON sopravvive però all'uscita del processo Claude Code**: alla ripresa di
  sessione era di nuovo giù. `setsid` sposta il problema, non lo toglie.
  **Avviarlo così, e ri-avviarlo a ogni ripresa di sessione:**

      setsid nohup npx vite --port 5199 > /tmp/vite-5199.log 2>&1 < /dev/null &

  Verificare comunque PRIMA di ogni run che costa GPU, **con `curl` e non con
  `pgrep`**: `curl -s -o /dev/null -w "%{http_code}" http://localhost:5199/<entry>.html`.
  `pgrep -f "vite --port 5199"` FA MATCH SULLA PROPRIA RIGA DI COMANDO e risponde
  sempre "vivo" — ci sono cascata il 2026-08-13, dichiarando vivo un server morto.
  Il gate dei kernel almeno NOMINA la causa (exit 2, «nessun server su …»); i
  runner di bench no, e per loro il sintomo resta un fallimento di caricamento
  pagina, cioè una causa travestita.
- **Il prefill "a chunk" è più LENTO del sequenziale** (34,4 contro 72,3 tok/s,
  it.1): instrada sul GEMV vecchio, mentre `step()` usa quello veloce. Un bench
  che non dichiara `prefillPath` non dice quale dei due ha misurato.
- **La cache del filesystem falsa il confronto col motore nativo del 35-41%.**
  Nessuno dei due artefatti dichiara il proprio stato di cache.
- **Mai confrontare misure fra macchine senza stato dell'host dichiarato**: la
  variazione fra due run identiche è ~2,4%, più della distanza dal riferimento.
- **Un tetto di 16 GiB di VRAM non esiste su questa scheda** (16.376 MiB
  totali): la richiesta sfonda.
- **Un campione da 22 posizioni non distingue niente**: ±1 colpo vale ±4,5
  punti, e it.50-51 ci ha perso due iterazioni. Prima di concludere da un
  conteggio di successi, guarda rango e log-prob del bersaglio: stessa
  informazione, varianza molto più bassa, spesso costo zero.
- **Leggere i parametri di un runner prima di spenderci sopra minuti di GPU.**
  Due volte in una sessione ho lanciato run con flag sbagliati dedotti invece
  che letti.
- **`q35-bench-run.mjs --prefill-m` NON attiva il gate di conformità: misura la
  velocità.** Quel runner passa SEMPRE `?bench=`, e la pagina prende il ramo
  bench prima di arrivare al gate. Il gate del prefill a chunk lo lancia
  **`q35-conf-run.mjs --prefill-m 16`**. Ci sono cascato in it.20 e ho prodotto
  un JSON il cui NOME diceva `q35-prefillchunk` e il cui `kind` era
  `q35-bench-4b-fullresident` sul prompt-idx 4 — cioè le due landmine qui sopra
  sommate. Rimosso in `9c04728`. **Prima di credere a un artefatto, leggi il suo
  `kind`, non il suo nome di file.**
- **UN `git checkout` DI UN RAMO VECCHIO PUÒ FAR NASCERE UN AGENTE FANTASMA nel
  tuo working tree.** Successo il 2026-08-14, ricostruito dal reflog e dal log
  del watchdog: alle 00:00:51 il checkout di
  `wip/riga2-cablaggio-non-verificato` ha riscritto
  `.harness/loop-state.json` con la copia `status: "active"` di quel ramo
  (`next_wake` 23:12:44); alle **00:01:34**, 43 secondi dopo,
  `harness-loop-watchdog.timer` (poll ogni 2 min) l'ha letto, ha visto un
  risveglio scaduto da 48 minuti e ha **rianimato una sessione che era stata
  fermata apposta** con `ScheduleWakeup{stop:true}` prima del riavvio. Due
  sessioni Claude hanno lavorato in parallelo sullo stesso albero e sulla stessa
  GPU per ~15 minuti, e me ne sono accorto solo perché `q35gpumodel.ts` è
  cambiato sotto di me.
  Il watchdog **non legge male**: legge `active` perché glielo ha appena scritto
  git. Il difetto era che un file di RUNTIME stesse in git — corretto in
  `c21648e` (`.gitignore` + `git rm --cached`).
  **RESIDUO APERTO: 19 rami portano ancora `status: "active"`** — tutti i
  `wip/riga2-*` e i `worktree-wf_*`. Toglierlo da main impedisce a main di
  ri-armare la miccia, **non** impedisce a un checkout di quei rami di
  riarmarla. La difesa robusta (rifiutare uno stato scaduto da N poll, e non
  saltare il controllo «transcript fresco» sul ramo `headless_dead`) sta in
  `~/Projects/harness/tools/loop-watchdog.sh`, **altro repo, non toccato**.
  Sintomo da cui riconoscerlo: file sorgente che cambiano senza che tu li abbia
  scritti. `ListAgents` mostra la sessione peer; `git log -1 --format=%h
  .harness/loop-state.json` contro `updated_epoch` dice se il file è stato
  riscritto da un checkout invece che dall'hook.
- **ASPETTARE UN WORKFLOW IN BACKGROUND FA DUPLICARE LA TUA SESSIONE. Sempre, e
  allungare il risveglio non serve.** Il watchdog usa il silenzio del transcript
  come test di morte; una sessione che aspetta un workflow è silenziosa **per
  costruzione**, perché il workflow non scrive sul transcript di chi l'ha
  lanciato. Misurato tre volte il 2026-08-14: 00:56:05 («transcript fermo da 28
  min») e 02:03:28 («fermo da 62 min») hanno duplicato la stessa sessione
  `ae3ad6a9`, la seconda volta **nonostante il risveglio fosse stato allungato a
  60 minuti apposta**. Allungare sposta l'istante, non toglie la condizione: il
  ramo `headless_dead` del watchdog **salta del tutto** il controllo «transcript
  fresco ⇒ sta lavorando», e quel ramo è vero per sempre una volta che il
  watchdog ha spinto una prima volta su quella directory.
  I duplicati **non sono innocui e non sono inutili**: hanno committato lavoro
  buono (l'integrazione di it.17 e l'attribuzione del gate rotto sono pulite),
  ma nessuno li coordina, lavorano sullo stesso albero e sulla stessa GPU, e
  chi li ha generati non può verificarli mentre corrono.
  **Finché il watchdog non è corretto: niente `/loop` con workflow in volo.**
  Il lavoro lungo va fatto con un umano presente, o `ScheduleWakeup{stop:true}`
  prima di lanciare il workflow — al prezzo che il loop poi non riparte da solo.
