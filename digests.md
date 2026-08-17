# Digests — browser-llm-lab

Narrativa per ciclo, append-only. L'HANDOFF e' una MAPPA di rientro e sta
sotto le 80 righe; la storia vive qui, le decisioni nei docket, la traccia
tecnica nei journal.

---

## Ciclo 2026-08-17 (pomeriggio/sera) — rilascio: misure, nome, split

Aperto sul docket 14-19 e chiuso col repo motore esistente. In ordine:

- **Igiene**: 5 worktree SDD (198 MB) + 41 branch rimossi, verificati file per
  file col diff contro main (non solo «zero commit»). Nessun lavoro perso.
- **Nome**: `webgguf`. Il PI ha rovesciato la linea evocativa — «i nomi
  evocativi non hanno successo su GitHub» — e ha ragione su un repo che cerca
  adozione. Libero su npm senza scope, zero repo omonimi.
- **Riferimento 35B Q2_K**: non esisteva. Dopo il passaggio a Q2_K non c'era un
  solo artefatto in `results/engine/`, e il 34,97 poggiava su artefatti che si
  autodichiarano NON riferimento. Tre repliche -> **34,602 tok/s**, spread 1,35%.
- **Head-to-head**: 322,6 contro WebLLM 89,5 same-day / 117,69 loro migliore.
- **shader-f16**: non ottenibile su Chrome/Linux (5 combinazioni di flag, driver
  capace). Firefox la espone.
- **Split eseguito**, e l'esecuzione ha trovato tre difetti che leggere il piano
  non poteva trovare: il typecheck ereditava i tipi node da `@wllama/wllama`;
  il manifest amputava `src/microbench/`; cinque strumenti DEL MOTORE vivevano
  in `.harness/` (spostati in `tools/harness/`).
- **Tre consulenze Fable**, che hanno corretto quattro premesse — due mie:
  «il decode e' memory-bound» (falso: il f32 satura la banda, i quantizzati
  stanno al 20%; il collo e' l'ALU della dequant) e «Firefox e' 9x piu' lenta»
  (quei numeri erano di WebLLM: il nostro motore non ci aveva mai girato).
- **Browser**: Chrome nudo 327,3 (il flag non serviva); Firefox 79,6 contro i
  9,9 di WebLLM. Degradiamo 4x dove loro degradano 9x.
- **Il 35B si carica su Firefox** (3,25 -> 4,94 tok/s su 2 turni).

---

## Cicli precedenti — archiviato dall'HANDOFF.md del 2026-08-17

# HANDOFF — browser-llm-lab   (aggiornato 2026-08-17)

## 1. Next decidable

**LA SESSIONE DEL 2026-08-17 HA CHIUSO IL GOAL SULLA VELOCITA' E APERTO LA FASE
DI RILASCIO.** Le decisioni aperte stanno nel docket, item **14-19**: licenza,
pesi da Hugging Face, inglese su tutto il pubblico, forma di distribuzione,
split delle repo, matrice dei dispositivi. Tre di quelle sono gia' DECISE dal PI
(15, 16, 19) e aspettano lavoro; tre aspettano un ruling (14, 17, 18).

**Leggi `docs/architettura/MECCANISMI.md` prima di progettare, e
`docs/architettura/VALUTAZIONE.md` prima di credere a un numero.**

### COSA E' CAMBIATO OGGI, in una riga per cosa

    la barra dei 30 tok/s      PASSATA in chat vera: 34,97 di regime su 10 turni
    la leva                    NON un kernel: la CAPIENZA (quant Q2_K, parco
                               17,07 -> 10,39 GiB, tutto dentro l'arena)
    il costo                   +0,13 bit/token, uniforme su quattro domini
    il motore e' fedele        top-1 99,04% vs llama.cpp, rango MASSIMO 2
    lo slab                    +3,7%: non serve, GGUF basta (docket 13)
    il thinking del 35B        SISTEMATO: la polarita' si deriva dal template
    tre documenti nuovi        QUANTIZZAZIONE.md, VALUTAZIONE.md, e i banchi

### LA PRIMA COSA DA FARE IN UNA SESSIONE FRESCA

**Lo SPLIT DELLE REPO** — deciso dal PI a chiusura del 2026-08-17: «faro' lo
split come prima cosa in una sessione fresca». Va fatto prima di tradurre e
prima di ripulire, per la ragione al punto 1 qui sotto.

*Igiene da fare li'*: restano 5 worktree del workflow SDD in `.claude/worktrees/`
(198 MB, **zero commit non presenti in main** — verificato a chiusura sessione:
sono copie di lavoro, tutto e' stato integrato per patch). Si tolgono con
`git worktree remove`, e vanno tolti prima dello split o finiscono nel rumore.

### IL LAVORO CHE DISCENDE, in ordine di dipendenza

1. **Split delle repo** (item 18) — perche' definisce cosa e' pubblico, e senza
   quel confine la traduzione in inglese non e' eseguibile.
2. **Traduzione del pubblico** (item 16), file per file, dentro lo split.
3. **Licenza + attribuzione** (item 14): senza, per default nessuno puo' usarlo.
4. **Pesi da HF** (item 15): URL nel registro, e VERIFICARE Range + CORS + redirect
   verso il CDN, che oggi non e' mai stato provato — leggiamo da localhost.
5. **API pubblica** (item 17): oggi la superficie e' `createQ35GpuModel` con venti
   opzioni e nomi interni.
6. **Rimisurare tutto al tag**, corpus CONGELATO e repliche (regola del progetto).

### CIO' CHE RESTA APERTO DEL GOAL, e non e' bloccante

- **riga 4**: la barra formale vuole un artefatto di riferimento su QUATTRO
  famiglie con warm-up scartato e >= 3 repliche. Oggi abbiamo la chat, che il PI
  ha dichiarato essere il numero che conta, ma non l'artefatto del contratto.
- **riga 7**: il consuntivo del goal.
- 4B, 9B e GLM non sono stati rimisurati sui dieci turni.

### IL NUMERO CHE CONTA E' QUELLO DELLA CHAT — e su DIECI TURNI

**Ruling del PI 2026-08-17: le prove in chat si fanno su 10 turni di follow-up,
mai due.** Due turni misurano l'arena che si RIEMPIE, non il regime.

    regime (media ultimi 5 turni di 10)     Q4_K_S 11,86  ->  Q2_K 34,97 tok/s
    miss per turno, al decimo                       429   ->            17
    TTFT                                      2,2-8,5 s   ->     0,7-1,2 s

**LA BARRA DEL GOAL (>= 30 tok/s a caldo) E' PASSATA IN CHAT VERA: 34,97.**
La leva non e' un kernel: e' la CAPIENZA. Col quant `bartowski Q2_K` il parco
expert passa da 17,07 a **10,391 GiB** ed entra tutto nell'arena; i miss vanno a
**0,05%** e ci restano, mentre sul Q4_K_S oscillano e non convergono mai.

Costo misurato: **+0,13 bit/token** (appaiato, teacher-forced, corpus in
`results/eval/`), -0,15 di logprob sulla matematica, indistinguibile da zero
sulla conoscenza. E' velocita' contro intelligenza, e i due si leggono insieme.

### COSA E' ENTRATO IERI NOTTE, e cosa ha insegnato

**Il 35B adotta `{ raw, slab }`** (it.50): `slabsource.ts` apre il file slab
servito via Range, `readMiss` in `q35gpumodel` e' la porta UNICA dei due siti di
fetch, e il fallback ai byte grezzi e' **dichiarato in `moeStats().slabSource`**
(quindi in ogni artefatto). Il file e' convertito e servito:

    ~/.cache/blab-models/q35/q35-a8138f183e3993f1.slabs.bin   17,07 GiB
    symlink in public/models/, convertito in 12,5 s, 16/16 slab verificati

**IL RISULTATO, A/B back-to-back sullo stesso host** (`results/chat/
chat-35b-slabAB-{A-noslab,B-slab}-2026-08-17.json`):

    tok/s        11,066 -> 11,472  (+3,7%)   ·  secondo turno +3,2%
    packMs        7.331 ->      0            <- binario, non una media
    fetchRepairMs 68.514 -> 71.639  (+4,6%)  <- LA SORPRESA
    risposte      IDENTICHE carattere per carattere, 1.310 token, temp 0

**Il conto di it.40 (7,11 s) valeva per meta'**: il pack sparisce davvero, ma
leggere UNO slab da 1,77 MB e' piu' lento di TRE tensori da 594 KB letti in
`Promise.all` — una richiesta grande non ha nessuno con cui sovrapporsi. E' il
reperto di it.33 (raggruppare non paga, 0,98x) che rientra da un'altra porta.
Netto: 4,2 s su 7,3.

### LA PROSSIMA COSA DA FARE, e nasce da questa misura

**Leggere uno slab in 2-4 sotto-range PARALLELI.** Il ginocchio della curva
banda/richieste-in-volo e' a 2-4 (it.33), `slabFileRange` e' gia' aritmetica, e
i contatori del parallelismo effettivo sono in `ggufrange`. Recupererebbe i
~3,1 s che il formato si e' ripreso. *E' una riga nuova, non una rifinitura.*

Dopo, in ordine di valore: **prefetch lookahead** (oracolo al 91,92% @K=8,
attacca i ~76 ms/token di tassa di residenza) e **spec-dec nella chat** (1,29x
proiettato, serve una testa MTP per il 35B che oggi non esiste).

### UNA TRAPPOLA PAGATA DI NUOVO

Il comando che it.47 aveva lasciato scritto (`node scripts/q35-slab-build.mjs`)
**non girava**: `ERR_MODULE_NOT_FOUND` sugli import senza estensione. L'hook
condiviso esisteva (`scripts/lib/tsimport.mjs`) e lo script non lo usava.
Corretto. E' la terza istanza della stessa regola: **un comando lasciato per la
ripresa si esegue come sta scritto prima di lasciarlo.**

### LE TRE LEVE, e come vanno lette

    22,58 -> 28,90 (kfan) -> 30,74 (router) -> 40,06 (rotta split-K)
    gate argmax 39/39 su tutte e tre

⚠️ **kfan e rotta NASCONO SPENTE** (i loro A/B le accendono un braccio alla
volta). **La chat le accende da it.39** ed e' li' che stanno i +37%: per giorni
«misurato» e «consegnato» non coincidevano e niente lo diceva. Ora
`model.levers` sta nell'artefatto di chat.

### COSA E' STATO ESCLUSO DALLA MISURA — non riproporlo senza leggere il journal

- **policy `tier`** (l'idea ds4: caldi fissi, freddi streammati): **esiste**, e'
  adottata, e **non paga** sul 35B (it.48). Miss +0,3%, tempo -2%. Il collo e'
  la CAPIENZA: LRU tiene gia' il 66% del parco, il cap del 12,5% tocca l'8%.
- **raggruppare le richieste I/O**: 0,98x fra continuo e raffiche (it.33). Il
  canale da' 460-740 MB/s in ogni configurazione; non e' il collo. **it.51 lo
  conferma dal verso opposto**: il raggruppamento IMPOSTO dal formato slab (una
  richiesta da 1,77 MB invece di tre da 594 KB in `Promise.all`) costa il 4,6%.
- **`idot` nel decode**: vince sul kernel (2,3%) e **perde nel motore**
  (-0,14/-0,66 ms) per il dispatch in piu'. Cablato e spento, con l'aritmetica
  al call-site.

### LE ALTRE DUE CELLE VUOTE

- **prefetch lookahead**: NON implementato, e l'oracolo e' misurato al
  **91,92% di recall @K=8**. Attacca i ~76 ms/token di tassa. E' l'idea piu'
  fuori-schema rimasta: usa una proprieta' del MODELLO (il routing e'
  prevedibile un layer prima) per battere un limite della PIATTAFORMA.
- **spec-dec MTP**: costruito, solo nel ktest, accept 50%, proiezione 1,29x.
  Serve una testa MTP per il 35B, che oggi non esiste.

### DECISIONI: nessuna ti blocca

Tutti gli item del docket sono chiusi o deprioritizzati dalla priorita' del PI
(«il massimo dal 35B; GLM secondario, lo butteremo»). Restano aperti solo due
item storici e ininfluenti (item 2: `blankNonCode` duplicato in 7 test; item 4:
gia' superato dal ruling successivo).

### LO STATO DELLE RIGHE

    riga 2b  I/O            CHIUSA sulla misura: raggruppare non paga
    riga 2c  kfan           atterrata
    riga 2d  rotta split-K  atterrata (40,06)
    riga 2e  router         atterrata
    riga 3   replay         done-when degenere (item 9), e vive nel regime sporco
    riga 4   la barra       serve il checkpoint su 4 famiglie
    riga 6   gate di merge  ktest verde; GLM fuori banda ma deprioritizzato
    riga 7   consuntivo     da scrivere

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
    BASE_URL=http://localhost:5199 node tools/harness/engine-ktest.mjs   # 111 PASS / 0 FAIL
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

---

## Ciclo 2026-08-18 — i due spike, e tre correzioni che li hanno raddrizzati

**Spike (1), `cost(M)`**: il ginocchio è a M=2 e il verdetto «spec-dec più
lento» cade — era corretto per i kernel su cui fu preso. Ma «non ha saturato a
M=16» era un **artefatto della metrica**: il tempo totale è affine in M, quindi
il costo *medio* per riga cala del 26% a ogni raddoppio per sempre, anche a
margine marginale zero. Residuo oltre M=16: **1,6×**, non un altro 30×. E il
banco misura un regime L2-resident (celle a 785 GB/s contro un tetto VRAM di
435) che il decode non vive.

**Spike (2), overlap del router**: correlato **11-12×** l'indipendente, con metà
dell'overlap che è componente topica e metà correlazione locale del modello.
`G(2) = 1,19×`. Banda di genere sotto l'1%, e il profilo per-layer correlato a
**r = 0,911** fra due testi diversi — quella è l'evidenza di struttura, non
l'uguaglianza degli aggregati. Il layer 0 è quasi indipendente, i profondi
correlati 12-15×.

**Le tre correzioni**, tutte dall'agente fable e tutte verificate sui dati prima
di accoglierle:
1. il prior che avevo citato (recall 82,67%) misurava predizione *fra layer
   dentro una posizione*, non overlap *fra token*. Un grafico pubblicato
   poggiava su di esso;
2. **doppio conteggio**: avevo moltiplicato 1,19 × 1,23. `G(M)` contiene già
   l'economia del kernel;
3. `G(M)>1` non dimostra che lo spec-dec paghi — vale se *tutti* gli M sono
   utili, cioè nel prefill. Break-even α ≥ 0,68 contro un'acceptance di 0,50:
   **lo spec-dec perde il 12%**. Il kernel si giustifica col **prefill** (2,11×).

Più: `webgguf` esiste su GitHub (privato); il 35B gira su Firefox a 9,97 tok/s;
Chrome nudo non costa nulla; e il ruling sulla portabilità — si alloca quello che
il device concede, il minimo di spec è il pavimento.
