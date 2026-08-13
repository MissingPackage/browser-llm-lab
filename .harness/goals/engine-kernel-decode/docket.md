# DOCKET — engine-kernel-decode

Il docket è il VERBALE, non il mezzo: le decisioni si chiedono in chat, qui si
registrano. Un item che potrei decidere da solo va deciso e registrato, non
escalato.

## item 1 — plan-check (PI)

`PHASES.md` scritto a iterazione 0. **Il goal non parte finché il PI non lo
approva**: è il gate "mostrami il file prima di eseguire". Le celle di misura
sono state test-fittate a tempo di scrittura (ruling C7) — v. item 2, che è
uscito proprio da lì.

STATO: **APPROVATO dal PI 2026-08-13** ("vai. Fai tutto con gli strumenti che
hai scelto. Usa loop, research-campaign, sdd-conductor, second-opinion, ecc.").
Autorizzati esplicitamente: loop autonomo, research-campaign, sdd-conductor,
second-opinion. Il PI è ASSENTE (dorme): tutto ciò che non è nel grant di
autorità del contratto si registra qui e si aspetta, non si decide.

## item 2 — il motore ha un tetto di contesto che i suoi limiti non dichiarano (io, fase 1)

TROVATO scrivendo PHASES, non eseguendo: `gpulimits.ts` calcola
`maxComputeWorkgroupStorageSize` per il path Qwen (`mlaAttention: false`) come
la costante `QWEN_WORKGROUP_STORAGE_BYTES = 30_848`, e il commento dice
«indipendente dal contesto». Ma `attnDecodeWgsl` — che è il kernel di
attenzione del path Qwen — dichiara `scores: array<f32, ctxMax>` + `red[64]`,
cioè **4·ctxMax + 256 B**.

I due combaciano per fortuna: a ctxMax 6400 servono 25.856 B < 30.848. Il
pareggio esatto è a **ctxMax 7648**; sopra, il kernel chiede più di quanto il
device abbia garantito e la creazione della pipeline fallisce.

Non è una domanda: è lavoro della fase 1, che quel `scores` lo toglie del tutto
(softmax in streaming). Registrato perché il difetto è nel MODULO DEI LIMITI —
il posto che esiste apposta per non avere sorprese — e perché il commento che
dichiara l'indipendenza dal contesto va corretto anche se il kernel cambia.


### it.1 — CORREZIONE all'item 2: è una TRAPPOLA, non un bug vivo

Verificato prima di scrivere il fix, e la mia formulazione era più grave del
vero. Il path q35 di produzione (`chat.worker.ts`, `q35conf.worker.ts`) NON
passa `mlaAttention: false`, quindi cade nel ramo che chiede
`max(30_848, 4·ctxMax+256)`: il limite giusto lo otteneva già — **attraverso un
campo che porta il nome dell'attenzione di GLM**, mentre il proprio consumatore
restava non dichiarato.

Il difetto vero è l'INVITO: il commento diceva che passare `false` "evita di
chiedere un limite per un consumatore che quel modello non ha", e non era vero.
Chi avesse seguito quel consiglio su un modello q35 avrebbe sotto-dichiarato, e
il motore sarebbe morto alla creazione della pipeline sopra ctxMax 7648.

CHIUSO in it.1: `attnDecodeWorkgroupStorageBytes(ctxMax)` esportata dal file del
KERNEL (una formula sola, dove sta il consumatore), contata SEMPRE dal modulo
dei limiti, commento corretto, e un test che rende la trappola non richiudibile:
spegnere l'MLA non può più far sparire il fabbisogno di Qwen.

## item 3 — il confronto `split` vs `split-gqa` è confuso (io, fase 1)

TROVATO misurando (fase 0). Le due varianti non differiscono solo per la fusione
GQA: differiscono anche per il numero di workgroup (256 contro 64), perché con
un workgroup per gruppo GQA le teste sono 4 invece di 16. `split` vince
(0,296 contro 0,325 ms) **leggendo quattro volte più byte**: il kernel è limitato
dal parallelismo, non dalla banda.

Non è una domanda: è lavoro della fase 1, che deve rifare il confronto **a parità
di occupancy** (`split-gqa` con 64 chunk invece di 16) prima di scegliere. Se a
parità di workgroup la fusione GQA vince, la fase 1 la adotta; se non vince, la
ridondanza GQA si lascia stare e si dichiara coi numeri.

Registrato perché la conclusione "la dedup GQA non serve" sarebbe **non
supportata** dai dati di fase 0 presi da soli.

## item 4 — dopo le fasi 1 e 2 la pendenza diventa la CODA (io, fuori scope)

Proiezione di fase 0 sulla scomposizione it.59: con le due leve applicate, i 7,6
ms di coda (lm_head + argmax + readback) passano dal 7,6% del token al 33-46%.

Fuori dallo scope di questo goal (che ha per bersaglio i due kernel caldi), ma
va davanti al PI **prima** della fase 3, non dopo: se la fase 3 arriva a 40-60
tok/s la soglia è presa e la coda non serve, se arriva a 28 la coda è l'unica
cosa rimasta e nessuno l'ha guardata.

## item 3 — il conductor tronca le patch a 16.000 caratteri e il sintomo sembra un conflitto di owns (io, harness)

FASE 1, primo tentativo: tutti e cinque i task BLOCKED. T1 per "conflitto
patch-apply", T2-T5 per dipendenza a monte.

**Non era un conflitto.** Lo script del workflow passa la patch all'integratore
con `r.patch.slice(0, 16000)`; la patch di T1 e' **40.209 caratteri**. Arrivava
tagliata a meta' di un file nuovo — un hunk che dichiara 88 righe e ne porta 77
— e `git apply` diceva `corrupt patch at line 369`.

L'integratore ha fatto la cosa giusta due volte: NON ha usato `--3way`, NON ha
risolto a mano (design §4: un conflitto e' un bug del piano `owns`, non roba da
aggiustare in integrazione), e ha isolato la causa vera facendo un `--check` sui
soli hunk arrivati interi — che combaciavano tutti. Senza quella diagnosi avrei
speso un'iterazione a ri-pianificare gli `owns` di un piano che non aveva
niente che non andasse.

**Perche' e' un difetto e non una svista**: il troncamento fa sembrare un
problema del CANALE un problema del PIANO. E' la stessa classe delle sentinelle
che questo progetto si e' dato altrove — uno strumento che tace, o peggio che
mente sulla causa, e' peggio che non averlo.

FIX (it.3): tolto lo `slice` nello script del run, e commento sul perche'. La
correzione vive nella COPIA di questo run
(`.claude/.../workflows/scripts/sdd-conductor-wf_f89a2754-af3.js`), non nel
workflow installato in `~/.claude/workflows/sdd-conductor.workflow.js`:
**quello e' ancora rotto e va corretto alla fonte**. Fuori dal grant di autorita'
di questo goal (non e' codice di progetto): registrato qui, da portare al PI.

## item 4 — il done-when (e) chiede piu' di quanto la riga 1 possa dare (io → PI, fase 1)

SCOPERTO ESEGUENDO (ruling C7: e' il caso che la verifica a tempo di scrittura
non ha intercettato). Il contratto chiede: «`maxComputeWorkgroupStorageSize`
richiesto dal motore < 16384 B a QUALUNQUE ctxMax».

Dopo T1 l'attenzione chiede **1.536 B costanti** (era 4·ctxMax+256): il tetto di
contesto e' sparito, che era l'INTENZIONE della clausola. Ma il totale richiesto
resta **30.848 B**, perche' un altro consumatore ci sta sopra da solo:
`rmsPairGemmSiluChunkFast`, il kernel FUSO DEL PREFILL (4·K·mMax + 256·mMax +
16·mMax, K=896, mMax=8), che non dipende dal contesto e appartiene al prefill —
cioe' al goal TTFT, non a questo.

Quindi la clausola, alla lettera, non e' soddisfacibile dalla riga 1; nella
sostanza lo e'. **Non la riscrivo io**: cambiare un done-when e' esplicitamente
fuori dal grant di autorita' ("must docket: cambiare i gate"). Il test
`tests/gpulimits.test.ts` intanto asserisce cio' che e' VERO e nomina cio' che
resta: il fabbisogno dell'attenzione e' costante e sotto la garanzia, il totale
non cresce piu' col contesto, e l'unico consumatore sopra i 16 KB e' quello
fuso, per nome.

**RULING RICHIESTO**, con la mia raccomandazione: spezzare (e) in due —
(e1) «il fabbisogno dell'ATTENZIONE e' costante in ctxMax e sotto i 16 KB» →
gia' soddisfatta e sotto test; (e2) «il TOTALE del motore sta sotto i 16 KB» →
spostata al goal TTFT insieme al kernel fuso del prefill. Se non arriva
risposta, procedo con (e1) come soddisfatta e (e2) registrata come debito
dichiarato, senza toccare il testo del contratto.

## item 5 — `glm-bench-run.mjs`: un `--out` ASSOLUTO diventa relativo alla root, e la run si perde (io, ergonomia dei runner)

Costato una run GPU da ~20 minuti a mezzanotte passata. `--out /tmp/x.json`
finisce in `<root>/tmp/x.json`: il runner fa `join(ROOT, out)` senza distinguere
i path assoluti. Il bench GIRA per intero, poi muore in scrittura con ENOENT — e
il lavoro e' perso, non solo il file.

Non e' una domanda: sono due righe (`isAbsolute(out) ? out : join(ROOT, out)`).
Registrato e non fatto perche' e' fuori dal brief di questo goal e il runner e'
di GLM; da fare quando si tocca quel path per altro. Nel frattempo: passare
sempre `--out results/engine/...` RELATIVO.

Vale anche per gli altri runner: `q35-bench-run.mjs` ha lo stesso schema
(`arg("out", join(ROOT, ...))`) e stanotte gli ho passato path /tmp — hanno
funzionato perche' li' l'`out` di default e' gia' assoluto e il mio veniva usato
tale e quale. Due runner della stessa famiglia che trattano lo stesso flag in
modo diverso: e' quello il difetto, piu' del path.

## item 6 — due call-site GLM nel ktest non sono coperti dal freeze (censimento it.8)

Il censimento di copertura (142/142, 100%) ha segnalato che
`ktest.worker.ts:1514` e `:1516` sono siti GLM protetti SOLO dalla convenzione:
il freeze sha256 del testo generato copre il down `scaledAccum` di riga 1519, non
loro. Oggi nessuno passa `vec4Rows2` li', quindi sono conformi — ma la garanzia
e' "nessuno lo fa", non "il test lo impedisce".

Caveat aggiuntivo del censimento, da tenere se un domani qualcuno volesse
adottarli: i loro binding sono SOTTO-RANGE di uno slab (offset+size), quindi la
size andrebbe verificata multipla di 16 B prima di poterli bindare come
`array<vec4<u32>>`.

Lavoro mio, non un ruling: estendere il freeze a quei due siti. Non fatto ora
perche' tocca il ktest di GLM a goal quasi chiuso e il rischio non lo giustifica.

## item 7 — `hostState.declared` e' una PROMESSA dell'operatore, e stanotte l'ho tradita (io, metodo)

La non-reg GLM di it.8 e' uscita rossa: decode 9,30 contro il gate 13,43,
prefill 10,98 contro 31,26, TTFT 41,98 s contro 14,74. Numeri da regressione
grave.

**Non e' una regressione: e' una misura invalida, e la colpa e' del mio
orchestratore.** L'ho lanciata mentre girava il censimento di copertura con 27
agenti. Il decode di GLM e' per il **61,7% fuori dalla GPU** (attribuzione del
runner stesso: wall 110,3 = gpuBusy 42,3 + stallo 30,5 + sync/CPU 37,6), cioe'
e' CPU-bound: 27 processi che leggono e grepano il repo lo affamano. Il piano di
dispatch e' rimasto identico (1449 dispatch/token, uguale al riferimento), e sul
path GLM il WGSL generato e' congelato da un test sha256 — il codice non poteva
spiegare un -65% sul prefill.

**Il difetto strutturale**: `--host-state quiescent` e' una stringa che il
runner scrive nel JSON senza poterla verificare. Campiona nvidia-smi (che vede
la GPU, non il carico CPU) e si fida dell'operatore. Un artefatto puo' quindi
dichiararsi "quiescent" essendo stato misurato sotto contesa, e nessuno se ne
accorge rileggendolo.

Proposta (lavoro mio, non un ruling): il runner campiona anche il carico CPU
(loadavg / conteggio processi) prima e dopo, e se la dichiarazione e'
"quiescent" ma il carico dice altro, il report esce con la stessa quarantena
`.INVALID` gia' usata per gli errori GPU. Registrato qui; da fare quando si
tocca quel runner.

Nel frattempo, regola operativa per me: **nessun bench mentre gira un
workflow**. E' la seconda volta in una notte che la mia orchestrazione invalida
una misura (la prima: due runner playwright sullo stesso profilo).
