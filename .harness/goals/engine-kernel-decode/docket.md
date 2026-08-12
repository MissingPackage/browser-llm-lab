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
