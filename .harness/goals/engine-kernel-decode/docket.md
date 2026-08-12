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
