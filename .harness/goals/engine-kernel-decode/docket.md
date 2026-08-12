# DOCKET — engine-kernel-decode

Il docket è il VERBALE, non il mezzo: le decisioni si chiedono in chat, qui si
registrano. Un item che potrei decidere da solo va deciso e registrato, non
escalato.

## item 1 — plan-check (PI)

`PHASES.md` scritto a iterazione 0. **Il goal non parte finché il PI non lo
approva**: è il gate "mostrami il file prima di eseguire". Le celle di misura
sono state test-fittate a tempo di scrittura (ruling C7) — v. item 2, che è
uscito proprio da lì.

STATO: in attesa.

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
