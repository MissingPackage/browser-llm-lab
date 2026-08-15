# PHASES — engine-35b-residency (decomposizione it.0, 2026-08-15)

Sequenziale. Le righe 2 e 3 toccano lo STESSO file (`q35gpumodel.ts`, il tail
ottimistico) e la stessa struttura (`residency.ts`): nessun `parallel-group`, la
proprietà esclusiva dei path non c'è. Stessa ragione dei tre goal precedenti.

**Metrica obiettivo del goal**: `decode.tokS` del 35B a caldo, select
`optimistic`, contesto dichiarato. **Barra < 30 tok/s → ≥ 30 tok/s**;
**nice-to-have ≥ 45 tok/s** (il token pulito misurato, 21,1 ms). Oggi: **8,34
tok/s** su un turno di chat NON-riferimento. Ogni riga dichiara quanto muove
QUESTA metrica, o si dichiara gate.

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
| 2 | **La fetch esce dal path seriale.** Meccanismo scelto sui numeri della riga 1, non qui: candidati misurati e non assunti — (a) sorgente non-HTTP (OPFS `FileSystemSyncAccessHandle` nel worker, che salta fetch+arrayBuffer+copia), (b) secondo livello di residency in RAM host sotto la VRAM (11,17 GB VRAM + ~7 GB host ≥ 17,67 GB totali: coprirebbe TUTTI gli expert su due livelli — ma `free -g` dà 21 GB disponibili su 31, quindi il budget host va CALCOLATO come quello d'arena, non costante), (c) una `readRange` che prende gate+up+down in una richiesta sola invece di tre. | tempo per miss **< 1,5 ms** (oggi 5,98) misurato dal contatore della riga 1 sullo stesso artefatto di riferimento, **oppure** `fetchRepairMs / tokenMs < 0,05` se il meccanismo la toglie dal path invece di accelerarla; nessuna regressione dei gate di correttezza; il budget host, se introdotto, è calcolato e dichiarato nell'artefatto come `expertBudgetBytes` lo è oggi. | none | `src/engine/q35expertstore.ts`, `src/engine/residency.ts`, `src/engine/chat/chat.worker.ts`, `src/engine/q35conf/**` | todo | 2-3 it |
| 3 | **L'unità di riparazione smette di essere il prefisso del pass.** Il router di un layer è noto prima che i suoi expert servano: il candidato è sovrapporre l'`ensure` del layer L+1 al calcolo del layer L, così il miss costa la fetch (già scesa in riga 2) e **non** 34,8 layer rigiocati. Il path `sync` esistente è il termine di paragone, non il bersaglio: a caldo faceva 132,8 ms/token contro 43,6 dell'ottimistico (`q35gpumodel.ts:2121-2133`) — ma quel confronto è PRIMA della riga 2, e la riga 2 ne cambia il segno. **Rimisurare i due bracci prima di scegliere.** | `replayLayers / (tokens × nLayer)` **≤ 0,20** (oggi 0,87) sull'artefatto di riferimento, **oppure** il replay a prefisso è sostituito e i suoi contatori dichiarati obsoleti con la ragione scritta; `dirtyTokens` resta un contatore valido e il suo valore è dichiarato; bit-identità del prefill MoE **rimisurata** (gate secco: qualunque cosa tocchi l'ordine delle somme la rompe). | none | `src/engine/q35gpumodel.ts` (tail ottimistico), `src/engine/moe.ts` | todo | 3-4 it |
| 4 | **La barra**: misura di chiusura del decode 35B. | artefatto `kind: "q35-residency-checkpoint"` con `decode.tokS` **≥ 30** a caldo, `decodeContext` dichiarato, host quiescente, warm-up scartato, ≥ 3 repliche, e il valore dichiarato anche contro il nice-to-have 45; la ripartizione del token con TUTTI i termini nominati (≥ 95%); `dirtyTokens`, `replays`, `replayLayers`, miss/token e byte/token accanto. | none | `scripts/`, `results/engine/` | todo | 1 it |
| 5 | **Il 35B riceve il suo default di ragionamento.** Gated sulla riga 4: si esegue solo a barra passata, per la ragione scritta nel contratto (col thinking acceso il 35B genera molti più token). | il prompt reso in-page rispetta la polarità del template per famiglia (Qwen3.5 default OFF, Qwen3.6 default ON — verificato sui `chatTemplateRaw` dei tre JSON del 2026-08-15); la scelta è **dichiarata** nel campo `params.chatTemplate` del JSON invece di essere implicita; un test SENZA GPU che pinna le due polarità e fallisce se un modello nuovo entra senza dichiarare la sua; un turno di chat 35B in artefatto che mostra il blocco `<think>` **non vuoto** e i tok/s con thinking acceso. | none | `src/engine/chat/**`, `tests/**`, `results/chat/` | todo | 1 it |
| 6 | **GATE DI MERGE** (riga di sola verifica, dichiarata tale). | `node .harness/tools/engine-ktest.mjs` tutti PASS; top-1 vs oracolo llama.cpp ≥ 1012/1024 su entrambi i bracci; sequenze generate identiche 8/8; **decode 4B ≥ 45,5 tok/s a ctx 6333**; **`prefill.ms + decode.firstMs` < 22.500 ms**; `gemm:deltanet-out` ≤ 2.000 ms e `gemm:ffn-down` ≤ 2.000 ms; GLM b12 optimistic entro ±5% di 13.172 / 31,26 / 14,74; bit-identità prefill MoE 35B; `npx vitest run` verde; `npx tsc --noEmit` pulito. Tutto su host dichiarato. | none | — (gate) | todo | 1 it |
| 7 | **Consuntivo e consegna.** | `docs/engine/35b-residency-consuntivo-<data>.md` clausola per clausola con l'artefatto accanto, la nuova ripartizione del token, e **il termine che diventa primo dopo questa leva nominato con la sua misura fresca**; `HANDOFF.md` §1 aggiornato; `GLOSSARY.md` coi termini coniati. | none | `docs/engine/`, `HANDOFF.md`, `GLOSSARY.md` | todo | 1 it |

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
