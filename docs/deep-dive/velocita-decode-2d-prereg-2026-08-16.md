# Pre-registrazione — la rotta split-K vale il decode? (riga 2d)

**Goal**: `engine-velocita-decode`, riga 2d. **Data**: 2026-08-16, iterazione 21.
**Scritta PRIMA di leggere i numeri**: la run che la accompagna è morta al
momento della scrittura dell'artefatto (tag assente da `PROV`), quindi nessuna
cella è stata vista. È la ragione per cui questo documento esiste adesso e non
dopo.

## La domanda, e perché non ha ancora una risposta

La riga 2d ha consumato **cinque iterazioni su 2-3 stimate** e il valore che ha
atterrato è sul *prefill* del 35B, che il goal ha dichiarato fuori scope. Ciò
che resta è **costruire la rotta split-K dentro il decode**: `gemv`
(`q35gpumodel.ts:862`) emette UN dispatch, la forma split-K ne vuole TRE
(quantizza x → GEMM a fette → combine) più due buffer che nel decode non
esistono. Stimato: 2-3 iterazioni.

**Il numero con cui quella spesa è stata finora giustificata è il "3,26× a
M=1", e it.17 ha scoperto che è misurato contro il kernel sbagliato**: il banco
confronta `splitk-idot` con `base-batch-z`, cioè
`gemvQuantWgsl({kind:"q8_0", batch:true})` — la forma del *prefill*, M righe
replicate su `wid.z`. Il decode emette `gemvQuantWgsl({kind:"q8_0", K, N,
hasBias:false})`, **senza `batch`**. È un rapporto ereditato dal banco invece
che misurato sul bersaglio, ed è la quinta inferenza non verificata di questo
goal.

## Cosa misura questa run

Due aggiunte a `src/microbench/ttKQuant.ts`, entrambe minime e dentro l'harness
che ha prodotto il 3,26× (nessun banco nuovo):

1. **Braccio `base-decode`**: `gemvQuantWgsl` / `gemvQ*Wgsl` **senza `batch`**,
   griglia `[gx, gy, 1]`. Emesso **solo a M=1**: a M>1 il decode non esiste, e
   un braccio senza `batch` con M righe misurerebbe una forma che nessuno
   emette.
2. **La shape che mancava**: `q8_0 K=2048, N=8192` — `attn_qkv`, il più grande
   dei quattro tensori di `ssmGemv` e da solo il **66,32%** dei loro byte
   (it.17). Il banco aveva solo `N=4096` (`attn_gate`, 33,16%).

Contesto della posta in gioco: dopo il kfan e il router parallelo, **`ssmGemv` è
il primo termine del decode del 35B** — 6,99 ms/token su 30 dispatch = **233 µs
a layer** — e ce l'hanno anche il 4B e il 9B, quindi una leva lì sarebbe
globale per costruzione.

## Le previsioni, numerate e falsificabili

- **P1** — `base-decode` sta **entro ±15%** di `base-batch-z` a M=1. A una riga
  sola la variante batch fa lo stesso lavoro con un termine di indice in più.
- **P2** — su `K=2048, N=4096`, `splitk-idot` contro **`base-decode`** dà fra
  **2,5× e 3,5×**: il 3,26× sopravvive al cambio di termine di paragone.
- **P3** — su `K=2048, N=8192` il vantaggio è **minore** che a N=4096, fra
  **1,5× e 3,0×**. Ragione: il GEMV semplice lancia ~N workgroup, quindi a
  N=8192 ne ha già 8.192 su 128 SM e **non gli manca parallelismo**; lo split-K
  paga soprattutto dove N è piccolo.

P3 è quella che mi aspetto di sbagliare più facilmente, ed è anche quella che
decide: N=8192 è la shape che porta i byte.

## La regola di decisione, fissata adesso

Sulla shape che pesa (`N=8192`), rapporto `splitk-idot` / **`base-decode`** a
M=1:

| misurato | conseguenza sulla riga 2d |
|---|---|
| **< 1,5×** | la rotta **non** vale 2-3 iterazioni: la riga 2d chiude su ciò che ha prodotto (il predicato su N, il q8_0 cablato) e il primo termine si attacca da un'altra parte |
| **1,5× – 2,0×** | zona grigia: si decide col costo, cioè quanti dei 233 µs a layer sono davvero nel GEMV e quanti nei tre dispatch in più che la rotta aggiunge |
| **≥ 2,0×** | la rotta si costruisce, e la stima di 2-3 iterazioni è confermata |

**Il rapporto contro `base-batch-z` non entra in questa tabella.** Resta
riportato accanto, come reperto di quanto il termine di paragone sbagliato
spostava il giudizio.

## Cosa questa run NON dice

- Non misura i **tre dispatch** che la rotta aggiunge nel decode (quantX,
  combine) né i due buffer da allocare: il banco cronometra il solo GEMM. Se il
  rapporto cade in zona grigia, quello è il conto che manca.
- Non dice **perché** `ssmGemv` costa 233 µs a layer. Dice solo se questa leva
  specifica lo sposta.

---

# ESITO — graduato contro le previsioni di sopra

**Artefatto**: `results/microbench/velocita-decode-2d-4090-linux-2026-08-16T02-56-25-413Z.json`
(host quiescente, p50 su campioni interleavati, warm-up scartato).

    shape            braccio        p50 ms    GB/s    % del picco (576 GB/s)
    K2048 N=4096     base-decode    0,1616     55,1      9,6%
    K2048 N=4096     splitk-idot    0,0489    182,2     31,6%
    K2048 N=8192     base-decode    0,1324    134,7     23,4%
    K2048 N=8192     splitk-idot    0,0340    524,4     91,0%

| previsione | banda dichiarata | misurato | esito |
|---|---|---|---|
| **P1** `base-decode` ≈ `base-batch-z` a M=1 | ±15% | +1,4% (N=4096), +1,1% (N=8192) | **CORRETTA** |
| **P2** split-K / `base-decode` a N=4096 | 2,5–3,5× | **3,30×** | **CORRETTA** |
| **P3** a N=8192 il vantaggio è MINORE, 1,5–3,0× | 1,5–3,0× | **3,89×** | **SBAGLIATA** |

## P1: il termine di paragone sbagliato non aveva spostato niente

`base-decode` e `base-batch-z` coincidono entro l'1,4%: a una riga sola la
variante batch fa lo stesso lavoro con un termine di indice in più, come
previsto. **Il 3,26× di it.13 era numericamente giusto** — contro
`base-decode` diventa 3,30×. it.17 aveva ragione a non ereditarlo, e la verifica
è costata un braccio di banco: il dubbio era corretto, la grandezza no.

## P3: sbagliata, e la ragione corregge il modello

Avevo previsto che a N=8192 lo split-K rendesse *meno*, perché il GEMV semplice
lancia ~N workgroup e a 8.192 su 128 SM «non gli manca parallelismo». Rende
**di più** (3,89×), e i GB/s dicono perché: **non era un problema di
occupazione, era di banda**. Il GEMV del decode sta al **9,6%** del picco a
N=4096 e al 23,4% a N=8192 — cioè migliora col numero di righe ma resta lontano
dal pavimento in entrambi i casi; lo split-K a N=8192 arriva al **91,0%**, che è
sostanzialmente ottimale.

**Il modello corretto**: il GEMV a un workgroup di 64 thread per riga di uscita
legge i pesi in modo che non satura il bus, e aumentare le righe non lo cura —
lo cura spezzare il K, che dà a ogni workgroup un accesso contiguo più lungo.
Il numero di workgroup non era la variabile.

## La regola di decisione, applicata

Sulla shape che pesa (`N=8192`, il 66,32% dei byte di `ssmGemv`): **3,89× ≥
2,0× ⇒ la rotta si costruisce.**

**Il guadagno atteso, dai numeri e non dall'intenzione:**

    per layer (attn_qkv + attn_gate)   0,2940 -> 0,0829 ms    3,55x
    ssmGemv nel motore                  6,99  ->  1,97 ms     risparmio 5,02 ms
    meno i dispatch aggiunti dalla rotta:
      quantX 1 per layer (x e' LO STESSO per i due tensori) + 2 combine
      = 90 dispatch/token x 8,65 us (misura di it.9)         -0,78 ms
    NETTO                                                    ~4,24 ms/token

    token 32,531 -> ~28,3 ms  =  ~35,4 tok/s   (barra 30, nice-to-have 45)

**Riserva dichiarata**: i 4,24 ms sono una proiezione da banco, e questo goal ha
già pagato quattro volte il prezzo di trattare una proiezione come una misura.
Vale come *criterio di spesa* — dice che 2-3 iterazioni sono giustificate — non
come risultato. Il risultato lo dirà l'A/B nel processo, come per il kfan.

**Il conto che ancora manca**: i due buffer che la rotta vuole e che nel decode
non esistono (le parziali N×fette f32 e la x quantizzata). Sono VRAM sottratta
all'arena expert, e su questo modello l'arena è il vincolo — va messo nel piano
della prossima iterazione, non scoperto costruendo.
