# La quantizzazione dei pesi, spiegata per intero

**A chi serve.** A chi deve *decidere* qualcosa che dipende dalla quantizzazione
— quale file scaricare, quale kernel scrivere, se un modello ci sta in memoria,
se un quant più aggressivo costa troppa intelligenza — e non ha già in testa la
differenza fra `Q4_K_M`, `IQ3_XXS` e `UD-Q2_K_XL`. Non presuppone niente oltre
l'idea che un modello sia un mucchio di matrici di numeri.

**Cosa non è.** Non è un tutorial su come si quantizza un modello (non lo
facciamo: consumiamo file già quantizzati) e non è una tabella di raccomandazioni
«usa questo». È la spiegazione dei *meccanismi*, con accanto i numeri veri di
questo progetto e l'indicazione di dove sono stati misurati.

**Provenienza dei numeri.** Ogni byte per blocco qui dentro viene da una di due
fonti, e sono dichiarate riga per riga: (a) `src/engine/quant.ts`, che implementa
i dequantizzatori e li verifica contro `ggml-quants.c`; (b) il controllo di
`scripts/q35-header-dump.mjs`, che somma i byte di tutti i tensori di un file e
li confronta con la taglia vera — se una taglia di blocco fosse sbagliata, quello
scarto non sarebbe dello 0,08%.

---

## Indice

1. [Perché si quantizza](#1-perché-si-quantizza)
2. [L'idea di base: scala, blocco, errore](#2-lidea-di-base-scala-blocco-errore)
3. [Famiglia 1 — i quant *legacy*](#3-famiglia-1--i-quant-legacy)
4. [Famiglia 2 — i **K-quant**](#4-famiglia-2--i-k-quant)
5. [Famiglia 3 — gli **i-quant**](#5-famiglia-3--gli-i-quant)
6. [Formati contro ricette: `_S`, `_M`, `_L`, `_XL`, `UD`](#6-formati-contro-ricette-_s-_m-_l-_xl-ud)
7. [La imatrix](#7-la-imatrix)
8. [Bit per peso: la tabella completa](#8-bit-per-peso-la-tabella-completa)
9. [Come si legge davvero un GGUF](#9-come-si-legge-davvero-un-gguf)
10. [Cosa cambia per un motore che gira nel browser](#10-cosa-cambia-per-un-motore-che-gira-nel-browser)
11. [Come si misura la qualità senza raccontarsela](#11-come-si-misura-la-qualità-senza-raccontarsela)
12. [Glossario](#12-glossario)
13. [Cosa in questo documento NON è verificato in casa](#13-cosa-in-questo-documento-non-è-verificato-in-casa)

---

## 1. Perché si quantizza

Un modello addestrato ha pesi in virgola mobile a 16 o 32 bit. Un modello da 35
miliardi di parametri a 16 bit occupa **70 GB**. Nessuna GPU consumer lo tiene, e
il problema non è solo *dove metterlo*: è **quanto ci vuole a leggerlo**.

Questa è la parte controintuitiva. Generare un token con un modello di questa
famiglia non è un problema di calcolo, è un problema di **banda di memoria**: per
produrre *un* token bisogna far passare davanti all'unità di calcolo tutti i pesi
che quel token attraversa. Se i pesi attivi sono 1,66 GB e la scheda legge 576
GB/s, il pavimento fisico è ~2,9 ms per token — e ogni bit tolto ai pesi abbassa
quel pavimento in proporzione diretta.

Quindi la quantizzazione compra **due cose insieme**:

| | |
|---|---|
| **capienza** | il modello ci sta, o non ci sta, nella memoria che abbiamo |
| **velocità** | meno byte da leggere per token = meno tempo per token |

E ne costa **una**: la **fedeltà**. Ogni peso viene sostituito da
un'approssimazione, e la domanda che conta non è «quanto è grande l'errore sui
pesi» ma «quanto cambia ciò che il modello dice». Le due cose non sono
proporzionali, ed è la ragione per cui la §11 esiste.

> **Il caso concreto di questo progetto.** Il nostro Qwen3.6-35B-A3B in
> `UD-Q4_K_S` occupa 19,46 GiB, di cui **17,07 GiB sono il parco degli expert**.
> La GPU ne può dedicare 11,17 all'arena: ci sta il **65%**, e il resto viene
> letto da disco durante la generazione. Su questa macchina il modello è
> *residency-bound sempre*, e quel 65% è la ragione per cui una chat vera fa
> 11,5 token/s invece dei 40 misurati a memoria piena.

---

## 2. L'idea di base: scala, blocco, errore

### 2.1 Quantizzare un numero

Prendiamo un gruppo di pesi in virgola mobile. L'idea più semplice: trovare il
massimo del gruppo, dividere tutti per lui, e memorizzare il risultato come
intero a pochi bit. Con 4 bit ci sono 16 livelli.

```
w  ≈  d · q          d = scala (un float), q = intero a n bit
```

Per ricostruire il peso servono `q` (pochi bit, per ogni peso) e `d` (un float,
condiviso). Alcuni formati aggiungono un offset:

```
w  ≈  d · q  +  m    m = minimo del gruppo (formati "affini")
```

La differenza fra i due conta quando i pesi di un gruppo non sono centrati sullo
zero: senza `m` metà dei 16 livelli viene sprecata su valori che non esistono.

### 2.2 Perché il *blocco*

Se la scala fosse una sola per tutta la matrice, un singolo peso anomalo —
e le reti ne sono piene — allargherebbe la scala e schiaccerebbe **tutti** gli
altri sui livelli bassi. Quindi la scala si tiene **locale**: si divide il
tensore in **blocchi** di 32 o 256 pesi e ognuno ha la sua.

Da qui nasce il primo compromesso di tutta la faccenda:

- blocchi **piccoli** → scale più aderenti, meno danno da outlier, ma **più
  scale da memorizzare** (una scala `f16` ogni 32 pesi costa 0,5 bit per peso);
- blocchi **grandi** → poche scale, ma un outlier fa danno su più pesi.

Le tre famiglie che seguono sono tre risposte diverse a questo compromesso.

### 2.3 Come si conta il costo: **bit per peso**

L'unità con cui si confrontano i formati è **bpw** (*bits per weight*): byte del
blocco × 8 ÷ pesi del blocco. Include tutto — valori, scale, maschere. È il
numero che dice quanto occuperà il modello, e non va confuso col nome del
formato: `Q4_K` non costa 4 bit per peso, ne costa **4,5**.

---

## 3. Famiglia 1 — i quant *legacy*

I primi formati di `llama.cpp`. Struttura elementare: **blocco da 32 pesi**, una
scala `f16`, i valori impacchettati.

| formato | byte/blocco | struttura | bpw |
|---|---|---|---|
| `Q4_0` | **18** | `d f16` + 32 nibble (4 bit) | 4,5 |
| `Q4_1` | **20** | `d f16` + `m f16` + 32 nibble | 5,0 |
| `Q8_0` | **34** | `d f16` + 32 interi a 8 bit | 8,5 |

*(byte da `src/engine/quant.ts`, che li implementa; esistono anche `Q5_0` e
`Q5_1`, che questo motore non incontra.)*

Il dequant è aritmetica pura: leggi il nibble, sottrai 8 (`Q4_0` è centrato),
moltiplica per la scala. È il motivo per cui i legacy restano utili: **il kernel
più semplice che esista**. `Q8_0` in particolare è quasi trasparente per la
qualità e viene usato ancora oggi, dentro file molto più aggressivi, per i
tensori che non sopportano approssimazione (v. §6).

**Il loro limite** è quello della §2.2: 32 pesi che condividono *una* scala, e
nessun modo di raffinarla localmente. A 4 bit si vede.

---

## 4. Famiglia 2 — i **K-quant**

L'innovazione dei K-quant è **la scala a due livelli**, ed è più furba di quanto
sembri.

Il blocco diventa un **super-blocco da 256 pesi**, diviso in **sotto-blocchi**
(di 16 o di 32, dipende dal formato). Poi:

- il super-blocco ha una scala `d` in `f16` (e, nei formati affini, una `dmin`);
- **ogni sotto-blocco ha la sua scala, ma memorizzata a 6 bit**, come multiplo
  della `d` del super-blocco.

Il guadagno è che una scala locale costa 6 bit invece di 16, quindi ci si può
permettere di averne **molte**: l'outlier resta confinato in 16 pesi invece di
32, e le scale non divorano il budget. A parità di bpw la fedeltà sale
sensibilmente.

### 4.1 I layout esatti

Tutti verificati in `src/engine/quant.ts`, che li dequantizza e li confronta con
`ggml-quants.c`. Un dettaglio che conviene vedere: **i layout non sono uniformi**
— cambia il numero di sotto-blocchi, e in `Q6_K` la scala del super-blocco sta
perfino *in coda*.

| formato | byte | sotto-blocchi | layout |
|---|---|---|---|
| `Q2_K` | **84** | 16 × 16 pesi | `scales 16 B` · `qs 64 B` · `d f16` · `dmin f16` |
| `Q3_K` | **110** | 16 × 16 | `hmask 32 B` · `qs 64 B` · `scales 12 B` · `d f16` |
| `Q4_K` | **144** | 8 × 32 | `d f16` · `dmin f16` · `scales 12 B` · `qs 128 B` |
| `Q5_K` | **176** | 8 × 32 | come `Q4_K` + `qh 32 B` (il 5º bit) |
| `Q6_K` | **210** | 16 × 16 | `ql 128 B` · `qh 64 B` · `scales 16 B int8` · `d f16` **in coda** |

Due cose da notare, perché sono esattamente ciò che rende noioso scrivere un
kernel:

1. **i bit di un peso non sono contigui.** In `Q3_K` ogni peso ha 2 bit bassi in
   `qs` e 1 bit alto in una maschera separata (`hmask`); in `Q6_K`, 4 bit in `ql`
   e 2 in `qh`. Ricomporre un valore richiede due letture e uno shift.
2. **le scale a 6 bit sono impacchettate a cavallo dei byte.** Nel nostro codice
   la ricomposizione per `Q3_K` è una funzione a sé (`q3kScale6`): 4 bit bassi
   nei primi 8 byte, 2 bit alti spalmati negli ultimi 4.

### 4.2 Il centraggio

`Q3_K` e `Q6_K` sono **simmetrici** (il valore va da −32 a +31: si sottrae un
offset fisso), `Q2_K`, `Q4_K` e `Q5_K` sono **affini** (hanno `dmin`, cioè un
minimo per sotto-blocco). Non è un dettaglio estetico: un errore su questo
produce **pesi plausibili e sbagliati**, che è la classe di difetto peggiore —
niente crash, solo un modello che risponde peggio senza che nessuno sappia
perché.

---

## 5. Famiglia 3 — gli **i-quant**

Sotto i 3 bit per peso, la strada dei K-quant si esaurisce: con 2 bit ci sono 4
livelli e nessuna scala per quanto furba li rende sufficienti. Gli i-quant
cambiano proprio l'oggetto memorizzato.

### 5.1 L'idea: non il valore, ma l'indice

Invece di quantizzare ogni peso per conto suo, si prendono **gruppi di 8 pesi** e
si cerca, in un **codebook fisso**, il vettore a 8 dimensioni che più gli
assomiglia. Nel file finisce **l'indice** di quel vettore (più bit di segno e le
scale).

Perché funziona meglio: quantizzare 8 pesi separatamente a 2 bit dà 4⁸ = 65.536
combinazioni tutte «a griglia», e la maggior parte di quelle combinazioni non
capita mai nei pesi veri. Un codebook sceglie qualche centinaio di punti
*distribuiti dove i vettori di pesi stanno davvero*. A parità di bit si ottiene
molta più risoluzione effettiva — è quantizzazione **vettoriale** invece che
scalare.

Il prezzo lo paga chi decodifica: il dequant smette di essere aritmetica e
diventa una **lettura da tabella**. Nel kernel il codebook deve stare da qualche
parte di veloce, e il ciclo interno guadagna un accesso indicizzato.

### 5.2 L'eccezione che conviene sapere: `IQ4_NL` e `IQ4_XS`

Portano la «I» nel nome ma **non sono a reticolo**. Usano una tabella di **16
valori non uniformi** — spaziati come i quantili di una distribuzione a campana
invece che equispaziati — perché i pesi di una rete non sono distribuiti
uniformemente. È una sostituzione di livelli, non un codebook vettoriale: come
costo di kernel stanno vicinissimi a un K-quant.

### 5.3 Il quadro pratico

| sotto-famiglia | meccanismo | costo per un kernel nuovo |
|---|---|---|
| `IQ1_*`, `IQ2_*`, `IQ3_*` | codebook a reticolo su gruppi di 8 | **alto**: tabella nel kernel + gather indicizzato |
| `IQ4_NL`, `IQ4_XS` | 16 livelli non uniformi | **basso**: simile a un K-quant |

---

## 6. Formati contro ricette: `_S`, `_M`, `_L`, `_XL`, `UD`

Qui nasce quasi tutta la confusione, perché **la stessa sigla significa due cose
diverse** a seconda di dove sta.

### 6.1 Dentro il nome di una famiglia I: è un formato

`IQ2_XXS` → `IQ2_XS` → `IQ2_S` → `IQ2_M` sono **formati di blocco diversi**:
byte diversi, bpw diversi, codebook diversi. Qui la sigla è sostanza.

### 6.2 In coda a un K-quant: è una **ricetta**

`Q3_K_S`, `Q3_K_M`, `Q3_K_L` **non** sono formati di blocco. Sono *ricette di
mescolanza*: quale tensore del modello prende quale formato. Un modello non è
omogeneo — ci sono tensori che sopportano 2 bit e tensori che se li togli
rovinano tutto — e la ricetta decide chi paga.

Non è teoria: si legge nei file. Dal nostro dump del 35B (§9):

```
bartowski Q3_K_S    120 tensori expert Q3_K                  + 3 Q8_0
bartowski Q3_K_M    100 tensori expert Q3_K  + 20 Q4_K       + 3 Q8_0
```

Stesso formato di blocco, ricetta diversa: il `_M` tiene venti tensori a
precisione più alta e pesa 0,7 GiB in più.

### 6.3 `_L`, `_XL`, `UD`: ricette di chi pubblica

`llama.cpp` definisce `_S/_M/_L`; le altre sigle sono di chi confeziona i file.

- **bartowski** aggiunge `_L`/`_XL`, che alzano la precisione su embedding e
  testa di output. E fa una cosa che vale la pena guardare: nel suo `Q2_K` il
  **router del MoE (`ffn_gate_inp`) resta in BF16**. Cioè: 2,6 bit per gli
  expert, 16 per la matrice che decide *quale* expert usare. Sbagliare la scelta
  costa più che sbagliare un peso.
- **Unsloth** marca `UD` («Dynamic»): sceglie il formato tensore per tensore
  guidandosi con la imatrix. Sul nostro 35B questo significa **i-quant sugli
  expert** tenendo l'attenzione a `Q5_K`/`Q6_K`.

> **La trappola che ne segue, ed è quella che ci ha spostato il candidato.**
> `Qwen3.6-35B-A3B-UD-Q2_K_XL` **non contiene un solo tensore `Q2_K` fra gli
> expert**: sono `IQ2_XS`, `IQ3_XXS` e `IQ4_XS`. Il nome dice la *taglia*, non i
> formati. Prima di stimare il lavoro di un kernel dal nome di un file, si apre
> l'header.

---

## 7. La imatrix

La **importance matrix** è ortogonale alle famiglie: si può usare (quasi) con
tutte.

L'idea: non tutti i pesi contano uguale. Si fa passare del testo di calibrazione
attraverso il modello **non quantizzato**, si registra quanto ogni peso viene
sollecitato dalle attivazioni, e si usa quel profilo come **peso dell'errore**
durante la quantizzazione. Il quantizzatore accetta più errore dove non si sente
e meno dove si sente.

Tre conseguenze pratiche:

1. gli i-quant a reticolo **praticamente la esigono**: sotto i 3 bit senza
   imatrix il danno è grosso;
2. **due file con lo stesso bpw non sono confrontabili** se uno ha la imatrix e
   l'altro no. È una variabile in più nel confronto, e va dichiarata;
3. la calibrazione introduce una dipendenza dal testo usato — un file
   calibrato su solo inglese può essere più fragile altrove. *Affermazione di
   meccanismo, non misurata qui: è una delle ragioni per cui il nostro corpus di
   valutazione ha una sezione in italiano (§11).*

---

## 8. Bit per peso: la tabella completa

Legacy e K dai byte in `src/engine/quant.ts`; gli i-quant dalla tabella di
ripiego di `scripts/q35-header-dump.mjs`, **validata dal controllo somma-tensori
vs taglia-file su tre modelli reali** (scarto 0,06–0,09%, che è header e
allineamenti).

| formato | byte/blocco | pesi/blocco | **bpw** | famiglia |
|---|---|---|---|---|
| `IQ1_S` | 50 | 256 | 1,56 | I (reticolo) |
| `IQ1_M` | 56 | 256 | 1,75 | I (reticolo) |
| `IQ2_XXS` | 66 | 256 | 2,06 | I (reticolo) |
| `IQ2_XS` | 74 | 256 | 2,31 | I (reticolo) |
| `IQ2_S` | 82 | 256 | 2,56 | I (reticolo) |
| **`Q2_K`** | **84** | 256 | **2,63** | K |
| `IQ3_XXS` | 98 | 256 | 3,06 | I (reticolo) |
| **`Q3_K`** | **110** | 256 | **3,44** | K |
| `IQ3_S` | 110 | 256 | 3,44 | I (reticolo) |
| `IQ4_XS` | 136 | 256 | 4,25 | I (tabella 16) |
| **`Q4_K`** | **144** | 256 | **4,5** | K |
| `Q4_0` | 18 | 32 | 4,5 | legacy |
| `IQ4_NL` | 18 | 32 | 4,5 | I (tabella 16) |
| `Q4_1` | 20 | 32 | 5,0 | legacy |
| **`Q5_K`** | **176** | 256 | **5,5** | K |
| **`Q6_K`** | **210** | 256 | **6,56** | K |
| `Q8_0` | 34 | 32 | 8,5 | legacy |

**La riga che spiega perché gli i-quant esistono**: `IQ3_S` e `Q3_K` costano
*esattamente* gli stessi bit, e l'i-quant è migliore. Il codebook non fa
risparmiare spazio, fa rendere meglio lo spazio.

---

## 9. Come si legge davvero un GGUF

La taglia del file **non basta** per decidere niente, perché mescola parti che
hanno vincoli diversi. In un MoE, in particolare, conta la separazione fra:

- il **parco expert** — enorme, e l'unica parte che si contende gli slot
  dell'arena in VRAM;
- **tutto il resto** (embedding, attenzione, shared expert, router) — piccolo, e
  sempre residente.

`scripts/q35-header-dump.mjs` dà l'istogramma per categoria e per formato. Legge
solo i primi 64 MB, e **accetta un URL**: valutare un quant candidato costa 64 MB
invece di scaricare 15 GB.

```
node scripts/q35-header-dump.mjs modello.gguf https://…/candidato.gguf --out dump.json
```

Il survey fatto il 2026-08-17 sul nostro 35B, che è anche l'esempio di lettura:

| file | taglia | **parco expert** | formati degli expert |
|---|---|---|---|
| `UD-Q4_K_S` *(in uso)* | 19,46 GiB | **17,07 GiB** | `Q4_K`, `Q6_K` |
| bartowski `Q3_K_M` | 15,94 | 14,35 | `Q3_K`, `Q4_K`, `Q8_0` |
| bartowski `Q3_K_S` | 15,28 | 13,69 | `Q3_K`, `Q8_0` |
| unsloth `UD-Q3_K_S` | 14,30 | 12,03 | `IQ3_S`, `IQ3_XXS`, `IQ4_XS` |
| **bartowski `Q2_K`** | 12,58 | **11,15** | `Q2_K`, `Q3_K`, `Q8_0` |
| unsloth `UD-Q2_K_XL` | 11,45 | 9,81 | `IQ2_XS`, `IQ3_XXS`, `IQ4_XS` |

Tre letture che dalla colonna «taglia» non si ricavano:

1. **il parco non scala col file**: fra `UD-Q3_K_S` e `Q3_K_S` ci sono 1 GiB di
   differenza sul file e 1,7 sul parco;
2. **i file più piccoli sono i-quant**, cioè quelli che chiedono la famiglia di
   kernel che non abbiamo;
3. **il `Q2_K` di bartowski è l'unico che sta sotto la nostra arena (11,17 GiB)
   restando in famiglia K.** È questo il motivo per cui è il candidato.

---

## 10. Cosa cambia per un motore che gira nel browser

Qui il discorso smette di essere generale, perché **un formato non è utilizzabile
finché non ha il suo kernel**, e nel browser i vincoli sono più stretti.

### 10.1 Il dequant avviene nello shader

Non si dequantizza il modello in memoria (raddoppierebbe l'occupazione, e ne
abbiamo troppo poca): i pesi restano quantizzati in VRAM e **ogni kernel legge
i byte grezzi e ricostruisce i valori nel ciclo interno**. Quindi il costo del
formato non è solo spazio: è aritmetica dentro il moltiplicatore matrice-vettore.

Conseguenza diretta: **bpw più basso non significa automaticamente più veloce.**
Meno byte da leggere aiuta, più lavoro di ricomposizione per byte no. Su un
kernel *bandwidth-bound* il primo effetto domina; su uno *dispatch-bound* può non
vedersi affatto.

### 10.2 Ogni formato costa un cablaggio, non solo un kernel

Nel nostro motore, aggiungere un formato tocca:

- il **dequant di riferimento in CPU** (che è il confronto contro cui il kernel
  si verifica);
- il **kernel GEMV/GEMM** nelle sue forme (la via a riga singola del decode e la
  forma multi-riga del prefill);
- il **layout dello slab** e le classi d'arena, se il formato compare fra gli
  expert (le taglie cambiano, e con esse la geometria degli slot);
- un caso di **conformance per formato** contro il riferimento CPU.

Per `Q2_K` e `Q3_K` la prima voce **esiste già** in `quant.ts` — dequantizzatore
*e* quantizzatore, verificati byte-identici a `llama-quantize`. È la ragione per
cui quei due sono un lavoro noto in taglia e gli `IQ2/IQ3` no.

### 10.3 La residenza, che è il vincolo vero di un MoE

In un MoE ogni token usa pochi expert (qui 8 su 256), ma *quali* cambia a ogni
token e a ogni layer. Se il parco non ci sta in VRAM, gli expert mancanti vanno
letti mentre il modello genera — ed è lì che finisce il tempo. La quantizzazione
degli expert è quindi la leva più diretta sulla velocità di un MoE: **non perché
il calcolo diventi più veloce, ma perché a un certo punto smette di servire il
disco.**

Sul nostro caso la soglia è netta e misurabile: parco 17,07 GiB contro arena
11,17 → 65% residente → ~11,5 token/s in chat. Con un parco sotto gli 11,17 il
regime diventa quello a zero miss, che al banco vale **40 token/s**.

---

## 11. Come si misura la qualità senza raccontarsela

La domanda «quanto costa in intelligenza» è la metà difficile, e ha tre trappole.

### 11.1 Le tabelle pubbliche di perplessità non sono un proxy valido

Sta scritto nel nostro stesso codice, da una fase precedente:

> *«senza pesi originali la catena è Q4_0 → f32 → Q3_K (errore COMPOSTO): le
> tabelle di perplexity pubbliche NON sono un proxy valido — si misura».*
> — `src/engine/quant.ts`

Le ragioni sono banali e decisive: quei numeri vengono da un altro modello, un
altro corpus, un'altra ricetta e spesso un'altra imatrix.

### 11.2 Si misura **appaiato**, non «due medie»

I due modelli devono vedere **gli stessi token nelle stesse posizioni**, e la
quantità da riportare è la differenza *per posizione*. La difficoltà del singolo
token — che è la sorgente di rumore dominante — si cancella nella differenza, e
l'intervallo di confidenza si stringe di parecchio.

E l'intervallo va calcolato con un **bootstrap a blocchi**: i token di un testo
sono autocorrelati, e un errore standard classico (`σ/√n`) sarebbe ottimista di
un fattore ignoto.

### 11.3 Sui compiti, non si contano le risposte giuste

Contare quante risposte sono corrette butta via quasi tutta l'informazione: su un
campione piccolo ±1 risposta muove il punteggio di parecchi punti. Si guardano
invece **la log-probabilità e il rango del bersaglio** — stessa informazione,
varianza molto più bassa, e a costo zero perché è teacher-forced.

### 11.4 Cosa c'è in casa

| strumento | cosa fa |
|---|---|
| `scripts/quant-eval-corpus.mjs` | costruisce il corpus: prosa italiana tecnica, codice, wikitext-2, + 200 compiti (GSM8K, MMLU). Ogni sorgente col suo sha |
| `scripts/quant-quality.py` | bits/token per sezione e log-prob + rango del bersaglio, **teacher-forced e deterministico**; scrive i vettori per posizione |
| `scripts/quant-quality-compare.mjs` | il confronto appaiato, col bootstrap a blocchi e il test di segno. **Rifiuta** corpus diversi, token diversi, oracoli diversi |

Il corpus ha una sezione in **italiano tecnico** perché il danno di una
quantizzazione non è uniforme fra domini, e il regime d'uso di questo progetto è
una chat in italiano — un numero preso solo su wikitext misurerebbe un regime che
non è il nostro.

*Riferimento misurato il 2026-08-17 sul quant in uso (`UD-Q4_K_S`), come esempio
di lettura: **2,594 bit/token su wikitext-2** (perplessità 6,04), 3,206 sul
codice, 4,097 sull'italiano tecnico.*

---

## 12. Glossario

| termine | significato |
|---|---|
| **bpw** | bit per peso: byte del blocco × 8 ÷ pesi del blocco. Include scale e maschere |
| **blocco / super-blocco** | il gruppo di pesi che condivide una scala (32 nei legacy, 256 nei K e negli I) |
| **sotto-blocco** | suddivisione del super-blocco con la sua scala a 6 bit (K-quant) |
| **scala (`d`)** | il float per cui si moltiplica l'intero per riottenere il peso |
| **`dmin`** | l'offset dei formati affini: `w = d·q − dmin·m` |
| **simmetrico / affine** | senza o con offset. Sbagliarlo dà pesi plausibili e sbagliati |
| **codebook / reticolo** | l'insieme di vettori a 8 dimensioni fra cui gli i-quant scelgono |
| **imatrix** | profilo di importanza dei pesi, usato per pesare l'errore in quantizzazione |
| **ricetta** | quale tensore prende quale formato (`_S`, `_M`, `_L`, `_XL`, `UD`) |
| **parco expert** | in un MoE, l'insieme dei pesi degli expert routed: la parte che si contende la VRAM |
| **arena** | la VRAM che il motore dedica agli expert, divisa in slot |
| **teacher-forced** | il modello non genera: si misura la sorpresa su token fissati. Deterministico |
| **GGUF** | il formato-contenitore di `llama.cpp`: header con metadati + tensori, ognuno col suo tipo |

---

## 13. Cosa in questo documento NON è verificato in casa

Sezione deliberata: un documento che non distingue ciò che ha misurato da ciò che
ha letto invita a fidarsi di entrambi allo stesso modo.

**Verificato dal nostro codice o dai nostri artefatti:**

- i byte di blocco e i layout interni di `Q4_0`, `Q4_1`, `Q8_0`, `Q2_K`, `Q3_K`,
  `Q4_K`, `Q5_K`, `Q6_K` — implementati e confrontati con `ggml-quants.c` in
  `src/engine/quant.ts`;
- le taglie di blocco degli i-quant citate nella §8 — **indirettamente**, dal
  controllo somma-tensori vs taglia-file su tre modelli veri (scarto ≤ 0,09%);
- tutte le composizioni per-tensore della §9 e i bits/token della §11.4 —
  artefatti in `results/eval/`.

**Preso dal disegno dei formati, non verificato qui:**

- il *meccanismo* interno degli i-quant a reticolo (codebook di vettori a 8
  dimensioni, bit di segno, scale) e la tabella a 16 valori di `IQ4_NL`/`IQ4_XS`.
  Il *peso* è verificato, la *struttura* no: questo motore non li legge.
  **Chi volesse scriverne un kernel deve partire da `ggml-quants.c`**, come la
  fase che ha portato `Q2_K`/`Q3_K` in `quant.ts` ha fatto — arrivando alla
  byte-identità con `llama-quantize`, che è il solo standard di prova accettabile
  qui;
- il funzionamento della imatrix come procedura di calibrazione;
- l'affermazione della §7 sulla fragilità linguistica dei file calibrati su un
  solo idioma: è plausibile e non misurata.

**Non trattato**: come si *produce* un file quantizzato (`llama-quantize`), la
quantizzazione delle attivazioni e della KV cache (altro problema), e i formati
non-GGUF (GPTQ, AWQ, EXL2, MXFP4).
