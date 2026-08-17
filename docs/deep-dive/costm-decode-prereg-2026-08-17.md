# Pre-registrazione — dove sta il ginocchio della curva cost(M)?

**Goal**: `engine-velocita-decode`, spike (1) di tre autorizzati dal PI il
2026-08-17. **Data**: 2026-08-17.
**Scritta PRIMA di leggere i numeri**: come per la 2d, la run che l'accompagna è
morta alla scrittura dell'artefatto (tag assente da `PROV`), quindi **nessuna
cella è stata vista**. È la ragione per cui questo documento esiste adesso.

## La domanda

Lo spec-dec è **parcheggiato** da un verdetto: il checkpoint B l'ha misurato
**più lento** (1,18×) e la proiezione di 1,29× non copriva il costo. Ma quel
verdetto è stato preso **coi kernel vecchi**, dove `headroom §3.1` misurava M
«quasi ininfluente, 1,22× a M=8»: la riga marginale costava ~0,85 di un pass
intero, e 1,5 token accettati su 1,8 di costo davano 0,83× — perdente.

Sui kernel di oggi la proprietà è diversa. `wgsl.ts:4609` (banco fase 0 di
`engine-kquant`, shape [2048,4096]) misura `splitk-idot` a **M=16 in 0,0376 ms
contro 0,0698 a M=1**: **sedici righe costano meno di una**, cioè il costo per
riga crolla di ~30×.

**Ciò che nessuno ha mai visto è dove quella proprietà si accende.** Sul q4_K il
misurato è 0,91× a M=1 e 3,20× a M=8: il ginocchio sta **fra 1 e 8**, e la
griglia storica `[1, 8, 16]` lo salta per costruzione. Lo spec-dec realistico
verifica **2-4 token per passata**, cioè esattamente nel buco.

## L'ipotesi, e l'aritmetica che la genera

A M=1 il GEMV quantizzato è **ALU-bound**, non memory-bound: il GEMV f32 satura
la banda misurata della scheda (435 GB/s, 100%) mentre i quantizzati stanno al
20%. Il costo per peso è quindi ≈ `t_load + t_dequant + M·t_mac`, e **dequant e
traffico pesi si pagano UNA volta per M righe**.

Con `t_dequant ≈ 4·t_mac` (shift+mask+convert per nibble contro una FMA), il
costo **per riga** scala come `(4+M)/5M`:

| M | costo/riga previsto |
|---|---|
| 1 | 100% |
| 2 | **60%** |
| 4 | **40%** |
| 8 | 30% |
| 16 | 25% |

## Le previsioni, in ordine di quanto mi espongono

1. **A M=1 la forma multi-riga PERDE** (~0,91× sul q4_K). Non è un difetto: è il
   motivo per cui il piano non la offre mai al decode. Se non perdesse, il
   modello di costo sopra sarebbe sbagliato in modo interessante.
2. **Il ginocchio è a M=2**: il costo per riga a M=2 sta **sotto il 70%** di
   quello a M=1. È la previsione che decide, ed è quella del consulente.
3. Il guadagno **satura**: fra M=8 e M=16 il costo per riga cala di meno del 20%
   relativo, perché lì il termine dequant è già ammortizzato e si comincia a
   sbattere sul tetto di banda.
4. Il ginocchio è **più netto sulla shape `down` (K=512)** che su `gate/up`
   (K=2048): con K=512 sono due soli superblocchi per riga, quindi a M=1
   l'occupancy è peggiore (2 lane attive su 64 contro 8) e c'è più da recuperare.

## Cosa FALSIFICA l'ipotesi, dichiarato prima

- **Se a M=2 il costo per riga non scende sotto il 90% di quello a M=1**,
  l'argomento dell'ammortamento non regge nel regime dello spec-dec, e lo
  spec-dec **resta parcheggiato**. Non si va a cercare M=8 «perché lì paga»: uno
  spec-dec che accetta 8 token di fila non è il nostro.
- **Se il ginocchio c'è ma solo su una shape su quattro**, non basta: il decode
  del 35B passa da gate/up E down a ogni expert selezionato.

## Cosa questa run NON può dire

- **Niente sul `q2_K`**, cioè sul quant che consegniamo davvero. Il banco copre
  q5_K/q4_1/q4_K/q6_K/q8_0 e non lui (docket item 24). Stesso generatore non è
  stessa geometria: il q2_K ha scale a 4 bit e blocco più stretto, quindi cambia
  il rapporto byte/lavoro — che è **esattamente ciò che decide dove sta il
  ginocchio**. Qualunque trasferimento è un'inferenza, non una misura.
- **Niente sul motore**: è un banco di kernel. La lezione già pagata quattro
  volte in questo progetto è che «il banco misura il kernel, il motore paga la
  rotta». Un ginocchio a M=2 sul banco autorizza a **misurare** lo spec-dec nel
  motore, non a dichiararlo vinto.
- **Niente sul segmento expert del MoE**, dove l'ammortamento dipende
  dall'**overlap del router** fra token adiacenti — che è lo spike (2), non
  questo.
