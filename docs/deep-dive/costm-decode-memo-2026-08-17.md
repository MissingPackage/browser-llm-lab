# Memo — la curva cost(M): il ginocchio è a M=2, e il verdetto sullo spec-dec cade

**Pre-registrazione**: `costm-decode-prereg-2026-08-17.md` (scritta prima di
vedere qualunque cella). **Artefatto**:
`results/microbench/costm-decode-4090-linux-2026-08-17T21-38-36-738Z.json`.
**Goal**: `engine-velocita-decode`, spike (1) di tre.

## Il risultato

Costo **per riga**, migliore variante per ogni M, normalizzato a M=1:

| shape | M=1 | **M=2** | M=4 | M=8 | M=16 |
|---|---|---|---|---|---|
| q4_K 2048×512 — expert gate/up | 100% | **60,3%** | 35,3% | 22,8% | 16,9% |
| q4_K 512×2048 — expert down | 100% | **56,8%** | 33,5% | 21,4% | 15,8% |
| q6_K 512×2048 — expert down | 100% | **57,0%** | 34,4% | 22,4% | 16,2% |
| q8_0 2048×4096 — attn q-proj | 100% | **38,9%** | 21,2% | 8,3% | 3,2% |

## Le previsioni, giudicate una a una

1. **«A M=1 la forma multi-riga perde»** — **CONFERMATA**, e più nettamente di
   come l'avevo scritta: a M=1 la variante migliore su gate/up è `base-batch-z`,
   e **da M=2 in poi vince `splitk-idot`**. Il sorpasso è esattamente al
   ginocchio.
2. **«Il ginocchio è a M=2, sotto il 70% del costo/riga di M=1»** —
   **CONFERMATA**: 56,8-60,3% sulle tre shape expert. Il modello aritmetico
   pre-registrato `(4+M)/5M` (dequant ≈ 4 FMA) prevedeva **60,0%**. Non era una
   previsione fortunata: prevede anche M=4 al 40% (misurato 33,5-35,3%) e M=8 al
   30% (misurato 21,4-22,8%), sbagliando **per difetto**, cioè l'ammortamento è
   un po' meglio del modello.
3. **«Satura fra M=8 e M=16, meno del 20% di calo relativo»** — **FALSIFICATA**:
   il calo è del 26% su tutte e tre le shape expert. A M=16 non ha ancora
   saturato. Conseguenza pratica: nessuna, perché lo spec-dec non arriva a M=16.
4. **«Il ginocchio è più netto su `down` (K=512) che su `gate/up` (K=2048)»** —
   **NON confermata in modo significativo**: 56,8% contro 60,3%. Avevo previsto
   un divario grosso, motivandolo con l'occupancy (2 lane attive su 64 contro 8).
   Il divario c'è ma è di 3,5 punti. **L'argomento dell'occupancy spiegava meno
   di quanto credessi.**

## Cosa ne discende per lo spec-dec

Costo TOTALE di verificare 2 token contro 1, sulle shape expert:

    gate/up  0,0075 / 0,0062 = 1,21x
    down     0,0043 / 0,0038 = 1,13x
    down q6  0,0099 / 0,0087 = 1,14x

Con accept 50% (1,5 token per passata) e prendendo la shape PEGGIORE:
**1,5 / 1,21 ≈ 1,24x**.

Contro il conto dei kernel vecchi, che è il verdetto oggi in vigore:
**1,5 / 1,8 ≈ 0,83x**, cioè perdente — ed è esattamente l'1,18x più lento
misurato dal checkpoint B.

**Il verdetto «spec-dec più lento» era corretto per i kernel su cui fu preso, ed
è caduto coi kernel di oggi.** Non perché qualcuno avesse sbagliato la misura:
perché la proprietà del kernel è cambiata sotto di essa.

## Cosa questo memo NON autorizza

- **Non autorizza a dichiarare lo spec-dec vinto.** È il BANCO. La lezione già
  pagata quattro volte in questo progetto è che «il banco misura il kernel, il
  motore paga la rotta»: 1,24x è il permesso di misurare nel motore, non il
  risultato.
- **Non dice niente sul `q2_K`**, il quant che consegniamo (docket item 24). Il
  banco non lo copre, e il ginocchio dipende dal rapporto byte/lavoro, che nel
  q2_K cambia (scale a 4 bit, blocco più stretto).
- **Non dice niente sul segmento expert del MoE**, ed è il buco più grande.
  Con top-8 su 256, se il routing di due token adiacenti fosse indipendente
  l'overlap atteso sarebbe **0,25 expert condivisi**: a M=2 si avrebbero ~15,75
  expert distinti per 16 selezioni, cioè **~1,02 righe per expert e ZERO
  ammortamento** — la curva qui sopra non si applicherebbe affatto al segmento
  che pesa di più. L'oracolo di lookahead misura recall **82,67% @K=8**, quindi
  il routing è fortemente correlato e l'overlap reale sarà molto maggiore di
  0,25 — ma **quanto** è precisamente lo spike (2), e senza quel numero il 1,24x
  vale solo per il segmento denso.

**Prossimo passo, invariato**: overlap del router top-8 a distanza 1-4 sul 35B.
Mezza giornata, riusa il router esistente, zero kernel nuovi.
