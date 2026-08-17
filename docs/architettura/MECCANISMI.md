# Meccanismi × path — cosa esiste, chi lo usa, cosa è misurato

**A cosa serve, e perché non è un documento di architettura.** Il diagramma
(`motore-architettura.excalidraw`) mostra come i pezzi si parlano. Questo mostra
**quali meccanismi esistono e chi NON li adotta** — cioè le celle vuote.

La ragione è un difetto osservato tre volte in una notte (2026-08-16): il
pre-pack degli slab, la policy `tier` e le leve `kfan`/`splitk` erano tutti
**già costruiti e non adottati dal path che ne aveva bisogno**, e io li ho
"scoperti" uno per uno come se fossero da scrivere. **Un grep dice cosa c'è, non
cosa manca.** Una tabella con le celle vuote sì.

> **Prima di progettare qualcosa, si cerca qui.** Se il meccanismo c'è ed è
> spento, il lavoro è adottarlo, non riscriverlo — è il ruling «riusa quello che
> c'è già, costruisci globale, poi estendi» e il principio `fix-dont-fence`.

**Questo file è tenuto onesto da un test.** `tests/engine-meccanismi.test.ts`
verifica contro il SORGENTE le colonne «dove vive» e «chi lo usa». Se qualcuno
adotta un meccanismo e non aggiorna la riga, il test fallisce. Una mappa che può
mentire è peggio di nessuna mappa: la si consulta e ci si fida.

---

## 1 · Residenza degli expert (solo MoE)

| meccanismo | dove vive | GLM | Qwen 35B | default | misurato |
|---|---|---|---|---|---|
| **arena + slotTable** | `residency.ts` | ✅ | ✅ | acceso | sì, entrambi |
| **policy `tier`** (autopin top-usage, LFRU, cap 12,5%) | `residency.ts` | ✅ | ✅ | **`lru`** | GLM 2026-08-09 · **35B it.48: NON paga** |
| **slab pre-impacchettato** `{ raw, slab }` | `residency.ts:110` | ✅ `glmmodel.ts:871` | ✅ `q35gpumodel` (`readMiss`, it.50) | acceso **se il file c'è** | GLM: −41,4 ms/token · 35B **it.51**: `packMs` 7.331 → **0**, ma fetch +5,3% ⇒ **netto +3,7% tok/s**, e risposte identiche |
| **sorgente OPFS** (import + `ensureSlabs`) | `glmsource.ts` | ✅ | ❌ **e non ci starebbe** | — | quota OPFS **10 GiB** < 17,07 richiesti (it.43) |
| **sorgente a Range** (file slab servito come il GGUF) | `slabsource.ts` | ❌ (usa OPFS) | ✅ | fallback ai byte grezzi, **con motivo** | apertura e rifiuti sotto test (it.50); file convertito e misurato in A/B (it.51) |
| **convertitore offline → slab** | `scripts/q35-slab-build.mjs` | ❌ | ✅ | — | verificato 8/8 slab (it.47) |

**Le due sorgenti sono due strade per UNA interfaccia**: il GLM genera il suo
file in OPFS all'import, il 35B legge un file già convertito via Range perché in
OPFS non ci sta. `ExpertCache` non sa quale delle due la sta servendo.

**Come si legge se è acceso davvero**: `moeStats().slabSource` — `file` non nullo
⇒ in uso; altrimenti `reason` dice perché (file assente, SHA di un altro GGUF,
taglia diversa). Sta nell'artefatto di ogni bench e di ogni chat, perché un miss
costa **una** richiesta Range con lo slab e **tre** senza: due run che
differiscono per quello non devono distinguersi solo da cosa c'era sul disco.

## 2 · Leve del decode

| leva | dove vive | default | chat | bench | guadagno misurato |
|---|---|---|---|---|---|
| **kfan** (topK expert in un giro) | `q35gpumodel` + `wgsl.ts` | **spenta** | ✅ da it.39 | `--kfan` | 22,58 → 28,90 tok/s |
| **router parallelo** | `wgsl.ts:routerTopKWgsl` | sempre attivo | ✅ | — | 28,90 → 30,74 |
| **rotta split-K nel decode** | `q35gpumodel:gemv` | **spenta** | ✅ da it.39 | `--splitk` | 30,74 → 40,06 |
| **pavimento N ≥ 2048** | `prefillgemmplan` | attivo | ✅ | — | +0,58 ms/token (it.38) |
| **via intera `idot` nel decode** | cablata, `DEC_SPLITK_IDOT=false` | **spenta** | ❌ | ❌ | **netto PEGGIORE** −0,14/−0,66 ms (it.«idot») |

⚠️ **Le prime due nascono spente** perché i loro A/B le accendono un braccio
alla volta. Per giorni la chat ha girato senza: «misurato» e «consegnato» non
coincidevano e niente lo diceva. Ora `model.levers` sta nell'artefatto.

## 3 · Prefill

| meccanismo | dove vive | stato |
|---|---|---|
| prefill a chunk (M=16) | `q35gpumodel:gemvB` | attivo con `prefillM` |
| rotta split-K di prefill | `prefillgemmplan` + `wgsl.ts` | kernel per **8 formati** (q4_0/q5_K/q4_1/q4_K/q6_K/q8_0/q2_K/q3_K), ma **instradati solo 4** (q4_0/q5_K/q4_1/q8_0): gli altri sono `wired: false` in `PREFILL_GEMM_SPEC` |
| **spec-dec MTP** | `q35gpumodel:mtpDraft`, `specVerify` | **costruito, solo nel ktest** — accept 50%, proiezione 1,29× · ❌ non nella chat, e serve una testa MTP separata (esiste per il 4B, non per il 35B) |

⚠️ **«Il kernel c'è» non è «il piano ci passa».** Sono due domande diverse dal
2026-08-14, e tenerle separate è ciò che permette di portare e verificare una
forma senza cambiare di una riga ciò che il 4B esegue. Chi vuole il conto vero
dei formati instradati chiede `PREFILL_GEMM_WIRED_KINDS`, non
`PREFILL_GEMM_KINDS.length`.

## 4 · Kernel per formato di peso (K-quant)

| meccanismo | dove vive | stato |
|---|---|---|
| **nucleo unico dei gemv K-quant** | `wgsl.ts:gemvKQuantWgsl` + `KQUANT_GEMV_DESC` | ✅ tutti e 5 i formati (q2_K/q3_K/q4_K/q5_K/q6_K) escono da UN generatore; i tre preesistenti emettono WGSL **identico carattere per carattere** a prima (fixture in `tests/fixtures/kquant-core/`) |
| gemv Q4_K / Q5_K / Q6_K | istanze del descrittore | ✅ in produzione (decode 35B, GLM) |
| **gemv Q2_K / Q3_K** | istanze del descrittore | ✅ **ESEGUITI SU GPU e in un turno di chat vero** (2026-08-17): ktest `gemv-q2_K`/`gemv-q3_K` su entrambe le shape degli expert, 4/4 PASS contro `dequantQ2_K`/`dequantQ3_K` (maxRel ≤ 8,1e-5), e il 35B `bartowski Q2_K` genera in chat — 13,44 tok/s al primo turno contro 11,47 del Q4_K_S |
| **selettore unico dei gemv K-quant** | `q35gpumodel.ts:q35KQuantGemvWgsl` + `q35KQuantKindOfGgml` | ✅ la scelta del kernel sta in UN posto (prima: tre catene di ternari); kernel, `blockBytes` e kind chiesto al piano di prefill discendono tutti dal tipo REALE del tensore |
| **prefill multi-riga Q2_K / Q3_K** | `wgsl.ts:prefillGemmQ2K*`/`Q3K*` | portati per analogia dai gemelli q4_K/q6_K, `wired: false`, **nessuna misura su device** (stessa postura di q4_K: portato e non instradato) |
| **riga di embedding per formato** | `q35gpumodel.ts:EMBD_ROW` | ✅ Q6_K/Q8_0/Q4_0/**Q2_K**. Era uno switch a 3 rami duplicato in DUE siti; il quarto formato ne avrebbe fatte sei copie. Il `token_embd` del `bartowski Q2_K` è Q2_K, ed è ciò che bloccava il caricamento |

Il perché di questi due formati sta nella capienza, non nel kernel: sul file
`bartowski Q2_K` il parco expert del 35B passa da 17,07 a **10,391 GiB**, cioè
dentro l'arena da 11,17 — il 100% residente contro il 65% di oggi.

**Misurato il 2026-08-17, e il regime a zero miss NON è ancora quello della
chat**: primo turno 13,44 tok/s (era 11,47) con **7.930 miss**, perché l'arena
si RIEMPIE durante il turno; secondo turno **17,96** con **362 miss** (0,58%
contro l'1,84% del Q4_K_S) e replay 3.072 layer contro 17.069. La direzione è
quella giusta e il collo si sta spostando, ma i 40 tok/s del banco pretendono
un'arena già calda: servono più turni, o il braccio caldo del banco.

## 5 · I/O

| meccanismo | dove vive | stato |
|---|---|---|
| lettore Range condiviso | `ggufrange.ts` | ✅ tutti e 5 i chiamanti (it.30) |
| contatori del lettore (parallelismo effettivo, latenza) | `ggufrange.ts` | ✅ sempre |
| **regime di lettura dichiarato** (`disk`/`os-cache`) | `residency.ts:readBandwidth` | ✅ glmbench · ❌ **q35conf non lo dichiara** |
| raggruppamento richieste | — | **misurato inutile**: continuo vs raffiche 0,98× (it.33) |
| **prefetch lookahead** | `scripts/q35-looka-run.mjs` (solo misura) | ❌ **non implementato** — recall 91,92% @K=8 |

## 6 · Strumenti di misura già pronti

Prima di scrivere un banco nuovo, questi esistono:

| strumento | cosa dà | costo |
|---|---|---|
| `--io-probe` | curva banda/richieste-in-volo, forma a raffiche, località, quota OPFS | secondi, **senza caricare il modello** |
| `--gpu-time` | ripartizione del token per categoria GPU | una run |
| `--logit-probe` | i due candidati di testa per token | una run (perturba il tempo) |
| `--kfan` / `--splitk` | A/B a caldo nello **stesso processo** | una run |
| `chat-smoke --prompt --policy` | la chat vera, due turni, export JSON | una run |
| `q35-slab-build --dry-run` | geometria e taglia del file slab | secondi |

---

## Le celle vuote, in ordine di valore misurato

1. **lo slab si legge in UNA richiesta e non si sovrappone a nessuno** — la
   cella nuova, aperta dalla misura di it.51. Il pre-pack ha reso 4,2 s netti
   invece di 7,3 perché la fetch è peggiorata del 5,3%: tre letture parallele
   erano più veloci di una grande. Il ginocchio della curva è a 2-4 richieste in
   volo (it.33) e `slabFileRange` è già aritmetica: leggere uno slab in 2-4
   sotto-range paralleli è la rifinitura che recupera quei 3,1 s.
2. **prefetch lookahead non implementato** → attacca i ~76 ms/token di tassa di
   residenza, con un oracolo già misurato al 91,92%.
3. **spec-dec non nella chat** → 1,29× proiettato, ma serve una testa MTP per il
   35B che oggi non esiste.
4. **Q2_K/Q3_K: kernel generati e instradati, zero esecuzioni su GPU** → è la
   cella aperta dal 2026-08-17. Il testo c'è, è verificato sul testo, e dal
   selettore unico il ramo expert lo sceglierebbe davvero per un GGUF Q2_K; ma
   finché non passa il ktest contro `dequantQ2_K`/`dequantQ3_K` e una misura,
   non è un guadagno — è un candidato. Il valore in palio è il 100% di
   residenza del parco expert.
5. `q35conf` non dichiara il regime di lettura → i suoi numeri non dicono se
   sono confrontabili, ed è il difetto che è costato il falso allarme del GLM.

## Cosa NON è una cella vuota, e perché

- **policy `tier`**: esiste, adottata, **misurata e non paga** sul 35B — il
  collo è la capienza (LRU tiene già il 66% del parco), non la politica.
- **raggruppamento delle richieste**: misurato, 0,98×. Il canale non è il collo.
- **`idot` nel decode**: cablato e spento con l'aritmetica accanto — vincerebbe
  sul kernel e perde nel motore per il dispatch in più.

*Tenere queste tre righe è deliberato: senza, fra un mese qualcuno le riproporrà
come idee nuove — che è esattamente ciò che è successo con `tier`.*
