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
| **slab pre-impacchettato** `{ raw, slab }` | `residency.ts:110` | ✅ `glmmodel.ts:871` | ❌ **non adottato** | — | GLM: −41,4 ms/token · 35B: **7,11 s/sessione da recuperare** |
| **sorgente OPFS** (import + `ensureSlabs`) | `glmsource.ts` | ✅ | ❌ | — | quota OPFS **10 GiB** < 17,07 richiesti (it.43) |
| **convertitore offline → slab** | `scripts/q35-slab-build.mjs` | ❌ | ✅ | — | verificato 8/8 slab (it.47) |

**La cella che conta oggi**: il 35B non adotta `{ raw, slab }`. L'interfaccia è
nel modulo condiviso da un goal intero.

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
| rotta split-K di prefill | `prefillgemmplan` + `wgsl.ts` | attiva, q4_0/q4_1/q4_K/q5_K/q6_K/q8_0 |
| **spec-dec MTP** | `q35gpumodel:mtpDraft`, `specVerify` | **costruito, solo nel ktest** — accept 50%, proiezione 1,29× · ❌ non nella chat, e serve una testa MTP separata (esiste per il 4B, non per il 35B) |

## 4 · I/O

| meccanismo | dove vive | stato |
|---|---|---|
| lettore Range condiviso | `ggufrange.ts` | ✅ tutti e 5 i chiamanti (it.30) |
| contatori del lettore (parallelismo effettivo, latenza) | `ggufrange.ts` | ✅ sempre |
| **regime di lettura dichiarato** (`disk`/`os-cache`) | `residency.ts:readBandwidth` | ✅ glmbench · ❌ **q35conf non lo dichiara** |
| raggruppamento richieste | — | **misurato inutile**: continuo vs raffiche 0,98× (it.33) |
| **prefetch lookahead** | `scripts/q35-looka-run.mjs` (solo misura) | ❌ **non implementato** — recall 91,92% @K=8 |

## 5 · Strumenti di misura già pronti

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

1. **35B non adotta `{ raw, slab }`** → 7,11 s per sessione, e 38.625 richieste
   Range → 12.875. Tutto il resto è pronto (formato, descrittore, convertitore).
2. **prefetch lookahead non implementato** → attacca i ~76 ms/token di tassa di
   residenza, con un oracolo già misurato al 91,92%.
3. **spec-dec non nella chat** → 1,29× proiettato, ma serve una testa MTP per il
   35B che oggi non esiste.
4. `q35conf` non dichiara il regime di lettura → i suoi numeri non dicono se
   sono confrontabili, ed è il difetto che è costato il falso allarme del GLM.

## Cosa NON è una cella vuota, e perché

- **policy `tier`**: esiste, adottata, **misurata e non paga** sul 35B — il
  collo è la capienza (LRU tiene già il 66% del parco), non la politica.
- **raggruppamento delle richieste**: misurato, 0,98×. Il canale non è il collo.
- **`idot` nel decode**: cablato e spento con l'aritmetica accanto — vincerebbe
  sul kernel e perde nel motore per il dispatch in più.

*Tenere queste tre righe è deliberato: senza, fra un mese qualcuno le riproporrà
come idee nuove — che è esattamente ciò che è successo con `tier`.*
