# JOURNAL — engine-kquant

Una voce per iterazione: cosa ho fatto, cosa ho misurato, cosa ho deciso e
perche'. Le decisioni che NON sono mie stanno nel docket.

## it.0 — decomposizione (2026-08-14)

Contratto in `GOAL.md` (chartered 2026-08-14, con la doppia barra del PI:
22.500 obbligatoria, 18.000 nice to have). Spina in `PHASES.md`: sette righe.

**Test-fit dei done-when prima della tabella — quattro correzioni**, tutte
scritte in testa a `PHASES.md`:

- **C0-1**: `prefillGemmSplitsFor` (`wgsl.ts:4150`) conta in blocchi da 32; i
  K-quant hanno il superblocco da 256 come unita' indivisibile (scale e min a
  6 bit condivisi). Va reso parametrico sulla famiglia, o il conto dei byte su
  cui poggia il goal smette di valere.
- **C0-2**: `prefillQuantXQ8Wgsl` quantizza per blocchi da 32 e i sotto-blocchi
  K-quant sono anch'essi da 32 ⇒ **la via intera e' disponibile ai K-quant
  senza un secondo quantizzatore e senza dispatch aggiuntivi**. Riuso, non
  lavoro nuovo.
- **C0-3**: K-quant e q4_1 hanno un termine costante per blocco che moltiplica
  Σx (non Σ(q·x)): nella via intera si ottiene con
  `dot4I8Packed(xq, 0x01010101)`. Ipotesi da confermare al banco, non promessa.
- **C0-4**: cella degenere evitata — il down-proj degli expert del 35B ha
  K=512 = 2 superblocchi per riga, quindi lo split-K a 4 fette **non esiste**
  su quella shape. Celle a `splits ∈ {1,2}`. Shape 35B verificate sull'header
  dump: dModel 2048, dFfnExpert 512, nExpert 256, 40 layer.

**Decisione mia, registrata e non escalata** (l'ordine e il meccanismo sono
miei): la famiglia di kernel si fa per intero in questo goal — Q5_K, Q4_1,
Q4_K, Q6_K, Q8_0, tutte misurate al banco e verificate col ktest — ma **solo
Q5_K e Q4_1 vengono cablate e misurate end-to-end**, perche' sono le uniche
che esistono nel 4B. Il 35B e' il goal successivo (deciso dal PI) e sara'
wiring + residency, non lavoro di kernel: 17,67 GB di expert su 16 GiB di
scheda, un piano diverso (`moeprefillplan.ts`) e nessuna baseline fresca da cui
partire. Cosi' quel goal parte da forme MISURATE invece che da forme da
inventare, e questo goal conserva UNA condizione di completamento.

**Reperto per il goal 35B**, dall'header dump 2026-08-10: il 35B non ha **un
byte** di q4_0 (expert Q4_K 17,67 GB · Q6_K 0,66 · attn Q8_0 1,09 · linear_attn
Q8_0 0,27 · head Q8_0 0,54). La via veloce attuale ne copre lo **0%**.

**Reperto per la scaletta dei modelli**: il **Qwen3.5-9B** ha la stessa identica
struttura del 4B (`linear_attn:Q5_K` 276,8 MB su 24 tensori, `ffn/shexp:Q4_1`
125,8 MB su 4). Questa leva vale li' **senza una riga di codice in piu'**.
Segnalato al PI in chat; non entra nel contratto (nessuna baseline 9B esiste, e
aggiungerlo raddoppierebbe le run di verifica).

Prossima: riga 1, fase 0 al banco.

## it.1 — riga 1: la fase 0 dice SI' a entrambe le famiglie cablabili (2026-08-14)

**Comando esatto** (letto prima di spendere, come impone il protocollo):

    setsid nohup npx vite --port 5199 > /tmp/vite-5199.log 2>&1 < /dev/null &
    curl -s -o /dev/null -w "%{http_code}" http://localhost:5199/ttbench.html   # 200
    BASE_URL=http://localhost:5199 node scripts/tt-microbench-run.mjs --label 4090-linux --host quiescent

Artefatto: `results/microbench/ttft-riga1-4090-linux-2026-08-14T18-54-05-813Z.json`
(18 celle `gemm-kquant-multirow`, **zero scartate**).

**I NUMERI, a M=16 — che e' il punto di lavoro del prefill (`PREFILL_M`):**

| famiglia | legacy (produzione) | splitk-idot | splitk-f32 |
|---|---|---|---|
| Q5_K `[4096, 2560]` | 1,2700 ms | **0,0452 ms = 28,07x** | 0,2058 ms = 6,17x |
| Q4_1 `[9216, 2560]` | 2,3483 ms | **0,1040 ms = 22,58x** | 0,1412 ms = 16,63x |

**La regola di stop (>= 1,5x) e' superata di un ordine di grandezza su
entrambe: entrambe si cablano.** La via f32 e' sempre piu' lenta dell'intera ma
sempre sopra la legacy a M >= 8: regge come fallback dichiarato.

**IL CONTROLLO CHE VALE PIU' DEI NUMERI: il banco riproduce la produzione.** Il
braccio legacy Q5_K misura **91 GB/s** di traffico pesi a M=16; il checkpoint di
`engine-ttft` misura **89,9 GB/s** sul segmento `gemm:deltanet-out` in
produzione. Due strumenti diversi, stessa cifra a meno dell'1,2%: il braccio di
paragone non e' un'imitazione del percorso vecchio, e' il percorso vecchio.

**PROIEZIONE sul segmento** (396 chunk = ceil(6333/16)):
- Q5_K: 24 layer x 396 x 0,0452 = **430 ms** contro 12.169 ⇒ **−11,7 s**
- Q4_1: 4 tensori x 396 x 0,1040 = **165 ms** contro ~3.720 (la quota Q4_1 dei
  4.971 di `gemm:ffn-down`) ⇒ **−3,6 s**
- TTFT 32.127 − 15,3 s ≈ **16,8 s**, cioe' sotto ANCHE la barra nice-to-have
  (18.000). Con la penale di lettura fredda misurata sul q4_0 (+13% su
  `splitk-coldw`) resta ~17,0 s.
**Non e' un risultato: e' una proiezione da microbench.** Vale finche' non la
smentisce il checkpoint della riga 5, ed e' li' che va verificata.

**M=1 DICE UNA COSA CHE SERVIRA' AL CABLAGGIO**: a M=1 la forma split-K Q5_K in
f32 e' PIU' LENTA della legacy (0,1491 contro 0,0829). Il combine dei parziali
non si ammortizza su una riga sola. Il piano non deve instradare M=1 sulla forma
multi-riga — oggi non lo fa per costruzione (la via veloce vive solo nel piano
gemello del prefill), ma se un domani qualcuno la offrisse anche al decode,
questo e' il numero che glielo vieta.

**DIFETTO INCONTRATO SUL PERCORSO, TOLTO (non recintato)**: il driver del banco
scriveva sempre `ttft-riga1-*` come nome file e `microbench-ttft-riga1` come
`kind`, anche ora che il banco porta le celle di un ALTRO goal. E' la landmine
del progetto («leggi il `kind`, non il nome del file») vista dal lato di chi
PRODUCE l'artefatto. Aggiunto `--tag`, che muove nome e `kind` INSIEME cosi' non
possono divergere, e l'unione dei `kind` in `mbSchema.ts` resta chiusa (un tag
nuovo si aggiunge al tipo, non si inventa sulla riga di comando).
**Conseguenza onesta**: l'artefatto di questa iterazione porta ancora il `kind`
vecchio, perche' e' stato scritto prima della correzione. Non lo riscrivo a
posteriori — un artefatto di misura ritoccato dopo il fatto non e' piu' una
misura. La riga 1 si chiude in it.2 con una run che porta tutte e cinque le
famiglie e il tag giusto, e QUELLA supersede questa.

**Gate**: `npx tsc --noEmit` pulito · `npx vitest run` **695 passed | 10
skipped** (erano 680, +15 dal test nuovo senza GPU). Nessun file di
`src/engine/` toccato oltre a due union di tipo: questa riga non tocca il motore
per contratto.

Prossima: it.2 — Q4_K, Q6_K, Q8_0 al banco (shape 35B dall'header dump, celle a
`splits` 1 o 2 dove il superblocco non si divide in 4) e run finale con
`--tag kquant-fase0`.
