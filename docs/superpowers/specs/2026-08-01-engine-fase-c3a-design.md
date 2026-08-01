# Spec — engine-fase-c3a (struttura: dal 4.68 al floor CPU)

Stato: **DRAFT, in attesa di ruling PI** (docket C3a item 6).
Contratto: `.harness/goals/engine-fase-c3a/GOAL.md` (v1 + emendamento 1).
Input di misura: `results/engine/bench-glm-4090-b12-attrib-2026-08-01.json`,
`results/engine/host-gpu-clocks-c3a-it1-2026-08-01.csv`, journal it.1.

---

## 1. Da dove si parte (misurato, non assunto)

Decode GLM-4.7-Flash, slab 12 GiB, p6 (461 token), nGen 64, macchina quiescente:
**4.678 tok/s = 215.0 ms/token**, scomposti così:

| voce | ms/token | come è stata ottenuta |
|---|---|---|
| `gpuBusy` | **78.2** | somma delle durate dei 94 compute pass (timestamp-query) |
| stallo residenza | **53.8** | contatori ExpertCache: pack 41.4 + read 4.8 + upload 7.8 |
| sync/CPU | **83.0** | residuo `wall − gpuBusy − stallo` |
| *(di cui readback vero)* | *7.6* | probe indipendente: mapAsync a GPU scarica 0.16 ms × 46 |
| encode CPU | 3.5 | segmento disgiunto misurato (identità chiusa allo 0.3%) |

Contesto host durante la run: clock SM **1746 MHz medi su 3105 di cap**,
utilizzo GPU **34.6%**, 61-66 °C.

**Il budget del gate.** 13.43 tok/s = **74.46 ms/token**. Con le leve 1 e 2 al
100% di efficacia si arriva a 98.2 ms/token = 10.18 tok/s: `gpuBusy` da solo
(78.2) eccede il budget. Da qui l'emendamento 1 e la quarta leva.

**Il budget per leva** che questa spec si dà (somma = 74.4 ms/token, gate preso
con margine zero — il margine deve venire dai clock):

| leva | da | a | risparmio |
|---|---|---|---|
| 1. repack all'import | pack 41.4 | ~0 | −41.4 |
| 2. sync router | 83.0 | 7.6 (floor) | −75.4 |
| 4. granularità dispatch | `gpuBusy` 78.2 | **≤ 54.5** | −23.7 |
| residuo residenza (read+upload) | 12.4 | 12.4 | — |

**Nota di onestà**: parte della riduzione di `gpuBusy` arriverà "gratis" dai
clock che salgono quando spariscono le bolle (34.6% di utilizzo è la firma).
La spec impone di **attribuire separatamente** le due cause nel report della
fase 4b — altrimenti si attribuisce al lavoro sui kernel un guadagno che è del
governor.

---

## 2. Leva 1 — repack all'import (fase 3)

**Problema.** `ExpertCache.ensure` chiama `packExpertSlab` a ogni miss: legge i
byte GGUF grezzi dei tre tensori e li ri-impacchetta nel layout `[qs | scales]`
del motore. Costa **9.5 ms per miss × 4.47 miss/token = 41.4 ms/token**, sul
path caldo, su CPU.

**Design.** Il repack si fa **una volta sola**, all'import, e produce un secondo
file OPFS in layout slab. `ensure` diventa `read → writeBuffer`, senza CPU.

- **File**: `GLM-4.7-Flash-Q4_0.slabs.bin` accanto al GGUF in OPFS.
- **Contenuto**: i 2.944 slab expert consecutivi, ognuno nel layout di
  `SLAB_DOWN_Q4_0` / `SLAB_DOWN_Q4_1` (invariati: il repack non cambia i byte,
  cambia solo QUANDO avviene). Indicizzati per `(layer, expert)` con l'ordine
  già usato da `expertKey`.
- **Taglia**: 2.688 × 5.308.416 + 256 × 5.505.024 = **15.68 GB**. Con il GGUF
  restano 32.89 GB in OPFS; verificato 1.2 TB liberi sul volume del profilo.
  **Il GGUF resta**: i tensori non-expert e gli harness di conformance/routing
  lo leggono ancora.
- **Header** (primo blocco da 4 KiB, allineato): magic `BLABSLAB`, versione del
  layout (u32), SHA-256 del GGUF sorgente, conteggi e taglie delle due
  size-class, offset della prima slab. **Invalidazione**: se magic, versione di
  layout o SHA non combaciano, il file si rigenera — mai si legge un file di
  cui non si è certi della provenienza.
- **Costo one-shot**: ~28 s di CPU per il repack (2.944 × 9.5 ms) più la
  lettura del GGUF e la scrittura di 15.68 GB. Accettabile: è un costo di
  import, non di inferenza. Va **misurato e riportato** (entra nel TTFT a
  freddo, che è metrica di C3b).
- **Fallimento parziale**: la generazione scrive su un file temporaneo e
  rinomina alla fine, così un'interruzione non lascia un file valido a metà.

**Done-when della fase 3** (da PHASES): `pack CPU < 1.0 ms/token` nel path
caldo, test verdi su produzione dell'artefatto e su invalidazione, correttezza
invariata (argmax ≡ cpuref-f64 256/256).

**Decisione chiesta al ruling**: nessuna — è la leva meno rischiosa e la più
dimensionata. Si segnala solo il consumo disco (+15.68 GB).

---

## 3. Leva 2 — i 46 sync del router (fase 4)

**Problema.** La selezione top-4 vive su CPU perché decide **quali slab
bindare**, e in WebGPU i bind group sono oggetti CPU. Quindi ogni layer MoE
spezza il token in un submit a sé: 46 readback + 47 submit per token.

**Il numero che cambia la scelta**: il readback in sé costa **7.6 ms/token**
(probe), cioè **meno del 10%** degli 83 ms. Gli altri ~75 ms sono
**frammentazione dei submit**: latenza submit→start, bolle in cui la GPU non ha
lavoro accodato, e il governor che di conseguenza tiene i clock a 1746 MHz.
**Conseguenza di design**: l'obiettivo non è "leggere più in fretta", è
**smettere di spezzare il token in 47 submit**. Una soluzione che elimina i
readback ma lascia 47 submit non incassa quasi nulla.

### 3.1 Vincolo WebGPU da verificare PRIMA di scegliere

Le tre vie note dipendono tutte da quanto parco si riesce a bindare in un colpo:

- `maxStorageBufferBindingSize` — oggi il codice negozia
  `min(limite adapter, 2 GiB)`; il limite reale dell'adapter **non è mai stato
  letto** `[VERIFY]`.
- `maxBufferSize` — idem.
- WGSL **non** indicizza dinamicamente tra binding diversi (niente binding
  array in core WebGPU), e i bind group non si costruiscono da GPU.

**Primo task della fase 4, prima di ogni implementazione**: un probe da ~20
righe che stampa i due limiti reali e quanti slab entrano in un binding. Il
risultato seleziona il design. Costo: minuti. Sceglierlo senza è tirare a
indovinare.

### 3.2 Vie, con il criterio di scelta esplicitato

- **(A) Selezione su GPU con parco bindato per size-class.** Il router scrive i
  4 id in un buffer storage; le catene expert leggono i pesi da un binding
  grande indicizzato per offset calcolato **sulla GPU**. Elimina readback *e*
  submit split. Praticabile solo se il parco residente sta in **poche**
  bindings; con slot da 5.3 MB, un binding da 2 GiB ne copre ~394, quindi
  servono ~7 binding per 2.419 slot. Poiché WGSL non indicizza tra binding, la
  variante realizzabile è **dispatchare la catena expert una volta per
  binding** (7×4 = 28 dispatch, di cui 24 escono subito con un test sull'id):
  legale, ma va misurata — 28 dispatch quasi-vuoti costano.
- **(B) Bind del parco per-layer.** I 64 expert di UN layer sono 340 MB: stanno
  in un binding solo. Ma richiede residenza per-layer contigua, cioè tutti e 64
  gli expert di ogni layer residenti — 15.68 GB, oltre la VRAM. Praticabile solo
  a layer parziale, quindi non elimina il caso "expert non residente".
- **(C) Pipelining a profondità >1.** Nel decode **non è applicabile fra token**
  (il token t+1 dipende da t) né fra layer (l+1 dipende da l). Resta applicabile
  al **prefill** (leva 3) e in regime speculativo (fase D). **Va scartata come
  soluzione al sync del decode**, e questa spec lo dichiara per chiudere
  l'opzione: era elencata fra le vie note nel contratto, ma la dipendenza
  sequenziale la esclude.

**Criterio di scelta** (dichiarato come chiede il DONE WHEN §1): si sceglie la
via che minimizza i **submit per token** a parità di correttezza del routing,
perché la misura dice che il costo sta nella frammentazione e non nel readback.
A parità di submit, si preferisce quella con meno dispatch aggiunti.
**Raccomandazione: (A)**, condizionata al probe di §3.1.

**Done-when della fase 4**: ≤ 2 sync/token (o la soglia ammortizzata dichiarata
col suo parametro), routing conformance non peggiore del riferimento C2, **più
la ri-misura obbligatoria di `gpuBusy` e dei clock** che dimensiona la 4b.

---

## 4. Leva 4 — granularità dei dispatch (fase 4b, emendamento 1)

**Il margine.** Per token si leggono **2.22 GB** di pesi (47 layer di attention
+ 46 × (4 expert + shared) + head Q6_K); la banda del device è **576 GB/s**
(9001 MHz × 2 × 256 bit, letta dal device) ⇒ floor memory-bound **3.85
ms/token**. `gpuBusy` misurato è **20× sopra**: 1.816 dispatch a 43 µs medi.
Il tempo GPU non è un muro fisico.

**Prima si misura DOVE, poi si fonde.** La telemetria di fase 1 somma le durate
dei 94 pass in un unico numero. Il primo task della 4b è **spezzare quella somma
per categoria** (attention, router, shexp, catene expert, head) usando i
timestamp già disponibili: cambia l'aggregazione, non la strumentazione. Fondere
kernel senza sapere quale categoria pesa è la ricetta per ottimizzare il 5%.

**Ipotesi da confermare o smentire con quel dato** (in ordine di sospetto):
1. **Dispatch minuscoli serializzati dalle dipendenze RW**: `rmsnorm` gira su
   **1 workgroup**, `silu`/`add` su 24-32; dentro un pass ogni dispatch legge
   ciò che il precedente scrive, quindi il driver li serializza. Sono decine per
   layer con la GPU quasi ferma.
2. **Le 4 catene expert sono 16 dispatch/layer** che fanno la stessa forma su
   pesi diversi: candidate naturali a un dispatch unico su 4 expert (una
   dimensione di griglia in più), che è anche il presupposto di (A) in §3.
3. **Fusione delle code**: `gate/up → silu → down` è una catena a tre stadi con
   attivazioni intermedie in VRAM; fonderla riduce dispatch e traffico.

**Done-when della fase 4b**: `gpuBusy ≤ 54.5 ms/token` (soglia derivata:
74.46 − 12.4 di stallo residuo − 7.6 di floor sync), con dispatch/token e clock
SM medio riportati **e attribuiti separatamente**, correttezza invariata.

---

## 5. Leva 3 — prefill batched M>1 (fase 5)

**Problema.** Il motore non ha un percorso M>1: il prefill esegue 461 forward
sequenziali ⇒ 5.235 tok/s e **TTFT 88.06 s** contro un budget UX di 4 s. Non è
una prestazione 12× peggiore del floor, è una **capacità mancante** (precisato
in docket C2 item 6 e ratificato dal ruling che ha fissato il TTFT).

**Design.** Forward su M token insieme: le GEMV diventano GEMM su M righe;
l'attention usa il percorso chunked già esistente (pattern `prefillplan` del
path Qwen); il MoE è il punto nuovo — **token diversi selezionano expert
diversi**, quindi la catena expert va eseguita per unione degli expert
selezionati nel chunk, con maschera per token. La residenza deve reggere
l'unione (più miss per chunk, ma ammortizzati su M token).

**Condizione di identità** (il gate di correttezza della fase): i logits di un
prompt processato con M>1 devono coincidere con quelli a M=1 entro la tolleranza
fissata qui: **argmax identico su tutte le posizioni** e differenza massima sui
logit entro il rumore f32 già accettato in C2 (`maxAbsDeltaLogit` resta metrica
di scala, mai gate — landmine C2). Test in `npm test`.

**M iniziale**: 16 `[ASSUMED]`, poi taratura sul TTFT. Con M=16 il TTFT
scenderebbe, a parità di tutto il resto, verso ~5-6 s — ancora sopra i 4 s, il
che dice che il prefill ha bisogno anche delle leve 1/2/4, non solo del
batching.

---

## 6. Protocollo di bench, TTFT, telemetria

**Invariato rispetto a C2, per parità**: prompt p6 (461 token), nGen 64, 3
repliche + warmup scartato, mediana come headline, slab 12 GiB, macchina
quiescente con stato host dichiarato nel report. Nessun confronto fuori parità
(lezione B2).

**Repliche di attribuzione separate**: la telemetria liv.1+2 accende
`performance.now()` e i `timestampWrites`; il loro wall **non entra nei gate**.
Headline sempre da repliche a telemetria spenta.

**TTFT — definizione operativa** (come chiede il DONE WHEN §1): tempo dal
**primo forward di prefill** al momento in cui il **primo token generato è
disponibile** (argmax dei logits dell'ultima posizione di prompt), **modello già
residente**. Il TTFT a freddo — che include load, import e popolamento della
slab — è metrica di **C3b** (instant-on) e non si confonde con questo.

**Obbligo di riporto** (contratto, ruling PI 2026-08-01): ogni report contiene
i gap da 30 tok/s, da 60 (thinking) e dal budget TTFT di 4 s. **L'assenza è un
FAIL di checklist**, non un dettaglio di forma.

---

## 7. Rischi dichiarati

1. **Il gate resta irraggiungibile anche con quattro leve.** La forbice misurata
   è 10.2-15.6 tok/s e il gate 13.43 ci sta dentro: l'esito dipende da quanto i
   clock salgono. La clausola di fallback è stata **rimandata dal PI a fine fase
   4** (docket item 2), quando la ri-misura l'avrà sciolta.
2. **§3.1 può invalidare la via (A).** Se i limiti reali di binding non
   consentono di coprire il parco, la leva 2 va ripensata e il ruling torna al
   PI. È il motivo per cui il probe viene prima del codice.
3. **Il repack aggiunge 15.68 GB in OPFS.** Verificato spazio, ma il costo di
   import cresce e si somma al TTFT a freddo di C3b.
4. **La fase 4b è kernel engineering**, che direction §2 mette per ultima nel
   ranking delle leve. Entra qui solo perché la misura dimostra che senza non si
   prende il gate — non perché sia diventata prioritaria in generale.
5. **Correttezza sotto rifattorizzazione pesante**: fondere kernel e cambiare il
   layout dei pesi tocca i path che C2 ha validato bit per bit. Il gate doppio
   (argmax ≡ cpuref-f64 sul campione ratificato + top-1 vs golden ≥ 98.83%) va
   rieseguito a **ogni** fase, non solo alla 6.

---

## 8. Decisioni chieste al ruling

1. **§2** — repack come secondo file OPFS (+15.68 GB), GGUF mantenuto: ok?
2. **§3** — criterio di scelta "minimizzare i submit, non i readback" e
   raccomandazione (A) condizionata al probe: ok? Si ratifica anche lo
   **scarto esplicito del pipelining** (via C) come soluzione al sync del
   decode?
3. **§4** — "prima misurare dove va `gpuBusy`, poi fondere", con l'obbligo di
   attribuire separatamente kernel e clock: ok?
4. **§5** — M=16 iniziale `[ASSUMED]` e condizione di identità "argmax identico
   su tutte le posizioni": ok?
5. Si conferma che il **gate resta 13.43 / 56.58** e che il TTFT ≤4 s resta
   riportato-non-gateato anche in questa fase?
