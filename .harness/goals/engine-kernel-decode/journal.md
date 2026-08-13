# JOURNAL — engine-kernel-decode

## it.1 (2026-08-13, fase 0 + fase 1 prep) — la fase 0 gira; intanto chiudo la trappola dei limiti

Fase 0 lanciata col veicolo dichiarato (`research-campaign`, prediction-gated:
pre-registra → esegui → grade indipendente → memo). Gira in background sul
micro-bench isolato, zero run di modello.

**Intanto, lavoro della riga 1 che NON dipende da quale variante vince**: il
modulo dei limiti diceva il falso sul path Qwen.

**E la mia formulazione dell'item 2 era più grave del vero.** Verificando prima
di scrivere il fix: il path q35 di produzione non passa mai `mlaAttention:
false`, quindi il limite giusto lo otteneva già — ma attraverso un campo che
porta il nome dell'attenzione di GLM, mentre il proprio consumatore restava
invisibile. Non un bug vivo: una trappola, e l'invito a caderci era scritto nel
commento ("un consumatore che quel modello non ha", falso).

**Fix, tre pezzi.** `attnDecodeWorkgroupStorageBytes(ctxMax)` esportata dal file
del KERNEL e non ricopiata nei limiti — una formula sola, dove sta il
consumatore, perché il difetto era esattamente due posti che divergevano. Il
modulo dei limiti la conta SEMPRE, indipendentemente da `mlaAttention`. E un
test che chiude la trappola: spegnere l'MLA non può più far sparire il
fabbisogno di Qwen, a nessun ctxMax, e il valore deve crescere col contesto —
cioè la frase "path Qwen indipendente dal contesto" è ora falsificabile da un
test invece che da una lettura attenta.

Gate: `npx tsc --noEmit` pulito, suite **443|10** (era 442|10: +1 test nuovo),
`tests/gpulimits.test.ts` 19/19.

Nota di traiettoria: questa iterazione NON muove la metrica obiettivo (9,95
tok/s a ctx 6333) ed è dichiarata come tale — è un done-when della riga 1
(portabilità) che si poteva chiudere in parallelo alla fase 0 invece di
aspettarla.

## it.2 (2026-08-13, fase 0 CHIUSA) — le leve esistono: attenzione 33,7x, gemv 4,0x

Campagna prediction-gated conclusa (`research-campaign`: pre-registrazione
committata `1d08061` alle 01:39, misura alle 01:49 — dieci minuti DOPO, e il
grade lo ha verificato dall'ordine dei commit prima di guardare i numeri).

**Verificato da me, non preso per buono**: `git diff 0cf6436..HEAD -- src/engine/`
**vuoto** (la fase non ha toccato il motore, come da vincolo), 32 celle e 0
skipped nel JSON, e soprattutto i **checksum di tutte le varianti entro ~1e-6
dalla baseline** — che è il gate che distingue "più veloce" da "salta lavoro".

**ATTENZIONE** (n=6333, byte unici 51,9 MB = 6333×8192 esatti):

| variante | GB/s | vs base |
|---|---|---|
| forma attuale | 5,2 | 1,0x |
| letture vec4 | 19,7 | 3,8x |
| softmax in streaming | 14,9 | 2,9x |
| streaming + fusione GQA | 12,6 | 2,4x |
| **split del contesto** | **175,3** | **33,7x** |
| split + fusione GQA | 159,8 | 30,7x |

**E il ribaltamento che conta per la fase 1**: `split-gqa` è PIÙ LENTO di
`split`. Fondere le quattro teste che condividono le stesse righe KV — la
dedup che avevo proposto io leggendo il kernel — **toglie parallelismo** una
volta che il contesto è già spezzato. La leva non era la ridondanza 4x: era che
16 workgroup non riempiono la scheda. La mia diagnosi era plausibile e
sbagliata, e la pre-registrazione l'ha fatta cadere invece di lasciarmela
raccontare a posteriori (P3 prevedeva proprio questo, ed è stata confermata).

Controllo `coldkv` su ogni variante: numeri identici ai warm ⇒ non è un
artefatto di L2.

**GEMV q4_0**: `vec4-rows2-sg` vince su tre forme dense (341,3 · 383,6 · 389,5
GB/s contro 83,9 · 89,9 · 98,4 della forma attuale); sulla lm_head — l'unica
forma il cui working set esce dalla L2, e quindi l'unica confrontabile col
motore — tutte le varianti si accalcano fra 255 e 271 GB/s contro 67,9: **4,0x**,
e restano sotto i ~500 G pesi/s di llama.cpp.

**Sonda delle feature, misurata e non assunta**: `subgroups` esposta e concessa,
`subgroupAdd` compila e dà il risultato giusto; `dot4I8Packed` PRESENTE (era la
sub-predizione dichiarata a bassa confidenza, e ha retto); **`shader-f16`
ASSENTE**, confermato — il vincolo scritto nel contratto è vero.

**Proiezione sull'obiettivo** [proiezione, non promessa]: a ctx 6333 la KV vale
65,8 ms su 100,5. A 33,7x scenderebbe a ~2,0 ⇒ token ~36,7 ms = **27 tok/s**.
Con il GEMV a 4x sul corpo (27,1 → ~6,8) ⇒ ~16,4 ms = **61 tok/s**. Il
trasferimento dal micro-bench al motore non è mai pieno: serve la fase 1 per
sapere quanto.

**FASE 0 CHIUSA, regola di stop non scattata.** Il piano non si riscrive: la
riga 1 parte, e parte con un progetto deciso dalla misura — split del contesto
+ softmax in streaming, SENZA fusione GQA.

## it.3 (2026-08-13, fase 1) — la build si e' fermata sul canale, non sul piano

Primo tentativo della riga 1 col conductor: 5 task su 5 BLOCKED. Il messaggio
diceva "conflitto patch-apply", che nel design del conductor significa "il piano
degli `owns` e' sbagliato" — e mi avrebbe mandato a ri-pianificare.

**Era falso, e l'ho verificato prima di crederci**: la patch di T1 e' 40.209
caratteri, lo script la passa all'integratore con `slice(0, 16000)`. Arrivava
tagliata dentro un file nuovo (hunk che dichiara 88 righe, ne porta 77), e
`git apply` rispondeva `corrupt patch at line 369`.

L'integratore aveva gia' fatto il lavoro di diagnosi al posto mio: niente
`--3way`, niente risoluzione manuale, e un `--check` sui soli hunk interi per
mostrare che combaciavano tutti con l'albero. La riga che chiude il caso e'
sua: "non e' un conflitto di contesto, e' il testo della patch che arriva
incompleto".

Fix e ripresa: tolto lo `slice` nella copia dello script di questo run, e
`resumeFromRunId` — i risultati degli implementatori rientrano dalla cache
(nessun lavoro ripetuto), si rifa' solo l'integrazione. Il workflow INSTALLATO
resta rotto: docket item 3, va corretto alla fonte e non e' codice di questo
goal.

Nessun commit durante la ripresa: `git add` sopra una patch applicata a meta'
sarebbe il modo piu' rapido di rovinare un albero pulito.

## it.5 (2026-08-13, fase 1 CHIUSA) — 2,72x a contesto vero: 9,95 -> 27,06 tok/s

Seconda ondata della build SDD integrata (5 task, 24 agenti). Verificato da me,
non preso per buono.

**I gate, tutti eseguiti ora**: ktest **100 PASS / 0 FAIL** · `q35-model-4b-argmax`
argmax **IDENTICO** all'oracolo (gate secco) · `q35-attn-full-real-blk3` contro
il cpuref f64 a **L2rel 2,07e-7** — meglio della tolleranza di ieri, nonostante
lo streaming cambi l'ordine delle somme · `dense-batch-attn-chunk` **BIT-IDENTICO**
(il prefill non e' stato sfiorato, come da contratto) · suite **472|10** · tsc
pulito.

**LA MISURA** (stesso host, stessa configurazione, `--vram-gib 8`, quiescent):

| | prima | dopo | |
|---|---|---|---|
| ctx 388 | 38,72 ms | **35,69** | 1,08x |
| ctx 6333 | 100,52 ms | **36,95** | **2,72x** |
| tok/s a ctx 6333 | 9,95 | **27,06** | |
| pendenza | 10,40 us/pos | **0,21** | 50x piu' piatta |
| scansione KV | 6,3 GB/s | **308** | 1,4% -> 71% del picco |

**Il token e' quasi indipendente dal contesto**: 35,69 a 388 posizioni contro
36,95 a 6333. La cosa che dominava il decode a contesto vero — 65,8 ms su 100,5
— e' scesa a ~1,3 ms.

**Obiettivo NON ancora raggiunto**: 27,06 contro 30 a ctx >= 6000. Mancano 3,
e la riga 2 (GEMV quantizzati, 4,0x nel micro-bench sul corpo da ~27 ms) e'
esattamente la leva che li copre — con margine, se anche solo meta' del fattore
si trasferisce.

**Nota di metodo, perche' e' la seconda volta in due giorni**: la fase 0 aveva
predetto la vittoria dello `split` (175,3 GB/s) e il micro-bench diceva 33,7x
sul kernel isolato. Nel motore il kernel rende 308 GB/s sulla scansione, ma il
guadagno END-TO-END e' 2,72x perche' il resto del token non e' cambiato. Il
micro-bench non mente e non basta: dice quanto vale il pezzo, non quanto vale
il tutto. Il numero del contratto e' sempre stato il secondo.

Artefatto con prima/dopo nello stesso JSON, hostState dichiarato e i gate:
`results/engine/attn-split-fase1-4090-2026-08-13T02-15-18-710Z.json`.
