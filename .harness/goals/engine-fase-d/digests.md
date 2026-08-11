# Digests — engine-fase-d

## it.0 (2026-08-10)

Goal aperto dopo il ruling PI sulla parità di ottimizzazioni. 9 fasi,
ordine del blocco A dal ROI MISURATO in q1; regola di misura scritta NEL
PIANO (micro-bench durante, bench pieni solo a fase 6 e 8).

## it.1 (2026-08-10)

`moe.ts` unificato: tabella di geometria dei quant, UN router con due
configurazioni (GLM sigmoid+bias+1.8 / Qwen softmax), UN builder di slab
(le classi GLM ora derivate, byte storici verificati). Nato il GATE
STRUTTURALE (ratchet delle duplicazioni). GLM bit-identico, ktest 84/84.

## it.2 (2026-08-10)

`residency.ts` parametrica NEI DERIVATI (MoeModelConfig → chiavi, parco,
minimi, slotTable; ExpertClass da unione chiusa a stringa). GLM
bit-identico (ktest 84/84, arena-vs-slotrange bit-a-bit). **Verifier PASS
con tre rilievi accolti**: il titolo era un overclaim (il MOTORE della
cache è ancora GLM-shaped → aggiunto un guard che rifiuta le config non
onorate); il gate strutturale dichiarava il falso sul router (cpuref.ts
era un terzo router non elencato → eccezione riscritta come CATEGORIA);
i predicati del gate sono firme testuali, non invarianti (docket item 4).
Next: it.3 = motore della cache cfg-driven + gate irrobustito.

## it.3 (2026-08-10)

Il MOTORE della residenza è cfg-driven: stati di classe, ripartizione del
budget, arena, repin, stats, destroy e il PATH CALDO (`ensure`) vengono
dalla config, non da `G.*`. Guard di it.2 rimosso: la config è onorata.
I campi compat legacy ora FALLISCONO sui K-quant invece di fabbricare un
offset finto. q35 non ha più bisogno di `slotsOverride`. GLM bit-identico
(ktest 84/84), suite 387. Restano: gate strutturale da irrobustire (4b) e
migrazione di q35gpumodel.

## it.4 (2026-08-10)

Gate strutturale da FIRME TESTUALI a INVARIANTI non aggirabili: (A) chi
nomina i tensori expert del GGUF e crea buffer GPU deve importare la
meccanica; (B) il clamp del router può stare solo in moe.ts. Allowlist con
razionale, e le voci "DEBITO NOTO" (solo q35gpumodel) sono ciò che la
fase 1 deve far sparire. Il gate PROVA SE STESSO su offender sintetici —
inclusa la variante di spaziatura che sfuggiva prima. ktest 84/84, suite
391. Docket item 4 chiuso per intero; resta la migrazione di q35gpumodel
(it.5), che chiude la fase 1.

## it.5 (2026-08-10)

Il verifier aveva bocciato il gate di it.4 ESEGUENDO tre evasioni: era una
congiunzione di indizi, non un invariante. Riscritto in tre invarianti
INDIPENDENTI su atti singoli (allocazione GPU / nomi expert anche costruiti
per parti / clamp del router), senza esenzioni per import, con guard
anti-scansione-vuota e col ciclo delle asserzioni a sua volta testato. Ho
riprovato io le tre evasioni: tutte rosse. Docket item 4 CHIUSO. Resta la
migrazione di q35gpumodel (it.6), che chiude la fase 1.

## it.6 (2026-08-10)

Terza bocciatura del gate (5 evasioni eseguite, una delle quali — un router
Qwen legittimo — NON catturabile da nessuna scansione). Diagnosi: stavo
cercando una porta che non esiste, invece di crearla. **L'invariante si
sposta nel sistema di tipi**: marchio di conio su `SlotRef` (solo
residency.ts lo CONIA — 11 sonde ostili del verifier rifiutate) + test di
tipo che va rosso se il marchio sparisce. Limite dichiarato: ferma la
contraffazione, non l'indifferenza — q35gpumodel oggi ignora SlotRef e tsc
è verde; il marchio diventa portante con la migrazione (it.7). La scansione resta un RATCHET, con la
pretesa ridimensionata per iscritto. N3/N4/N5 chiuse (scansione su tutto
src/, ancorata, estensioni; predicato di allocazione corretto). Lezione:
tre iterazioni su un poliziotto mentre ciò che elimina la duplicazione è la
MIGRAZIONE — che è it.7 e chiude la fase 1.

## it.7 (2026-08-10) — FASE 1 CHIUSA

`q35gpumodel` non possiede più niente della residenza expert: arena, LRU,
ensure, repack e router propri sono spariti, sostituiti dalla `ExpertCache` di
GLM con una `MoeModelConfig` dedotta dal GGUF e da `routerSelect`. Nato
`q35expertstore.ts`, gemello di `expertstore.ts`: le tre voci DEBITO NOTO sono
sparite davvero (gate: `debiti == []`, 21/21). Prove: parità slab CPU-only 6/6
(stessi byte agli stessi offset), ktest 84/84 con GLM bit-identico
(arena-vs-slotrange maxRel 0), e conformance smoke 35B sul path migrato
**top1 5/5** con hits/misses/uploadedBytes IDENTICI al run pre-migrazione
(8846 / 3314 / 5.916.950.528). Corretto un bug del meccanismo condiviso: il
controllo del limite di binding mancava il segmento down dei K-quant.

## it.8 (2026-08-10) — fase 2: pack 6,24x

Misurato prima di scrivere: il repack K-quant costava 3,50 ms/miss = 22% del
prompt sullo smoke 35B. Due cause: `repackKQuant` ricostruiva le parole con un
`|=` per byte (ma su little-endian e' una COPIA travestita) e
`packExpertSlab` passava da un array temporaneo (ogni byte toccato 3 volte).
Fatto: memcpy + `repackKQuantInto` che scrive dritto nello slab. Pack
11.585 -> 1.856 ms (**6,24x**), 3,50 -> 0,56 ms/miss, prompt 53,0 -> 42,5 s,
con hits/miss e top1 IDENTICI. ktest 84/84 con valori identici cifra per
cifra (il repack veloce e' bit-a-bit lo stesso). Nuovo test con oracolo
scalare indipendente + il caso del padding Q6_K. Il done-when e' centrato
nell'obiettivo ma non nella lettera (il repack non e' uscito dal path, e'
diventato quasi gratis dentro): **docket item 5** al PI, col parere che
spostarlo all'import non conviene (~4% al prezzo di ~18 GB su disco).

## it.10 (2026-08-10) — fase 3 sui densi [NUMERO CORRETTO IN it.12: -3,3, non -15,0]

Misurato prima: 50,5 ms/token con sync per token contro 35,4 accodando senza
attese = 15,0 ms di pura serializzazione (il `readbackMs` di 44,6 NON e' il
costo dei 604 KB: e' l'attesa che include il lavoro GPU). Fatto
`decodeBatch`: K token teacher-forced in un submit, argmax su GPU, un
readback di K·4 byte. **35,5 ms/token**, sul tetto misurato. Gate secco:
argmax identici al path a readback su tutti i 39 token, dentro il `pass` del
ktest. Errore mio corretto in corsa: batchare anche il prefill era piu'
LENTO (li' non c'era attesa da togliere e l'argmax su GPU si aggiungeva).
Il MoE NON puo' usarlo — 41 submit/token per il routing su CPU: docket
item 8, il pezzo piu' grosso rimasto.

## it.11-12 (2026-08-10) — arena K-quant, e la correzione di un numero mio

it.11: `gemvQ4K`/`gemvQ6K` accettano l'indirizzamento d'ARENA (slot da `Sel`,
buffer bindato intero), riusando la testa di GLM. Gate nuovo sul 35B reale:
l'uscita degli 8 expert coi due regimi di indirizzamento e' **BIT-A-BIT
identica** su entrambe le classi. ktest 86/86.

it.12: **il -15,0 ms/token di it.10 era un artefatto di misura** — il braccio
"con sync" era la prima passata dopo il load (a freddo), quello "batch" la
terza (a caldo): ~8 ms/token di warm-up tutti nel braccio lento. Micro-bench
riscritto con warm-up scartato, bracci interleavati, 3 ripetizioni, mediana e
dispersione. **Valore onesto: -3,33 ms/token (-8,3%)**, spread 0,19/0,10. La
serializzazione vera a caldo e' 4,58 ms/token, non 15, e il batch ne recupera
il 73%. Il done-when della fase 3 chiedeva >= -5,1: **NON e' soddisfatto**,
docket item 9 al PI. Regola nuova per l'harness a docket item 10.

## it.13 (2026-08-10) — fase 3b fetta 2: router+resolve Qwen su GPU

`routerTopKWgsl` parametrico sul gating (sigmoid GLM / softmax Qwen), col
binding `bias` tenuto in entrambi i casi e un buffer di zeri per chi non lo
usa — cosi' il layout non dipende dalla famiglia e per GLM il testo emesso
resta byte-identico. Gate nuovo su 64 estrazioni 256x8: **0 flip d'insieme,
0 flip d'ordine, 0 resolve errati**, errore sui pesi 2,32e-7 contro soglia
1e-5. Il resolve si prova con una slotTable finta che ha UN MISS di
proposito: slot e flag sono interi, confronto secco. ktest 87/87. Progetto
della fetta 3 (cablaggio in q35gpumodel) scritto nel journal PRIMA di
iniziarla, con l'ordine di montaggio 3a/3b/3c e il rischio identificato
(l'arena cappa i buffer anche col limite di BINDING, non solo di taglia).

## it.14 (2026-08-11) — fase 3b fetta 3a: q35 in regime d'arena

Ruling PI su docket item 9 incassato (opzione (a)): la fase 3 sui densi e'
chiusa col numero misurato, -3,33 ms/token, e la soglia ex-ante cade.

`q35gpumodel` monta la `ExpertCache` in regime `arena + slotTable`: i buffer
di classe si bindano INTERI e l'indirizzo dello slab lo ricava il kernel dallo
slot che legge in `Sel`. Bind group layout ESPLICITO (`hasDynamicOffset` non
esiste con `layout: "auto"`) ⇒ **3 bind group per classe** invece di 3 per
slot. Le costanti dell'ABI di `Sel`/`MoeIdx` si spostano accanto alle struct
WGSL: una sola verita' per le due famiglie.

**GATE, due bracci nella stessa sessione (worktree su HEAD vs modificato,
stesso golden, stesso budget 10 GiB dichiarato)**: **istogramma di routing
IDENTICO chiave per chiave** (3314 chiavi, 12.160 selezioni = 38 token x 40
layer x 8), argmax generati identici, top1 5/5, hits/misses 8846/3314,
uploadedBytes identici, 782 dispatch/token. Il routing e' il numero che porta
il peso: dipende dallo stato nascosto di ogni layer, quindi un offset d'arena
sbagliato divergerebbe subito.

**Rischio di it.13 confermato reale**: la classe q4k si spezza in 5-6 buffer ⇒
8-9 storage binding contro i 7 del path Qwen. `q35conf.worker` ora negozia
`arenaNeeds` (buffer e finestra) con la config dedotta dal GGUF; senza, la
pipeline sarebbe stata invalida al primo bind.

**OOM a 12 GiB**: prima diagnosi, poi codice — il braccio PRIMA (HEAD, non
modificato) muore allo stesso budget sullo stesso host ⇒ non e' la modifica, e'
il tetto VRAM (docket item 11: il budget del 35B non e' ctx-aware come quello
di GLM). Tempi non misurati per bene (un campione per braccio): non si
dichiara nulla.

ktest 87/87, suite 410|9, tsc pulito.

## it.15 (2026-08-11) — fase 3b fetta 3b: router GPU in ombra sui layer VERI

`Sel` raddoppia (produzione + ombra nello stesso buffer), il router+resolve di
it.13 gira nello stesso submit del segmento statico e scrive l'ombra; la
selezione di produzione resta della CPU e si confrontano. Opt-in
(`routerShadow` / `--shadow`), spento di default.

**Smoke 35B, 1520 confronti (38 token x 40 layer), 12.160 selezioni**: 0 flip
d'insieme, 0 flip d'ordine, errore max sui pesi 3,80e-7 (soglia 1e-5),
**slotMismatch 0**, miss GPU/CPU 3314/3314 con 0 disaccordi. Produzione
IDENTICA a it.14 (routing, argmax, top1): l'ombra e' davvero un'ombra.

**Il numero che qualifica il gate e' il MARGINE, non gli zeri**: il caso
peggiore aveva 1,0e-5 fra l'ultimo expert preso e il primo scartato (sui
logit), cioe' una decade sopra la risoluzione dell'f32 a quella scala. Sotto
~1e-6 sarebbe testa-o-croce e in questo corpus non e' capitato: non si conclude
"non flippera' mai", si conclude "su 12.160 selezioni vere non ha flippato, col
caso peggiore a 1e-5". Da rifare sul corpus pieno alla fase 6.

`slotMismatch = 0` prova il resolve end-to-end: per ogni expert residente la
GPU ha letto dalla slotTable lo slot che la CPU ha poi usato — `tableBase` e il
mantenimento della tabella dentro `ensure` sono giusti sul modello vero, non
solo nel caso finto di it.13.

tsc pulito, suite 410|9, 0 gpu-error.

## it.16 (2026-08-11) — misura prima della 3c: chi e' sporco, e quando

Il path ottimistico di GLM ha una PRECONDIZIONE (`optimisticMinResidency`
0.8): sotto quella residenza ogni token e' sporco e il replay costa piu' del
sync. Il 35B a 10 GiB sta al 58,7% del parco — sotto la soglia. Misurato
invece di assumere, con `--misstrace` (due passate sullo stesso prompt:
`resetState` azzera lo stato ricorrente, non la cache expert).

**Smoke 35B, 39 token, 320 selezioni/token**: pass FREDDO = **39/39 token
sporchi**, mediana 68 miss/token, hit rate 73,2%. Pass CALDO = **0/39 token
sporchi, 0 miss, hit rate 100%**.

Due conclusioni opposte da tenere insieme: (1) a residenza raggiunta il decode
ottimistico da' **1 submit/token**, che e' alla lettera il done-when — ed e'
misurabile su questo stesso smoke con la seconda passata; (2) a cache fredda
il repair+replay e' il REGIME, non il caso limite, e il path ottimistico nudo
sarebbe una regressione sul prefill.

⇒ la fetta 3c diventa DUE cose: il path a submit unico (Sel di produzione dal
router, dirty, hiddenCkpt, repair+replay) E una policy d'ingresso tarata sui
numeri (sync finche' i miss/token stanno sopra soglia). Gate della 3c: due
passate, submit/token riportati SEPARATI per freddo e caldo — mediarli
nasconderebbe il fenomeno. E la 3c non sara' bit-identica per costruzione: i
pesi vengono dal router GPU (3,80e-7 misurato in it.15), quindi il gate e'
l'argmax identico + routing e miss invariati, come dice il contratto.

## it.17 (2026-08-11) — fetta 3c: il token intero in un submit

La `Sel` di produzione la scrive il ROUTER SU GPU (resolve dalla slotTable nello
stesso dispatch, `dirtyB` sui miss). 40 segmenti statici + 40 router + 320
dispatch expert in UN encoder, UN submit; il routing si ricostruisce alla fine
dalla `Sel` letta in coda. I kernel expert non cambiano di una riga — e' cio'
per cui l'indirezione della fetta 3a esisteva.

Due pezzi nuovi: `clearBuffer(moeAcc)` dentro l'encoder (la `writeBuffer` e'
ordinata prima dell'INTERO submit: i 40 layer vedrebbero un solo azzeramento) e
`axpySelWgsl`, l'axpy col peso preso da `Sel.w` — i pesi ora nascono su GPU.
Piu' `ExpertCache.noteResidentHit`: senza, la LRU non verrebbe mai toccata (nel
path nuovo `ensure` non si chiama mai) e degraderebbe a FIFO.

**Smoke 35B, 39 token, 10 GiB** — tre regimi separati:
submit/token **81 → 1**, readback/token **41 → 1**, argmax **39/39 identico**,
routing **identico chiave per chiave** (3341 chiavi, 0 diff), miss 0 vs 0.
ms/token A PARITA' di residenza (bracci interleavati, prima coppia scartata,
docket item 10): **143,49 [142,51-145,94] → 71,50 [71,05-71,57] = −71,99
(−50,2%)**. Il primo giro del gate confrontava freddo contro caldo (1192,9 vs
71,5): e' servita una terza passata perche' il numero fosse onesto.

Correzione di fatto al docket item 8: i submit erano **81**, non 41 (ogni layer
MoE ne fa due). I readback erano giusti.

NON coperto, e precondizione della prossima fetta: il replay di GLM **non e'
portabile su un modello ricorrente**. 30 layer su 40 sono deltanet e aggiornano
`convSt`/`stateS` IN PLACE: un replay li applicherebbe due volte, senza errore e
con numeri plausibili. Via d'uscita: snapshot per token (62,8 MiB, calcolati) +
restore dei soli layer >= firstDirty, da MISURARE contro i 71,5 ms/token.

## it.18 (2026-08-11) — repair+replay, e la fase 3b si chiude

Il replay di GLM non era portabile: 30 layer su 40 sono deltanet e aggiornano lo
stato IN PLACE, quindi rigiocarli lo applicherebbe due volte (senza errore, con
numeri plausibili). Snapshot per token — costa poco perche' stato-all-ingresso-
del-layer == stato-all-ingresso-del-token — 62,8 MiB, restore dei soli layer
>= firstDirty.

BUG PRESO DAL GATE: avevo usato `startLayer === 0` come sinonimo di "primo
giro". A cache vuota il primo layer sporco E' lo zero, quindi il replay finiva
nel ramo dello snapshot e `x` restava l'uscita del giro prima invece
dell'embedding. Ipotesi formulata dal codice prima di toccarlo, corretta
separando i due concetti, ri-verificata contro lo stesso repro.

**Smoke 35B, 39 token, 10 GiB** — ottimistico FREDDO: 3,79 submit/token, 39/39
token sporchi, 109 replay (2,79/token, 32,3 layer su 40 ciascuno = 80,7% del
token rigiocato), 3742 miss contro i 3341 del sync (**+12,0%**: un expert
riparato puo' non finire nella Sel definitiva — 910 fetch mai usati), repair
484,3 ms/token = 65,6% del token, 738,6 ms/token. CALDO: **81 → 1 submit/token,
41 → 1 readback**, 141,18 → 72,58 ms/token (**−68,60, −48,6%**).
**argmax IDENTICO 39/39 anche fra freddo-ottimistico e sync-caldo**: 109 restore
dello stato ricorrente e la deriva non c'e' — su un modello ricorrente sarebbe
cumulativa e divergerebbe subito.

La previsione di it.16 ("a freddo l'ottimistico e' una regressione") NON e'
confermata: 738,6 contro 1192,9 del sync freddo. Ma sono un campione per braccio
in due run diversi: per il docket item 10 non e' una misura, e la soglia della
fase 5 va tarata su un bench fatto apposta.

FASE 3b CHIUSA sui suoi quattro done-when; (d) vale a residenza piena, che e' il
campione che il contratto nomina — a freddo il +12% di fetch e' inerente al
meccanismo e si pubblica come fatto.
