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

## it.10 (2026-08-10) — fase 3: -15,0 ms/token sui densi

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
