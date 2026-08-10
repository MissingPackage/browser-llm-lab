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
