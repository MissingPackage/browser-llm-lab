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
