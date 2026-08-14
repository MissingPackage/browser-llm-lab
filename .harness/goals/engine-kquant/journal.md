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
