# HANDOFF — browser-llm-lab   (aggiornato 2026-08-18)

## 1. Next decidable

**SPIKE (1) E (2) FATTI: il GEMM multi-riga in arena vale — ma per il PREFILL
(2,11× sul segmento expert), NON per lo spec-dec**, che a M=2 costa 1,68× per
1,5 token utili e quindi *perde* l'11%. Break-even α ≥ 0,68 contro un'acceptance
misurata di 0,50. Documenti (con grafici):
`docs/deep-dive/{costm-ricerca,router-overlap}-2026-08-18.md`, docket item 25-29.

**LA DECISIONE CHE ASPETTA TE**: si scrive quel kernel? Il divieto
`batch && arena` (`wgsl.ts:2176-2190`) è per costruzione. Se sì, la politica
giusta è **per-expert** (`m_e ≥ 2` → multi-riga), non per-layer — la multiplicity
è già nota all'encode in `pinUnion`/`encodeExperts`.

Per decidere lo **spec-dec** mancano due misure: l'**overlap nel DECODE** (~10
min, stesso script, i 128 token golden teacher-forced — finora è misurato solo il
prefill) e l'**acceptance della testa MTP del 35B**, mai misurata (quella del 4B
non si trasferisce: stessa lezione del 91,92% di GLM).

### ⚠️ IN UNA SESSIONE FRESCA: APRI UN GOAL E FAI PARTIRE IL LOOP NOTTURNO

Il lavoro di ottimizzazione **non è una sequenza di spike: è un goal**. La
classifica completa, col ragionamento e cosa NON fare, sta nel docket **item 21**
(§«la classifica») e **22**. In ordine di valore:

1. **prefetch lookahead** — l'unica che compra **bit** invece di ms: porta una
   quant più ricca sopra i 30 tok/s. Nessun concorrente ce l'ha.
2. **riscrittura GEMV quantizzati** — vec4 + `subgroupAdd` + 2-4 righe/WG +
   `dot4I8Packed`: headroom **3,7×** dimostrato.
3. **attenzione a contesto lungo** — il kernel KV gira all'**1,4% del picco**; a
   ctx 6333 il 4B crolla 25,9 → 9,95. KV f16 (`pack2x16float`) è WGSL core.
4. **GEMM multi-riga in arena** (2,11× sul prefill) · 5. **i ~11 ms fuori dai
   pass** (sonda prima) · 6. **selezione kernel per tier** (item 28).
7. slab in sotto-range · storage/cache (leve 3-4 di WebLLM): prodotto, non regime.
8. **LoRA-over-GGUF hot-swap** — terzo asse dell'obiettivo, gap verificato.

**Aperte e in attesa di TE**: item 24 il banco non copre `q2_K` · item 26 tre
celle `q8_0/splitk-idot` fuori tolleranza, errore **crescente con M** · item 21
subgroup-matrix · traduzione (item 16) · `webgguf-bench` · tre repliche per il
Firefox.

**Leggi `MECCANISMI.md` prima di progettare e `VALUTAZIONE.md` prima di credere a
un numero** (entrambi in `docs/architettura/`).

## 2. Mappa

**Destinazione** (item 22): tre assi — il motore più veloce sui modelli che
supportiamo; girare bene sui browser vanilla; hot-swap LoRA-over-GGUF.
**Distanza oggi**: 35B a **34,6 tok/s** (riferimento, 3 repliche) · 0.5B a
**322,6** contro WebLLM 89,5-117,7 · Chrome nudo **327,3** · Firefox: 35B a
**9,97**, 0.5B **79,6 contro 9,9** — degradiamo 4× dove loro 9× · LoRA da fare.

**Decisioni prese** — indice, nel docket citato sopra: 14 Apache-2.0 · 15 pesi da
HF · 16 pubblico in inglese · 17 npm + Space HF · 18 split (fonte di verità = il
lab) · 19 matrice dispositivi · 20 nome **`webgguf`** · 22 obiettivo a tre assi ·
23 split compiuto · **28 portabilità = si alloca quello che il device concede**.
Cosa va in quale repo: `docs/publishing/split-manifest.md`.

**Nebbia**: regime di Firefox su una run sola · traduzione mai dimensionata ·
primo utilizzo (12,6 GB) mai cronometrato · contesto lungo del 35B (8k misurati
contro 262k) · generalizzazione dell'overlap a code/json (~20 min).

**Fuori scope**: subgroup-matrix · TVM · sviluppo su Mac · policy `tier` ·
raggruppamento I/O · `idot` nel decode · slab · vocab ridotto · **M oltre 16 per
il decode** (residuo 1,6×, non un altro 30×).

## 3. Landmines

- **`--help` non esiste sui runner e fa PARTIRE il bench.** Flag dal sorgente.
  Mai due runner GPU insieme; quiescenza sui processi **browser**. Mai pipe
  sull'output: maschera l'exit code.
- **Mai backtick dentro `git commit -m "..."`**: la shell li ESEGUE, e `batch` è
  un comando vero che schedula un job e si appende su stdin. Fatto due volte il
  18. Usare heredoc `<<'EOF'`.
- **Le leve `kfan`/`splitk` nascono SPENTE** (le accende la chat, non il bench);
  il path bench di `q35conf` dichiara di sé «frame PRE-ottimizzazioni»; il
  thinking è ACCESO di default → passare `--thinking` esplicito.
- **Non aggregare celle di shape diverse** (minimo per shape+M) e **leggi il
  MARGINALE, non la media**: una media con intercetta fissa cala per sempre e
  sembra headroom che non c'è. Tre numeri sbagliati in due giorni, fra i due.
- **Non riscrivere i registri**: si correggono le *istruzioni vive*, mai journal,
  docket e `results/*.json`. `~/Projects/webgguf/.harness/` è gitignorato apposta.
