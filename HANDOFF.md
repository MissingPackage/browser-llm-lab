# HANDOFF — browser-llm-lab   (aggiornato 2026-08-18)

## 1. Next decidable

**SPIKE (1) E (2) FATTI. La conclusione non è quella che sembrava a metà strada:
il GEMM multi-riga in arena vale — ma per il PREFILL, non per lo spec-dec.**
Documenti: `docs/deep-dive/costm-ricerca-2026-08-18.md` e
`router-overlap-2026-08-18.md` (con grafici). Docket item 25-29.

| | prefill | spec-dec |
|---|---|---|
| M | **16, strutturale** | 2-4 |
| righe utili | tutte | solo le accettate, α≈0,5 |
| guadagno segmento expert | **2,11×** | **0,89× — perde** |

Il break-even dello spec-dec è **α ≥ 0,68 a M=2**; l'acceptance misurata (sul
4B) è 0,50. Restano **due incognite misurabili** prima di poterlo decidere:
1. **overlap del router nel DECODE** — ~10 min, stesso script con i 128 token
   golden teacher-forced. Finora è misurato solo il **prefill**.
2. **acceptance della testa MTP del 35B** — mai misurata, e quella del 4B non si
   trasferisce (stessa lezione del 91,92% di GLM).

**LA DECISIONE CHE ASPETTA TE**: si scrive il GEMM multi-riga in regime d'arena?
Il divieto `batch && arena` (`wgsl.ts:2176-2190`) è per costruzione. Se sì, la
politica giusta è **per-expert** (`m_e ≥ 2` → multi-riga), non per-layer: la
multiplicity è già nota all'encode in `pinUnion`/`encodeExperts`.

**Altre aperte** (titoli; il contenuto è nel docket): item 24 il banco non copre
`q2_K`, il quant che consegniamo · item 26 tre celle `q8_0/splitk-idot` fuori
tolleranza, e l'errore **cresce con M** · item 21 subgroup-matrix · item 22
classifica delle ottimizzazioni · traduzione del repo motore (item 16) ·
estrazione di `webgguf-bench` · tre repliche per rendere citabile il Firefox.

**Leggi `MECCANISMI.md` prima di progettare e `VALUTAZIONE.md` prima di credere a
un numero** (entrambi in `docs/architettura/`).

## 2. Mappa

**Destinazione** (docket item 22): tre assi — il motore più veloce al mondo sui
modelli che supportiamo; girare bene sui browser vanilla; hot-swap LoRA-over-GGUF.
**Distanza oggi**: 35B a **34,6 tok/s** (riferimento, 3 repliche) · 0.5B a
**322,6** contro WebLLM 89,5-117,7 · Chrome nudo **327,3** (asse 2 soddisfatto su
Chrome) · Firefox: 35B a **9,97**, 0.5B a **79,6 contro 9,9 di WebLLM** —
degradiamo 4× dove loro degradano 9× · LoRA non iniziato, gap verificato.

**Decisioni prese** — indice, tutte in
`.harness/goals/engine-velocita-decode/docket.md`:
- 14 Apache-2.0 · 15 pesi da HF · 16 pubblico in inglese · 17 npm + Space HF ·
  18 split (fonte di verità = il lab) · 19 matrice dispositivi · 20 nome
  **`webgguf`** · 22 obiettivo a tre assi · 23 split compiuto · **28 portabilità
  = si alloca quello che il device concede; il minimo di spec è il PAVIMENTO**
- Cosa va in quale repo: `docs/publishing/split-manifest.md`

**Nebbia**: il regime di Firefox su una run sola · il costo della traduzione mai
dimensionato · il primo utilizzo (12,6 GB) mai cronometrato · la pendenza a
contesto lungo del 35B (misurato fino a 8k contro 262k) · la generalizzazione
dell'overlap a code/json (test di rottura, ~20 min).

**Fuori scope**: subgroup-matrix · TVM · sviluppo su Mac · policy `tier` ·
raggruppamento I/O · `idot` nel decode · slab · vocab ridotto · **spingere M
oltre 16 per il decode** (il residuo è 1,6×, non un altro 30×).

## 3. Landmines

- **`--help` non esiste sui runner e fa PARTIRE il bench.** Flag dal sorgente.
- **Mai due runner GPU insieme**; quiescenza sui processi **browser**.
- **Mai pipe sull'output di un runner**: maschera l'exit code.
- **Mai backtick dentro `git commit -m "..."`**: la shell li ESEGUE. Il 18 hanno
  mutilato un messaggio e lanciato `batch`, che è un comando vero e schedula
  job. Usare heredoc `<<'EOF'`.
- **Le leve `kfan`/`splitk` nascono SPENTE**: le accende la chat, non il bench.
- **Il path bench di `q35conf` dichiara di sé** «frame PRE-ottimizzazioni».
- **Il thinking è ACCESO di default**: passare `--thinking` esplicito.
- **Non aggregare celle di shape diverse** in un artefatto multi-variante: prendi
  il minimo per (shape, M). Mi ha prodotto due numeri sbagliati in due giorni.
- **Leggi il MARGINALE, non la media**: una media con intercetta fissa cala per
  sempre e sembra headroom che non c'è.
- **Non riscrivere i registri**: si correggono le *istruzioni vive*, mai i
  journal, i docket e `results/*.json`.
- `~/Projects/webgguf/.harness/` è **gitignorato di proposito**.
