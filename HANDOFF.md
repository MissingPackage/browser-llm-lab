# HANDOFF — browser-llm-lab   (aggiornato 2026-08-17)

## 1. Next decidable

**Gli spike economici sullo spec-dec, in questo ordine** — già autorizzati dal PI
(«prima gli spike, i micro-bench e le prove in chat; la misura intera solo alla
versione finale, come numero da paper»):

1. **curva `cost(M)`** per M ∈ {1,2,4,8,16} sulle shape del decode (K=2048 e
   K=512 dei gemv expert), celle nuove in `src/microbench/`. **Mezza giornata,
   zero run di modello.** Decide tutto il resto.
2. **overlap del router top-8 a distanza 1-4 sul 35B** — mezza giornata, riusa il
   router esistente. Decide se lo spec-dec paga anche sul MoE o solo sul denso.

Perché contano: il verdetto «spec-dec più lento» fu misurato **coi kernel
vecchi**, dove M era quasi ininfluente (1,22x a M=8). Coi nuovi, `wgsl.ts` misura
splitk-idot a **M=16 in 0,0376 ms contro 0,0698 a M=1** — sedici righe costano
meno di una. Se la curva conferma il ginocchio a M=2, una riga parcheggiata torna
in gioco.

**Aperte e in attesa di TE** (titoli; il contenuto sta nel docket):
item 21 priorità del subgroup-matrix · item 22 ruling sulla classifica delle
ottimizzazioni · traduzione in inglese del repo motore (item 16) · estrazione di
`webgguf-bench` · `paper/` non esiste, quindi il terzo repo non è estraibile.

**Leggi `docs/architettura/MECCANISMI.md` prima di progettare, e `VALUTAZIONE.md`
prima di credere a un numero.**

## 2. Mappa

**Destinazione** (docket item 22, dichiarata dal PI): tre assi — il motore più
veloce al mondo sui modelli che decidiamo di supportare; girare bene sui browser
vanilla; hot-swap di adapter LoRA-over-GGUF.
**Distanza oggi**: 35B MoE a **34,6 tok/s** (riferimento, 3 repliche) · 0.5B a
**322,6** contro WebLLM 89,5-117,7 · Chrome nudo **327,3**, quindi l'asse 2 è già
soddisfatto su Chrome · LoRA non iniziato, ma registrato come gap verificato
(`ideas-ledger.md:74`: WebLLM no, wllama TODO, MediaPipe solo Gemma).

**Decisioni prese** — indice, una riga ciascuna, tutte in
`.harness/goals/engine-velocita-decode/docket.md`:
- item 14 licenza **Apache-2.0** + NOTICE · 15 pesi da HF · 16 pubblico in
  inglese · 17 distribuzione **npm + Space HF** · 18 split, fonte di verità = il
  lab · 19 matrice dispositivi
- item 20 nome **`webgguf`**, namespace personale, scelta reversibile
- item 21 mappa delle capability + consulenza · 22 funzione obiettivo a tre assi
  · 23 il 35B su Firefox e lo split compiuto
- Cosa finisce in quale repo: `docs/publishing/split-manifest.md` — §7 sono i tre
  difetti che solo **eseguire** lo split ha trovato.

**Nebbia** — non ancora specificato:
- il **regime di Firefox è ignoto**: due turni non sono regime, 4,94 non si
  estrapola. Serve una run da 10 turni, ~15 min.
- il costo della **traduzione** non è mai stato dimensionato: 60 file di sorgente,
  21 di `docs/engine`, 6 di `docs/architettura`.
- il **primo utilizzo** (download da 12,6 GB) non è mai stato cronometrato, e la
  quota OPFS (10 GiB) non basta a cacheare il file.
- la pendenza **µs/posizione del 35B a contesto lungo**: misurato fino a 8k contro
  262k dichiarati dal modello.

**Fuori scope, deliberatamente** — smesso di riproporre:
subgroup-matrix (il probe committato dice che è inutilizzabile qui, ed è anche
dietro flag) · adottare TVM · trasferire lo sviluppo su Mac · policy `tier` ·
raggruppamento I/O · `idot` nel decode · formato slab · vocab ridotto sulla
lm_head.

## 3. Landmines

- **`--help` non esiste sui runner e fa PARTIRE il bench** coi default. I flag si
  leggono dal sorgente: `grep -n 'arg("' scripts/<runner>.mjs`.
- **Mai due runner GPU insieme**, e la quiescenza si misura sui processi
  **browser**, non sui server MCP.
- **Mai pipe sull'output di un runner**: bufferizza e maschera l'exit code.
- **Le leve `kfan`/`splitk` nascono SPENTE**: le accende la chat, non il bench. Un
  riferimento preso senza accenderle misura un motore che non consegniamo.
- **Il path bench di `q35conf` dichiara di sé** «frame PRE-ottimizzazioni»: è di
  grado riferimento ma non misura ciò che consegniamo.
- **Il thinking è ACCESO di default** dal 2026-08-17 (polarità derivata dal
  template): due run in modalità diverse non si confrontano. Passare
  `--thinking 0|1|auto` esplicito.
- **`results/chat/` contiene sia artefatti auto-dichiarati non-riferimento sia le
  tre repliche che lo sono**, e il nome della cartella non distingue.
- **Non riscrivere i registri**: journal, docket e `results/*.json` erano veri
  quando sono stati scritti. Si correggono le *istruzioni vive*, mai i registri.
- `~/Projects/webgguf/.harness/` e il suo `HANDOFF.md` sono **gitignorati di
  proposito**: processo su disco, fuori dalla storia pubblica.
