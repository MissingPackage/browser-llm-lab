# HANDOFF — browser-llm-lab   (aggiornato 2026-08-17)

## 1. Next decidable

**SPIKE (1) FATTO il 2026-08-17 — il ginocchio di `cost(M)` è a M=2, e il
verdetto «spec-dec più lento» è caduto.** Artefatto
`results/microbench/costm-decode-4090-linux-2026-08-17T21-38-36-738Z.json`,
prereg e memo in `docs/deep-dive/costm-decode-*-2026-08-17.md`. Costo per riga,
miglior variante per ogni M, normalizzato a M=1:

    q4_K 2048x512  gate/up    100 | 60,3 | 35,3 | 22,8 | 16,9
    q4_K 512x2048  down       100 | 56,8 | 33,5 | 21,4 | 15,8
    q6_K 512x2048  down       100 | 57,0 | 34,4 | 22,4 | 16,2
    q8_0 2048x4096 attn       100 | 38,9 | 21,2 |  8,3 |  3,2

**Ricalcolata in modo indipendente dall'artefatto grezzo: riproduce esatta.**
Il salto grosso è il PRIMO (1→2). Verificare 2 token costa **1,21x** uno solo
sulla shape peggiore (2 × 60,3%). Il vecchio verdetto era **corretto per i suoi
kernel** ed è caduto perché la proprietà del kernel è cambiata sotto di esso —
misura scaduta, non misura sbagliata.
**Il sorpasso di variante vale per q4_K 2048x512 SOLTANTO** (a M=1 vince
`base-batch-z`, da M=2 `splitk-idot`); sulle altre tre `splitk-idot` vince già a
M=1. La frase generale è più forte del dato, ed è quella che verrebbe ricitata.

**PROSSIMO — spike (2), senza ruling del PI**: **overlap del router top-8 a
distanza 1-4 sul 35B**, mezza giornata, riusa il router esistente. Decide se lo
spec-dec paga **anche sul MoE o solo sul denso**. Il banco dice che verificare 2
token è ammortizzabile; **non** dice che lo spec-dec paghi sul MoE.

**Aperte e in attesa di TE**, aggiunte il 2026-08-17: **item 24** — il banco
copre cinque famiglie ma NON il q2_K, cioè il quant che consegniamo (mezza
giornata, geometria + kernel gemello) · **item 26** — una cella scartata per
checksum, `q8_0/splitk-idot@M1` su K=512 N=2048, relDiff 3,489e-2 contro
tolleranza 2e-2. Non è rotto in generale (stesso kernel a maxRel 5,96e-4 al
ktest su 2048x200): sballa su QUELLA shape a M=1. Diventa urgente proprio ora,
perché il ginocchio a M=2 può far entrare quella forma nel decode · **tre
repliche** per rendere citabile il Firefox (~1 ora di scheda; oggi 9,97 tok/s
sul 35B contro 34,6 di Chrome = 3,47x, **una run sola, indicativa**).

**Aperte e in attesa di TE** (titoli; il contenuto sta nel docket): item 21
priorità del subgroup-matrix · item 22 ruling sulla classifica delle
ottimizzazioni · traduzione del repo motore (item 16) · estrazione di
`webgguf-bench` · `paper/` non esiste, quindi il terzo repo non è estraibile.

**Leggi `MECCANISMI.md` prima di progettare e `VALUTAZIONE.md` prima di credere a
un numero** (entrambi in `docs/architettura/`).

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
- Cosa va in quale repo: `docs/publishing/split-manifest.md` (§7 = i tre difetti
  che solo **eseguire** lo split ha trovato)

**Nebbia** — non ancora specificato:
- il **regime di Firefox è ignoto**: due turni non sono regime, 4,94 non si
  estrapola. Serve una run da 10 turni, ~15 min.
- il costo della **traduzione** non è mai stato dimensionato: 60 file di sorgente
  più 27 di documentazione.
- il **primo utilizzo** (download da 12,6 GB) non è mai stato cronometrato, e la
  quota OPFS (10 GiB) non basta a cacheare il file.
- la pendenza **a contesto lungo del 35B**: misurato fino a 8k contro 262k.

**Fuori scope, deliberatamente** — smesso di riproporre: subgroup-matrix (il
probe committato dice che è inutilizzabile qui, ed è anche dietro flag) · TVM ·
sviluppo su Mac · policy `tier` · raggruppamento I/O · `idot` nel decode ·
formato slab · vocab ridotto sulla lm_head.

## 3. Landmines

- **`--help` non esiste sui runner e fa PARTIRE il bench** coi default. I flag si
  leggono dal sorgente: `grep -n 'arg("' scripts/<runner>.mjs`.
- **Mai due runner GPU insieme**, e la quiescenza si misura sui processi
  **browser**, non sui server MCP.
- **Mai pipe sull'output di un runner**: bufferizza e maschera l'exit code.
- **Le leve `kfan`/`splitk` nascono SPENTE**: le accende la chat, non il bench —
  un riferimento senza accenderle misura un motore che non consegniamo.
- **Il path bench di `q35conf` dichiara di sé** «frame PRE-ottimizzazioni»: è di
  grado riferimento ma non misura ciò che consegniamo.
- **Il thinking è ACCESO di default** dal 2026-08-17 (polarità derivata dal
  template): due modalità diverse non si confrontano. Passare `--thinking` esplicito.
- **`results/chat/` contiene sia artefatti auto-dichiarati non-riferimento sia le
  tre repliche che lo sono**, e il nome della cartella non distingue.
- **Non riscrivere i registri**: journal, docket e `results/*.json` erano veri
  quando sono stati scritti. Si correggono le *istruzioni vive*, mai i registri.
- `~/Projects/webgguf/.harness/` e il suo `HANDOFF.md` sono **gitignorati di
  proposito**: processo su disco, fuori dalla storia pubblica.
