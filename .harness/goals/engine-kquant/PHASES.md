# PHASES — engine-kquant (decomposizione it.0, 2026-08-14)

Sequenziale. Le righe 2, 3 e 4 toccano lo STESSO file
(`src/engine/kernels/wgsl.ts`), lo stesso piano (`prefillgemmplan.ts`) e lo
stesso ktest: nessun `parallel-group`, la proprieta' esclusiva dei path non
c'e'. Stessa ragione dei due goal precedenti.

**Metrica obiettivo del goal**: `prefill.ms + decode.firstMs` sul prompt-idx 0
(6333 token), a modello caldo. **Barra del goal < 22.500 ms**; **nice to have
< 18.000 ms** (ruling PI 2026-08-14). Oggi: **32.127 ms**. Ogni riga dichiara
quanto muove QUESTA metrica, o si dichiara gate.

---

## Punto di partenza — quattro correzioni trovate nel test-fit dei done-when

Prima di mettere una clausola in tabella ho verificato che sia eseguibile COME
SCRITTA. Quattro non lo erano.

**(C0-1) `prefillGemmSplitsFor` conta in blocchi da 32, e i K-quant non hanno
blocchi da 32.** `wgsl.ts:4150` fa `bpr = K/32` e sceglie 4 fette se
`bpr % 8 == 0`. Su un K-quant l'unita' indivisibile e' il **superblocco da 256**
(8 sotto-blocchi con scale e min a 6 bit condivisi): una fetta che ne taglia
meta' deve comunque leggere l'header delle scale del superblocco intero, e il
conto dei byte — quello su cui poggia tutto il goal — smetterebbe di valere.
La funzione va resa **parametrica sulla famiglia** (unita' = 32 pesi per i
formati legacy, 256 per i K-quant). Non e' un blocco: e' un requisito di design
della riga 2, ed e' scritto li'.

**(C0-2) La quantizzazione delle attivazioni si riusa TALE E QUALE, e questa e'
una buona notizia da mettere per iscritto.** `prefillQuantXQ8Wgsl` produce
`xq`/`xsc` per blocchi da **32**, e i sotto-blocchi Q5_K/Q4_K/Q6_K sono
anch'essi da 32 (otto per superblocco). La via intera `dot4I8Packed` e' quindi
disponibile per i K-quant **senza un secondo quantizzatore e senza dispatch in
piu'**: e' lo stesso buffer che il prefill gia' riempie.

**(C0-3) Serve un termine che la q4_0 non ha: Σx per sotto-blocco.** Q5_K/Q4_K
sono `w = d·sc_i·q − dmin·m_i` e il q4_1 e' `w = d·q + m`: entrambi hanno un
termine costante per blocco, che moltiplica **la somma delle attivazioni**, non
il loro prodotto coi pesi. Nella via intera si ottiene con un
`dot4I8Packed(xq, 0x01010101)` per sotto-blocco — zero dispatch aggiuntivi, ma
e' aritmetica che il kernel q4_0 non contiene. **Ipotesi da confermare al banco
in riga 1, non promessa**: e' esattamente cio' che la fase 0 esiste per
misurare.

**(C0-4) Una cella del banco sarebbe degenere per costruzione, ed e' il
precedente che la skill cita.** Il down-proj degli expert del 35B ha
**K=512 = 2 superblocchi per riga**: lo split-K a 4 fette **non esiste** su
quella shape. Le celle 35B-down vanno a `splits ∈ {1,2}`; scriverle a 4
produrrebbe un kernel che rifiuta (giustamente) e una cella vuota eseguita per
conformita'. Shape reali verificate sull'header dump 2026-08-10
(`qwen35moe`: dModel **2048**, dFfnExpert **512**, nExpert **256**, 40 layer):
expert gate/up `[2048, 512]` Q4_K · expert down `[512, 2048]` Q4_K · attn Q8_0
K=2048.

---

## Tabella

| # | phase | done-when (mechanical) | authority delta | owns | status | stima |
|---|-------|------------------------|-----------------|------|--------|-------|
| 1 | **FASE 0 — il banco delle famiglie non-q4_0.** Estende `ttGemm`/`ttRunner` con le forme multi-riga candidate per Q5_K, Q4_1, Q4_K, Q6_K, Q8_0 (via intera + fallback f32), contro la forma legacy attuale come termine di paragone. **Nessuna riga di produzione toccata.** | JSON in `results/microbench/` con **G pesi/s e GB/s per (famiglia × variante × M)**, M ∈ {1,8,16}, sulle shape reali: Q5_K `[4096,2560]` · Q4_1 `[9216,2560]` · Q4_K `[2048,512]` e `[512,2048]` (splits ∈ {1,2}, C0-4) · Q6_K `[512,2048]` · Q8_0 `[2048,4096]`; gate di checksum del banco verde su ogni cella; `npx vitest run` verde sul test SENZA GPU che verifica la lista delle varianti. **REGOLA DI STOP per famiglia**: nessuna variante ≥ **1,5×** sulla legacy ⇒ quella famiglia si chiude col numero e non si cabla (non fa fallire il goal: fa fallire il suo cablaggio, e va nel consuntivo). | none | `src/microbench/**`, `scripts/tt-microbench-run.mjs`, `tests/microbench-*.test.ts` | **done (it.1-it.3)** | 2 it stimate, **3 consumate** |
| 2 | **Q5_K in produzione — fetta verticale** (kernel → `prefillGemmSplitsFor` parametrico (C0-1) → predicato del piano → wiring `gemvB` → ktest → copertura). Veicolo: **`sdd-conductor`**. | `[6c]` di `tests/engine-prefillgemmplan.test.ts` **≥ 10,9×** con le 24 `ssm_out` fra i `multirow`; `[6d]` cancellato con la sua ragione scritta; **caso ktest nuovo PASS** contro il cpuref; **floor test** che ri-deriva la tolleranza dal testo WGSL generato con e senza contrazione FMA; `tests/gpulimits.test.ts` verde con la formula del kernel nuovo dentro il `Math.max` calcolato; `npx tsc --noEmit` pulito; `npx vitest run` verde; nessun secondo predicato di ammissibilita' fuori da `prefillgemmplan.ts` (gate strutturale gia' esistente). | none | `src/engine/kernels/wgsl.ts`, `src/engine/prefillgemmplan.ts`, `src/engine/q35gpumodel.ts`, `src/engine/ktest/ktest.worker.ts`, `tests/**` | **done (it.4)** | 3-4 it |
| 3 | **Q4_1 in produzione — fetta verticale**, stessa forma della riga 2 (aritmetica diversa: `d·Σ(q·x) + m·Σx`, C0-3). Veicolo: **`sdd-conductor`**. | `[6c]` **≥ 15,5×** con anche le 4 `ffn_down` Q4_1 fra i `multirow`; caso ktest Q4_1 PASS; floor test esteso; `gpulimits` verde; tsc + vitest verdi. | none | idem riga 2 | **done (it.5-it.7)** | 1-2 it, **3 consumate** |
| 4 | **Le tre forme del 35B: misurate e verificate, NON cablate** (Q4_K, Q6_K, Q8_0) + **scheda di consegna** al goal successivo. | un caso ktest **PASS** per ciascuna delle tre forme contro il cpuref; un test che verifica che il piano **NON le instrada** in produzione (il predicato resta ai soli kind cablati — una forma verificata e non instradata non e' una forma inventata); `docs/engine/kquant-consegna-35b-<data>.md` con, per ciascuna: numeri di riga 1, shape misurate, e cosa manca per cablarla (baseline 35B assente, `moeprefillplan`, residency dei 17,67 GB). | none | `src/engine/kernels/wgsl.ts`, `src/engine/ktest/**`, `tests/**`, `docs/engine/` | **done (it.8)** | 1-2 it, **1 consumata** |
| 5 | **La misura di chiusura**: checkpoint fresco sul prompt-idx 0 e le barre del contratto. Include il debito di `build-ttft-checkpoint.mjs:108` (i byte del segmento derivati dal meter invece che ricopiati). | JSON `kind: "q35-ttft-kernel-checkpoint"` con `gemm:deltanet-out` **≤ 2.000 ms**, `gemm:ffn-down` **≤ 2.000 ms**, byte e GB/s per segmento; `prefill.ms + decode.firstMs` **< 22.500 ms** (e il valore dichiarato contro la soglia nice-to-have 18.000); `prefill.tokS > 282`; `prefill.tokS > decode.tokS`; `loadMs`/`prefill.ms`/`decode.firstMs` scomposti; `hostState.declared = "quiescent"`; un test che verifica che i byte del checkpoint vengano da `prefillbytes.ts` e non da una costante. | none | `scripts/build-ttft-checkpoint.mjs`, `results/engine/`, `tests/**` | ready | 1 it |
| 6 | **GATE DI MERGE** (riga di sola verifica, dichiarata tale). | `node .harness/tools/engine-ktest.mjs` **tutti PASS**; top-1 contro l'oracolo llama.cpp **≥ 1012/1024** su ENTRAMBI i bracci; sequenze generate **identiche 8/8**; decode 4B **≥ 45,5 tok/s** a ctx 6333; GLM b12 optimistic entro ±5% di 13.172 / 31,26 / 14,74; `npx vitest run` verde; `npx tsc --noEmit` pulito. Tutto su host dichiarato. | none | — (gate) | ready | 1 it |
| 7 | **Consuntivo e consegna.** | `docs/engine/kquant-consuntivo-<data>.md` voce per voce con l'artefatto accanto a ogni clausola del DONE WHEN, la nuova ripartizione del tempo per segmento, e **il termine che diventa primo dopo questa leva nominato con la sua misura fresca**; `HANDOFF.md` §1 aggiornato; `GLOSSARY.md` aggiornato coi termini coniati. | none | `docs/engine/`, `HANDOFF.md`, `GLOSSARY.md` | ready | 1 it |

**Conto dei gate**: una riga su sette e' gate puro (la 6), una e' fase 0 (la 1,
che non muove la metrica ma ha il potere di chiudere il goal). Cinque righe su
sette producono o misurano il risultato — il piano non fotografa lo stato.

**Contributo alla metrica obiettivo, dichiarato per riga**: riga 2 **−11,4 s**
proiettati · riga 3 **−3,3 s** · riga 4 **0 s sul 4B** (e' il ponte al goal
successivo, e il PI l'ha chiesto esplicitamente) · righe 1, 5, 6, 7 non muovono
la metrica e non pretendono di farlo.

**Nessuna riga ha un ramo "oppure dichiara il debito".** L'unica clausola che ci
somiglia — la regola di stop della riga 1 — non e' un ramo libero: costa una
misura completa PRIMA di poter essere invocata, e il suo esito e' un numero
pubblicato, non una dichiarazione.
