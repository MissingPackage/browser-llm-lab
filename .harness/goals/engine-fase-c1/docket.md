# Docket — engine-fase-c1 (decisioni PI pendenti)

1. ~~**PLAN-CHECK**~~ RISOLTO (2026-07-30, ruling PI in chat: "Parti pure. Per
   il GGUF usa quello che userà il nostro motore, ovviamente"). Decomposizione
   approvata senza modifiche; criterio GGUF fissato = layout-compatibilità coi
   kernel del motore ⇒ variante **Q4_0** (i kernel fase A dequant-fusi leggono
   q4_0; stesso criterio di qwen2.5-0.5b-instruct-q4_0.gguf dell'oracolo
   attuale). Se Q4_0 puro non esiste per GLM-4.7-Flash nei repo GGUF candidati,
   la scelta del fallback più vicino torna QUI come nuova entry, non si decide
   unilateralmente. Fasi 1 sbloccata; 3-6 restano gated dal ruling di spec
   (fase 2).

2. **FINDING fuori-scope (it.1, 2026-07-30) — GGUF GLM-4.7-Flash è arch
   `deepseek2` e SENZA testa NextN/MTP.** La conversione unsloth Q4_0 mappa il
   modello sull'architettura deepseek2 (MLA: kv_lora 512, rope 64 — conferma
   direction §3) e NON contiene chiavi/tensori NextN: la testa MTP del
   checkpoint upstream è droppata. Implicazioni PI-gated: (a) fase D
   ("spec-dec: prima la MTP nativa", direction §7) — la testa va presa da una
   conversione glm4moe-arch o estratta dal checkpoint HF a parte; (b) C2: il
   reader GGUF del motore leggerà layout deepseek2 (notizia buona: il codice
   KV di ds4/colibri è sullo stesso lineage). Nessuna azione in C1: la traccia
   routing non dipende dalla MTP. Riferimento: gguf-dump del file in journal
   it.1; ledger §H/§I da aggiornare in fase 6 (sintesi).

3. ~~**RULING RICHIESTO — spec C1**~~ RISOLTO (2026-07-31, ruling PI in chat:
   "ok (a)-(f)" dopo walkthrough delle decisioni). Spec approvata senza
   modifiche; fasi 3-6 SBLOCCATE (3 ready, 4-6 a cascata). Testo originale:

   **RULING RICHIESTO — spec C1** (2026-07-30, fase 2): approvazione di
   `docs/superpowers/specs/2026-07-30-engine-fase-c1-design.md`. Decisioni
   proposte da ratificare:
   (a) predittore LOOKA = router di L+1 applicato a `ffn_norm(L)` (analogo
       colibri kind-1; two-step NON in v1, solo rimando se il recall delude);
   (b) LOOKA calcolato ONLINE nel tool (estrazione pesi router ~24 MB alla
       load + replica esatta sigmoid→+bias→top-k), niente dump hidden da 6 GB;
       autotest hard: predire lo stesso layer ⇒ recall ≥0.999;
   (c) corpus 8 prompt eterogenei committati (~12k prefill + 4k decode ≈ 16k
       posizioni, ~9-12 min/run); recall headline decode-only @4 e @8;
   (d) simulatore TypeScript+vitest nel repo (npm test), policy {LRU, LFRU
       tier.h, LFRU+pin split-temporale anti-leakage, LFRU+pin+prefetch con
       replay delle predizioni vere e guard anti-eviction}, griglia budget
       6 punti (1/16 → tutto il parco, ~1-15.6 GB);
   (e) timebox fase 4 = 3 iterazioni (fallback ROUTE_TRACE-only da contratto);
   (f) NESSUN gate numerico sul recall (misura, non target): il numero decide
       il go/no-go PILOT in fase 6, decisione PI.

4. ~~**PROPOSTA go/no-go PILOT per C2/C3**~~ RISOLTO (2026-07-31, ruling PI in
   chat: **"Adotta, ma WP comunque"** — GO prefetch / NO-GO pinning adottato
   come input di C2/C3; il WP banda fredda **browser** si fa comunque, come
   parte/precondizione di C3, NON blocca C2. Prossimo goal deciso: **C2**.
   Propagazione del fatto nuovo eseguita: ledger §A righe paging/OPFS,
   direction §8.3). Testo originale:

   **PROPOSTA go/no-go PILOT per C2/C3** (2026-07-31, fase 6 — DECISIONE DEL PI,
   qui solo istruita). Raccomandazione: **GO sul prefetch predittivo, NO-GO sul
   learned pinning come leva primaria.** Evidenza:
   - **Recall misurato** (C1 it.4): 92.0% @K=8, 87.7% @6, 77.5% @4 sul decode;
     baseline temporale 32.3%. Il segnale del router batte la persistenza di
     2.8× (colibri su GLM-5.2: 1.7×). Prefill quasi identico ⇒ proprietà del
     modello, non del regime.
   - **Costo-miss** (results/opfs-bench, 4090/NVMe): read warm 10.7-11.7 GB/s a
     blocchi 64 KB-1 MB ⇒ un expert da 5.33 MB costa ~0.46 ms warm; il regime
     freddo è disco-bound e NON caratterizzato (drop_caches impossibile dal
     browser — rischio dichiarato in direction §8.3).
   - **Aritmetica della decisione**: a 30 tok/s il budget è 33 ms/token; 184
     accessi/token con hit-rate 90.8% ⇒ ~17 miss/token × 0.46 ms = **7.8 ms**
     warm (24% del budget). Con hit-rate 84.7% (LRU nudo) ⇒ 28 miss = 13 ms
     (39%). Il delta prefetch vale ~5 ms/token: significativo, ma il numero
     dipende dalla banda FREDDA, che manca.
   - **Perché no-go sul pinning**: skew debole misurato (top-4/layer 21.8%,
     working-set 1.663 in 32 token); a pin 50% LRU vince, a pin 12.5% il pin è
     ~neutro. Il pin resta utile solo per il non-routed (1.53 GB) = instant-on.
   Cosa NON è deciso qui e serve a C3: il guadagno di LATENZA del prefetch (C1
   misura occupazione di cache, non tempo) e la banda OPFS a freddo.
   Domanda al PI: si adotta questa raccomandazione come input di C2/C3, o si
   vuole prima un WP sulla banda fredda (misura ~mezza giornata) per rendere
   l'aritmetica sopra non-condizionale?

   **ISTRUTTORIA AGGIUNTA (2026-07-31, sessione post-C1)** — la banda fredda è
   stata bracketata SENZA il WP browser: misura lato OS con fadvise(DONTNEED)
   (`tools/cold-read-bench.py`, `results/opfs-bench/cold-read-os-4090-linux-2026-07-31.json`).
   990 PRO a freddo: **1.63 GB/s su read random da 5.33 MB (p50 3.74 ms/expert,
   8× il warm 0.46 ms)**; seq 3.22 GB/s. Warm re-read 10-11 GB/s = coincide col
   bench browser ⇒ metodo validato; il freddo browser è ≤ di questi numeri.
   Aritmetica non più condizionale (184 accessi/token, budget 33 ms @30 tok/s):
   - **spillover RAM-backed (warm)**: 2× regge — tuned 90.8% ⇒ 7.8 ms (24%),
     LRU 84.7% ⇒ 12.9 ms (39%). Come già nel punto sopra.
   - **spillover disk-backed (cold)**: 2× a 30 tok/s NON regge — tuned 16.9
     miss × 3.74 ms = 63 ms/token (1.9× il budget); floor di banda 90 MB/token
     ⇒ servirebbero 2.7 GB/s > 1.63 misurati. Rate max bandwidth-bound ~18
     tok/s; per 30 tok/s servirebbe hit ≥94.5% CON overlap perfetto (= solo il
     prefetch può darlo, il pinning no).
   - **working point dev (2208 slot, 87%)**: regge anche a freddo con prefetch
     (98.9% ⇒ 2 miss ⇒ 7.3 ms, 22%); LRU nudo 96.4% ⇒ 24.7 ms (75%, al limite).
   Conseguenze per la decisione: (1) il GO-prefetch/NO-GO-pinning è INVARIANTE
   rispetto alla banda fredda — meno banda ⇒ prefetch più necessario, e il
   valore del pinning dipende dallo skew (misurato debole), non dalla banda ⇒
   l'item è decidibile ORA; (2) il WP browser da mezza giornata non serve più
   per DECIDERE (il bound disco è misurato; il browser può solo peggiorarlo) —
   resta utile a C3 per raffinare il modello di banda, dove direction §7 già lo
   colloca; (3) NUOVO FATTO per ledger §A: la headline "modello ~2× la memoria"
   va condizionata alla struttura dei tier — regge come "2× la VRAM con
   spillover coperto dalla RAM/page-cache" (dev box 31 GB, M4 48 GB: sì), NON
   come "2× la memoria host totale" (regime disk-bound: max ~18 tok/s, o hit
   ≥94.5%). Da propagare a ledger §A e direction §8.3 dopo il ruling.

5. **Nota per C2 (non una decisione)**: la matrice usage grezza 46x64 NON e'
   serializzata (il report ha solo le curve cumulative top-N, come da spec
   approvata) — e' ricalcolabile in secondi dalla traccia committata con
   `usageCounts()` del simulatore. Se C2 la vuole come file, e' una riga di
   script, non una nuova run.
