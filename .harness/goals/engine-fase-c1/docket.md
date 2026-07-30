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

3. **RULING RICHIESTO — spec C1** (2026-07-30, fase 2): approvazione di
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
   Le fasi 3-6 restano gated fino al ruling.
