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
