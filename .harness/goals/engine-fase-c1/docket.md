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
