# Baseline RED — fase 1 (creazione skill bottleneck-brainstorm), 2026-07-27

Output dei 2 run di baseline (subagent SENZA skill) usati per il ciclo TDD di writing-skills.
Sono materiale grezzo di qualità per le fasi 3 (buffer-limit-2gb) e 5 (kv-cache-layout):
le citazioni sono state verificate dagli agenti nel repo, ma vanno RI-VERIFICATE quando
questi testi entrano in un doc — non copiarli alla cieca.

Esito RED registrato: la baseline NON fallisce in capacità (prior-art, scoring, citazioni
già presenti, con lo spec su disco). Fallisce in FORMA: varianza di struttura tra i run
(prosa+lista vs tabella, dimensioni di valutazione non uniformi, routing espresso in modi
diversi). La skill è quindi un contratto di output (ricetta), non una disciplina.

Finding load-bearing dal run A (per fase 3): WebLLM richiede hardcoded 1 GiB di
maxStorageBufferBindingSize con fallback 128 MiB (bundle index.js:4067-4082, @mlc-ai/web-llm
0.2.x) — il soffitto per-binding operativo è 1 GiB, non i ~2 GiB del probe. Confermato dal
run LLVMPIPE (134217728 = 1<<27 in results/4090-linux-...-firefox152-LLVMPIPE-CPU.json).
