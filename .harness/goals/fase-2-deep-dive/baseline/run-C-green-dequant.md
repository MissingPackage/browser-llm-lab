# GREEN test — scenario dequant, CON skill bottleneck-brainstorm (2026-07-27)

<!-- Output verbatim del subagent GREEN (fase 1, ciclo TDD writing-skills). Scenario nuovo
     (non usato in baseline) per testare la generalizzazione del contratto. Esito: contratto
     rispettato in tutti e 5 i punti. Bonus: il passo 1 della skill ("inchioda i numeri") ha
     corretto la premessa del prompt — il bench usa q4f32_1 ovunque, non q4f16_1, e l'S22
     espone shader-f16 mai usato. Da ri-verificare prima dell'uso in fase 4. -->

Valutazione contratto (fatta dal main loop):
1. Sottosezioni ### con numeri inchiodati — SÌ (e correzione della premessa errata del prompt)
2. Tabella unica 5 colonne esatte — SÌ
3. Instradamento a 3 valori con ragioni nelle celle — SÌ (1 esperimento, 2 engine-notes, 3 scartate)
4. Paragrafo di chiusura con UNA raccomandazione motivata — SÌ (swap q4f32_1→q4f16_1 su S22)
5. Claim citati o [VERIFY] — SÌ
Extra richiesti dal processo: sweep ≥3 famiglie — SÌ; idea fuori dagli schemi — SÌ (texture
compression ASTC/ETC2, valutata e scartata con ragione); ds4 considerato-ed-escluso con ragione.

---

## Output verbatim

## Bottleneck & vie d'uscita

### Banda di memoria nel decode, non il calcolo di dequant

Il numero di riferimento: su S22 Ultra (Xclipse 920, WebGPU, Chrome 150), WebLLM su
`Qwen2.5-0.5B-Instruct-q4f32_1-MLC` genera a **6.99 tok/s** in decode (media su 3 replicati,
stdev 0.09 — `warmupPolicy: "always"`, `warmupApplied: true`, quindi non è un artefatto di
cold-start; `results/s22-ultra-2026-07-27T00-34-09-931Z.json`, cella `webllm`).

**Correzione rispetto alla premessa**: la run citata usa quant `q4f32_1`, non `q4f16_1`. La
config di bench attuale fissa `q4f32_1` su tutti i device, S22 incluso
(`scripts/seq-bench.mjs:20`, `src/conformance/page.ts:26`). Le uniche celle `q4f16_1` nel
repo sono due run esplorative su 4090-linux/Firefox, non rappresentative per il confronto
mobile: `results/4090-linux-2026-07-25T20-08-26-375Z-firefox152-LLVMPIPE-CPU.json` (1.82
tok/s, rasterizer software, da scartare) e
`results/4090-linux-2026-07-25T20-17-03-334Z-firefoxPW.json` (9.91 tok/s, WebGPU reale). Il
meccanismo di dequant-in-kernel discusso sotto vale per entrambe le varianti — cambia solo
il dtype del compute a valle, non lo schema di lettura dei pesi 4-bit — ma il numero di
targa ~7 tok/s è misurato su f32.

Confronto same-quant, same-model, desktop: su 4090-linux (Chrome branded) lo stesso
`q4f32_1` gira a **85–117 tok/s** su più run (es.
`results/4090-linux-2026-07-25T21-17-40-926Z-chrome-branded.json`,
`results/4090-linux-2026-07-26T19-54-55-278Z.json`). Rapporto ~13-17x, coerente con un gap
di banda tra GDDR6X desktop e LPDDR mobile condivisa con la CPU (probe S22:
`deviceMemoryGB: 8` totali, non dedicati alla GPU) — non con differenze di conteggio FLOP,
che a parità di modello sono identiche.

Meccanica: sia `q4f16_1` che `q4f32_1` usano group size 32 per la quantizzazione dei pesi a
4 bit (schema `qAfB_id`: A = bit peso, B = dtype di calcolo, non del peso — confermato in
`mlc_llm/quantization/quantization.py` del repo upstream mlc-ai/mlc-llm). Il kernel dequant
è fuso nel matmul: legge il buffer di pesi packed 4-bit + gli scale per gruppo, li
despacchetta in registri e li scala, poi esegue subito la moltiplicazione — nessun buffer
intermedio dequantizzato scritto in VRAM. A batch=1 (decode) questo significa che l'intera
matrice pesi va letta una volta per token indipendentemente da quanto "costi" la dequant in
ALU: l'intensità aritmetica è già dominata dal solo trasferimento, testbook memory-bound.

### Percorso f32 dove il device supporta f16 (leva concreta, non teorica)

Il probe S22 elenca esplicitamente `shader-f16` tra le `features` supportate
(`results/s22-ultra-2026-07-27T00-34-09-931Z.json`, `probe.features`), ma nessuna run nel
repo usa `q4f16_1` su questo device — la config bench lo fissa a `q4f32_1` ovunque
(citazioni sopra). Compute a f16 dimezza tipicamente registri/banda per gli intermedi
rispetto a f32 a parità di pesi 4-bit letti; è un delta plausibile ma non ancora misurato
sul device target.

---

| Idea | Prior art | Fattibilità / costo | Rischio | Instradamento |
|---|---|---|---|---|
| Passare a `q4f16_1` sui device con `shader-f16` confermato (S22 già lo espone) | Catalogo quantizzazione MLC-LLM: entrambe le varianti esistono già, stesso group size 32, differiscono solo nel dtype di calcolo ([mlc-llm/quantization.py](https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_llm/quantization/quantization.py)) | Bassissima — solo `modelId` in `scripts/seq-bench.mjs:20` / `src/conformance/page.ts:26`; riusa l'infrastruttura del micro-bench-matmul già pianificato nello spec di fase | Basso — verificare conformance numerica sull'output (`src/conformance/`) | **esperimento** |
| Kernel dequant+GEMM fuso con overlap latenza/calcolo via async copy e layout ottimizzato (stile Marlin) | Marlin, GPTQModel/vLLM: nasconde il costo di dequant sovrapponendolo al GEMM, layout in memoria dedicato ([arXiv:2408.11743](https://arxiv.org/pdf/2408.11743)) | Alta — richiederebbe WGSL custom; TVM/MLC oggi non genera questo pattern per il backend WebGPU | Alto — subgroup ops WebGPU ancora parziali su GPU mobile | engine-notes |
| Block layout pensato per accessi coalesced nel dequant, come `mul_mat_q` di llama.cpp | llama.cpp, kernel quantizzati CUDA/Metal con layout a blocchi ottimizzato per warp/coalescing `[VERIFY: non letto il sorgente llama.cpp in questa sessione, riferimento da prior art nota]` | Media-alta — tocca il layout dei pesi generato da TVM, non solo la config | Medio | engine-notes |
| Delegare il matmul quantizzato a WebNN (driver nativo: DirectML/CoreML/NNAPI, operatori QDQ dedicati) | WebNN espone `quantizeLinear`/`dequantizeLinear` e delega a driver nativi ([webmachinelearning/webnn#623](https://github.com/webmachinelearning/webnn/issues/623), [WebNN DirectML, Microsoft](https://blogs.windows.com/windowsdeveloper/2024/05/24/introducing-the-webnn-developer-preview-with-directml/)) | Molto alta — adapter WebNN completo, nuovo | Alto — supporto mobile/Android ancora incompleto | **scartata** (fuori scope esplicito: WebNN è nei "Deferred" dello spec madre, non un aggiramento di questo goal) |
| Dequantizzare una volta e cachare i pesi f16/f32 in un buffer persistente tra token | Pattern usato in sistemi di offload (es. FlexGen) ma per batching/prefill, non per singolo token in decode `[VERIFY: nessun prior art trovato per il caso batch=1]` | Bassa da implementare | Alto — a batch=1 quadruplica i byte da leggere/scrivere in VRAM rispetto al 4-bit, peggiorando esattamente la metrica memory-bound che si vuole migliorare | **scartata** (l'analisi mostra che aggrava il bottleneck, non lo risolve) |
| Impaccare i pesi 4-bit in formati texture compressi hardware (ASTC/ETC2) e leggerli via texture sampler invece che storage buffer + ALU | Nessun motore LLM lo fa: transfer speculativo dalle unità di decompressione texture della GPU (dominio grafica/game, non ML) `[VERIFY: idea speculativa, nessuna fonte upstream]`; il probe S22 conferma comunque `texture-compression-astc`/`texture-compression-etc2` tra le feature disponibili sull'hardware target | Molto alta — i formati texture hanno schemi di quantizzazione fissi per errore percettivo (immagini), non per errore numerico sui pesi; servirebbe validare l'accuratezza e TVM non genera oggi binding sampled per i pesi | Molto alto — nessuna garanzia di correttezza numerica | **scartata** (rischio di correttezza troppo alto per un esperimento a basso costo; eventuale ricerca separata, fuori da questo goal) |

`antirez/ds4` (SSD streaming) è stato considerato nello sweep ma escluso dalla tabella: risolve un problema di capacità VRAM (fase di caricamento/paging), non il costo di banda per-token nel decode già residente in VRAM — non transfer utile per *questo* bottleneck specifico (è più pertinente a `buffer-limit-2gb.md`).

**Raccomandazione**: l'esperimento da proporre come voce docket è il primo — cambiare `q4f16_1`/`q4f32_1` sul device S22, che espone `shader-f16` ma non lo usa mai nella config attuale. È l'unica idea con costo prossimo a zero (un solo `modelId` nei due file di config citati, nessun rework del motore), l'unica direttamente verificabile senza toccare `src/adapters/webllm.ts` (esplicitamente vietato dallo spec di fase), e chiude un buco di verifica aperto in questa stessa sezione: oggi non sappiamo quanto del gap 4090→S22 sia banda intrinseca e quanto sia percorso f32 non necessario. Si appoggia inoltre al micro-bench-matmul già pianificato come deliverable della fase, quindi il costo marginale è quasi nullo.
