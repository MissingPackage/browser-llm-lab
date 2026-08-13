// Orchestratore GPU full-model per Qwen 3.5 DENSO (4B/9B) — q1 fase 4
// slice 3. Assemblatore "correttezza prima" (pattern del bring-up
// gpuforward): decode teacher-forced, piano di step PRECOSTRUITO una volta
// (pipeline cache + bind group statici su scratch condivisi), un submit per
// token, readback dei logits e argmax su CPU. Full residency by design (il
// 4B Q4 sta in VRAM: è il regime del tier 16 GB; il paging resta al MoE).
//
// Ogni pezzo è GIÀ provato: kernel ktestati (79/79) e assembly dei due tipi
// di layer con pesi reali == cpuref a L2rel ~1e-7 (it.7). Qui c'è SOLO
// l'orchestrazione: loop 32 layer + embed row + ffn + head + residui.
import {
  addInPlaceWgsl, ARGMAX_CHUNK, argmaxStage1Wgsl, argmaxStage2Wgsl,
  attnDecodeCombineWgsl, attnDecodeWgsl, axpyWgsl,
  gemvF32Wgsl, gemvGrid, gemvQ4KWgsl, gemvQ5KWgsl,
  gemvQ6KWgsl, gemvQuantWgsl, gemvQuantGrid, gemvQuantVec4Rows2Ok, kvAppendWgsl, rmsnormWgsl, ropeNeoxWgsl, sigmoidMulWgsl,
  siluMulWgsl, stridedCopyWgsl, routerTopKWgsl,
  SEL_BYTES, MOE_IDX_BYTES, MOE_IDX_STRIDE, type KArenaOpts,
} from "./kernels/wgsl";
import { expertArenaBindings } from "./gpulimits";
import type { SlabTensorLayout } from "./moe";
import { deltaNetConvWgsl, deltaNetCoreWgsl, deltaNetGatesWgsl } from "./kernels/deltanet";
import { GGML_TYPE, tensorByteSize, type GgufTensorInfo } from "./gguf";
import { dequantQ4_0, dequantQ6_K, dequantQ8_0, repackKQuant, repackQ4_0, repackQ4_1, repackQ8_0, Q5_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES } from "./quant";
import { q35AttnPartialsFloats, q35AttnSplitPlan } from "./q35attnsplit";
import { q35IsFullAttn, type Q35Shape } from "./q35shape";
import { ROUTER_QWEN35MOE, routerSelect, WEIGHTS_SUM_CLAMP_MIN, type RouterConfig } from "./moe";
import {
  ExpertCache, moeParkOf, type ExpertClass, type ExpertRawBytes, type SlotRef,
} from "./residency";
import { q35ExpertReader, q35MoeConfig } from "./q35expertstore";
import { gemvCapsFor } from "./gemvcaps";

export interface Q35RawReader {
  shape: Q35Shape;
  info(name: string): GgufTensorInfo;
  /** byte GREZZI del tensore (il chiamante può scartarli dopo l'upload) */
  read(name: string): Promise<Uint8Array>;
  /** sotto-range di un tensore (it.17: slab expert on-miss) — OBBLIGATORIO per MoE */
  readRange?(name: string, off: number, len: number): Promise<Uint8Array>;
}

interface Step { pipe: GPUComputePipeline; bind: GPUBindGroup; wgs: [number, number, number] }

/**
 * Esito del confronto ROUTER GPU (in ombra) vs selezione CPU, sui layer VERI
 * (goal fase-D fase 3b, fetta 3b). it.13 ha già gateato lo stesso kernel su 64
 * estrazioni sintetiche 256x8: qui la domanda è diversa e la sintetica non la
 * poteva porre — se la DISTRIBUZIONE vera dei logits del 35B produce margini
 * sotto la risoluzione dell'f32, la GPU (f32) e la CPU (f64) sceglierebbero
 * expert diversi. Per questo si riporta anche il margine minimo osservato: un
 * conteggio di flip a zero senza sapere quanto era stretto il caso peggiore
 * non dice se il gate ha visto il caso difficile.
 */
export interface Q35RouterShadowStats {
  /** confronti = layer MoE x token */
  comparisons: number;
  /** selezioni confrontate = comparisons x topK */
  picks: number;
  /** l'INSIEME dei top-K differisce: è il difetto che conta */
  setFlips: number;
  /** stesso insieme, ordine diverso (i pesi seguono l'expert, ma l'ordine
   *  decide quale slot finisce in quale entry di Sel) */
  orderFlips: number;
  /** errore relativo massimo sui pesi di mixing (CPU f64 come riferimento) */
  maxWeightRelErr: number;
  /** margine minimo sui logit fra l'ultimo preso e il primo scartato: la
   *  softmax è monotona, quindi è questo il numero che decide l'ordinamento */
  minLogitMargin: number;
  /** slot risolti dalla GPU diversi da quelli che la CPU ha poi usato, ESCLUSI
   *  i miss: l'ombra gira PRIMA degli `ensure` del suo layer, quindi un expert
   *  che sta per essere caricato si risolve MISS ed è corretto così */
  slotMismatch: number;
  /** miss visti dalla GPU (flag di Sel) e dalla CPU (`isResident`) */
  missGpu: number; missCpu: number; missDisagree: number;
}

export interface Q35GpuModel {
  /**
   * Un token teacher-forced; ritorna l'argmax dei logits. Con read=false
   * NON copia/mappa i logits (prefill sequenziale: gli step si accodano
   * senza sync; tornare -1) — la conformance legge solo dove serve.
   */
  step(token: number, pos: number, read?: boolean): Promise<number>;
  /**
   * K token TEACHER-FORCED in UN SOLO submit, con l'argmax calcolato su GPU
   * (goal fase-D fase 3). `step` aspetta il readback dei logits a ogni token
   * — 604 KB e, soprattutto, un round-trip in cui CPU e GPU si aspettano a
   * vicenda: misurato 51,5 ms/token contro 35,8 senza sync sul 4B. Qui gli
   * step si accodano tutti, ogni step scrive il proprio argmax in un buffer,
   * e si legge UNA volta sola K·4 byte.
   *
   * Teacher-forced: i token di ingresso sono noti in anticipo (conformance,
   * golden). La generazione libera ha bisogno del feedback su GPU (embed
   * gather), che è un passo a parte.
   *
   * `null` sui modelli MoE: la selezione degli expert passa dalla CPU a ogni
   * layer, quindi il batch non è possibile finché il routing non è su GPU.
   */
  decodeBatch: ((tokens: ArrayLike<number>, posStart: number) => Promise<number[]>) | null;
  /**
   * DRAFT della testa MTP per la posizione appena eseguita (fase 7, it.53).
   * `null` se il file non porta la testa o se il modello non e' stato costruito
   * con `opts.mtp`.
   *
   * VINCOLO D'USO, e non e' negoziabile: va chiamata SUBITO dopo lo `step` che
   * ha prodotto `nextToken`, perche' legge `x` — il residuo finale di QUELLA
   * posizione — che il token successivo sovrascrive. `nextToken` e' il token
   * che il modello ha appena predetto (t_{i+1}); il draft e' la sua ipotesi su
   * t_{i+2}, cioe' un token gratis per ogni token vero.
   *
   * Tiene una cache KV SUA: la testa e' un layer in piu', non un layer del
   * modello, e la sua attenzione guarda la sequenza degli h' — non quella del
   * modello.
   */
  mtpDraft: ((nextToken: number) => Promise<number>) | null;
  /**
   * VERIFICA SPECULATIVA (fase 7, it.54): esegue in UN submit la posizione
   * `pos` col token CERTO e la posizione `pos+1` col token PROPOSTO dalla
   * testa, e ritorna i due argmax. Il secondo vale SOLO se il draft coincide
   * col primo — altrimenti la riga speculativa e' spazzatura e va disfatta con
   * `specRollback`.
   *
   * `null` sui MoE: la selezione degli expert passa dalla CPU a ogni layer.
   */
  specVerify: ((certain: number, draft: number, pos: number) => Promise<[number, number]>) | null;
  /**
   * Disfa la riga speculativa: rimette lo stato ricorrente dei layer DeltaNet e
   * il residuo `x` a com'erano DOPO la posizione certa. Da chiamare quando il
   * draft e' stato rifiutato, prima di proseguire.
   */
  specRollback: (() => void) | null;
  /**
   * VERIFICA SPECULATIVA A PESI LETTI UNA VOLTA (fase 7, it.55). Come
   * `specVerify`, ma sul piano a 2 righe: i kernel batch leggono ogni peso una
   * volta sola per entrambe le posizioni, che e' l'unico modo in cui la
   * speculazione puo' pagare in un decode memory-bound (docket item 29).
   *
   * Richiede il modello costruito con `mtp: true` e `prefillM: 2`. Dopo la
   * chiamata si DEVE invocare `specCommit(accettato)`: e' li' che il residuo
   * giusto torna in `x` e, se il draft e' stato rifiutato, che lo stato
   * ricorrente torna a dopo la riga certa.
   */
  specVerifyBatched: ((certain: number, draft: number, pos: number) => Promise<[number, number]>) | null;
  /** Chiude la passata speculativa: residuo della riga giusta in `x`, e stato ricorrente riparato se rifiutata. */
  specCommit: ((accepted: boolean) => void) | null;
  /** azzera gli stati ricorrenti (conv + S) di tutti i layer linear: nuovo prompt. */
  resetState(): void;
  /**
   * Dispatch per token, SCOMPOSTI (docket item 12). `static` sono gli step del
   * piano precostruito; `dynamic` sono quelli per-expert, che il piano non
   * contiene perche' dipendono dalla selezione. Il campo `total` e' quello che
   * il nome prometteva e che finora non c'era: sul 35B sono 782 statici e
   * 1.320 dinamici, cioe' 2.102 — non 782.
   */
  dispatchesPerToken: number;
  dispatchBreakdown: { static: number; dynamic: number; total: number };
  /** stats MoE (null sui densi): arena, routing esatto, residenza. */
  moeStats: (() => {
    hits: number; misses: number; uploadedBytes: number;
    /** scomposizione del costo per MISS (fase D fase 2): dove vanno i ms */
    readMs: number; packMs: number; uploadMs: number;
    routing: Record<string, number>; nSlots: Record<string, number>;
    parkSlots: Record<string, number>; slotBytes: Record<string, number>;
    /** policy di residenza ATTIVA: due run che differiscono per la policy si
     *  distinguevano solo dal comando (docket item 12, rilievo di it.37) */
    policy: "lru" | "tier";
  }) | null;
  /**
   * Scomposizione del costo per token (fase D fase 3): dove vanno i ms del
   * decode. `embedMs` = dequant CPU della riga di embedding + writeBuffer;
   * `readbackMs` = attesa del readback dei logits (604 KB sul 4B);
   * `argmaxMs` = il max su `vocab` float in CPU. Tutti e tre spariscono col
   * pattern decodeBatch — misurarli PRIMA e' come si decide se conviene.
   *
   * `submits` e `readbacks` (fetta 3c) sono i CONTATORI su cui la fase 3b si
   * chiude: `submits` conta le `queue.submit`, `readbacks` i punti di attesa
   * GPU→CPU (piu' mapAsync concorrenti risolte insieme valgono UNO: e' un
   * round-trip solo). Si contano su ENTRAMBI i path, altrimenti il "prima" del
   * done-when sarebbe una stima.
   *
   * `dirtyTokens`/`replays`/`replayLayers`/`repairMs` sono la TASSA del
   * repair+replay: token con almeno un miss, giri rigiocati (piu' di uno per
   * token = cascata), somma dei layer MoE rigiocati, e il tempo CPU di
   * fetch+upload+flush. La tassa GPU del replay sta gia' dentro `submits`.
   */
  perf(): {
    tokens: number; embedMs: number; readbackMs: number; argmaxMs: number;
    submits: number; readbacks: number;
    dirtyTokens: number; replays: number; replayLayers: number; repairMs: number;
    /**
     * FASE 4-TER (it.28): il token FUORI dai pass GPU, decomposto.
     * `encodeMs` = registrare i ~2400 dispatch nel command buffer e fare il
     * submit (CPU pura, e la GPU non ha ancora cominciato: e' seriale col
     * lavoro GPU, non sovrapposta). `readbackMs` = attesa delle mapAsync, che
     * NON e' overhead ma per lo piu' la GPU che lavora. `tailCpuMs` = la
     * contabilita' di fine token (derivazione dei miss da Sel, routing,
     * noteResidentHit). `embedMs` e `argmaxMs` erano gia' li'. `tokenMs` e' il
     * totale di parete, cosi' il residuo non misurato si vede per differenza
     * invece di essere assunto zero.
     */
    encodeMs: number; tokenMs: number; tailCpuMs: number;
    /**
     * TESTA MTP (it.53): tempo di parete e conteggio dei draft, FUORI da
     * `tokenMs`. Il draft e' un submit a parte con un'attesa sua, e sommarlo al
     * token mescolerebbe il costo del modello con quello della speculazione —
     * che e' esattamente il rapporto che la fase 8 deve pesare.
     */
    mtpMs: number; mtpDrafts: number;
    /**
     * VERIFICA SPECULATIVA (it.54): passate a due righe, tempo di parete, e
     * quante sono finite in rifiuto (`specRejects` = rollback dello stato
     * ricorrente). `specPasses - specRejects` sono i token guadagnati.
     */
    specMs: number; specPasses: number; specRejects: number;
  };
  /**
   * Accende/spegne il path a submit unico a caldo (solo se il modello e' stato
   * costruito con `select: "optimistic"`; altrimenti LANCIA — non esiste il
   * cablaggio). E' quello che permette al gate di misurare le due passate sulla
   * STESSA cache: passata sync a freddo, passata ottimistica a caldo.
   */
  setOptimistic(on: boolean): void;
  /**
   * PREFILL A CHUNK (fase 4): M token in un solo submit, con i logits della
   * SOLA ultima riga — il prefill non ha bisogno degli altri. `null` se il
   * modello non e' stato costruito con `prefillM`. Il chunk deve essere PIENO:
   * le taglie dei dispatch e il numero di vettori sono cotti nel piano gemello,
   * quindi un chunk parziale vorrebbe un secondo piano (e non serve: la coda
   * del prompt la fa `step`).
   */
  prefillChunk: ((tokens: ArrayLike<number>, posStart: number) => Promise<Float32Array>) | null;
  /**
   * I logits dell'ULTIMO `step(..., read=true)`. Esiste perche' il gate della
   * fase 4 confronta i LOGITS e non l'argmax: due vettori possono avere lo
   * stesso massimo ed essere diversi ovunque, e il done-when chiede i logits.
   */
  lastLogits(): Float32Array | null;
  /**
   * Come si e' arrivati al budget dell'arena expert: tetto, allocato davvero,
   * riserva, budget. `null` sui densi (non c'e' arena). Sta nel report perche'
   * un budget che nessuno vede e' un budget che nessuno controlla.
   */
  vramPlan(): { ceilingBytes: number | null; allocatedBytes: number; reserveBytes: number; expertBudgetBytes: number; ctxMax: number } | null;
  /**
   * SOLO HARNESS (fase 5, it.36): rimette la cache expert a VUOTA. Serve a
   * misurare due path nello STESSO processo partendo entrambi da freddo — che
   * e' l'unico modo di confrontarli davvero, visto che la cache fredda esiste
   * una volta sola per processo e finora i due bracci stavano in due run.
   * `null` sui densi (non c'e' arena).
   */
  debugEvictAll: (() => void) | null;
  /**
   * Tempo GPU per CATEGORIA accumulato sui token girati col path ottimistico e
   * la sonda accesa; `null` se la sonda non e' attiva (o `timestamp-query` non
   * concessa dal device). `ms` e' la somma dei pass, `n` il numero di pass.
   */
  gpuTimeStats: (() => { tokens: number; overflow: number; byCat: Record<string, { ms: number; n: number }> }) | null;
  /** DEBUG (it.17): dopo step(), hidden x a valle del layer indicato (solo MoE). */
  readTap(layer: number): Promise<Float32Array>;
  /** Esito dell'OMBRA del router GPU; `null` se non è stata accesa. */
  routerShadowStats: (() => Q35RouterShadowStats) | null;
  destroy(): void;
}

export interface Q35GpuModelOpts {
  /**
   * Accende il router+resolve su GPU in OMBRA (fase 3b, fetta 3b): gira in
   * coda al GEMV del router, nello stesso submit, e scrive una regione
   * PARALLELA di `Sel` che nessun kernel expert legge. La selezione di
   * produzione resta quella della CPU: qui si misura la fedeltà sui layer
   * veri, non si cambia il risultato. Costa un dispatch e una copia da 128 B
   * per layer, e si spegne di default.
   */
  routerShadow?: boolean;
  /**
   * Costruisce la TESTA MTP (NextN) accanto al modello (fase 7, it.53): pesi
   * di `blk.<nLayer>`, cache KV sua, e un secondo piano di dispatch che si
   * lancia a parte. Spenta di default perche' costa VRAM (120,6 M parametri +
   * due cache KV) a chi non fa spec-dec, e perche' i GGUF senza testa non
   * hanno quei tensori.
   */
  mtp?: boolean;
  /**
   * Path di selezione degli expert (fase 3b, fetta 3c). PORT del `select` di
   * `glmmodel`, con gli stessi nomi:
   *
   * - `"cpu"` (default): il path di oggi — un submit e un readback dei logits
   *   del router PER LAYER, la CPU sceglie, `ensure`, e riempie `Sel`.
   * - `"optimistic"`: il token intero in UN submit. La `Sel` di produzione la
   *   scrive il router su GPU, che risolve expert→slot dalla `slotTable` e
   *   marca `dirtyB` quando trova un MISS; nessuno sulla CPU vede la selezione
   *   mentre il token gira, e il routing si ricostruisce dalla `Sel` letta in
   *   coda.
   *
   * OPT-IN per costruzione: finché la residenza non è raggiunta il path sync
   * costa meno (it.16: a cache fredda 39 token su 39 sono sporchi), e la
   * SOGLIA d'ingresso è materia della fase 5 — qui c'è il meccanismo, non la
   * policy. Costruire con `"optimistic"` costruisce ENTRAMBI i path e accende
   * il secondo; `setOptimistic(false)` torna al primo a caldo (è così che il
   * gate misura le due passate sulla stessa cache).
   */
  select?: "cpu" | "optimistic";
  /**
   * DECOMPOSIZIONE DEL TEMPO GPU del token (fase 4, it.19), solo sul path a
   * submit unico. Spezza il pass di ogni layer in tre — statico / router /
   * expert — e li cronometra con `timestamp-query`. E' opt-in e PERTURBA la
   * misura: tre pass per layer invece di uno significa tre barriere invece di
   * una, quindi il token cronometrato e' un po' piu' lento di quello vero. Si
   * riporta anche il totale con la sonda spenta, cosi' la perturbazione si
   * vede invece di essere assunta trascurabile.
   *
   * Serve a decidere la fase 4: il batching M>1 toglie DISPATCH, e vale la
   * pena solo se il tempo sta dove i dispatch sono tanti.
   */
  telemetryGpu?: boolean;
  /**
   * SOLO HARNESS (fase 4-ter, it.28): salta lo SNAPSHOT dello stato ricorrente
   * (62,8 MiB di copyBufferToBuffer per token, nati in it.18 per il replay).
   * Serve a MISURARE quanto costa quella copia, che e' la piu' grossa
   * operazione GPU fuori dai pass e quindi il sospetto numero uno per gli 11,9
   * ms che la decomposizione CPU non spiega.
   *
   * Rompe il replay per costruzione — senza snapshot il restore leggerebbe
   * ombre stantie — quindi il path LANCIA se un token risulta sporco invece di
   * ripararlo con dati sbagliati. Si usa solo a cache calda, dove i token
   * sporchi sono 0 per misura (it.16).
   */
  debugNoStateSnapshot?: boolean;
  /**
   * PREFILL A CHUNK (fase 4): costruisce, ACCANTO al piano per riga, un piano
   * gemello che fa M token insieme. Assente = non cambia una riga, e il decode
   * resta esattamente quello che era: la non-regressione e' per costruzione.
   * Oggi cablato sul tipo di layer DENSO (4B/9B); il MoE e' il passo dopo.
   */
  prefillM?: number;
  /**
   * TETTO VRAM (fase 5, it.35, docket item 11). Con questo, il budget
   * dell'arena expert si DERIVA — tetto meno cio' che il modello ha davvero
   * allocato meno una riserva — invece di essere un parametro fisso che
   * nessuno controlla. Il default storico (12 GiB) su un host con 14,4 GiB
   * liberi sfondava, e il fallimento era `VK_ERROR_OUT_OF_DEVICE_MEMORY` alla
   * createBuffer: rumoroso solo grazie al listener, coi buffer invalidi che
   * lasciavano girare il modello su numeri plausibili (it.14).
   * Se assente, resta il comportamento di prima.
   */
  vramCeilingBytes?: number;
  /** riserva sopra il derivato (default 512 MiB): driver, frammentazione, staging */
  vramReserveBytes?: number;
  /**
   * Policy di residenza (fase 5): "lru" (default, il comportamento storico) o
   * "tier" = LRU + AUTOPIN, che pinna gli expert caldi e li protegge
   * dall'eviction. Su q35 non era cablata: la riga 5 chiede il delta misurato
   * o l'esclusione coi numeri, e senza il cablaggio non c'e' ne' l'uno ne'
   * l'altra.
   */
  expertPolicy?: "lru" | "tier";
}

export async function createQ35GpuModel(
  device: GPUDevice, r: Q35RawReader, ctxMax = 64, arenaBudgetBytes = 12 * (1 << 30),
  opts: Q35GpuModelOpts = {},
): Promise<Q35GpuModel> {
  const S = r.shape;
  const isMoe = S.arch === "qwen35moe";
  if (isMoe && !r.readRange) throw new Error("q35gpumodel: MoE richiede reader.readRange");
  const d = S.dModel;
  const hd = S.headDim;
  const qDim = S.nHead * hd;
  const kvDim = S.nKvHead * hd;
  const qkvDim = (2 * S.linKHead + S.linVHead) * S.linHeadDim;
  const inner = S.linVHead * S.linHeadDim;
  const dFfn = (S.dFfn ?? 0) as number;

  const pipes = new Map<string, GPUComputePipeline>();
  const pipe = (code: string): GPUComputePipeline => {
    let p = pipes.get(code);
    if (!p) {
      p = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
      pipes.set(code, p);
    }
    return p;
  };
  /**
   * BYTE ALLOCATI dal modello prima dell'arena expert (fase 5, it.35). Non e'
   * una stima dei pesi non-expert: e' la SOMMA di cio' che questo file ha
   * chiesto al device, contata dove si chiede. Serve al budget ctx-aware —
   * GLM lo deriva sottraendo termini calcolati (`slabBudgetCtxAware`), qui si
   * puo' fare di meglio perche' il modello sa esattamente quanto ha preso.
   */
  let allocBytes = 0;
  let vramPlanOut: {
    ceilingBytes: number | null; allocatedBytes: number; reserveBytes: number;
    expertBudgetBytes: number; ctxMax: number;
  } | null = null;
  const track = (b: GPUBuffer, bytes: number): GPUBuffer => { allocBytes += Math.max(16, bytes); return b; };
  const sbuf = (data: Float32Array | Uint32Array): GPUBuffer => {
    const b = device.createBuffer({ size: Math.max(16, data.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    device.queue.writeBuffer(b, 0, data as BufferSource);
    return track(b, data.byteLength);
  };
  const empty = (bytes: number): GPUBuffer =>
    track(device.createBuffer({ size: Math.max(16, bytes), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC }), bytes);

  // pesi quantizzati → upload (i raw si scartano subito: streaming)
  const q40 = async (name: string): Promise<{ qs: GPUBuffer; scales: GPUBuffer; k: number; n: number; kind: "q4_0" | "q4_1" }> => {
    const t = r.info(name);
    const raw = await r.read(name);
    const nBlocks = (t.dims[0] / 32) * t.dims[1];
    const kind = t.type === GGML_TYPE.Q4_1 ? "q4_1" : "q4_0";
    if (t.type !== GGML_TYPE.Q4_0 && t.type !== GGML_TYPE.Q4_1) throw new Error(`q35gpumodel: ${name} tipo ${t.type}, atteso q4_0/q4_1`);
    const { qs, scales } = kind === "q4_1" ? repackQ4_1(raw, 0, nBlocks) : repackQ4_0(raw, 0, nBlocks);
    return { qs: sbuf(qs), scales: sbuf(scales), k: t.dims[0], n: t.dims[1], kind };
  };
  const q80 = async (name: string): Promise<{ qs: GPUBuffer; scales: GPUBuffer; k: number; n: number }> => {
    const t = r.info(name);
    const raw = await r.read(name);
    const { qs, scales } = repackQ8_0(raw, 0, (t.dims[0] / 32) * t.dims[1]);
    return { qs: sbuf(qs), scales: sbuf(scales), k: t.dims[0], n: t.dims[1] };
  };
  const kquant = async (name: string, blockBytes: number): Promise<{ blocks: GPUBuffer; k: number; n: number }> => {
    const t = r.info(name);
    const raw = await r.read(name);
    return { blocks: sbuf(repackKQuant(raw, 0, (t.dims[0] / 256) * t.dims[1], blockBytes)), k: t.dims[0], n: t.dims[1] };
  };
  const f32buf = async (name: string): Promise<GPUBuffer> => {
    const raw = await r.read(name);
    return sbuf(new Float32Array(raw.slice().buffer, 0, raw.byteLength / 4));
  };
  /** Loader GENERICO type-driven (it.17): gemv col kernel giusto per il tipo
   *  REALE del tensore (35B: attn/ssm_out/shexp Q8_0, alpha/beta F32). */
  const loadW = async (name: string): Promise<{
    n: number; k: number;
    push: (src: GPUBuffer, dst: GPUBuffer) => void;
    /** gemello a M righe (fase 4): no-op senza `prefillM` */
    pushB: (src: GPUBuffer, dst: GPUBuffer) => void;
  }> => {
    const t = r.info(name);
    const [k, n] = [t.dims[0], t.dims[1]];
    if (t.type === GGML_TYPE.F32) {
      const w = await f32buf(name);
      return {
        n, k,
        push: (src, dst) => push(gemvF32Wgsl({ K: k, N: n }), [w, src, dst], gemvGrid(n)),
        pushB: (src, dst) => pushB(gemvF32Wgsl({ K: k, N: n, batch: true }), [w, src, dst], [gemvGrid(n)[0], gemvGrid(n)[1], M_MAX]),
      };
    }
    if (t.type === GGML_TYPE.Q4_0 || t.type === GGML_TYPE.Q4_1) {
      const w = await q40(name);
      return { n, k, push: (src, dst) => gemv(w, src, dst), pushB: (src, dst) => gemvB(w, src, dst) };
    }
    if (t.type === GGML_TYPE.Q8_0) {
      const w = await q80(name);
      return { n, k, push: (src, dst) => gemv(w, src, dst, "q8_0"), pushB: (src, dst) => gemvB(w, src, dst, "q8_0") };
    }
    if (t.type === GGML_TYPE.Q4_K || t.type === GGML_TYPE.Q5_K || t.type === GGML_TYPE.Q6_K) {
      const blockBytes = t.type === GGML_TYPE.Q4_K ? 144 : t.type === GGML_TYPE.Q5_K ? Q5_K_BLOCK_BYTES : Q6_K_BLOCK_BYTES;
      const w = await kquant(name, blockBytes);
      const code = t.type === GGML_TYPE.Q4_K ? gemvQ4KWgsl({ K: k, N: n }) : t.type === GGML_TYPE.Q5_K ? gemvQ5KWgsl({ K: k, N: n }) : gemvQ6KWgsl({ K: k, N: n });
      // Sul 35B i pesi statici sono tutti Q8_0/F32 e i K-quant stanno solo
      // negli expert (it.24) — ma il 4B ha `ssm_out` in Q5_K, quindi il gemello
      // a M righe serve anche qui: l'inventario valeva per un modello, non per
      // la famiglia (it.32).
      const codeB = t.type === GGML_TYPE.Q4_K
        ? gemvQ4KWgsl({ K: k, N: n, batch: true })
        : t.type === GGML_TYPE.Q5_K ? gemvQ5KWgsl({ K: k, N: n, batch: true }) : gemvQ6KWgsl({ K: k, N: n, batch: true });
      return {
        n, k,
        push: (src, dst) => push(code, [w.blocks, src, dst], gemvGrid(n)),
        pushB: (src, dst) => pushB(codeB, [w.blocks, src, dst], [gemvGrid(n)[0], gemvGrid(n)[1], M_MAX]),
      };
    }
    throw new Error(`q35gpumodel: loadW tipo ${t.type} non gestito (${name})`);
  };

  // uniform (pos, nPast) — uno solo, aggiornato per token
  const uni = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  // scratch condivisi fra i layer (i bind group restano statici)
  const x = empty(d * 4);
  const xn = empty(d * 4);
  const attnY = empty(d * 4);
  const qkv = empty(qkvDim * 4), z = empty(inner * 4);
  const bRaw = empty(S.linVHead * 4), aRaw = empty(S.linVHead * 4), bSig = empty(S.linVHead * 4), gVal = empty(S.linVHead * 4);
  const convOut = empty(qkvDim * 4), gated = empty(inner * 4);
  const qFull = empty(2 * qDim * 4), kCur = empty(kvDim * 4), vCur = empty(kvDim * 4);
  const qB = empty(qDim * 4), gateB = empty(qDim * 4), qN = empty(qDim * 4), kN = empty(kvDim * 4);
  const attnO = empty(qDim * 4);
  /**
   * PARZIALI dell'attention decode split (pass 1 -> pass 2), UNA coppia per
   * tutto il modello. Il sizing viene da `q35AttnPartialsFloats` — la stessa
   * funzione da cui il kernel ricava la propria geometria, non una copia — e
   * passa da `empty`, quindi finisce in `allocBytes` e nel `vramPlan` come ogni
   * altro scratch.
   *
   * PERCHE' SI POSSONO CONDIVIDERE fra i layer full-attn E la testa MTP. I due
   * dispatch (split + combine) sono accodati come step CONSECUTIVI, e in WebGPU
   * i dispatch dello stesso compute pass sono ordinati fra loro con barriera
   * implicita (e i pass dello stesso submit a maggior ragione: la sonda dei
   * timestamp puo' spezzare il pass su una marca senza cambiare nulla). Nessun
   * altro step legge o scrive questi due buffer nel mezzo: il layer l+1
   * sovrascrive i parziali del layer l solo DOPO che il combine del layer l li
   * ha gia' letti. La testa MTP gira in un submit separato (mtpSteps, dopo il
   * piano del token), quindi non si sovrappone per costruzione. Costo evitato:
   * nHead*splits*(headDim+2) float per OGNI layer invece che una volta sola.
   */
  const attnParts = q35AttnPartialsFloats({ nHead: S.nHead, headDim: hd, ctxMax });
  const attnPartOut = empty(attnParts.out * 4);
  const attnPartMS = empty(attnParts.ms * 4);
  /** Y del dispatch del pass 1 — stesso piano che il kernel ha baked dentro. */
  const { splits: attnSplits } = q35AttnSplitPlan(ctxMax);
  const gateF = empty(dFfn ? dFfn * 4 : 16), upF = empty(dFfn ? dFfn * 4 : 16);
  // --- MoE (it.17): scratch dedicati (i dinamici non toccano quelli shexp) ---
  const nE = S.nExpert ?? 0, topK = S.nExpertUsed ?? 0, dE = S.dFfnExpert ?? 0;
  const routerLogits = empty(Math.max(nE, 4) * 4);
  const routerStaging = device.createBuffer({ size: Math.max(nE, 4) * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const moeAcc = empty(d * 4);
  const gateS = empty(Math.max(dE, 4) * 4), upS = empty(Math.max(dE, 4) * 4), dnS = empty(d * 4), shScalar = empty(16);
  const gateE = empty(Math.max(dE, 4) * 4), upE = empty(Math.max(dE, 4) * 4);
  /** confini dei segmenti statici: segmento i = steps[cuts[i-1]..cuts[i]) */
  const cuts: number[] = [];
  /**
   * Layer ASSOLUTO dell'i-esimo segmento MoE. Nella famiglia Qwen 3.5/3.6 gli
   * expert ci sono da blk.0 (`denseLead: 0`) e i due indici coincidono, ma la
   * chiave della slotTable e la classe dello slab si leggono col layer
   * assoluto: tenerlo esplicito costa un array di 40 interi ed evita che un
   * modello della famiglia con dei layer densi in testa indirizzi la classe
   * sbagliata SENZA fallire.
   */
  const moeLayerAbs: number[] = [];

  // embedding: raw Q6_K tenuto CPU-side per il gather della riga; head = stesso
  // tensore (tied) o output.weight, su GPU
  // embd: Q6_K (4B) o Q4_0 (9B — che tiene invece la HEAD in Q6_K: inverso
  // del 4B, scoperto dal file in it.11)
  const embdInfo = r.info("token_embd.weight");
  if (embdInfo.type !== GGML_TYPE.Q6_K && embdInfo.type !== GGML_TYPE.Q4_0 && embdInfo.type !== GGML_TYPE.Q8_0) {
    throw new Error(`q35gpumodel: embd tipo ${embdInfo.type}, atteso Q6_K/Q4_0/Q8_0`);
  }
  const embdRaw = await r.read("token_embd.weight");
  // head: Q6_K (4B tied) o Q4_0 (9B non-tied, output.weight) — it.11
  const headName = S.tiedEmbeddings ? "token_embd.weight" : "output.weight";
  const headT = r.info(headName);
  let headStep: (src: GPUBuffer, dst: GPUBuffer) => void;
  if (headT.type === GGML_TYPE.Q6_K) {
    const head = await kquant(headName, Q6_K_BLOCK_BYTES);
    headStep = (src, dst) => push(gemvQ6KWgsl({ K: head.k, N: head.n }), [head.blocks, src, dst], gemvGrid(head.n));
  } else if (headT.type === GGML_TYPE.Q4_0) {
    const head = await q40(headName);
    headStep = (src, dst) => gemv(head, src, dst);
  } else {
    throw new Error(`q35gpumodel: head ${headName} tipo ${headT.type}, atteso Q6_K/Q4_0`);
  }
  const outNorm = await f32buf("output_norm.weight");
  const logits = empty(S.vocab * 4);
  const staging = device.createBuffer({ size: S.vocab * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

  const steps: Step[] = [];
  // ---- PREFILL A CHUNK (fase 4, it.32): il piano gemello a M righe ----
  const M_MAX = Math.max(0, Math.min(16, Math.floor(opts.prefillM ?? 0)));
  const prefillOn = M_MAX > 0;
  /** scratch a M righe: row-major, passo di riga = la taglia per riga. */
  const bM = (perRow: number): GPUBuffer => empty(perRow * Math.max(1, M_MAX));
  const PB = prefillOn ? {
    x: bM(d * 4), xn: bM(d * 4), attnY: bM(d * 4),
    qkv: bM(qkvDim * 4), z: bM(inner * 4),
    bRaw: bM(S.linVHead * 4), aRaw: bM(S.linVHead * 4), bSig: bM(S.linVHead * 4), gVal: bM(S.linVHead * 4),
    convOut: bM(qkvDim * 4), gated: bM(inner * 4),
    qFull: bM(2 * qDim * 4), kCur: bM(kvDim * 4), vCur: bM(kvDim * 4),
    qB: bM(qDim * 4), gateB: bM(qDim * 4), qN: bM(qDim * 4), kN: bM(kvDim * 4),
    attnO: bM(qDim * 4), gateF: bM(dFfn ? dFfn * 4 : 16), upF: bM(dFfn ? dFfn * 4 : 16),
    // MoE (it.34): il segmento STATICO si batcha (shexp + router GEMV); la
    // catena expert resta per riga, ed e' il motivo per cui il gate resta la
    // bit-identita'.
    gateS: bM(Math.max(dE, 4) * 4), upS: bM(Math.max(dE, 4) * 4), dnS: bM(d * 4),
    shScalar: bM(16), moeAcc: bM(d * 4), routerLogits: bM(Math.max(nE, 4) * 4),
    /** posizione per riga: la leggono rope, kvAppend e l'attenzione a chunk */
    rowPos: device.createBuffer({ size: Math.max(16, M_MAX * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
    /** indice di riga per la RICORRENZA (uniform, una entry ogni 256 B) */
    rowUni: device.createBuffer({ size: M_MAX * 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
    steps: [] as Step[],
    /** confini dei segmenti statici nel piano gemello (uno per layer MoE) */
    cuts: [] as number[],
  } : null;
  if (PB) {
    const u = new Uint32Array(M_MAX * 64);
    for (let m = 0; m < M_MAX; m++) u[m * 64] = m;
    device.queue.writeBuffer(PB.rowUni, 0, u as unknown as BufferSource);
  }
  /**
   * ETICHETTE per la sonda dei tempi (fase 4, it.23): `{at, cat}` dice che dallo
   * step `at` in poi si entra nella categoria `cat`. Sono MARCHE su intervalli e
   * non un campo per step, cosi' i ~25 siti di `push` non si toccano.
   *
   * Serve a due domande insieme: quanto del segmento statico e' ROW-PARALLEL
   * (cioe' quanto la fase 4 potrebbe comprimere batchando M righe) e quanto e'
   * RICORRENTE (deltanet: resta M anche batchato); e dove sia finito il +1,62 ms
   * che il segmento statico ha preso con la 4-bis (docket item 16).
   */
  const segMarks: { at: number; cat: string }[] = [];
  const mark = (cat: string): void => {
    if (segMarks.length > 0 && segMarks[segMarks.length - 1].at === steps.length) segMarks.pop();
    segMarks.push({ at: steps.length, cat });
  };
  /**
   * Stato RICORRENTE dei layer linear (deltanet). Il `layer` non è decorazione:
   * il replay del path ottimistico rigioca i layer da un certo punto in giù e
   * deve rimettere a posto lo stato di QUELLI, non di tutti — quelli a monte
   * non si rigiocano e il loro stato è già quello giusto.
   */
  const stateBufs: { buf: GPUBuffer; bytes: number; layer: number }[] = [];
  /**
   * DOVE finiscono gli step accodati. Esiste per la testa MTP (it.53): la testa
   * riusa `loadW`/`gemv`, che chiamano `push`, ma i suoi dispatch NON vanno nel
   * piano del token — si lanciano a parte, dopo. Si sposta il bersaglio per il
   * tempo della costruzione e lo si rimette; il piano del modello non cambia di
   * un dispatch (`dispatchesPerToken` si misura prima).
   */
  let stepTarget: Step[] = steps;
  const push = (code: string, bufs: GPUBuffer[], wgs: number | [number, number], withUni = false): void => {
    const p = pipe(code);
    const entries: GPUBindGroupEntry[] = bufs.map((b, i) => ({ binding: i, resource: { buffer: b } }));
    if (withUni) entries.push({ binding: bufs.length, resource: { buffer: uni } });
    const bind = device.createBindGroup({ layout: p.getBindGroupLayout(0), entries });
    stepTarget.push({ pipe: p, bind, wgs: typeof wgs === "number" ? [wgs, 1, 1] : [wgs[0], wgs[1], 1] });
  };
  /**
   * ATTENTION DECODE in DUE dispatch (split sul contesto + combine log-sum-exp).
   * Un helper solo perche' i due call-site — i layer full-attn del piano token e
   * la testa MTP — devono restare per costruzione la stessa aritmetica: se
   * divergono, il draft si accetta contro un'attenzione diversa da quella che lo
   * verifica.
   *
   * GRIGLIA FISSA IN ctxMax, LAVORO NO. `q35AttnSplitPlan` sceglie i chunk col
   * criterio dichiarato in q35attnsplit.ts: chunk di almeno
   * Q35_ATTN_MIN_CHUNK = 512 posizioni (sotto, il combine e i parziali costano
   * piu' di quanto renda l'occupancy in piu') e al piu'
   * Q35_ATTN_MAX_SPLITS = 64 chunk (oltre, cresce il chunk e NON il numero di
   * workgroup: e' il punto dello split, tenere i buffer parziali costanti in
   * ctxMax). Il dispatch e' quindi (nHead, splits) su ctxMax e non su nPast+1, e
   * a contesto corto lancia workgroup che non hanno niente da fare. E' voluto: i
   * bind group e le griglie sono PRECOSTRUITI una volta sola (nessuna
   * ricreazione per token), e un chunk oltre nPast+1 esce col loop vuoto
   * scrivendo m = -3.0e38, s = 0, acc = 0, che il combine annulla
   * (exp(-3e38 - gm) = 0, nessun NaN). Si paga qualche workgroup vuoto;
   * l'alternativa era ricostruire griglia e bind group a ogni token.
   *
   * Binding congelati dai kernel: pass 1 = [q, kCache, vCache, partOut, partMS]
   * con l'uniform a binding 5 (`withUni`, che `push` mette a `bufs.length`);
   * pass 2 = [partOut, partMS, out] e NESSUN uniform — quanti chunk esistono e'
   * baked nel WGSL, `nPast` al combine non serve.
   */
  function pushAttnDecodeSplit(qSrc: GPUBuffer, kCache: GPUBuffer, vCache: GPUBuffer, dst: GPUBuffer): void {
    push(
      attnDecodeWgsl({ nHead: S.nHead, nKvHead: S.nKvHead, headDim: hd, ctxMax }),
      [qSrc, kCache, vCache, attnPartOut, attnPartMS], [S.nHead, attnSplits], true,
    );
    push(
      attnDecodeCombineWgsl({ nHead: S.nHead, headDim: hd, ctxMax }),
      [attnPartOut, attnPartMS, dst], S.nHead,
    );
  }
  /**
   * FORMA `vec4-rows2` (fase 2): load vettoriali + `dot()`, DUE righe per
   * workgroup, riduzione con `subgroupAdd` dove `gemvCapsFor` la dichiara
   * SICURA (feature presente E subgroup fisso a 32 — v. gemvcaps.ts: una
   * mappatura riga→subgroup sbagliata somma le lane sbagliate senza sollevare
   * errori). Ammessa solo su q4_0 nudo: ogni altra combinazione torna al testo
   * di prima, byte per byte, ed e' cosi' che GLM e i K-quant non regrediscono.
   *
   * La griglia NON si calcola a mano: `gemvQuantGrid` la deriva dalle righe per
   * workgroup del testo generato, e le due decisioni restano una sola
   * (`gemvQuantVec4Rows2Ok`).
   */
  const gemvCaps = gemvCapsFor(device);
  const gemv = (w: { qs: GPUBuffer; scales: GPUBuffer; k: number; n: number; kind?: "q4_0" | "q4_1" | "q8_0" }, src: GPUBuffer, dst: GPUBuffer, kind?: "q4_0" | "q4_1" | "q8_0"): void => {
    const kk = kind ?? w.kind ?? "q4_0";
    const opts = { kind: kk, K: w.k, N: w.n, hasBias: false, vec4Rows2: kk === "q4_0", sg: gemvCaps.sg };
    const ok = gemvQuantVec4Rows2Ok(opts);
    const use = ok ? opts : { kind: kk, K: w.k, N: w.n, hasBias: false };
    push(gemvQuantWgsl(use), [w.qs, w.scales, src, dst], gemvQuantGrid(use));
  };
  // ---- gemelli a M righe (fase 4). `pushB` scrive SOLO nel piano gemello. ----
  const pushB = (code: string, bufs: GPUBuffer[], wgs: [number, number, number]): void => {
    if (!PB) return;
    const p = pipe(code);
    const bind = device.createBindGroup({
      layout: p.getBindGroupLayout(0),
      entries: bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })),
    });
    PB.steps.push({ pipe: p, bind, wgs });
  };
  /**
   * La RICORRENZA: M step, uno per riga, IN ORDINE. Un bind group per riga, e
   * l'unica differenza fra loro e' l'offset dell'uniform che porta l'indice —
   * i buffer sono gli stessi (it.30).
   */
  const pushBRows1 = (code: string, bufs: GPUBuffer[], wg: number, row: number): void => {
    if (!PB) return;
    const p = pipe(code);
    const bind = device.createBindGroup({
      layout: p.getBindGroupLayout(0),
      entries: [
        ...bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })),
        { binding: bufs.length, resource: { buffer: PB.rowUni, offset: row * 256, size: 16 } },
      ],
    });
    PB.steps.push({ pipe: p, bind, wgs: [wg, 1, 1] });
  };
  const gemvB = (w: { qs: GPUBuffer; scales: GPUBuffer; k: number; n: number; kind?: "q4_0" | "q4_1" | "q8_0" }, src: GPUBuffer, dst: GPUBuffer, kind?: "q4_0" | "q4_1" | "q8_0"): void => {
    const kk = kind ?? w.kind ?? "q4_0";
    // `batch` NON e' ammesso dalla forma a 2 righe (fase 0 non l'ha misurata):
    // qui il kernel resta quello di prima, byte per byte. La griglia pero' si
    // deriva lo stesso da `gemvQuantGrid` invece che da `gemvGrid` a mano —
    // oggi danno lo stesso numero (1 riga per workgroup sul batch), ma un TERZO
    // posto che decide le righe-per-workgroup e' esattamente la forma del bug
    // trovato in it.7, dove due posti che decidevano la stessa cosa la
    // decidevano diversamente. Uno solo, e non si ripresenta.
    const opts = { kind: kk, K: w.k, N: w.n, hasBias: false, batch: true };
    const [gx, gy] = gemvQuantGrid(opts);
    pushB(gemvQuantWgsl(opts), [w.qs, w.scales, src, dst], [gx, gy, M_MAX]);
  };

  /**
   * SNAPSHOT DELLO STATO RICORRENTE DENTRO IL PIANO BATCH (fase 7, it.55).
   *
   * La verifica speculativa a 2 righe vuole poter tornare allo stato "dopo la
   * riga certa". Nel piano a M righe quel momento NON e' un punto solo della
   * sequenza di dispatch — ogni layer aggiorna il proprio stato prima di
   * passare al successivo — quindi lo snapshot si prende PER LAYER, subito dopo
   * la riga 0, con un kernel di copia (le `copyBufferToBuffer` non possono
   * stare dentro un compute pass, un dispatch si').
   *
   * Vive solo se il modello e' costruito per lo spec-dec (`mtp`) con M=2: il
   * prefill a chunk vero gira con M piu' grandi e non paga niente.
   */
  const specSnap: { src: GPUBuffer; dst: GPUBuffer; n: number }[] | null =
    (opts.mtp && prefillOn && M_MAX === 2) ? [] : null;

  for (let l = 0; l < S.nLayer; l++) {
    const b = `blk.${l}.`;
    const attnNorm = await f32buf(`${b}attn_norm.weight`);
    const postNorm = await f32buf(`${b}post_attention_norm.weight`);
    mark("norm");
    push(rmsnormWgsl(d, S.rmsEps), [x, attnNorm, xn], 1);
    if (PB) pushB(rmsnormWgsl(d, S.rmsEps, true), [PB.x, attnNorm, PB.xn], [M_MAX, 1, 1]);

    if (q35IsFullAttn(S, l)) {
      mark("attn");
      const wq = await loadW(`${b}attn_q.weight`);
      const wk = await loadW(`${b}attn_k.weight`);
      const wv = await loadW(`${b}attn_v.weight`);
      const wo = await loadW(`${b}attn_output.weight`);
      const qNormW = await f32buf(`${b}attn_q_norm.weight`);
      const kNormW = await f32buf(`${b}attn_k_norm.weight`);
      const kCache = empty(ctxMax * kvDim * 4);
      const vCache = empty(ctxMax * kvDim * 4);
      wq.push(xn, qFull);
      wk.push(xn, kCur);
      wv.push(xn, vCur);
      if (PB) { wq.pushB(PB.xn, PB.qFull); wk.pushB(PB.xn, PB.kCur); wv.pushB(PB.xn, PB.vCur); }
      push(stridedCopyWgsl({ nVec: S.nHead, len: hd, srcStride: 2 * hd, srcOffset: 0, dstStride: hd, dstOffset: 0 }), [qFull, qB], Math.ceil(qDim / 64));
      push(stridedCopyWgsl({ nVec: S.nHead, len: hd, srcStride: 2 * hd, srcOffset: hd, dstStride: hd, dstOffset: 0 }), [qFull, gateB], Math.ceil(qDim / 64));
      push(rmsnormWgsl(hd, S.rmsEps, true), [qB, qNormW, qN], S.nHead);
      push(rmsnormWgsl(hd, S.rmsEps, true), [kCur, kNormW, kN], S.nKvHead);
      if (PB) {
        // APPIATTIMENTO (it.31): questi kernel sono PER-VETTORE, e il vettore
        // (riga, head) sta all'indice riga*nVec + head. Si dispatchano nVec*M
        // vettori e l'aritmetica degli offset torna da sola — provato
        // bit-identico alla geometria vera.
        pushB(stridedCopyWgsl({ nVec: S.nHead * M_MAX, len: hd, srcStride: 2 * hd, srcOffset: 0, dstStride: hd, dstOffset: 0 }), [PB.qFull, PB.qB], [Math.ceil((M_MAX * qDim) / 64), 1, 1]);
        pushB(stridedCopyWgsl({ nVec: S.nHead * M_MAX, len: hd, srcStride: 2 * hd, srcOffset: hd, dstStride: hd, dstOffset: 0 }), [PB.qFull, PB.gateB], [Math.ceil((M_MAX * qDim) / 64), 1, 1]);
        pushB(rmsnormWgsl(hd, S.rmsEps, true), [PB.qB, qNormW, PB.qN], [S.nHead * M_MAX, 1, 1]);
        pushB(rmsnormWgsl(hd, S.rmsEps, true), [PB.kCur, kNormW, PB.kN], [S.nKvHead * M_MAX, 1, 1]);
      }
      push(ropeNeoxWgsl(S.nHead, hd, S.ropeFreqBase, S.ropeDims), [qN], Math.ceil((S.nHead * S.ropeDims / 2) / 64), true);
      push(ropeNeoxWgsl(S.nKvHead, hd, S.ropeFreqBase, S.ropeDims), [kN], Math.ceil((S.nKvHead * S.ropeDims / 2) / 64), true);
      push(kvAppendWgsl(kvDim), [kN, kCache], Math.ceil(kvDim / 64), true);
      push(kvAppendWgsl(kvDim), [vCur, vCache], Math.ceil(kvDim / 64), true);
      pushAttnDecodeSplit(qN, kCache, vCache, attnO);
      push(sigmoidMulWgsl(qDim), [attnO, gateB], Math.ceil(qDim / 64));
      wo.push(attnO, attnY);
      if (PB) {
        pushB(ropeNeoxWgsl(S.nHead, hd, S.ropeFreqBase, S.ropeDims, true), [PB.qN, PB.rowPos], [Math.ceil((S.nHead * S.ropeDims / 2) / 64), M_MAX, 1]);
        pushB(ropeNeoxWgsl(S.nKvHead, hd, S.ropeFreqBase, S.ropeDims, true), [PB.kN, PB.rowPos], [Math.ceil((S.nKvHead * S.ropeDims / 2) / 64), M_MAX, 1]);
        pushB(kvAppendWgsl(kvDim, true), [PB.kN, kCache, PB.rowPos], [Math.ceil(kvDim / 64), M_MAX, 1]);
        pushB(kvAppendWgsl(kvDim, true), [PB.vCur, vCache, PB.rowPos], [Math.ceil(kvDim / 64), M_MAX, 1]);
        // l'append PRECEDE l'attenzione: e' cosi' che la riga m vede le righe
        // precedenti del chunk, e la causalita' non ha bisogno di maschere (it.26)
        pushB(attnDecodeWgsl({ nHead: S.nHead, nKvHead: S.nKvHead, headDim: hd, ctxMax, batch: true }), [PB.qN, kCache, vCache, PB.attnO, PB.rowPos], [S.nHead, M_MAX, 1]);
        pushB(sigmoidMulWgsl(qDim, true), [PB.attnO, PB.gateB], [Math.ceil(qDim / 64), M_MAX, 1]);
        wo.pushB(PB.attnO, PB.attnY);
      }
    } else {
      mark("ssmGemv");
      const wqkv = await loadW(`${b}attn_qkv.weight`);
      const wz = await loadW(`${b}attn_gate.weight`);
      const wb = await loadW(`${b}ssm_beta.weight`);
      const wa = await loadW(`${b}ssm_alpha.weight`);
      const convW = await f32buf(`${b}ssm_conv1d.weight`);
      const aBuf = await f32buf(`${b}ssm_a`);
      const dtBuf = await f32buf(`${b}ssm_dt.bias`);
      const nrmBuf = await f32buf(`${b}ssm_norm.weight`);
      const wOut = await loadW(`${b}ssm_out.weight`);
      const convSt = sbuf(new Float32Array((S.linConvK - 1) * qkvDim));
      const stateS = sbuf(new Float32Array(S.linVHead * S.linHeadDim * S.linHeadDim));
      stateBufs.push({ buf: convSt, bytes: (S.linConvK - 1) * qkvDim * 4, layer: l });
      stateBufs.push({ buf: stateS, bytes: S.linVHead * S.linHeadDim * S.linHeadDim * 4, layer: l });
      wqkv.push(xn, qkv);
      wz.push(xn, z);
      wb.push(xn, bRaw);
      wa.push(xn, aRaw);
      if (PB) {
        wqkv.pushB(PB.xn, PB.qkv);
        wz.pushB(PB.xn, PB.z);
        wb.pushB(PB.xn, PB.bRaw);
        wa.pushB(PB.xn, PB.aRaw);
        pushB(deltaNetGatesWgsl(S.linVHead, true), [PB.bRaw, PB.aRaw, aBuf, dtBuf, PB.bSig, PB.gVal], [1, M_MAX, 1]);
        // LA RICORRENZA: M step IN ORDINE, riga da uniform. Lo stato non e' per
        // riga — e' la memoria che attraversa il chunk (it.30).
        for (let m = 0; m < M_MAX; m++) {
          pushBRows1(deltaNetConvWgsl(qkvDim, S.linConvK, true), [convSt, PB.qkv, convW, PB.convOut], Math.ceil(qkvDim / 64), m);
          pushBRows1(deltaNetCoreWgsl({ hd: S.linHeadDim, nK: S.linKHead, nV: S.linVHead, eps: S.rmsEps }, true), [PB.convOut, stateS, PB.bSig, PB.gVal, PB.z, nrmBuf, PB.gated], S.linVHead, m);
          // subito dopo la riga CERTA: lo stato di questo layer finisce in ombra
          if (m === 0 && specSnap) {
            for (const [buf, n] of [
              [convSt, (S.linConvK - 1) * qkvDim],
              [stateS, S.linVHead * S.linHeadDim * S.linHeadDim],
            ] as [GPUBuffer, number][]) {
              const dst = empty(n * 4);
              specSnap.push({ src: buf, dst, n });
              pushB(stridedCopyWgsl({ nVec: 1, len: n, srcStride: n, srcOffset: 0, dstStride: n, dstOffset: 0 }), [buf, dst], [Math.ceil(n / 64), 1, 1]);
            }
          }
        }
      }
      // Da qui la RICORRENZA: conv shifta lo stato e core aggiorna S. E' la
      // parte che il batching M>1 non puo' comprimere (le righe di un chunk
      // sono token consecutivi), quindi va misurata da sola: e' il pavimento
      // della fase 4.
      mark("ssmRec");
      push(deltaNetGatesWgsl(S.linVHead), [bRaw, aRaw, aBuf, dtBuf, bSig, gVal], 1);
      push(deltaNetConvWgsl(qkvDim, S.linConvK), [convSt, qkv, convW, convOut], Math.ceil(qkvDim / 64));
      push(deltaNetCoreWgsl({ hd: S.linHeadDim, nK: S.linKHead, nV: S.linVHead, eps: S.rmsEps }), [convOut, stateS, bSig, gVal, z, nrmBuf, gated], S.linVHead);
      mark("ssmOut");
      wOut.push(gated, attnY);
      if (PB) wOut.pushB(PB.gated, PB.attnY);
    }
    mark("resid");
    push(addInPlaceWgsl(d), [x, attnY], Math.ceil(d / 64));
    if (PB) pushB(addInPlaceWgsl(d, true), [PB.x, PB.attnY], [Math.ceil(d / 64), M_MAX, 1]);

    if (!isMoe) {
      const wg = await q40(`${b}ffn_gate.weight`);
      const wu = await q40(`${b}ffn_up.weight`);
      const wd = await q40(`${b}ffn_down.weight`);
      push(rmsnormWgsl(d, S.rmsEps), [x, postNorm, xn], 1);
      gemv(wg, xn, gateF);
      gemv(wu, xn, upF);
      push(siluMulWgsl(dFfn), [gateF, upF], Math.ceil(dFfn / 64));
      gemv(wd, gateF, attnY);
      push(addInPlaceWgsl(d), [x, attnY], Math.ceil(d / 64));
      if (PB) {
        pushB(rmsnormWgsl(d, S.rmsEps, true), [PB.x, postNorm, PB.xn], [M_MAX, 1, 1]);
        gemvB(wg, PB.xn, PB.gateF);
        gemvB(wu, PB.xn, PB.upF);
        pushB(siluMulWgsl(dFfn, true), [PB.gateF, PB.upF], [Math.ceil(dFfn / 64), M_MAX, 1]);
        gemvB(wd, PB.gateF, PB.attnY);
        pushB(addInPlaceWgsl(d, true), [PB.x, PB.attnY], [Math.ceil(d / 64), M_MAX, 1]);
      }
    } else {
      // MoE (it.17): il segmento STATICO del layer chiude con shexp (statico:
      // non dipende dalla selezione, accumula in moeAcc col gate sigmoid su
      // GPU) + router → routerLogits. La parte per-expert è DINAMICA (readback
      // della selezione su CPU, correttezza-prima: 1 sync/layer DICHIARATO).
      mark("shexp");
      push(rmsnormWgsl(d, S.rmsEps), [x, postNorm, xn], 1);
      if (PB) pushB(rmsnormWgsl(d, S.rmsEps, true), [PB.x, postNorm, PB.xn], [M_MAX, 1, 1]);
      const shGate = await loadW(`${b}ffn_gate_shexp.weight`);
      const shUp = await loadW(`${b}ffn_up_shexp.weight`);
      const shDown = await loadW(`${b}ffn_down_shexp.weight`);
      const shScalarW = await f32buf(`${b}ffn_gate_inp_shexp.weight`);
      shGate.push(xn, gateS);
      shUp.push(xn, upS);
      push(siluMulWgsl(dE), [gateS, upS], Math.ceil(dE / 64));
      shDown.push(gateS, dnS);
      push(gemvF32Wgsl({ K: d, N: 1 }), [shScalarW, xn, shScalar], 1);
      if (PB) {
        shGate.pushB(PB.xn, PB.gateS);
        shUp.pushB(PB.xn, PB.upS);
        pushB(siluMulWgsl(dE, true), [PB.gateS, PB.upS], [Math.ceil(dE / 64), M_MAX, 1]);
        shDown.pushB(PB.gateS, PB.dnS);
        pushB(gemvF32Wgsl({ K: d, N: 1, batch: true }), [shScalarW, PB.xn, PB.shScalar], [1, 1, M_MAX]);
      }
      push(axpyWgsl(d, true), [moeAcc, dnS, shScalar], Math.ceil(d / 64));
      if (PB) pushB(axpyWgsl(d, true, true), [PB.moeAcc, PB.dnS, PB.shScalar], [Math.ceil(d / 64), M_MAX, 1]);
      mark("routerGemv");
      const router = await loadW(`${b}ffn_gate_inp.weight`);
      router.push(xn, routerLogits);
      if (PB) { router.pushB(PB.xn, PB.routerLogits); PB.cuts.push(PB.steps.length); }
      cuts.push(steps.length);
      moeLayerAbs.push(l);
    }
  }

  // CONFINE DELLA CODA (fase 4, it.19): da qui in giu' ci sono solo la norma
  // finale e la HEAD, e servono unicamente a chi legge i logits. Il prefill
  // gira con `read=false` e li calcolava lo stesso, buttandoli: la sonda dei
  // timestamp li ha misurati in **6,86 ms su 73,85 (9,3%) per token** sul 35B —
  // il pezzo piu' grosso che si toglie senza cambiare un solo numero, perche'
  // la head scrive `logits` (che nessuno legge) e la norma scrive `xn`, uno
  // scratch: il residuo `x` non viene sfiorato.
  const headCut = steps.length;
  push(rmsnormWgsl(d, S.rmsEps), [x, outNorm, xn], 1);
  headStep(xn, logits);

  const embdKind = embdInfo.type === GGML_TYPE.Q6_K ? "q6k" : embdInfo.type === GGML_TYPE.Q8_0 ? "q80" : "q40";
  const rowBlocks = embdKind === "q6k" ? d / 256 : d / 32;
  const rowBytes = embdKind === "q6k" ? (d / 256) * 210 : (d / 32) * (embdKind === "q80" ? 34 : 18);
  const embRow = new Float32Array(d);
  const dispatchesPerToken = steps.length;
// ====== residenza expert: LA MECCANICA UNICA (goal fase-D fase 1, it.7) ======
// L'arena a chunk + LRU + bind group scritte a mano qui in q1 it.16 sono
// SPARITE: questa è la stessa `ExpertCache` che serve GLM, guidata da una
// `MoeModelConfig` costruita dai metadata del GGUF. Il layout degli slab che
// ne esce coincide BYTE PER BYTE con quello che questo file calcolava a mano
// (test `q35-slab-parity`, CPU-only), quindi la migrazione non sposta un byte
// in VRAM: cambia il proprietario del meccanismo — e con lui arrivano a Qwen
// le ottimizzazioni che GLM aveva e questo path no (ruling direction §7-ter).
  interface MoeRt {
    nSlots: Record<string, number>; parkSlots: Record<string, number>; slotBytes: Record<string, number>;
    routing: Map<number, number>;
    stats(): { hits: number; misses: number; uploadedBytes: number; readMs: number; packMs: number; uploadMs: number };
    runLayer(l: number, logitsF32: Float32Array): Promise<void>;
    /**
     * OMBRA (fase 3b, fetta 3b): accoda il router+resolve su GPU nello STESSO
     * encoder del segmento statico, subito dopo il GEMV che ha scritto i
     * logits, e copia la `Sel` d'ombra nella staging. `null` se l'ombra è
     * spenta — il path di produzione non cambia di una riga.
     */
    shadowEncode: ((enc: GPUCommandEncoder, m: number) => void) | null;
    /** Confronta l'ombra col la selezione CPU del layer appena eseguito. */
    shadowCompare: (() => Promise<void>) | null;
    shadowStats: (() => Q35RouterShadowStats) | null;
    /**
     * IL TOKEN INTERO IN UN SUBMIT (fetta 3c). `null` se il modello non è stato
     * costruito con `select: "optimistic"`. Ritorna i logits se `read`, e fa da
     * sé tutta la contabilità di fine token (routing, hit, miss): nel path
     * sync quella contabilità la fa `runLayer`, qui non c'è nessun momento in
     * cui la CPU veda la selezione prima della fine.
     */
    runTokenOptimistic: ((read: boolean) => Promise<Float32Array | null>) | null;
    /**
     * La META' CPU di `runLayer`: selezione, `ensure` dei mancanti, `Sel`,
     * flush della slotTable. Separata perche' il prefill a chunk la chiama per
     * OGNI riga del chunk e poi encoda tutte le catene expert; la selezione
     * resta identica a quella del path sequenziale, ed e' il motivo per cui il
     * gate della fase 4 e' la bit-identita' (it.34).
     */
    prepLayer(m: number, logitsF32: Float32Array, row?: number, pinAll?: Set<number>): Promise<void>;
    /** La meta' GPU: i dispatch expert del layer, in un pass gia' aperto. */
    encodeExperts(pass: GPUComputePassEncoder, m: number, addBg: GPUBindGroup, row?: number): void;
    /**
     * L'UNIONE degli expert scelti da tutte le righe del chunk per un layer.
     * Vive qui perche' `routerSelect` e la cache stanno in questa chiusura, e
     * serve PRIMA di qualunque `ensure`: e' il pin che impedisce alla riga r+1
     * di evincere uno slot che i dispatch gia' encodati della riga r leggeranno.
     */
    pinUnion(m: number, logitsAll: Float32Array, rows: number, nExpertLogits: number): Set<number>;
    /** SOLO HARNESS: svuota la cache expert (misura a freddo ripetibile). */
    evictAll(): void;
    /** cablaggio del prefill a chunk; `null` senza `prefillM` */
    chunkRt: { bgAddRow: GPUBindGroup[]; routerStagingM: GPUBuffer } | null;
    destroy(): void;
  }
  let moe: MoeRt | null = null;
  if (isMoe) {
    // i NOMI dei tensori expert e i loro byte stanno in `q35expertstore`, il
    // gemello di `expertstore` (GLM): qui non si nomina nessun tensore expert
    // e non si calcola nessun offset di slab.
    const cfg = q35MoeConfig(S, (n) => r.info(n));
    const readExpert = q35ExpertReader(S, (n) => r.info(n), r.readRange!.bind(r));
    // I limiti sono quelli NEGOZIATI col device (il conf worker chiede
    // slabClassBytes = 2 GiB): usarli invece di una costante evita il buffer
    // monolitico che in it.17 falliva in silenzio oltre il cap dell'adapter.
    // BUDGET DERIVATO (it.35): il tetto meno cio' che e' gia' stato allocato
    // meno la riserva. `allocBytes` non e' una stima — e' la somma contata dove
    // i buffer si chiedono, quindi comprende pesi, KV, scratch e (se acceso) il
    // piano a M righe del prefill, senza che nessuno debba ricordarsene.
    const reserveB = opts.vramReserveBytes ?? 512 * (1 << 20);
    const budgetDerived = opts.vramCeilingBytes !== undefined
      ? opts.vramCeilingBytes - allocBytes - reserveB
      : arenaBudgetBytes;
    if (opts.vramCeilingBytes !== undefined && budgetDerived <= 0) {
      throw new Error(
        `q35 budget expert: ${budgetDerived} B <= 0 (tetto ${opts.vramCeilingBytes} − allocati ` +
        `${allocBytes} − riserva ${reserveB}) — servono piu' VRAM o meno contesto, e dirlo qui e' ` +
        "meglio che scoprirlo con una createBuffer che fallisce a meta' caricamento");
    }
    vramPlanOut = {
      ceilingBytes: opts.vramCeilingBytes ?? null, allocatedBytes: allocBytes,
      reserveBytes: reserveB, expertBudgetBytes: budgetDerived, ctxMax,
    };
    const cache = new ExpertCache(device, {
      budgetBytes: budgetDerived,
      maxBindingBytes: device.limits.maxStorageBufferBindingSize,
      maxBufferBytes: device.limits.maxBufferSize,
      cfg,
      // telemetria liv.1 SEMPRE accesa su q35: sono 4 performance.now() per
      // MISS (non per token, non sugli hit) contro una lettura Range da ~1,7 MB
      // — non misurabile. In cambio ogni run di conformance porta con sé la
      // scomposizione read/pack/upload, che è il numero su cui la fase 2
      // decide.
      timing: true,
      // REGIME D'ARENA (fase 3b, fetta 3a). I buffer di classe si bindano
      // INTERI e l'indirizzo dello slab lo ricava il KERNEL dallo slot che
      // legge in `Sel`. È la precondizione per togliere il readback per layer:
      // finché è la CPU a calcolare i sotto-range, deve conoscere la selezione
      // PRIMA di poter accodare il layer, e quindi deve leggere i logits del
      // router. Non cambia né quanti slot ci sono né quanta VRAM costano
      // (`expertSlots` non guarda il regime): cambia la taglia dei buffer di
      // classe, e con essa quanti sono.
      policy: opts.expertPolicy ?? "lru",
      arena: true,
      // La tabella expertKey→slot in VRAM non la legge ancora nessuno in
      // questa fetta (il router è su CPU). Si accende qui perché è lo stesso
      // `ensure` a tenerla aggiornata: in fetta 3b il resolve su GPU la trova
      // popolata dal primo token invece che dal secondo.
      slotTable: true,
    });
    const routerCfg: RouterConfig = { ...ROUTER_QWEN35MOE, nUsed: topK };
    const shadowOn = opts.routerShadow === true;
    if (shadowOn && opts.select === "optimistic") {
      throw new Error(
        "q35gpumodel: routerShadow e select \"optimistic\" insieme non hanno senso — l'ombra " +
        "confronta il router GPU con la selezione CPU, e in optimistic la selezione CPU non esiste");
    }
    const nMoeLayer = cuts.length;
    const nSel = nMoeLayer * topK;
    // `MoeIdx` si binda a dynamic offset: la spaziatura delle entry deve essere
    // un multiplo dell'allineamento CONCESSO, non del 256 di spec.
    if (MOE_IDX_STRIDE % device.limits.minUniformBufferOffsetAlignment !== 0) {
      throw new Error(
        `q35 MoE arena: stride ${MOE_IDX_STRIDE} non multiplo di ` +
        `minUniformBufferOffsetAlignment ${device.limits.minUniformBufferOffsetAlignment}`);
    }
    // `Sel`: una entry per (layer MoE, k) — {id, slot, w, flags}. In questa
    // fetta la riempie la CPU dopo gli `ensure`, esattamente come faceva con i
    // sotto-range; in fetta 3c ci scriverà il router su GPU e la CPU non
    // toccherà più questo buffer nel path caldo. È dimensionata per tutto il
    // token (40×8 entry, 5 120 B) e non per un layer: il layout non deve
    // cambiare quando i 40 submit diventano uno.
    // In modo OMBRA il buffer RADDOPPIA (design §4 di GLM): [0, nSel) è la
    // produzione — quella che i kernel expert leggono e che riempie la CPU —,
    // [nSel, 2·nSel) è l'ombra dove scrive il resolve GPU. Due regioni dello
    // stesso buffer e non due buffer: lo stesso kernel serve la fetta 3c
    // cambiando solo la entry di uniform che lo indirizza.
    // Nel prefill a chunk la `Sel` ha una dimensione in piu': la RIGA. Senza,
    // le M scritture del layer finirebbero tutte allo stesso offset e — visto
    // che `queue.writeBuffer` e' ordinata PRIMA del submit — ogni riga
    // userebbe la selezione dell'ULTIMA. E' il primo dei due bug che il gate
    // bit-identico ha preso (it.34).
    const selRows = prefillOn ? M_MAX : 1;
    const nSelTot = (shadowOn ? 2 * nSel : nSel) * selRows;
    const selBuf = device.createBuffer({
      size: nSelTot * SEL_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    const selScratch = new ArrayBuffer(topK * SEL_BYTES);
    const selU32 = new Uint32Array(selScratch);
    const selF32 = new Float32Array(selScratch);
    // `MoeIdx` è STATICA: contenuto noto al load, una scrittura sola. Il dynamic
    // offset sceglie la entry, la entry dice quale `Sel` leggere. `tableBase` è
    // la base del layer nella slotTable e la userà il resolve GPU (fetta 3b).
    // In ombra servono nMoeLayer entry IN PIÙ, una per layer, che indirizzano
    // il router: stesso `tableBase`, ma `selIdx` già spostato nell'ombra. È
    // così che il kernel di resolve resta identico fra ombra e produzione —
    // cambia la entry, non il WGSL.
    const nIdx = nSel * selRows + (shadowOn ? nMoeLayer : 0);
    const moeIdxUni = device.createBuffer({
      size: nIdx * MOE_IDX_STRIDE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    {
      const u = new Uint32Array(nIdx * (MOE_IDX_STRIDE / 4));
      for (let row = 0; row < selRows; row++) {
        for (let m = 0; m < nMoeLayer; m++) {
          const tableBase = moeLayerAbs[m] * cfg.nExpert;
          for (let k = 0; k < topK; k++) {
            const selIdx = (row * nMoeLayer + m) * topK + k;
            const w = selIdx * (MOE_IDX_STRIDE / 4);
            u[w] = selIdx; u[w + 1] = tableBase; u[w + 2] = m; u[w + 3] = 0;
          }
        }
      }
      if (shadowOn) {
        for (let m = 0; m < nMoeLayer; m++) {
          const w = (nSel + m) * (MOE_IDX_STRIDE / 4);
          u[w] = nSel + m * topK; u[w + 1] = moeLayerAbs[m] * cfg.nExpert; u[w + 2] = m; u[w + 3] = 0;
        }
      }
      device.queue.writeBuffer(moeIdxUni, 0, u as unknown as BufferSource);
    }
    // i kernel si scelgono dal LAYOUT della classe, non da un'assunzione sul
    // formato: se un GGUF della famiglia arrivasse con gate/up diversi, qui
    // si ferma con un messaggio invece di dequantizzare byte sbagliati.
    const gk = cfg.layout(cfg.classes[0]).gate.kind;
    if (gk !== "q4_K") throw new Error(`q35 MoE: nessun kernel gemv per gate/up ${gk}`);
    /**
     * La catena expert di UNA classe in regime d'arena (PORT da `glmmodel`).
     * Il bind group layout è ESPLICITO e non `"auto"`: `hasDynamicOffset` non
     * si esprime con layout auto, ed è l'offset dinamico a scegliere la entry
     * di `Sel`. Il risultato è che i bind group sono TRE per classe — gate, up,
     * down — invece di tre per slot: l'indirizzo non è più nel bind group.
     */
    const mkExpertClass = (cls: ExpertClass) => {
      const geo = cache.arenaGeometry(cls);
      const need = expertArenaBindings(geo.nBuf);
      if (need > device.limits.maxStorageBuffersPerShaderStage) {
        throw new Error(
          `q35 MoE arena: la classe ${cls} ha ${geo.nBuf} buffer d'arena ⇒ ${need} storage binding, ` +
          `il device ne concede ${device.limits.maxStorageBuffersPerShaderStage} — negoziare ` +
          "maxStorageBuffersPerShaderStage con arenaBuffers (gpulimits/arenaNeeds)");
      }
      const L = cfg.layout(cls);
      const dk = L.down.kind;
      if (dk !== "q4_K" && dk !== "q6_K") throw new Error(`q35 MoE: nessun kernel gemv per down ${dk}`);
      const kar = (t: SlabTensorLayout): KArenaOpts => ({
        nBuf: geo.nBuf, slabWords: geo.slabWords, slabsPerBuf: geo.slabsPerBuf, tensorWords: t.data / 4,
      });
      const bgl = device.createBindGroupLayout({
        entries: [
          ...Array.from({ length: geo.nBuf }, (_, j) => ({
            binding: j, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" as const },
          })),
          { binding: geo.nBuf, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" as const } },
          { binding: geo.nBuf + 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } },
          { binding: geo.nBuf + 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" as const } },
          {
            binding: geo.nBuf + 3, visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "uniform" as const, hasDynamicOffset: true, minBindingSize: MOE_IDX_BYTES },
          },
        ],
      });
      const layout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
      const mkPipe = (code: string): GPUComputePipeline => device.createComputePipeline({
        layout, compute: { module: device.createShaderModule({ code }), entryPoint: "main" },
      });
      const bg = (src: GPUBuffer, dst: GPUBuffer): GPUBindGroup => device.createBindGroup({
        layout: bgl,
        entries: [
          ...cache.arenaBuffers(cls).map((b, j) => ({ binding: j, resource: { buffer: b } })),
          { binding: geo.nBuf, resource: { buffer: src } },
          { binding: geo.nBuf + 1, resource: { buffer: dst } },
          { binding: geo.nBuf + 2, resource: { buffer: selBuf } },
          // `size` ESPLICITA: con hasDynamicOffset la validazione pretende
          // offset+dynamicOffset+size <= buffer.size, e senza `size` il binding
          // varrebbe l'intero buffer ⇒ qualunque offset dinamico > 0 sarebbe
          // illegale (glmmodel:1007 documenta la stessa trappola).
          { binding: geo.nBuf + 3, resource: { buffer: moeIdxUni, offset: 0, size: MOE_IDX_BYTES } },
        ],
      });
      return {
        nBuf: geo.nBuf,
        pGate: mkPipe(gemvQ4KWgsl({ K: d, N: dE, arena: kar(L.gate) })),
        pUp: mkPipe(gemvQ4KWgsl({ K: d, N: dE, arena: kar(L.up) })),
        // il down ACCUMULA in `moeAcc` col peso da `Sel` (fase 4, it.21):
        // l'axpy che seguiva non esiste piu' — un dispatch in meno per expert,
        // 320 per token sul 35B, e bit-identico per costruzione.
        pDown: mkPipe(dk === "q6_K"
          ? gemvQ6KWgsl({ K: dE, N: d, arena: kar(L.down), accum: true })
          : gemvQ4KWgsl({ K: dE, N: d, arena: kar(L.down), accum: true })),
        bgGate: bg(xn, gateE), bgUp: bg(xn, upE), bgDown: bg(gateE, moeAcc),
      };
    };
    const expertCls: Record<ExpertClass, ReturnType<typeof mkExpertClass>> = {};
    for (const c of cfg.classes) expertCls[c] = mkExpertClass(c);
    const pSilu = pipe(siluMulWgsl(dE));
    const pAdd = pipe(addInPlaceWgsl(d));
    const mkBg = (p2: GPUComputePipeline, entries: (GPUBuffer | GPUBufferBinding)[]): GPUBindGroup =>
      device.createBindGroup({
        layout: p2.getBindGroupLayout(0),
        entries: entries.map((e, i) => ({ binding: i, resource: (e as GPUBufferBinding).buffer ? (e as GPUBufferBinding) : { buffer: e as GPUBuffer } })),
      });
    const bgSilu = mkBg(pSilu, [gateE, upE]);
    const bgAddRes = mkBg(pAdd, [x, moeAcc]);
    // Nel prefill a chunk il residuo va sulla RIGA: stesso kernel, stesso
    // `moeAcc`, solo la destinazione cambia (sotto-range di PB.x).
    const bgAddRow = PB ? Array.from({ length: M_MAX }, (_, m2) => device.createBindGroup({
      layout: pAdd.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: PB.x, offset: m2 * d * 4, size: d * 4 } },
        { binding: 1, resource: { buffer: moeAcc } },
      ],
    })) : null;
    const routerStagingM = PB
      ? device.createBuffer({ size: Math.max(16, M_MAX * nE * 4), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
      : null;
    // ---- OMBRA: router + resolve su GPU accanto alla selezione CPU (fetta 3b) ----
    // Il kernel è quello di it.13, in configurazione qwen35moe (softmax). Il
    // binding `bias` resta dichiarato anche qui e ci si lega un buffer di ZERI:
    // `probs[i] + 0.0` è esatto ed è letteralmente ciò che `routerSelect` fa
    // quando la config non usa il bias — così il layout non dipende dalla
    // famiglia (scelta motivata in it.13).
    const shadow = shadowOn ? ((): {
      encode: (enc: GPUCommandEncoder, m: number) => void;
      compare: () => Promise<void>;
      stats: () => Q35RouterShadowStats;
      note: (sel: { experts: ArrayLike<number>; weights: ArrayLike<number> }, slots: SlotRef[], missing: Set<number>, logits: Float32Array) => void;
    } => {
      const bgl = device.createBindGroupLayout({
        entries: [
          ...[0, 1].map((b) => ({ binding: b, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" as const } })),
          ...[2, 3, 4].map((b) => ({ binding: b, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } })),
          { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" as const } },
          {
            binding: 6, visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "uniform" as const, hasDynamicOffset: true, minBindingSize: MOE_IDX_BYTES },
          },
        ],
      });
      const code = routerTopKWgsl({
        nExpert: nE, nUsed: topK, weightsScale: routerCfg.weightsScale,
        clampMin: WEIGHTS_SUM_CLAMP_MIN, gating: routerCfg.gating,
        resolve: { nExpert: nE, nUsed: topK },
      });
      const pipeR = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
        compute: { module: device.createShaderModule({ code }), entryPoint: "main" },
      });
      const zeroBias = sbuf(new Float32Array(nE));
      const ids = empty(topK * 4), wts = empty(topK * 4);
      const bind = device.createBindGroup({
        layout: bgl,
        entries: [
          { binding: 0, resource: { buffer: routerLogits } },
          { binding: 1, resource: { buffer: zeroBias } },
          { binding: 2, resource: { buffer: ids } },
          { binding: 3, resource: { buffer: wts } },
          { binding: 4, resource: { buffer: selBuf } },
          { binding: 5, resource: { buffer: cache.slotTableBuffer() } },
          { binding: 6, resource: { buffer: moeIdxUni, offset: 0, size: MOE_IDX_BYTES } },
        ],
      });
      const selStaging = device.createBuffer({
        size: topK * SEL_BYTES, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      const st: Q35RouterShadowStats = {
        comparisons: 0, picks: 0, setFlips: 0, orderFlips: 0, maxWeightRelErr: 0,
        minLogitMargin: Infinity, slotMismatch: 0, missGpu: 0, missCpu: 0, missDisagree: 0,
      };
      // ultima selezione CPU del layer, messa da parte per il confronto
      let cpuIds: number[] = [], cpuW: number[] = [], cpuSlots: number[] = [], cpuMiss: boolean[] = [];
      let cpuMargin = Infinity;
      return {
        encode: (enc, m) => {
          const p2 = enc.beginComputePass();
          p2.setPipeline(pipeR);
          p2.setBindGroup(0, bind, [(nSel + m) * MOE_IDX_STRIDE]);
          p2.dispatchWorkgroups(1);
          p2.end();
          enc.copyBufferToBuffer(selBuf, (nSel + m * topK) * SEL_BYTES, selStaging, 0, topK * SEL_BYTES);
        },
        note: (sel, slots, missing, lg) => {
          cpuIds = Array.from(sel.experts as ArrayLike<number>).slice(0, topK);
          cpuW = Array.from(sel.weights as ArrayLike<number>).slice(0, topK);
          cpuSlots = slots.map((s) => s.idx);
          cpuMiss = cpuIds.map((e) => missing.has(e));
          // margine sui LOGIT fra l'ultimo preso e il primo scartato: la
          // softmax è monotona, quindi l'ordinamento lo decide questo numero.
          // È la misura di "quanto era difficile il caso", senza la quale uno
          // zero flip non dice se il gate ha visto il caso stretto.
          const taken = new Set(cpuIds);
          let lastIn = Infinity, firstOut = -Infinity;
          for (let i = 0; i < nE; i++) {
            if (taken.has(i)) { if (lg[i] < lastIn) lastIn = lg[i]; }
            else if (lg[i] > firstOut) firstOut = lg[i];
          }
          cpuMargin = lastIn - firstOut;
        },
        compare: async () => {
          await selStaging.mapAsync(GPUMapMode.READ);
          const raw = selStaging.getMappedRange().slice(0);
          selStaging.unmap();
          const u = new Uint32Array(raw), fl = new Float32Array(raw);
          st.comparisons++;
          st.picks += topK;
          if (cpuMargin < st.minLogitMargin) st.minLogitMargin = cpuMargin;
          const gpuIds: number[] = [];
          for (let k = 0; k < topK; k++) gpuIds.push(u[k * 4]);
          const same = new Set(gpuIds);
          let setEq = same.size === cpuIds.length;
          if (setEq) for (const e of cpuIds) if (!same.has(e)) { setEq = false; break; }
          if (!setEq) st.setFlips++;
          else if (gpuIds.some((e, k) => e !== cpuIds[k])) st.orderFlips++;
          for (let k = 0; k < topK; k++) if (cpuMiss[k]) st.missCpu++;
          // Slot e miss si appaiano per EXPERT, non per posizione: con un flip
          // d'ordine la posizione k porterebbe expert diversi e il confronto
          // sarebbe un falso allarme (o peggio, un falso via libera).
          const cpuSlotOf = new Map<number, number>();
          const cpuMissOf = new Map<number, boolean>();
          const cpuWOf = new Map<number, number>();
          cpuIds.forEach((e, i) => { cpuSlotOf.set(e, cpuSlots[i]); cpuMissOf.set(e, cpuMiss[i]); cpuWOf.set(e, cpuW[i]); });
          for (let k = 0; k < topK; k++) {
            const e = gpuIds[k], slotGpu = u[k * 4 + 1], missFlag = (u[k * 4 + 3] & 1) === 1;
            if (missFlag) st.missGpu++;
            if (!cpuSlotOf.has(e)) continue; // expert non scelto dalla CPU: è già un setFlip
            const wGpu = fl[k * 4 + 2], wCpu = cpuWOf.get(e)!;
            if (Math.abs(wCpu) > 1e-12) {
              const rel = Math.abs(wGpu - wCpu) / Math.abs(wCpu);
              if (rel > st.maxWeightRelErr) st.maxWeightRelErr = rel;
            }
            // L'ombra gira PRIMA degli `ensure` del layer: un expert che la CPU
            // sta per caricare si risolve MISS, ed è la residenza parziale vista
            // da GPU, non un errore. Il disaccordo è l'altro verso: la GPU dice
            // MISS su un expert che era GIÀ residente, o dà uno slot diverso da
            // quello che la CPU ha poi usato per un expert residente.
            if (missFlag !== cpuMissOf.get(e)) st.missDisagree++;
            else if (!missFlag && slotGpu !== cpuSlotOf.get(e)) st.slotMismatch++;
          }
        },
        stats: () => ({ ...st }),
      };
    })() : null;
    // ---- PATH A SUBMIT UNICO (fetta 3c): la `Sel` di produzione la scrive la GPU ----
    // Cambia UN pezzo rispetto alla fetta 3a: chi riempie `Sel`. I kernel
    // expert sono gli stessi, il bind group è lo stesso, l'offset dinamico è lo
    // stesso — l'indirezione era stata costruita apposta perché questo passo
    // costasse una entry di uniform e non una riscrittura (design §1).
    const opt = opts.select === "optimistic" ? ((): {
      dirtyB: GPUBuffer; dirtyStaging: GPUBuffer; selStaging: GPUBuffer;
      pipeR: GPUComputePipeline; bindR: GPUBindGroup;
      hiddenCkpt: GPUBuffer;
      stateShadow: { src: GPUBuffer; dst: GPUBuffer; bytes: number; layer: number }[];
      stateShadowBytes: number;
      destroy(): void;
    } => {
      const dirtyB = device.createBuffer({
        size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      const dirtyStaging = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const selStaging = device.createBuffer({
        size: nSel * SEL_BYTES, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      // Router+resolve di PRODUZIONE: lo stesso kernel dell'ombra con in più il
      // binding 7 (`dirtyB`) — [0] = primo layer MoE con miss (atomicMin), [1] =
      // conteggio (atomicAdd). È l'unica differenza di WGSL fra ombra e
      // produzione; la differenza di CABLAGGIO è la entry di `MoeIdx` che
      // l'offset dinamico seleziona, e quella di produzione è la (layer, k=0),
      // il cui `selIdx` punta alla regione che i kernel expert leggono.
      const bgl = device.createBindGroupLayout({
        entries: [
          ...[0, 1].map((b) => ({ binding: b, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" as const } })),
          ...[2, 3, 4].map((b) => ({ binding: b, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } })),
          { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" as const } },
          {
            binding: 6, visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "uniform" as const, hasDynamicOffset: true, minBindingSize: MOE_IDX_BYTES },
          },
          { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } },
        ],
      });
      const pipeR = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
        compute: {
          module: device.createShaderModule({
            code: routerTopKWgsl({
              nExpert: nE, nUsed: topK, weightsScale: routerCfg.weightsScale,
              clampMin: WEIGHTS_SUM_CLAMP_MIN, gating: routerCfg.gating,
              resolve: { nExpert: nE, nUsed: topK, dirty: true },
            }),
          }),
          entryPoint: "main",
        },
      });
      const zeroBias = sbuf(new Float32Array(nE));
      const ids = empty(topK * 4), wts = empty(topK * 4);
      const bindR = device.createBindGroup({
        layout: bgl,
        entries: [
          { binding: 0, resource: { buffer: routerLogits } },
          { binding: 1, resource: { buffer: zeroBias } },
          { binding: 2, resource: { buffer: ids } },
          { binding: 3, resource: { buffer: wts } },
          { binding: 4, resource: { buffer: selBuf } },
          { binding: 5, resource: { buffer: cache.slotTableBuffer() } },
          { binding: 6, resource: { buffer: moeIdxUni, offset: 0, size: MOE_IDX_BYTES } },
          { binding: 7, resource: { buffer: dirtyB } },
        ],
      });
      // hiddenCkpt: l'hidden di INGRESSO di ogni segmento MoE. È l'input del
      // replay — il rientro deve essere BIT-IDENTICO a ciò che il layer aveva
      // visto nel giro ottimistico, altrimenti il replay non ripara: ricalcola
      // un'altra cosa.
      const hiddenCkpt = empty(nMoeLayer * d * 4);
      // stateShadow: LA DIFFERENZA STRUTTURALE RISPETTO A GLM. Il replay di GLM
      // è idempotente perché i suoi layer sono senza stato (il KV append
      // riscrive la stessa posizione con gli stessi indici). Qui 30 layer su 40
      // sono deltanet e aggiornano `convSt`/`stateS` IN PLACE: rigiocarli
      // applicherebbe l'aggiornamento DUE VOLTE, e non con un errore — con uno
      // stato ricorrente sbagliato e numeri plausibili.
      //
      // Lo snapshot costa poco perché ogni layer tocca il proprio stato UNA
      // volta per token: lo stato all'ingresso del LAYER è lo stato all'ingresso
      // del TOKEN. Quindi una copia sola a inizio token serve a tutti i round di
      // replay, che ripartono sempre più in basso.
      const stateShadow = stateBufs.map((s) => ({
        src: s.buf,
        dst: device.createBuffer({ size: Math.max(16, s.bytes), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC }),
        bytes: s.bytes, layer: s.layer,
      }));
      const stateShadowBytes = stateShadow.reduce((a, s) => a + s.bytes, 0);
      return {
        dirtyB, dirtyStaging, selStaging, pipeR, bindR,
        hiddenCkpt, stateShadow, stateShadowBytes,
        destroy: () => {
          dirtyB.destroy(); dirtyStaging.destroy(); selStaging.destroy(); hiddenCkpt.destroy();
          for (const s of stateShadow) s.dst.destroy();
        },
      };
    })() : null;
    /** sentinel di `dirtyB`: [0] è un atomicMin sul layer MoE, [1] un atomicAdd */
    const dirtyInit = new Uint32Array([0xffffffff, 0, 0, 0]);
    const routing = new Map<number, number>();
    const parkSlots = moeParkOf(cfg);
    const slotBytes: Record<string, number> = {};
    for (const c of cfg.classes) slotBytes[c] = cfg.layout(c).bytes;
    moe = {
      nSlots: cache.stats().slots, parkSlots, slotBytes, routing,
      stats: () => {
      const st = cache.stats();
      return {
        hits: st.hits, misses: st.misses, uploadedBytes: st.bytesUploaded,
        readMs: st.readMs, packMs: st.packMs, uploadMs: st.uploadMs,
      };
    },
      destroy: () => { opt?.destroy(); cache.destroy(); },
      shadowEncode: shadow ? (enc, m) => shadow.encode(enc, m) : null,
      shadowCompare: shadow ? () => shadow.compare() : null,
      shadowStats: shadow ? () => shadow.stats() : null,
      chunkRt: PB && bgAddRow && routerStagingM ? { bgAddRow, routerStagingM } : null,
      runTokenOptimistic: opt ? async (read: boolean): Promise<Float32Array | null> => {
        // ---- UN GIRO: encode da `startLayer` + submit + readback di coda ----
        // `startLayer` 0 è il giro normale; > 0 è il REPLAY, che rientra
        // dall'hidden checkpointato e rimette a posto lo stato ricorrente dei
        // soli layer che sta per rigiocare.
        const runPass = async (startLayer: number, first: boolean): Promise<{
          su: Uint32Array; lg: Float32Array | null; missCount: number; firstDirty: number;
        }> => {
          // reset di `dirtyB` PER GIRO: `queue.writeBuffer` è ordinata prima
          // dell'intero submit che segue, quindi arriva prima di ogni resolve.
          device.queue.writeBuffer(opt.dirtyB, 0, dirtyInit as unknown as BufferSource);
          // Error scope come CONTRATTO di ogni encode nuovo (landmine B1): una
          // pipeline invalida fa droppare il submit IN SILENZIO e i readback
          // tornano stale-ma-plausibili.
          device.pushErrorScope("validation");
          device.pushErrorScope("out-of-memory");
          const tEnc = performance.now();
          const enc = device.createCommandEncoder();
          if (first) {
            if (opts.debugNoStateSnapshot !== true) {
            // Snapshot dello stato ricorrente all'ingresso del TOKEN. Uno solo
            // per token basta a tutti i round: ogni layer tocca il proprio
            // stato una volta sola, quindi stato-all-ingresso-del-layer ==
            // stato-all-ingresso-del-token.
              for (const s of opt.stateShadow) enc.copyBufferToBuffer(s.src, 0, s.dst, 0, s.bytes);
            }
          } else if (opts.debugNoStateSnapshot === true) {
            throw new Error(
              "q35 optimistic: replay richiesto con debugNoStateSnapshot — lo stato ricorrente non e' " +
              "stato salvato, quindi il restore leggerebbe ombre stantie. La sonda si usa solo a cache " +
              "calda, dove i token sporchi sono 0.");
          } else {
            // Rientro del replay: x torna a essere l'hidden che il layer aveva
            // visto nel giro sporco, BIT-IDENTICO — è la condizione per cui il
            // router del layer riscelga gli stessi expert e, coi miss ormai
            // riparati, li risolva puliti.
            //
            // `first` e `startLayer === 0` NON sono la stessa cosa, e confonderli
            // e' stato un bug vero (it.18, misurato al primo token a cache
            // fredda): a cache vuota il primo layer sporco E' lo zero, quindi il
            // replay riparte proprio da li' — e senza questo rientro `x`
            // conterrebbe l'uscita del giro precedente invece dell'embedding.
            // Il router di layer 0 sceglierebbe altri 8 expert, non residenti,
            // e il round successivo troverebbe di nuovo il layer 0 sporco: il
            // guard sul progresso alza, ma la causa e' qui.
            enc.copyBufferToBuffer(opt.hiddenCkpt, startLayer * d * 4, x, 0, d * 4);
            // E lo stato ricorrente dei layer RIGIOCATI torna quello di
            // partenza. Senza, `deltaNetConv`/`deltaNetCore` applicherebbero
            // l'aggiornamento due volte: nessun errore, uno stato sbagliato.
            // I layer a monte NON si rigiocano e il loro stato è già giusto.
            const fromAbs = startLayer === 0 ? 0 : moeLayerAbs[startLayer - 1] + 1;
            for (const s of opt.stateShadow) {
              if (s.layer >= fromAbs) enc.copyBufferToBuffer(s.dst, 0, s.src, 0, s.bytes);
            }
          }
          const gg2 = gemvGrid(d);
          // Pass per CATEGORIA (fase 4, it.19). Con la sonda spenta `usePass`
          // non spezza mai: resta UN pass per layer, cioe' esattamente la forma
          // che gira in produzione. Con la sonda accesa ne apre tre e li
          // cronometra — piu' barriere, quindi un token un po' piu' lento: la
          // perturbazione si riporta, non si assume trascurabile.
          let tsIdx = 0;
          const cats: string[] = [];
          let cur: GPUComputePassEncoder | null = null;
          let curCat = "";
          const endPass = (): void => { if (cur) { cur.end(); cur = null; } };
          const usePass = (cat: string): GPUComputePassEncoder => {
            if (cur && canGpuTs && cat !== curCat) endPass();
            if (cur) return cur;
            curCat = cat;
            if (canGpuTs && tsIdx < TSQ_PASSES) {
              cats[tsIdx] = cat;
              cur = enc.beginComputePass({
                timestampWrites: {
                  querySet: tsqSet!, beginningOfPassWriteIndex: tsIdx * 2, endOfPassWriteIndex: tsIdx * 2 + 1,
                },
              });
              tsIdx++;
            } else {
              if (canGpuTs) tsqOverflow++;
              cur = enc.beginComputePass();
            }
            return cur;
          };
          for (let m = startLayer; m < nMoeLayer; m++) {
            // checkpoint dell'hidden di INGRESSO del segmento, fuori dal pass
            // (i dispatch che hanno prodotto questo x sono già chiusi).
            // Come lo snapshot, serve SOLO al replay: la sonda di misura lo
            // salta insieme a quello (fase 4-ter, it.28).
            if (opts.debugNoStateSnapshot !== true) {
              enc.copyBufferToBuffer(x, 0, opt.hiddenCkpt, m * d * 4, d * 4);
            }
            // `moeAcc` si azzera DENTRO l'encoder. La `queue.writeBuffer` del
            // path sync qui non funzionerebbe: è ordinata PRIMA dell'intero
            // submit, quindi i 40 layer vedrebbero un solo azzeramento —
            // l'ultimo.
            enc.clearBuffer(moeAcc, 0, d * 4);
            // Con la sonda spenta e' UN pass "static" per layer; accesa, si
            // spezza sulle marche (norm/attn/ssm/resid/shexp/routerGemv) — e'
            // l'unico modo di sapere quanto del segmento e' ricorrente.
            let pass = usePass("static");
            for (let i = m === 0 ? 0 : cuts[m - 1]; i < cuts[m]; i++) {
              if (canGpuTs) {
                const mk = markAt.get(i);
                if (mk !== undefined) pass = usePass(mk);
              }
              const st = steps[i];
              pass.setPipeline(st.pipe);
              pass.setBindGroup(0, st.bind);
              pass.dispatchWorkgroups(st.wgs[0], st.wgs[1], st.wgs[2]);
            }
            // router+resolve: legge i logits che il GEMV appena accodato ha
            // scritto e pubblica `Sel` per gli 8 dispatch che seguono, nello
            // stesso pass. L'entry di `MoeIdx` è la (layer, k=0): il suo
            // `selIdx` è la base della regione di PRODUZIONE del layer.
            pass = usePass("router");
            pass.setPipeline(opt.pipeR);
            pass.setBindGroup(0, opt.bindR, [m * topK * MOE_IDX_STRIDE]);
            pass.dispatchWorkgroups(1);
            const E = expertCls[cfg.classOf(moeLayerAbs[m])];
            pass = usePass("expert");
            for (let k2 = 0; k2 < topK; k2++) {
              const dyn = (m * topK + k2) * MOE_IDX_STRIDE; // decode: riga 0
              pass.setPipeline(E.pGate); pass.setBindGroup(0, E.bgGate, [dyn]); pass.dispatchWorkgroups(dE, 1, 1);
              pass.setPipeline(E.pUp); pass.setBindGroup(0, E.bgUp, [dyn]); pass.dispatchWorkgroups(dE, 1, 1);
              pass.setPipeline(pSilu); pass.setBindGroup(0, bgSilu); pass.dispatchWorkgroups(Math.ceil(dE / 64), 1, 1);
              pass.setPipeline(E.pDown); pass.setBindGroup(0, E.bgDown, [dyn]); pass.dispatchWorkgroups(gg2[0], gg2[1], 1);
            }
            pass.setPipeline(pAdd); pass.setBindGroup(0, bgAddRes); pass.dispatchWorkgroups(Math.ceil(d / 64), 1, 1);
            endPass();
          }
          const tail = usePass("tail");
          for (let i = cuts[nMoeLayer - 1]; i < (read ? steps.length : headCut); i++) {
            const st = steps[i];
            tail.setPipeline(st.pipe);
            tail.setBindGroup(0, st.bind);
            tail.dispatchWorkgroups(st.wgs[0], st.wgs[1], st.wgs[2]);
          }
          endPass();
          if (canGpuTs && tsIdx > 0) {
            enc.resolveQuerySet(tsqSet!, 0, tsIdx * 2, tsqResolve!, 0);
            enc.copyBufferToBuffer(tsqResolve!, 0, tsqStaging!, 0, tsIdx * 2 * 8);
          }
          if (read) enc.copyBufferToBuffer(logits, 0, staging, 0, S.vocab * 4);
          // `Sel` INTERA in coda: è il solo momento in cui le regioni dei 40
          // layer sono complete, ed è insieme la decisione del router e ciò che
          // i kernel hanno letto davvero. Nel replay le entry dei layer non
          // rigiocati restano quelle del giro prima, in VRAM: la copia è sempre
          // dell'intera `Sel` e quindi le riporta comunque.
          enc.copyBufferToBuffer(selBuf, 0, opt.selStaging, 0, nSel * SEL_BYTES);
          enc.copyBufferToBuffer(opt.dirtyB, 0, opt.dirtyStaging, 0, 16);
          device.queue.submit([enc.finish()]);
          // CPU pura, e la GPU non ha ancora cominciato: e' seriale col lavoro
          // GPU, non sovrapposta (fase 4-ter, it.28).
          perfAcc.encodeMs += performance.now() - tEnc;
          perfAcc.submits++;
          // I1: da qui al readback la slotTable è INTOCCABILE — il resolve l'ha
          // già letta e gli slot che ha pubblicato in `Sel` devono restare
          // quelli. Il repair sta DOPO, al confine.
          cache.setInFlight(true);
          // ORDINE (fase 4-ter, it.28): le mapAsync PRIMA, gli error scope DOPO.
          // `popErrorScope` si risolve quando il device ha processato il lavoro,
          // quindi awaitarlo PRIMA del readback e' una seconda attesa della
          // stessa cosa. Il contratto non cambia — l'errore si cattura comunque
          // e si alza prima di usare i byte letti; cambia che si aspetta una
          // volta sola.
          const tRb = performance.now();
          const maps: Promise<undefined>[] = [
            opt.selStaging.mapAsync(GPUMapMode.READ),
            opt.dirtyStaging.mapAsync(GPUMapMode.READ),
          ];
          if (read) maps.push(staging.mapAsync(GPUMapMode.READ));
          // La mapAsync dei timestamp parte DOPO il submit — mai prima,
          // altrimenti Dawn droppa il command buffer (known-issue fase A).
          if (canGpuTs && tsIdx > 0) maps.push(tsqStaging!.mapAsync(GPUMapMode.READ));
          await Promise.all(maps);
          const errOom = await device.popErrorScope();
          const errVal = await device.popErrorScope();
          if (errOom ?? errVal) throw new Error(`q35 optimistic error scope: ${(errOom ?? errVal)!.message.slice(0, 300)}`);
          perfAcc.readbacks++;
          perfAcc.readbackMs += performance.now() - tRb;
          cache.setInFlight(false); // confine di giro: la tabella torna toccabile
          if (canGpuTs && tsIdx > 0) {
            const ts = new BigUint64Array(tsqStaging!.getMappedRange().slice(0));
            tsqStaging!.unmap();
            for (let i = 0; i < tsIdx; i++) {
              const dt = Number(ts[i * 2 + 1] - ts[i * 2]) / 1e6;
              const c = cats[i];
              const a = tsqAcc.get(c) ?? { ms: 0, n: 0 };
              a.ms += dt; a.n++;
              tsqAcc.set(c, a);
            }
            tsqTokens++;
          }
          const su = new Uint32Array(opt.selStaging.getMappedRange().slice(0));
          opt.selStaging.unmap();
          const du = new Uint32Array(opt.dirtyStaging.getMappedRange().slice(0));
          opt.dirtyStaging.unmap();
          let lg: Float32Array | null = null;
          if (read) {
            lg = new Float32Array(staging.getMappedRange().slice(0, S.vocab * 4));
            staging.unmap();
          }
          // I MISS si derivano dalla `Sel` e si incrociano col flag aggregato
          // che ha scritto il kernel: due strade indipendenti verso lo stesso
          // fatto. Una divergenza è un bug strutturale (il resolve ha scritto
          // `Sel` e `dirtyB` in disaccordo), mai un dato da interpretare.
          // Si guardano solo i layer RIGIOCATI: sopra `startLayer` la `Sel` è
          // quella del giro precedente, i cui miss sono già stati riparati, e
          // `dirtyB` è stato azzerato per questo giro.
          let missCount = 0, firstDirty = -1;
          for (let m = startLayer; m < nMoeLayer; m++) {
            for (let k2 = 0; k2 < topK; k2++) {
              if ((su[(m * topK + k2) * 4 + 3] & 1) !== 0) {
                missCount++;
                if (firstDirty < 0) firstDirty = m;
              }
            }
          }
          const kernelFirst = du[0] === 0xffffffff ? -1 : du[0];
          if (du[1] !== missCount || kernelFirst !== firstDirty) {
            throw new Error(
              `q35 optimistic: dirtyB (${du[1]} miss, primo layer MoE ${kernelFirst}) != derivazione ` +
              `da Sel (${missCount} miss, primo layer MoE ${firstDirty}) — il resolve ha scritto Sel e ` +
              "dirtyB in disaccordo: bug strutturale, token non interpretabile");
          }
          return { su, lg, missCount, firstDirty };
        };

        let cur = await runPass(0, true);
        // ---- contabilità PER PREFISSO + repair/replay iterativo ----
        // I layer sopra il primo sporco sono DEFINITIVI: il replay riparte da
        // `firstDirty` e non li tocca più. Contabilizzarli subito evita di
        // doverli proteggere dall'eviction del repair, ed è anche il motivo per
        // cui il pin-for-replay può limitarsi ai layer da `firstDirty` in giù.
        const repaired = new Set<number>();
        let accounted = 0;
        const account = (upTo: number, su: Uint32Array): void => {
          for (let m = accounted; m < upTo; m++) {
            const l = moeLayerAbs[m];
            const experts = new Uint32Array(topK);
            for (let k2 = 0; k2 < topK; k2++) {
              const e = su[(m * topK + k2) * 4];
              experts[k2] = e;
              const key = cache.keyOf(l, e);
              routing.set(key, (routing.get(key) ?? 0) + 1);
              // Gli expert RIPARATI in questo token sono già passati da
              // `ensure` e contati come MISS: contarli anche come hit
              // gonfierebbe le richieste. Gli altri non hanno visto `ensure`
              // affatto — la contabilità e il touch della LRU sono qui.
              if (!repaired.has(key)) cache.noteResidentHit(l, e);
            }
            cache.noteSelection(l, experts);
          }
          accounted = upTo;
        };
        // `lastFirst` parte a -1 e non a 0: a cache fredda il primo layer sporco
        // E' lo zero, ed e' legittimo — il progresso stretto si pretende fra un
        // round e il successivo, non fra il giro iniziale e il primo round.
        const tTail = performance.now();
        let rounds = 0, lastFirst = -1;
        while (cur.missCount > 0) {
          if (rounds === 0) perfAcc.dirtyTokens++;
          account(cur.firstDirty, cur.su);
          // La convergenza è per PREFISSO: `firstDirty` deve crescere
          // STRETTAMENTE a ogni round, quindi i round sono <= nMoeLayer. Se non
          // cresce, o l'eviction ha toccato uno slot pinnato o il flush della
          // tabella non ha seguito le writeBuffer degli slab: è un bug
          // strutturale, non un caso da ritentare.
          if (cur.firstDirty <= lastFirst) {
            throw new Error(
              `q35 optimistic: replay round ${rounds + 1} sporco allo STESSO layer MoE ` +
              `${cur.firstDirty} (precedente ${lastFirst}) — progresso violato`);
          }
          if (++rounds > nMoeLayer + 1) {
            throw new Error(`q35 optimistic: ${rounds} round di repair — oltre il cap teorico (${nMoeLayer} layer MoE)`);
          }
          lastFirst = cur.firstDirty;
          // pin-for-replay del ROUND: gli slot che la `Sel` corrente
          // referenzia dai layer >= firstDirty (miss inclusi, che stanno lì)
          // non possono diventare vittime mentre si ripara.
          const pinned = new Set<number>();
          const misses: { l: number; e: number }[] = [];
          for (let m = cur.firstDirty; m < nMoeLayer; m++) {
            const l = moeLayerAbs[m];
            for (let k2 = 0; k2 < topK; k2++) {
              const o = (m * topK + k2) * 4;
              const e = cur.su[o];
              pinned.add(cache.keyOf(l, e));
              if ((cur.su[o + 3] & 1) !== 0) misses.push({ l, e });
            }
          }
          const tRep = performance.now();
          // L'I/O sta FUORI dalla cache (il 35B legge per Range): si `await`ta
          // solo ciò che manca e poi si consegnano i byte già in mano.
          const got = await Promise.all(misses.map((ms) => readExpert(ms.l, ms.e)));
          misses.forEach((ms, i) => {
            cache.ensure(ms.l, ms.e, () => got[i], pinned);
            repaired.add(cache.keyOf(ms.l, ms.e));
          });
          // UN flush, DOPO le writeBuffer degli slab: il dato prima della
          // tabella che lo indirizza (R5 del design d'arena).
          cache.flushSlotTable();
          perfAcc.repairMs += performance.now() - tRep;
          perfAcc.replays++;
          perfAcc.replayLayers += nMoeLayer - cur.firstDirty;
          cur = await runPass(cur.firstDirty, false);
        }
        account(nMoeLayer, cur.su);
        perfAcc.tailCpuMs += performance.now() - tTail;
        return cur.lg;
      } : null,
      async runLayer(m: number, logitsF32: Float32Array): Promise<void> {
        await moe!.prepLayer(m, logitsF32);
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        moe!.encodeExperts(pass, m, bgAddRes);
        pass.end();
        device.queue.submit([enc.finish()]);
        perfAcc.submits++;
      },
      evictAll(): void {
        const keys: number[] = [];
        for (let l2 = cfg.denseLead; l2 < cfg.nLayer; l2++) {
          for (let e2 = 0; e2 < cfg.nExpert; e2++) keys.push(cache.keyOf(l2, e2));
        }
        cache.debugMarkMiss(keys);
        routing.clear();
      },
      pinUnion(m: number, logitsAll: Float32Array, rows: number, nExpertLogits: number): Set<number> {
        const l = moeLayerAbs[m];
        const out = new Set<number>();
        for (let r2 = 0; r2 < rows; r2++) {
          const selR = routerSelect(logitsAll.subarray(r2 * nExpertLogits, (r2 + 1) * nExpertLogits), null, routerCfg);
          for (const e of selR.experts) out.add(cache.keyOf(l, e));
        }
        return out;
      },
      async prepLayer(m: number, logitsF32: Float32Array, row = 0, pinAll?: Set<number>): Promise<void> {
        const l = moeLayerAbs[m];
        // Selezione: IL router unico, in configurazione qwen35moe (softmax,
        // niente bias, niente scale). Stessa matematica del cpuref, che prima
        // era ricopiata qui a mano.
        const sel = routerSelect(logitsF32, null, routerCfg);
        // L'I/O sta FUORI dalla cache: `ensure` è sincrona (GLM legge da
        // memoria), il 35B no. Guardo chi manca, `await`to solo quelli, e poi
        // consegno i byte già in mano. Sugli hit non si legge niente.
        const raw = new Map<number, ExpertRawBytes>();
        const missing: number[] = [];
        for (const e of sel.experts) if (!cache.isResident(l, e)) missing.push(e);
        if (missing.length > 0) {
          const got = await Promise.all(missing.map((e) => readExpert(l, e)));
          missing.forEach((e, i) => raw.set(e, got[i]));
        }
        // i top-K del token devono coesistere: nessuno di loro può essere
        // vittima di eviction per far posto agli altri.
        // `pinAll` (prefill a chunk): il pin copre l'UNIONE di tutte le righe
        // del layer, non i soli top-K di questa. Senza, l'`ensure` della riga
        // r+1 potrebbe evincere uno slot che i dispatch GIA' ENCODATI della
        // riga r leggeranno — e la writeBuffer del nuovo slab arriva prima del
        // submit. E' il secondo dei due bug che il gate ha preso (it.34).
        const pinned = pinAll ?? new Set<number>();
        if (!pinAll) for (const e of sel.experts) pinned.add(cache.keyOf(l, e));
        const slots: SlotRef[] = [];
        for (const e of sel.experts) {
          const key = cache.keyOf(l, e);
          routing.set(key, (routing.get(key) ?? 0) + 1);
          slots.push(cache.ensure(l, e, (_ly, ex) => raw.get(ex)!, pinned).slot);
        }
        cache.noteSelection(l, sel.experts);
        // l'ombra confronta contro CIÒ CHE LA CPU HA DECISO, e i miss vanno
        // presi PRIMA degli `ensure`: dopo, `isResident` direbbe sempre sì.
        shadow?.note(sel, slots, new Set(missing), logitsF32);
        // `Sel` del layer. L'indirizzo dell'expert è `slot.idx` — l'indice
        // GLOBALE dello slot nella classe — e da qui in poi è il KERNEL a
        // ricavarne (binding, base): la CPU non calcola più sotto-range. Lo
        // `SlotRef` resta il solo modo di ottenerlo (marchio di conio,
        // `residency.ts`): l'indirizzo cambia rappresentazione, non provenienza.
        for (let k2 = 0; k2 < topK; k2++) {
          selU32[k2 * 4] = sel.experts[k2];
          selU32[k2 * 4 + 1] = slots[k2].idx;
          selF32[k2 * 4 + 2] = sel.weights[k2];
          selU32[k2 * 4 + 3] = 0;
        }
        device.queue.writeBuffer(selBuf, (row * nMoeLayer + m) * topK * SEL_BYTES, selScratch);
        // La tabella si pubblica DOPO gli `ensure` del layer: le writeBuffer
        // degli slab sono già in coda, quindi la tabella arriva dopo il dato che
        // indirizza (R5 del design d'arena — l'ordine inverso pubblicherebbe uno
        // slot ancora vuoto). Qui non la legge nessuno: serve dalla fetta 3b.
        cache.flushSlotTable();
      },
      encodeExperts(pass: GPUComputePassEncoder, m: number, addBg: GPUBindGroup, row = 0): void {
        const E = expertCls[cfg.classOf(moeLayerAbs[m])];
        const disp = (p2: GPUComputePipeline, b2: GPUBindGroup, wg: [number, number, number], dyn?: number): void => {
          pass.setPipeline(p2);
          if (dyn === undefined) pass.setBindGroup(0, b2);
          else pass.setBindGroup(0, b2, [dyn]);
          pass.dispatchWorkgroups(wg[0], wg[1], wg[2]);
        };
        const gg2 = gemvGrid(d);
        for (let k2 = 0; k2 < topK; k2++) {
          // l'offset dinamico sceglie la entry (layer, k) di MoeIdx, che punta
          // alla Sel di quell'expert: è l'unico parametro per-expert rimasto.
          const dyn = ((row * nMoeLayer + m) * topK + k2) * MOE_IDX_STRIDE;
          disp(E.pGate, E.bgGate, [dE, 1, 1], dyn);
          disp(E.pUp, E.bgUp, [dE, 1, 1], dyn);
          disp(pSilu, bgSilu, [Math.ceil(dE / 64), 1, 1]);
          disp(E.pDown, E.bgDown, [gg2[0], gg2[1], 1], dyn);
        }
        disp(pAdd, addBg, [Math.ceil(d / 64), 1, 1]); // x += moeAcc (shexp + expert)
      },
    };
  }
  let lastLg: Float32Array | null = null;
  const perfAcc = {
    tokens: 0, embedMs: 0, readbackMs: 0, argmaxMs: 0, submits: 0, readbacks: 0,
    dirtyTokens: 0, replays: 0, replayLayers: 0, repairMs: 0,
    encodeMs: 0, tokenMs: 0, tailCpuMs: 0,
    // testa MTP (it.53): tenuti FUORI da `tokenMs`, perche' il draft e' un
    // submit a parte e sommarlo al token confonderebbe due costi diversi.
    mtpMs: 0, mtpDrafts: 0, specMs: 0, specPasses: 0, specRejects: 0,
  };
  /**
   * Path attivo: `true` se costruito con `select: "optimistic"`.
   *
   * NIENTE SOGLIA D'INGRESSO, e la decisione e' misurata (fase 5, it.36). Il
   * progetto di it.16 prevedeva una policy — sync finche' i miss/token stanno
   * sopra una soglia, ottimistico sotto — perche' a cache fredda ogni token e'
   * sporco e il replay sembrava dover costare piu' del sync. Misurato nello
   * STESSO processo, svuotando la cache fra i bracci e ripetendo nei due ordini:
   * a FREDDO l'ottimistico fa 651-661 ms/token contro 1098-1112 del sync
   * (**1,68x piu' veloce**), e a caldo 43,6 contro 132,8 (3,05x). Il replay
   * costa — 109 replay, +12% di fetch — ma costa MENO dei 77 round-trip per
   * token che il path sync paga comunque. La soglia non serve: e' esclusa coi
   * numeri, che e' cio' che il done-when della riga 5 ammette.
   */
  let optimisticOn = opts.select === "optimistic";
  // ---- sonda di decomposizione del tempo GPU (fase 4, it.19) ----
  // Con le marche di it.23 le categorie per layer diventano 7 (norm, attn|ssm,
  // resid, shexp, routerGemv, router, expert): 40x7+1 = 281 sul 35B. 512 da'
  // un margine 1,8x — e l'overflow si CONTA, perche' una sonda che perde i
  // pass finali in silenzio farebbe sembrare piu' economici gli ultimi layer.
  const TSQ_PASSES = 512;
  let tsqOverflow = 0;
  const canGpuTs = opts.telemetryGpu === true && device.features.has("timestamp-query");
  const tsqSet = canGpuTs ? device.createQuerySet({ type: "timestamp", count: TSQ_PASSES * 2 }) : null;
  const tsqResolve = canGpuTs
    ? device.createBuffer({ size: TSQ_PASSES * 2 * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC })
    : null;
  const tsqStaging = canGpuTs
    ? device.createBuffer({ size: TSQ_PASSES * 2 * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
    : null;
  const tsqAcc = new Map<string, { ms: number; n: number }>();
  let tsqTokens = 0;
  /** indice step → categoria, dalle marche: lookup O(1) dentro l'encoder */
  const markAt = new Map<number, string>(segMarks.map((m) => [m.at, m.cat]));
  const zeroAcc = new Float32Array(d);
  const tapStaging = device.createBuffer({ size: d * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  let tapLayer = -1;
  let tapWanted = -1;
  let tapValue: Float32Array | null = null;

  // --- argmax su GPU + batch teacher-forced (fase D fase 3) ---
  const N_PARTIALS = Math.ceil(S.vocab / ARGMAX_CHUNK);
  const amaxPartMax = empty(N_PARTIALS * 4);
  const amaxPartIdx = empty(N_PARTIALS * 4);
  const amaxOut = empty(16);
  const pAmax1 = pipe(argmaxStage1Wgsl(S.vocab));
  const pAmax2 = pipe(argmaxStage2Wgsl(N_PARTIALS));
  const bgAmax1 = device.createBindGroup({
    layout: pAmax1.getBindGroupLayout(0),
    entries: [logits, amaxPartMax, amaxPartIdx].map((b, i) => ({ binding: i, resource: { buffer: b } })),
  });
  const bgAmax2 = device.createBindGroup({
    layout: pAmax2.getBindGroupLayout(0),
    entries: [amaxPartMax, amaxPartIdx, amaxOut].map((b, i) => ({ binding: i, resource: { buffer: b } })),
  });
  // ====== TESTA MTP (NextN) — fase 7, it.53 ======
  // Un SECONDO piano di dispatch, non un pezzo del token. Legge `x` (il residuo
  // finale del modello, PRIMA di `output_norm`) e l'embedding grezzo del token
  // appena predetto, e produce il draft per la posizione dopo. Vive qui in
  // fondo perche' gli servono `headStep` (la lm_head CONDIVISA) e le pipeline
  // di argmax, che nascono sopra.
  const mtpSteps: Step[] = [];
  let mtpDraftFn: ((nextToken: number) => Promise<number>) | null = null;
  if (opts.mtp && S.mtpLayers >= 1) {
    const bm = `blk.${S.nLayer}.`;
    const enormW = await f32buf(`${bm}nextn.enorm.weight`);
    const hnormW = await f32buf(`${bm}nextn.hnorm.weight`);
    const shNormW = await f32buf(`${bm}nextn.shared_head_norm.weight`);
    const attnNormM = await f32buf(`${bm}attn_norm.weight`);
    const postNormM = await f32buf(`${bm}post_attention_norm.weight`);
    const qNormM = await f32buf(`${bm}attn_q_norm.weight`);
    const kNormM = await f32buf(`${bm}attn_k_norm.weight`);
    // `loadW` type-driven: `eh_proj` e' q8_0 mentre il resto del blocco e'
    // q4_0 — letto dal file, non dedotto (it.52).
    const ehW = await loadW(`${bm}nextn.eh_proj.weight`);
    const wqM = await loadW(`${bm}attn_q.weight`);
    const wkM = await loadW(`${bm}attn_k.weight`);
    const wvM = await loadW(`${bm}attn_v.weight`);
    const woM = await loadW(`${bm}attn_output.weight`);
    const wgM = await loadW(`${bm}ffn_gate.weight`);
    const wuM = await loadW(`${bm}ffn_up.weight`);
    const wdM = await loadW(`${bm}ffn_down.weight`);
    const dFfnM = S.dFfn as number;

    const mEmb = empty(d * 4), mEn = empty(d * 4), mHn = empty(d * 4), mCat = empty(2 * d * 4);
    const mHp = empty(d * 4), mXn = empty(d * 4), mY = empty(d * 4), mOut = empty(d * 4);
    const mQFull = empty(2 * qDim * 4), mKCur = empty(kvDim * 4), mVCur = empty(kvDim * 4);
    const mQB = empty(qDim * 4), mGateB = empty(qDim * 4), mQN = empty(qDim * 4), mKN = empty(kvDim * 4);
    const mAttnO = empty(qDim * 4), mGateF = empty(dFfnM * 4), mUpF = empty(dFfnM * 4);
    const mKCache = empty(ctxMax * kvDim * 4), mVCache = empty(ctxMax * kvDim * 4);
    const mLogits = empty(S.vocab * 4);

    stepTarget = mtpSteps;
    // h' = eh_proj([norm_e(emb(t_{i+1})) ; norm_h(x_i)]) — embedding PRIMO
    // (ordine deciso per misura in it.49, confermato su vLLM in it.51).
    push(rmsnormWgsl(d, S.rmsEps), [mEmb, enormW, mEn], 1);
    push(rmsnormWgsl(d, S.rmsEps), [x, hnormW, mHn], 1);
    push(stridedCopyWgsl({ nVec: 1, len: d, srcStride: d, srcOffset: 0, dstStride: 2 * d, dstOffset: 0 }), [mEn, mCat], Math.ceil(d / 64));
    push(stridedCopyWgsl({ nVec: 1, len: d, srcStride: d, srcOffset: 0, dstStride: 2 * d, dstOffset: d }), [mHn, mCat], Math.ceil(d / 64));
    ehW.push(mCat, mHp);
    // da qui e' un layer full identico a quelli del modello, con cache KV SUA
    push(rmsnormWgsl(d, S.rmsEps), [mHp, attnNormM, mXn], 1);
    wqM.push(mXn, mQFull);
    wkM.push(mXn, mKCur);
    wvM.push(mXn, mVCur);
    push(stridedCopyWgsl({ nVec: S.nHead, len: hd, srcStride: 2 * hd, srcOffset: 0, dstStride: hd, dstOffset: 0 }), [mQFull, mQB], Math.ceil(qDim / 64));
    push(stridedCopyWgsl({ nVec: S.nHead, len: hd, srcStride: 2 * hd, srcOffset: hd, dstStride: hd, dstOffset: 0 }), [mQFull, mGateB], Math.ceil(qDim / 64));
    push(rmsnormWgsl(hd, S.rmsEps, true), [mQB, qNormM, mQN], S.nHead);
    push(rmsnormWgsl(hd, S.rmsEps, true), [mKCur, kNormM, mKN], S.nKvHead);
    push(ropeNeoxWgsl(S.nHead, hd, S.ropeFreqBase, S.ropeDims), [mQN], Math.ceil((S.nHead * S.ropeDims / 2) / 64), true);
    push(ropeNeoxWgsl(S.nKvHead, hd, S.ropeFreqBase, S.ropeDims), [mKN], Math.ceil((S.nKvHead * S.ropeDims / 2) / 64), true);
    push(kvAppendWgsl(kvDim), [mKN, mKCache], Math.ceil(kvDim / 64), true);
    push(kvAppendWgsl(kvDim), [mVCur, mVCache], Math.ceil(kvDim / 64), true);
    // stessi parziali dei layer del token: la testa gira in un submit a parte,
    // dopo, e i suoi due dispatch sono consecutivi come quelli dei layer.
    pushAttnDecodeSplit(mQN, mKCache, mVCache, mAttnO);
    push(sigmoidMulWgsl(qDim), [mAttnO, mGateB], Math.ceil(qDim / 64));
    woM.push(mAttnO, mY);
    push(addInPlaceWgsl(d), [mHp, mY], Math.ceil(d / 64));
    push(rmsnormWgsl(d, S.rmsEps), [mHp, postNormM, mXn], 1);
    wgM.push(mXn, mGateF);
    wuM.push(mXn, mUpF);
    push(siluMulWgsl(dFfnM), [mGateF, mUpF], Math.ceil(dFfnM / 64));
    wdM.push(mGateF, mY);
    push(addInPlaceWgsl(d), [mHp, mY], Math.ceil(d / 64));
    push(rmsnormWgsl(d, S.rmsEps), [mHp, shNormW, mOut], 1);
    headStep(mOut, mLogits);
    stepTarget = steps;

    // argmax del draft su GPU: stesse pipeline del modello, bind group suoi.
    // I buffer parziali si riusano — il draft gira in un submit a parte, dopo
    // che l'argmax del token e' gia' stato letto.
    const mAmaxOut = empty(16);
    const bgAmax1M = device.createBindGroup({
      layout: pAmax1.getBindGroupLayout(0),
      entries: [mLogits, amaxPartMax, amaxPartIdx].map((b, i) => ({ binding: i, resource: { buffer: b } })),
    });
    const bgAmax2M = device.createBindGroup({
      layout: pAmax2.getBindGroupLayout(0),
      entries: [amaxPartMax, amaxPartIdx, mAmaxOut].map((b, i) => ({ binding: i, resource: { buffer: b } })),
    });
    const mStagingId = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const mEmbRow = new Float32Array(d);

    mtpDraftFn = async (nextToken: number): Promise<number> => {
      const t0 = performance.now();
      dequantRow(nextToken, mEmbRow, 0);
      device.queue.writeBuffer(mEmb, 0, mEmbRow);
      // `uni` porta ancora (pos, pos) dello step appena eseguito: la testa sta
      // alla stessa posizione. Non e' un'approssimazione — il rope e' relativo
      // e uno shift uniforme non cambia un bit (it.51).
      device.pushErrorScope("validation");
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      for (const st of mtpSteps) {
        pass.setPipeline(st.pipe);
        pass.setBindGroup(0, st.bind);
        pass.dispatchWorkgroups(st.wgs[0], st.wgs[1], st.wgs[2]);
      }
      pass.setPipeline(pAmax1); pass.setBindGroup(0, bgAmax1M); pass.dispatchWorkgroups(N_PARTIALS);
      pass.setPipeline(pAmax2); pass.setBindGroup(0, bgAmax2M); pass.dispatchWorkgroups(1);
      pass.end();
      enc.copyBufferToBuffer(mAmaxOut, 0, mStagingId, 0, 4);
      device.queue.submit([enc.finish()]);
      const err = await device.popErrorScope();
      if (err) throw new Error(`q35 mtpDraft error scope: ${err.message.slice(0, 300)}`);
      await mStagingId.mapAsync(GPUMapMode.READ);
      const id = new Uint32Array(mStagingId.getMappedRange())[0];
      mStagingId.unmap();
      perfAcc.mtpMs += performance.now() - t0;
      perfAcc.mtpDrafts++;
      return id;
    };
  }

  const BATCH_MAX = 32;
  // Le righe di embedding e gli uniform dei K step NON si possono scrivere con
  // writeBuffer dentro l'encoder: `queue.writeBuffer` e' ordinata PRIMA del
  // submit, quindi tutti gli step vedrebbero l'ultimo valore scritto. Si
  // impacchettano in un buffer solo e si copiano nello scratch all'inizio di
  // ogni step, DENTRO l'encoder (stesso schema di `dbSlots` in gpuforward).
  const embBatch = empty(BATCH_MAX * d * 4);
  const uniBatch = device.createBuffer({ size: BATCH_MAX * 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const idsBatch = empty(BATCH_MAX * 4);
  const stagingIds = device.createBuffer({ size: BATCH_MAX * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const embRowsCpu = new Float32Array(BATCH_MAX * d);
  const uniCpu = new Uint32Array(BATCH_MAX * 4);
  const dequantRow = (token: number, dst: Float32Array, at: number): void => {
    const sub = dst.subarray(at, at + d);
    if (embdKind === "q6k") dequantQ6_K(embdRaw, token * rowBytes, rowBlocks, sub);
    else if (embdKind === "q80") dequantQ8_0(embdRaw, token * rowBytes, rowBlocks, sub);
    else dequantQ4_0(embdRaw, token * rowBytes, rowBlocks, sub);
  };

  // ====== VERIFICA SPECULATIVA (fase 7, it.54) ======
  // Due posizioni in UN submit: quella CERTA (il token che il modello ha
  // predetto) e quella PROPOSTA dalla testa MTP. Se la proposta coincide con
  // cio' che il modello produce alla posizione certa, il token e' guadagnato;
  // se no, si torna indietro.
  //
  // TORNARE INDIETRO NON E' GRATIS, ed e' il motivo per cui questo codice
  // esiste invece di due `step`: 24 layer su 32 del 4B sono DeltaNet e
  // aggiornano `convSt`/`stateS` IN PLACE. Una riga speculativa sbagliata non
  // lascia un errore che si sovrascrive — lascia uno stato ricorrente sbagliato
  // e numeri plausibili (stessa lezione del replay ottimistico, it.18). Quindi
  // lo snapshot si prende DENTRO l'encoder, fra la riga certa e quella
  // speculativa: nessun round-trip, e lo stato salvato e' esattamente quello
  // "dopo il token certo".
  //
  // Anche `x` va salvato: la testa MTP legge il residuo finale, e dopo la riga
  // speculativa `x` contiene quello della posizione SBAGLIATA. Sono d float, il
  // costo e' rumore accanto allo stato ricorrente.
  //
  // La cache KV NON si ripara: `kvAppend` scrive all'indice della posizione e
  // l'attenzione legge solo fino a `pos`, quindi la riga rifiutata viene
  // riscritta dalla posizione vera prima che qualcuno la legga.
  const specShadow = (opts.mtp && !isMoe)
    ? stateBufs.map((s) => ({
      src: s.buf, bytes: s.bytes,
      dst: track(device.createBuffer({ size: Math.max(16, s.bytes), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC }), s.bytes),
    }))
    : null;
  const specXShadow = specShadow ? empty(d * 4) : null;
  const specStaging = specShadow
    ? device.createBuffer({ size: 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
    : null;
  const specVerifyFn = specShadow && specXShadow && specStaging
    ? async (certain: number, draft: number, pos: number): Promise<[number, number]> => {
      const t0 = performance.now();
      dequantRow(certain, embRowsCpu, 0);
      dequantRow(draft, embRowsCpu, d);
      uniCpu[0] = pos; uniCpu[1] = pos;
      uniCpu[4] = pos + 1; uniCpu[5] = pos + 1;
      device.queue.writeBuffer(embBatch, 0, embRowsCpu, 0, 2 * d);
      device.queue.writeBuffer(uniBatch, 0, uniCpu, 0, 8);
      device.pushErrorScope("validation");
      const enc = device.createCommandEncoder();
      for (let i = 0; i < 2; i++) {
        enc.copyBufferToBuffer(embBatch, i * d * 4, x, 0, d * 4);
        enc.copyBufferToBuffer(uniBatch, i * 16, uni, 0, 16);
        const pass = enc.beginComputePass();
        for (const st of steps) {
          pass.setPipeline(st.pipe);
          pass.setBindGroup(0, st.bind);
          pass.dispatchWorkgroups(st.wgs[0], st.wgs[1], st.wgs[2]);
        }
        pass.setPipeline(pAmax1); pass.setBindGroup(0, bgAmax1); pass.dispatchWorkgroups(N_PARTIALS);
        pass.setPipeline(pAmax2); pass.setBindGroup(0, bgAmax2); pass.dispatchWorkgroups(1);
        pass.end();
        enc.copyBufferToBuffer(amaxOut, 0, idsBatch, i * 4, 4);
        if (i === 0) {
          enc.copyBufferToBuffer(x, 0, specXShadow, 0, d * 4);
          for (const s of specShadow) enc.copyBufferToBuffer(s.src, 0, s.dst, 0, s.bytes);
        }
      }
      enc.copyBufferToBuffer(idsBatch, 0, specStaging, 0, 8);
      device.queue.submit([enc.finish()]);
      const err = await device.popErrorScope();
      if (err) throw new Error(`q35 specVerify error scope: ${err.message.slice(0, 300)}`);
      await specStaging.mapAsync(GPUMapMode.READ);
      const ids = new Uint32Array(specStaging.getMappedRange());
      const pair: [number, number] = [ids[0], ids[1]];
      specStaging.unmap();
      perfAcc.specMs += performance.now() - t0;
      perfAcc.specPasses++;
      return pair;
    }
    : null;
  const specRollbackFn = specShadow && specXShadow
    ? (): void => {
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(specXShadow, 0, x, 0, d * 4);
      for (const s of specShadow) enc.copyBufferToBuffer(s.dst, 0, s.src, 0, s.bytes);
      device.queue.submit([enc.finish()]);
      perfAcc.specRejects++;
    }
    : null;

  const runSeg = (from: number, to: number, tail?: (enc: GPUCommandEncoder) => void): void => {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    for (let i = from; i < to; i++) {
      const st = steps[i];
      pass.setPipeline(st.pipe);
      pass.setBindGroup(0, st.bind);
      pass.dispatchWorkgroups(st.wgs[0], st.wgs[1], st.wgs[2]);
    }
    pass.end();
    tail?.(enc);
    device.queue.submit([enc.finish()]);
    perfAcc.submits++;
  };
  const readStaging = async (b: GPUBuffer, bytes: number): Promise<Float32Array> => {
    perfAcc.readbacks++;
    await b.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(b.getMappedRange().slice(0, bytes));
    b.unmap();
    return out;
  };

  return {
    dispatchesPerToken,
    dispatchBreakdown: {
      static: dispatchesPerToken,
      // per layer MoE: topK x (gate, up, silu, down) + l'add del residuo, piu'
      // il dispatch del router quando la selezione la fa la GPU.
      dynamic: isMoe ? cuts.length * (topK * 4 + 1 + (opts.select === "optimistic" ? 1 : 0)) : 0,
      get total(): number { return this.static + this.dynamic; },
    },
    perf: () => ({ ...perfAcc }),
    mtpDraft: mtpDraftFn,
    specVerify: specVerifyFn,
    specRollback: specRollbackFn,
    specVerifyBatched: (PB && specSnap && specStaging) ? async (certain: number, draft: number, pos: number): Promise<[number, number]> => {
      if (pos + 2 > ctxMax) throw new Error("q35 specVerifyBatched: contesto pieno");
      const t0 = performance.now();
      const rows = new Float32Array(2 * d), rp = new Uint32Array(2);
      dequantRow(certain, rows, 0);
      dequantRow(draft, rows, d);
      rp[0] = pos; rp[1] = pos + 1;
      device.queue.writeBuffer(PB.x, 0, rows as unknown as BufferSource);
      device.queue.writeBuffer(PB.rowPos, 0, rp as unknown as BufferSource);
      device.pushErrorScope("validation");
      const enc = device.createCommandEncoder();
      const pz = enc.beginComputePass();
      for (const st of PB.steps) {
        pz.setPipeline(st.pipe);
        pz.setBindGroup(0, st.bind);
        pz.dispatchWorkgroups(st.wgs[0], st.wgs[1], st.wgs[2]);
      }
      pz.end();
      // La head sulle DUE righe: si copia l'hidden della riga nel buffer per
      // riga e si riusano gli step di coda del decode, invariati (stesso
      // schema di `prefillChunk`, che pero' ne serve una sola). Due letture
      // della lm_head restano — batcharle e' la prossima leva, non questa.
      for (let m = 0; m < 2; m++) {
        enc.copyBufferToBuffer(PB.x, m * d * 4, x, 0, d * 4);
        const p2 = enc.beginComputePass();
        for (let i = headCut; i < steps.length; i++) {
          const st = steps[i];
          p2.setPipeline(st.pipe);
          p2.setBindGroup(0, st.bind);
          p2.dispatchWorkgroups(st.wgs[0], st.wgs[1], st.wgs[2]);
        }
        p2.setPipeline(pAmax1); p2.setBindGroup(0, bgAmax1); p2.dispatchWorkgroups(N_PARTIALS);
        p2.setPipeline(pAmax2); p2.setBindGroup(0, bgAmax2); p2.dispatchWorkgroups(1);
        p2.end();
        enc.copyBufferToBuffer(amaxOut, 0, idsBatch, m * 4, 4);
      }
      enc.copyBufferToBuffer(idsBatch, 0, specStaging, 0, 8);
      device.queue.submit([enc.finish()]);
      perfAcc.submits++;
      const err = await device.popErrorScope();
      if (err) throw new Error(`q35 specVerifyBatched error scope: ${err.message.slice(0, 300)}`);
      await specStaging.mapAsync(GPUMapMode.READ);
      const ids = new Uint32Array(specStaging.getMappedRange());
      const pair: [number, number] = [ids[0], ids[1]];
      specStaging.unmap();
      perfAcc.specMs += performance.now() - t0;
      perfAcc.specPasses++;
      return pair;
    } : null,
    specCommit: (PB && specSnap) ? (accepted: boolean): void => {
      const enc = device.createCommandEncoder();
      // il residuo della riga GIUSTA torna in `x`: e' quello che la testa MTP
      // legge al giro dopo. Su rifiuto e' la riga 0 (la posizione certa).
      enc.copyBufferToBuffer(PB.x, (accepted ? 1 : 0) * d * 4, x, 0, d * 4);
      if (!accepted) for (const sn of specSnap) enc.copyBufferToBuffer(sn.dst, 0, sn.src, 0, sn.n * 4);
      device.queue.submit([enc.finish()]);
      if (!accepted) perfAcc.specRejects++;
    } : null,
    decodeBatch: moe ? null : async (tokens: ArrayLike<number>, posStart: number): Promise<number[]> => {
      const k = tokens.length;
      if (k < 1 || k > BATCH_MAX) throw new Error(`q35 decodeBatch: k=${k} fuori da [1, ${BATCH_MAX}]`);
      const tEmb = performance.now();
      for (let i = 0; i < k; i++) {
        dequantRow(tokens[i] as number, embRowsCpu, i * d);
        uniCpu[i * 4] = posStart + i;
        uniCpu[i * 4 + 1] = posStart + i;
      }
      device.queue.writeBuffer(embBatch, 0, embRowsCpu, 0, k * d);
      device.queue.writeBuffer(uniBatch, 0, uniCpu, 0, k * 4);
      perfAcc.embedMs += performance.now() - tEmb;
      perfAcc.tokens += k;
      // Error scope come CONTRATTO di ogni percorso di encode nuovo (landmine
      // B1: una pipeline invalida fa droppare i submit IN SILENZIO, e i
      // readback tornano valori stale ma plausibili).
      device.pushErrorScope("validation");
      device.pushErrorScope("out-of-memory");
      const enc = device.createCommandEncoder();
      for (let i = 0; i < k; i++) {
        enc.copyBufferToBuffer(embBatch, i * d * 4, x, 0, d * 4);
        enc.copyBufferToBuffer(uniBatch, i * 16, uni, 0, 16);
        const pass = enc.beginComputePass();
        for (const st of steps) {
          pass.setPipeline(st.pipe);
          pass.setBindGroup(0, st.bind);
          pass.dispatchWorkgroups(st.wgs[0], st.wgs[1], st.wgs[2]);
        }
        pass.setPipeline(pAmax1); pass.setBindGroup(0, bgAmax1); pass.dispatchWorkgroups(N_PARTIALS);
        pass.setPipeline(pAmax2); pass.setBindGroup(0, bgAmax2); pass.dispatchWorkgroups(1);
        pass.end();
        enc.copyBufferToBuffer(amaxOut, 0, idsBatch, i * 4, 4);
      }
      enc.copyBufferToBuffer(idsBatch, 0, stagingIds, 0, k * 4);
      device.queue.submit([enc.finish()]);
      const errOom = await device.popErrorScope();
      const errVal = await device.popErrorScope();
      if (errOom ?? errVal) throw new Error(`q35 decodeBatch error scope: ${(errOom ?? errVal)!.message.slice(0, 300)}`);
      await stagingIds.mapAsync(GPUMapMode.READ);
      const ids = [...new Uint32Array(stagingIds.getMappedRange()).subarray(0, k)];
      stagingIds.unmap();
      return ids;
    },
    async step(token: number, pos: number, read = true): Promise<number> {
      const tTok = performance.now();
      const tEmb = performance.now();
      if (embdKind === "q6k") dequantQ6_K(embdRaw, token * rowBytes, rowBlocks, embRow);
      else if (embdKind === "q80") dequantQ8_0(embdRaw, token * rowBytes, rowBlocks, embRow);
      else dequantQ4_0(embdRaw, token * rowBytes, rowBlocks, embRow);
      device.queue.writeBuffer(x, 0, embRow);
      perfAcc.embedMs += performance.now() - tEmb;
      perfAcc.tokens++;
      device.queue.writeBuffer(uni, 0, new Uint32Array([pos, pos, 0, 0]));
      if (moe && optimisticOn) {
        // UN submit per il token intero: nessun readback del router, la
        // selezione la fa e la risolve la GPU. I logits arrivano dalla stessa
        // attesa che porta `Sel` e `dirtyB` (fetta 3c).
        const lgOpt = await moe.runTokenOptimistic!(read);
        if (!read || !lgOpt) return -1;
        lastLg = lgOpt;
        const tAmOpt = performance.now();
        let bestO = -Infinity, biO = -1;
        for (let i = 0; i < S.vocab; i++) if (lgOpt[i] > bestO) { bestO = lgOpt[i]; biO = i; }
        perfAcc.argmaxMs += performance.now() - tAmOpt;
        perfAcc.tokenMs += performance.now() - tTok;
        return biO;
      }
      if (!moe) {
        // stesso taglio della coda sui densi: il prefill del 4B/9B gira anche
        // lui con `read=false` e pagava la head per buttarla. (`decodeBatch` no:
        // li' l'argmax si calcola su GPU DAI logits, quindi la head serve.)
        runSeg(0, read ? steps.length : headCut, read ? (enc) => enc.copyBufferToBuffer(logits, 0, staging, 0, S.vocab * 4) : undefined);
      } else {
        tapWanted = tapLayer;
        // MoE segmentato (correttezza-prima, 1 readback router/layer DICHIARATO):
        // per layer: zero moeAcc → segmento statico (attn+shexp+router) → read
        // logits router → selezione CPU + ensure + dispatch dinamici.
        let from = 0;
        for (let li = 0; li < cuts.length; li++) {
          device.queue.writeBuffer(moeAcc, 0, zeroAcc);
          runSeg(from, cuts[li], (enc) => {
            // OMBRA (fetta 3b): il router GPU gira nello STESSO submit del
            // segmento statico, in coda al GEMV che ha appena scritto i logits.
            // Scrive la regione d'ombra di Sel, che nessun kernel expert legge.
            moe.shadowEncode?.(enc, li);
            enc.copyBufferToBuffer(routerLogits, 0, routerStaging, 0, nE * 4);
            if (tapWanted === -100 - li) enc.copyBufferToBuffer(x, 0, tapStaging, 0, d * 4);
          });
          if (tapWanted === -100 - li) tapValue = await readStaging(tapStaging, d * 4);
          const lg = await readStaging(routerStaging, nE * 4);
          await moe.runLayer(li, lg);
          // dopo runLayer: il confronto vuole la selezione CPU e i suoi slot
          await moe.shadowCompare?.();
          if (tapWanted === li) {
            const enc2 = device.createCommandEncoder();
            enc2.copyBufferToBuffer(x, 0, tapStaging, 0, d * 4);
            device.queue.submit([enc2.finish()]);
            tapValue = await readStaging(tapStaging, d * 4);
          }
          from = cuts[li];
        }
        runSeg(from, read ? steps.length : headCut, read ? (enc) => enc.copyBufferToBuffer(logits, 0, staging, 0, S.vocab * 4) : undefined);
      }
      if (!read) return -1;
      const tRb = performance.now();
      const lg = await readStaging(staging, S.vocab * 4);
      lastLg = lg;
      const tAm = performance.now();
      perfAcc.readbackMs += tAm - tRb;
      let best = -Infinity, bi = -1;
      for (let i = 0; i < S.vocab; i++) if (lg[i] > best) { best = lg[i]; bi = i; }
      perfAcc.argmaxMs += performance.now() - tAm;
      return bi;
    },
    async readTap(layer: number): Promise<Float32Array> {
      if (optimisticOn) {
        throw new Error(
          "q35gpumodel: readTap non e' cablato sul path a submit unico — il tap vive nel ramo " +
          "sync di `step`. Restituire un array vuoto sarebbe uno strumento di debug che tace " +
          "quando dovrebbe urlare (docket item 12)");
      }
      tapLayer = layer;
      const v = tapValue;
      tapValue = null;
      return v ?? new Float32Array(0);
    },
    setOptimistic(on: boolean): void {
      if (on && !moe?.runTokenOptimistic) {
        throw new Error('q35gpumodel: setOptimistic(true) su un modello costruito senza select "optimistic"');
      }
      optimisticOn = on;
    },
    gpuTimeStats: canGpuTs
      ? () => ({ tokens: tsqTokens, overflow: tsqOverflow, byCat: Object.fromEntries([...tsqAcc.entries()].map(([k2, v2]) => [k2, { ...v2 }])) })
      : null,
    lastLogits: () => lastLg,
    vramPlan: () => vramPlanOut,
    debugEvictAll: moe?.evictAll ?? null,
    prefillChunk: PB ? async (tokens: ArrayLike<number>, posStart: number): Promise<Float32Array> => {
      if (tokens.length !== M_MAX) {
        throw new Error(`q35 prefillChunk: chunk di ${tokens.length} righe, il piano gemello e' cotto su ${M_MAX}`);
      }
      if (posStart + M_MAX > ctxMax) throw new Error("q35 prefillChunk: contesto pieno");
      const tEmb = performance.now();
      const rows = new Float32Array(M_MAX * d);
      const pos = new Uint32Array(M_MAX);
      for (let m = 0; m < M_MAX; m++) {
        dequantRow(tokens[m] as number, rows, m * d);
        pos[m] = posStart + m;
      }
      device.queue.writeBuffer(PB.x, 0, rows as unknown as BufferSource);
      device.queue.writeBuffer(PB.rowPos, 0, pos as unknown as BufferSource);
      perfAcc.embedMs += performance.now() - tEmb;
      perfAcc.tokens += M_MAX;
      device.pushErrorScope("validation");
      device.pushErrorScope("out-of-memory");
      const runSegB = (from: number, to: number, tail?: (e: GPUCommandEncoder) => void): void => {
        const e2 = device.createCommandEncoder();
        const pz = e2.beginComputePass();
        for (let i = from; i < to; i++) {
          const st = PB.steps[i];
          pz.setPipeline(st.pipe);
          pz.setBindGroup(0, st.bind);
          pz.dispatchWorkgroups(st.wgs[0], st.wgs[1], st.wgs[2]);
        }
        pz.end();
        tail?.(e2);
        device.queue.submit([e2.finish()]);
        perfAcc.submits++;
      };
      if (moe) {
        // MoE: per layer, UN readback dei logit del router per TUTTE le M righe
        // (invece di uno per token), poi selezione ed `ensure` per riga sulla
        // CPU — identiche al path sequenziale, ed e' per questo che il gate e'
        // la bit-identita' — e le M catene expert in un submit solo (it.34).
        const cr = moe.chunkRt!;
        let fromB = 0;
        const zeroM = new Float32Array(M_MAX * d);
        for (let li = 0; li < PB.cuts.length; li++) {
          // `moeAcc` per riga va azzerato PRIMA del segmento statico: l'axpy
          // dello shexp accumula, come nel path per riga (dove lo azzera una
          // writeBuffer per layer). Ogni layer ha il suo submit, quindi la
          // writeBuffer cade nel posto giusto.
          device.queue.writeBuffer(PB.moeAcc, 0, zeroM as unknown as BufferSource);
          runSegB(fromB, PB.cuts[li], (e2) => e2.copyBufferToBuffer(PB.routerLogits, 0, cr.routerStagingM, 0, M_MAX * nE * 4));
          const lgAll = await readStaging(cr.routerStagingM, M_MAX * nE * 4);
          // PRIMO giro: le selezioni di TUTTE le righe, per costruire il pin
          // dell'unione prima di qualunque `ensure`.
          const pinAll = moe.pinUnion(li, lgAll, M_MAX, nE);
          const e3 = device.createCommandEncoder();
          for (let m2 = 0; m2 < M_MAX; m2++) {
            await moe.prepLayer(li, lgAll.subarray(m2 * nE, (m2 + 1) * nE), m2, pinAll);
            // la riga entra nei buffer PER RIGA che la catena expert gia' usa:
            // stessi kernel, stessi bind group, zero varianti nuove. Le copie
            // sono operazioni di ENCODER, quindi il pass si apre dopo di loro —
            // un pass per riga, e l'ordine dei dispatch lo garantisce l'encoder.
            e3.copyBufferToBuffer(PB.xn, m2 * d * 4, xn, 0, d * 4);
            e3.copyBufferToBuffer(PB.moeAcc, m2 * d * 4, moeAcc, 0, d * 4);
            const pz = e3.beginComputePass();
            moe.encodeExperts(pz, li, cr.bgAddRow[m2], m2);
            pz.end();
          }
          device.queue.submit([e3.finish()]);
          perfAcc.submits++;
          fromB = PB.cuts[li];
        }
      } else {
        runSegB(0, PB.steps.length);
      }
      const enc = device.createCommandEncoder();
      // la head serve sulla SOLA ultima riga: si copia il suo hidden nel
      // buffer per riga e si riusano gli step di coda del decode, invariati.
      enc.copyBufferToBuffer(PB.x, (M_MAX - 1) * d * 4, x, 0, d * 4);
      const p2 = enc.beginComputePass();
      for (let i = headCut; i < steps.length; i++) {
        const st = steps[i];
        p2.setPipeline(st.pipe);
        p2.setBindGroup(0, st.bind);
        p2.dispatchWorkgroups(st.wgs[0], st.wgs[1], st.wgs[2]);
      }
      p2.end();
      enc.copyBufferToBuffer(logits, 0, staging, 0, S.vocab * 4);
      device.queue.submit([enc.finish()]);
      perfAcc.submits++;
      const tRb = performance.now();
      await staging.mapAsync(GPUMapMode.READ);
      const errOom = await device.popErrorScope();
      const errVal = await device.popErrorScope();
      if (errOom ?? errVal) throw new Error(`q35 prefillChunk error scope: ${(errOom ?? errVal)!.message.slice(0, 300)}`);
      const lg = new Float32Array(staging.getMappedRange().slice(0, S.vocab * 4));
      staging.unmap();
      perfAcc.readbacks++;
      perfAcc.readbackMs += performance.now() - tRb;
      return lg;
    } : null,
    routerShadowStats: moe?.shadowStats ? () => moe!.shadowStats!() : null,
    moeStats: moe
      ? () => ({
          ...moe!.stats(),
          routing: Object.fromEntries([...moe!.routing.entries()].map(([k2, v2]) => [String(k2), v2])),
          nSlots: moe!.nSlots, parkSlots: moe!.parkSlots, slotBytes: moe!.slotBytes,
          policy: opts.expertPolicy ?? "lru",
        })
      : null,
    resetState(): void {
      for (const s of stateBufs) device.queue.writeBuffer(s.buf, 0, new Float32Array(s.bytes / 4));
    },
    destroy(): void {
      for (const [, p] of pipes) void p;
      moe?.destroy(); // l'arena expert è VRAM vera: senza questa il modello la teneva
    },
  };
}

/** byte attesi di un tensore (comodo per i reader Range). */
export function q35TensorBytes(t: GgufTensorInfo): number {
  return tensorByteSize(t);
}
