// Forward GLM-4.7-Flash multi-layer (goal C2 fase 5 slice 3): l'orchestratore
// che unisce l'attention MLA absorbed di fase 4 (pattern glmforward), il blocco
// MoE di slice 1 (moe.ts) e la residenza minima di slice 2 (residency.ts).
//
// Struttura per token (decode): il piano è statico DENTRO i segmenti, ma la
// selezione expert vive su CPU (decide quali slab bindare), quindi ogni layer
// MoE ha un punto di sincronizzazione GPU→CPU:
//   [submit] attention(l) → ffn_norm → router GEMV f32 → copy logits a staging
//   [sync]   mapAsync → routerSelect (sigmoid+bias+top4+norm×1.8, replica it.6)
//   [CPU]    ensure×4 sulla cache LRU (pinned: i top-4 devono coesistere)
//   [encode] shexp (Q5_K/Q6_K) scrive moe_out; 4 catene expert accumulano
//            (down scaledAccum ×peso); residuo — prosegue nel submit successivo
// Il costo del sync è dichiarato e si misura in fase 6 (journal it.6).
//
// Con `select:"gpu"` (C3a fase 4 slice C) quella struttura sparisce: il router e
// il resolve girano su GPU e scrivono `Sel` direttamente dove i kernel expert la
// leggono, quindi il token è UN SOLO encode e UN SOLO submit — zero readback del
// router, zero `ensure` nel forward, `routing[]` ricostruito al tail dalla copia
// di Sel che si faceva già. Ammesso solo a residenza TOTALE: senza, un errore
// esplicito al load (senza sync la CPU non pinna più niente, e uno slot risolto
// potrebbe essere evinto dopo la risoluzione dentro lo stesso token).
//
// Pipeline WGSL condivise tra i 47 layer (shape identiche); bind group fissi
// per-layer precostruiti al load.
// Dalla fase 4 (strato 1) anche gli expert hanno bind group FISSI: la cache
// espone i suoi buffer come arena bindata intera e lo slot viaggia in `Sel`
// (VRAM) invece che nel bind group. Erano fino a ~2,5k bind group per pipeline
// costruiti a runtime, e soprattutto costruibili solo DOPO aver saputo quale
// expert il router aveva scelto: è quel legame — non la banda — che teneva
// l'encode del blocco expert dietro al sync per layer.
import { GLM47_FLASH as G } from "./shape";
import { repackQ4_0, repackQ4_1, repackQ8_0, repackKQuant, Q5_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES } from "./quant";
import { routerSelect, ROUTER_GLM47, WEIGHTS_SUM_CLAMP_MIN, SLAB_DOWN_Q4_0, SLAB_DOWN_Q4_1, type SlabLayout } from "./moe";
import { MLA_CHUNK_P, mlaSMax, mlaPartialsLen, mlaSplitWorkgroupStorageBytes } from "./mlasplit";
import {
  pairGemvSiluQ5KFastWorkgroupStorageBytes, gemvQ6KFastWorkgroupStorageBytes,
} from "./kquantfast";
import { downIsQ4_1 } from "./expertstore";
import { ExpertCache, expertKey, expertSlots, modelExpertPark, type ArenaGeometry, type ExpertClass, type ExpertRawBytes, type ExpertReader, type SlotRef, type BindRange } from "./residency";
import { expertArenaBindings } from "./gpulimits";
import { type CoreCounters } from "./telemetry";
import { planMoeChunk, GLM_PREFILL_M } from "./moeprefillplan";
import {
  addInPlaceWgsl, gemvF32Wgsl, gemvGrid, gemvQ6KFastWgsl, gemvQ8HeadsWgsl,
  gemvQuantWgsl, kvAppendWgsl, mlaAttnSplitPartWgsl, mlaAttnSplitReduceWgsl,
  ropeMlaNormWgsl, rmsnormWgsl,
  siluMulWgsl, stridedCopyWgsl, pairGemvSiluFastWgsl, pairGemvSiluQ5KFastWgsl,
  gemvAccumFastWgsl, routerTopKWgsl, type ArenaOpts,
  pairGemvSiluGatherWgsl, gemvDownSlotsWgsl, moeCombineWgsl,
  SEL_BYTES, MOE_IDX_BYTES, MOE_IDX_STRIDE,
} from "./kernels/wgsl";

const HL = G.qkNope + G.ropeDims; // 256

// Indirezione dell'arena expert (C3a fase 4 strato 1, design §2).
// `Sel` = una entry per (layer MoE, k): {id, slot, w già ×1.8, flags}, 16 B.
// Le taglie stanno in kernels/wgsl accanto alle struct WGSL che descrivono
// (fase-D it.14): q35gpumodel scrive la STESSA Sel, e due costanti locali
// sarebbero due ABI che nessun compilatore confronta.

// Sorgente pesi: byte GREZZI GGUF. In produzione è OPFS (ExpertOpfsStore +
// GgufExpertIndex); nei test un mock sintetico. `nonExpert` riceve il nome
// completo ("blk.3.ffn_norm.weight", "token_embd.weight", …).
export interface GlmWeightSource {
  nonExpert(name: string): Uint8Array;
  expert(layer: number, expert: number): ExpertRawBytes;
  /**
   * Slab GIA' impacchettato (repack all'import, C3a fase 3). Se presente, la
   * cache salta `packExpertSlab` nel path caldo. Opzionale: i mock dei test e
   * le sorgenti senza file slab espongono solo `expert`.
   */
  expertSlab?(layer: number, expert: number): Uint8Array;
  hasSlabs?: boolean;
}

export interface GlmModelOpts {
  nLayer?: number;  // default G.nLayer; i test usano 2 (blk.0 denso + blk.1 MoE)
  ctxMax: number;
  // Output head (fase 6): final output_norm F32 + output.weight Q6_K. vocab
  // parametrico SOLO per i ktest (head sintetica ridotta); default G.vocab.
  head?: boolean;
  vocab?: number;
  cache: { budgetBytes: number; maxBindingBytes: number; maxBufferBytes: number; slotsOverride?: { q4_0: number; q4_1: number }; timing?: boolean; policy?: "lru" | "tier" };
  /**
   * Chi riempie `Sel`, cioe' chi decide quali expert bindare (C3a fase 4).
   * - "cpu" (default): la selezione e' il routerSelect f64 dopo il sync per layer.
   * - "shadow" (slice B): il router+resolve gira anche su GPU e scrive Sel in una
   *   regione OMBRA, ma il forward resta bit-per-bit quello di "cpu" — serve a
   *   misurare la fedelta' del router GPU sul corpus vero prima di fidarsene.
   *   Costa 1 dispatch per layer MoE, 12 KB di slotTable e un readback di 5,7 KB
   *   su staging dedicata, mappata INSIEME a quella dell'hidden (un round-trip
   *   host solo, non uno in piu').
   * - "gpu" (slice C): il router+resolve scrive Sel nella regione di PRODUZIONE e
   *   i kernel expert la leggono direttamente. Sparisce tutto il blocco di sync
   *   per layer (copy dei logits, submit, mapAsync, routerSelect, ensure): un
   *   UNICO submit per token, zero readback del router. `routing[]` si ricostruisce
   *   al tail dalla copia di Sel che gia' si faceva.
   *   AMMESSO SOLO A RESIDENZA TOTALE: senza, e' un errore esplicito in
   *   createGlmModel, mai una degradazione silenziosa (vedi la precondizione).
   * - "optimistic" (C3b, spec 2026-08-07): la struttura di "gpu" (un submit,
   *   zero readback del router) a residenza PARZIALE nel regime near-total.
   *   Il resolve puo' trovare MISS: i kernel expert saltano (guardia `ok`,
   *   degrado DEFINITO), il resolve marca `dirtyB` (piggyback sul tail, zero
   *   sync in piu') e il forward riporta `dirty` invece di alzare — il token
   *   sporco NON va campionato (invariante I2: repair+replay in fase 3).
   *   Precondizione build-time: slots/parco >= optimisticMinResidency per
   *   classe, altrimenti errore esplicito (il regime di scarsita' e' C3c).
   */
  select?: "cpu" | "gpu" | "shadow" | "optimistic";
  /**
   * Soglia della precondizione del modo "optimistic" (default 0.88 = regime
   * near-total di WP-0). Override PER I TEST: in produzione non si abbassa
   * senza docket (contratto C3b).
   */
  optimisticMinResidency?: number;
  /**
   * Prefetch IN-FORWARD (C3c fase 4, spec §3): al router del layer MoE L si
   * incoda ANCHE il GEMV del router di L+1 sullo stesso hidden normato (fnB)
   * — stesso pass, stessa copy di staging, stessa mapAsync: zero sync in più.
   * Le predizioni (top-K=4 via routerSelect coi bias di L+1) si consumano al
   * submit successivo, PRIMA dell'await: il fetch OPFS+writeBuffer dei
   * predetti avviene mentre la GPU esegue il lavoro appena sottomesso — è il
   * tempo in cui la CPU oggi aspetta. Solo path sync per-posizione
   * (`select:"cpu"`): l'ottimistico C3b (I1-I5) non si tocca. NIENTE
   * predizioni oltre il confine di token (predictor al confine FALSIFICATO —
   * WP-0): l'ultimo layer MoE non ha tap. Default off = path bit-identico.
   */
  prefetch?: "inforward";
  /**
   * Checkpoint dell'hidden di INGRESSO di ogni layer (C3b spec §3c): una
   * copyBufferToBuffer per layer (x → hiddenCkpt[l]) nell'encoder del token —
   * zero dispatch in piu', l'input del replay dal primo layer sporco.
   * Solo nei modi a submit unico ("gpu"/"optimistic"): nel path sync il
   * confine di encoder per layer lo renderebbe ambiguo. Aggiunge un confine
   * di pass per layer: con telemetryGpu i pass aumentano (stessa avvertenza
   * del byCat — le quote restano confrontabili, il totale va dichiarato).
   */
  checkpointHidden?: boolean;
  /**
   * SOLO HARNESS: `false` disattiva repair+replay in "optimistic" (il forward
   * riporta `dirty` e restituisce il degrado definito — ktest fase 2). In
   * produzione resta il default `true`: un token sporco non si campiona mai.
   */
  optimisticRepair?: boolean;
  // Telemetria di attribuzione (goal C3a fase 1). Zero-overhead da spenta
  // (contratto CONSTRAINTS): senza `telemetry` non si chiama nemmeno
  // performance.now(). `telemetryGpu` aggiunge timestampWrites a ogni compute
  // pass (richiede la feature "timestamp-query" negoziata sul device) — è il
  // livello 2 del pattern Qwen (gpuforward, docs/engine/tsq-diag-2026-07-29).
  telemetry?: boolean;
  telemetryGpu?: boolean;
}

export interface GlmRouting {
  layer: number; experts: Int32Array; weights: Float64Array;
  /**
   * Selezione del router GPU, letta da Sel al tail. Presente SOLO con
   * `select: "shadow"`, dove non decide niente: e' il termine di confronto con
   * `experts`/`weights` (che restano quelli della CPU f64). `slots` e' cio' che
   * il resolve ha trovato nella slotTable e `flags` bit 0 dice che non c'era —
   * in shadow e' atteso, perche' il router gira prima degli `ensure` del layer.
   */
  gpu?: { experts: Int32Array; weights: Float64Array; slots: Uint32Array; flags: Uint32Array };
  /**
   * La regione di PRODUZIONE di Sel riletta dalla VRAM (stessi campi, in modo
   * `shadow` e in modo `gpu`): e' cio' che i kernel expert hanno letto davvero.
   * In modo `gpu` e' anche la SORGENTE di `experts`/`weights` — li' non esiste
   * un'altra opinione da confrontare: la decisione e la lettura coincidono, e
   * `slots`/`flags` sono l'unico modo di vedere che la residenza ha retto.
   * Confrontarla
   * con `experts`/`weights` e con gli slot degli `ensure` e' l'unica verifica
   * diretta che la writeBuffer della CPU sia arrivata dove e quando doveva
   * (rischio R5 del design, lato produzione). Non costa un readback in piu': la
   * copia era gia' dell'intera Sel.
   */
  vram?: { experts: Int32Array; weights: Float64Array; slots: Uint32Array; flags: Uint32Array };
}

/**
 * Esito di sporcizia di UN token in modo "optimistic" (C3b spec §3b/§4).
 * `firstDirtyLayer` e' il layer ASSOLUTO (−1 = token pulito); `misses` viene
 * dal readback Sel (id con flag MISS), `missCount`/`firstDirtyLayer` sono
 * CROSS-VERIFICATI col buffer dirtyB scritto dal kernel — una divergenza fra
 * i due e' un bug strutturale e alza, non si sceglie un lato.
 */
export interface GlmDirtyInfo {
  firstDirtyLayer: number;
  missCount: number;
  misses: { layer: number; id: number; k: number }[];
  /** true = il token e' stato RIPARATO e rigiocato: hidden/logits sono puliti. */
  repaired?: boolean;
  /**
   * Round di repair+replay del token (>= 1 se repaired). Piu' di 1 = cascata:
   * l'hidden riparato ha cambiato selezioni a valle (I4 emendata, it.4).
   */
  repairRounds?: number;
}

// Scomposizione del wall per token (contatori CUMULATIVI: il chiamante prende
// le differenze per finestra, come per cacheStats). Identità di costruzione:
//   wall = encodeCpuMs + ensureMs + routerWaitMs + tailWaitMs + residuo
// dove `residuo` (= sync non attribuito) si ottiene per differenza dal wall
// misurato fuori, e gpuBusyMs è la somma delle durate dei compute pass.
// Estende il nucleo unico dei due path (telemetry.ts, fase 4d): forwards,
// encodeCpuMs, submits, dispatches, gpuBusyMs, gpuPasses vengono da li' —
// qui restano solo i campi specifici del path GLM.
export interface GlmTelemetry extends CoreCounters {
  on: boolean;
  /**
   * Prefetch in-forward (C3c fase 4), null quando spento. `preds` =
   * predizioni consumate (fetch al submit successivo), `fetches` = quelle
   * NON residenti (I/O reale), `resident` = gia' in cache; recall in-engine:
   * hits4/hits8 su `recallPreds` predizioni × 4 expert veri (confronto col
   * 92.0% @K=8 dell'oracolo C1 — spiegato, non gateato).
   */
  prefetch: {
    preds: number; fetches: number; resident: number; prefetchMs: number;
    recallPreds: number; recallHits4: number; recallHits8: number;
  } | null;
  ensureMs: number;       // tempo dentro ExpertCache.ensure (residenza)
  routerWaitMs: number;   // tempo negli await di readback del router (46/token)
  tailWaitMs: number;     // tempo nell'await finale (hidden + logits)
  routerSyncs: number;    // readback router EFFETTIVI (non una costante)
  /**
   * Entry di Sel di PRODUZIONE trovate col flag MISS al tail (R6 del design):
   * expert che i kernel hanno dovuto saltare perche' il resolve non ha trovato
   * uno slot pubblicato. Deve essere 0 SEMPRE — in modo "cpu"/"shadow" la Sel di
   * produzione la scrive la CPU dopo gli `ensure`, in modo "gpu" la residenza e'
   * totale per precondizione. In modo "gpu" un valore > 0 rende il token
   * INVALIDO e il forward alza un errore invece di restituire numeri sbagliati.
   */
  selMiss: number;
  /**
   * Decode ottimistico (C3b): token con >= 1 miss (`dirtyTokens`), replay
   * eseguiti (`replays` — con repair attivo coincidono coi dirty), tempo CPU
   * del repair (fetch+upload+flush, `repairMs`). P(dirty) = dirtyTokens /
   * forwards; la tassa GPU del replay sta gia' dentro submits/dispatches/
   * gpuBusy, che contano TUTTI i giri.
   */
  dirtyTokens: number;
  replays: number;
  replayLayers: number; // somma dei layer rigiocati: E[replayFrac|dirty] = replayLayers/(replays*nLayer)
  repairMs: number;
  gpuPassOverflow: number;  // pass non strumentati per ring pieno (atteso 0)
  // Attribuzione di gpuBusyMs per CATEGORIA di kernel (spec §4, primo task
  // della fase 4b). Non nulla solo con setTelemetry(on, gpu, byCat=true), che
  // spezza i compute pass ai confini di categoria: fuori da quella modalita' i
  // pass ne contengono piu' d'una e la somma per categoria non esiste. La
  // modalita' aggiunge confini di pass, quindi il suo gpuBusy TOTALE non e'
  // confrontabile alla lettera con quello del bench headline — si confrontano
  // le QUOTE, e il totale serve a dichiarare quanto costano i confini.
  gpuByCatMs: Record<string, number> | null;
  gpuByCatPasses: Record<string, number> | null;
  // NOTA su `dispatches` (dal core): CONTATI a runtime, non la formula del
  // piano — confronta con `dispatchesPerTokenPlanned`.
}

export interface GlmModel {
  // hidden = stato post-ultimo-layer (readback: harness di conformance, non
  // bench). readLogits richiede opts.head: aggiunge final norm + lm_head e
  // ritorna i logits interi (154.880 × f32 = 620 KB/readback).
  forward(x: Float32Array, pos: number, readLogits?: boolean): Promise<{ hidden: Float32Array; logits?: Float32Array; routing: GlmRouting[]; dirty?: GlmDirtyInfo }>;
  /**
   * Readback dell'intero buffer di checkpoint (nLayer × dModel f32) — SOLO
   * harness/ktest (un sync dedicato). Richiede opts.checkpointHidden.
   */
  readHiddenCkpt(): Promise<Float32Array>;
  /**
   * SOLO HARNESS (ktest C3b): forza MISS deterministici marcando le coppie
   * (layer assoluto, expert id) nella slotTable. Richiede un modo col router
   * GPU. Da chiamare FRA i token, mai con un submit in volo.
   */
  debugMarkMiss(misses: { layer: number; id: number }[]): void;
  // ATTENZIONE alla semantica: questo e' il valore DERIVATO dal piano statico
  // (formula sui conteggi di layer), non un conteggio. Il numero misurato sta
  // in `telemetry().dispatches`. I due divergono: la formula non contiene la
  // testa (rms + lm_head = 2 dispatch), che pero' viene eseguita a ogni token
  // con readLogits. Tenuti entrambi e nominati per quello che sono.
  dispatchesPerTokenPlanned: number;
  cacheStats(): ReturnType<ExpertCache["stats"]>;
  /**
   * Prefill batched M>1 (fase 5): un chunk di M righe gia' embeddate
   * (M·dModel f32) alle posizioni posStart..posStart+M-1. Path ADDITIVO:
   * forward() e' intoccato. Ritorna il routing per riga e, sull'ULTIMA riga,
   * hidden e (opzionale) logits — riusando i passi head del decode.
   * Identita' col per-token garantita per costruzione (kernel bit-identici,
   * router CPU, combine k-order): v. docs/engine/glmprefill-wiring-plan.
   */
  prefillChunk(xRows: Float32Array, posStart: number, readLogitsLast?: boolean):
    Promise<{ hidden: Float32Array; logits?: Float32Array; routing: GlmRouting[][] }>;
  // Drena i batch di timestamp in volo e ritorna i contatori cumulativi.
  telemetry(): Promise<GlmTelemetry>;
  // Accende/spegne la REGISTRAZIONE (la capacità è fissata da opts). Da spenta
  // il forward non chiama performance.now() né scrive timestamp.
  // byCat: spezza i compute pass ai confini di categoria per attribuire
  // gpuBusy a attn/router/shexp/experts/addMoe/head. Costa confini in più:
  // si usa in repliche dedicate, mai nelle finestre che alimentano i gate.
  setTelemetry(on: boolean, gpu?: boolean, byCat?: boolean): void;
  destroy(): void;
}

export function createGlmModel(device: GPUDevice, src: GlmWeightSource, opts: GlmModelOpts): GlmModel {
  const nLayer = opts.nLayer ?? G.nLayer;
  const ctxMax = opts.ctxMax;
  // Guardia sul workgroup storage: il MASSIMO fra tutti i kernel del modello
  // che usano shared memory. Non solo lo split MLA — i kernel fast K-quant
  // (it.13) tengono x intero in shared, quindi il loro fabbisogno CRESCE con K
  // (8 448 B di solo x a K=2048) e va nel conto per ogni K di produzione.
  // ONESTA' SU COSA E': sui path attuali NON puo' scattare. La negoziazione
  // chiede sempre almeno i 30 848 B del path Qwen fuso (requisito hard,
  // gpulimits) e il massimo qui sotto e' ~9 KB. Resta come guardia per gli
  // EDIT FUTURI dei kernel: se qualcuno allarga un tile, alza un K o rimette
  // un array [sMax] in shared, si ferma qui con un messaggio invece che a
  // runtime con un errore di validazione criptico.
  // Cio' che NON e' piu': un vincolo sul CONTESTO. Il monolitico MLA teneva
  // scores[ctxMax] in shared (4·ctxMax+256 B, che tagliava ctxMax a 8128 sul
  // default 16 KiB, it.10); ora nessuno dei kernel dipende da ctxMax.
  const wgNeeds: Array<[string, number]> = [ // formule dai moduli, non riscritte
    ["mlaAttnSplitPart", mlaSplitWorkgroupStorageBytes(G.nHead)],
    ["pairGemvSiluQ5KFast (shexp gate/up)", pairGemvSiluQ5KFastWorkgroupStorageBytes(G.dModel)],
    ["gemvQ6KFast (shexp down)", gemvQ6KFastWorkgroupStorageBytes(G.dFfnExpert)],
    ["gemvQ6KFast (testa)", gemvQ6KFastWorkgroupStorageBytes(G.dModel)],
  ];
  const worst = wgNeeds.reduce((a, b) => (b[1] > a[1] ? b : a));
  if (worst[1] > device.limits.maxComputeWorkgroupStorageSize) {
    throw new Error(
      `glmmodel: ${worst[0]} richiede ${worst[1]} B di workgroup storage ` +
      `(limite ${device.limits.maxComputeWorkgroupStorageSize}) — negoziare ` +
      `maxComputeWorkgroupStorageSize`);
  }

  // ---- telemetria di attribuzione (C3a fase 1) ----
  // Livello 1 (CPU): contatori a costo di una performance.now() per segmento,
  // accesi solo con opts.telemetry. Livello 2 (GPU): timestampWrites su ogni
  // compute pass, un resolve per token nel submit finale — la mapAsync parte
  // SEMPRE dopo il submit (root-cause del known-issue fase A, tsq-diag).
  // `telemetry`/`telemetryGpu` sono la CAPACITÀ (allocano il query set); la
  // registrazione si accende/spegne a runtime con setTelemetry, così le
  // repliche headline girano a overhead nullo sullo stesso modello (una
  // seconda istanza GLM raddoppierebbe la VRAM: non ci sta a slab 12 GiB —
  // qui divergiamo dal pattern Qwen "secondo engine dedicato", motivo VRAM).
  const canGpuTs = opts.telemetryGpu === true && device.features.has("timestamp-query");
  let telemOn = opts.telemetry === true;
  let wantGpuTs = telemOn && canGpuTs;
  let wantByCat = false; // attribuzione per categoria: spezza i pass, off di default
  const TSQ_PASSES = 512; // ≈3 pass/layer × 47 layer = ~141: margine 3.6×
  const querySet = canGpuTs ? device.createQuerySet({ type: "timestamp", count: TSQ_PASSES * 2 }) : null;
  const tsqResolve = canGpuTs
    ? device.createBuffer({ label: "glm-tsq-resolve", size: TSQ_PASSES * 2 * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC })
    : null;
  const pendingTsq: Promise<{ ms: number; passes: number; byCat: Map<string, number>; byCatN: Map<string, number> }>[] = [];
  const T = {
    forwards: 0, encodeCpuMs: 0, ensureMs: 0, routerWaitMs: 0, tailWaitMs: 0,
    routerSyncs: 0, submits: 0, selMiss: 0, gpuBusyMs: 0, gpuPasses: 0, gpuPassOverflow: 0,
    dirtyTokens: 0, replays: 0, replayLayers: 0, repairMs: 0,
    dispatches: 0,
    // prefetch in-forward (C3c fase 4)
    prefetchPreds: 0, prefetchFetches: 0, prefetchResident: 0, prefetchMs: 0,
    recallPreds: 0, recallHits4: 0, recallHits8: 0,
  };
  const catMs = new Map<string, number>();
  const catPasses = new Map<string, number>();
  const nowT = (): number => (telemOn ? performance.now() : 0);
  const armTsq = (staging: GPUBuffer, passes: number, cats?: string[]): void => {
    // `cats` è una COPIA dell'etichetta per indice di pass del token: la
    // mapAsync si risolve molto dopo, quando l'array vivo è già di un altro
    // token.
    pendingTsq.push(
      (async () => {
        await staging.mapAsync(GPUMapMode.READ);
        const ts = new BigUint64Array(staging.getMappedRange().slice(0));
        staging.destroy();
        let ms = 0;
        const byCat = new Map<string, number>();
        const byCatN = new Map<string, number>();
        for (let i = 0; i < passes; i++) {
          const d = Number(ts[i * 2 + 1] - ts[i * 2]) / 1e6;
          ms += d;
          const c = cats?.[i];
          if (c !== undefined) {
            byCat.set(c, (byCat.get(c) ?? 0) + d);
            byCatN.set(c, (byCatN.get(c) ?? 0) + 1); // pass, non batch
          }
        }
        return { ms, passes, byCat, byCatN };
      })(),
    );
  };

  // ---- upload helper (come glmforward) ----
  const upload = (data: Uint32Array | Float32Array): GPUBuffer => {
    const b = device.createBuffer({ size: Math.max(16, data.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(b, 0, data as BufferSource);
    return b;
  };
  const storage = (bytesLen: number): GPUBuffer =>
    device.createBuffer({ size: Math.max(16, bytesLen), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const f32Of = (raw: Uint8Array): Float32Array => {
    const copy = new Uint8Array(raw.length); // il source può riusare il buffer
    copy.set(raw);
    return new Float32Array(copy.buffer, 0, copy.length / 4);
  };
  const quantBufs = (raw: Uint8Array, kind: "q4_0" | "q4_1" | "q8_0") => {
    const bb = kind === "q4_0" ? 18 : kind === "q4_1" ? 20 : 34;
    const rp = (kind === "q4_0" ? repackQ4_0 : kind === "q4_1" ? repackQ4_1 : repackQ8_0)(raw, 0, raw.length / bb);
    return { qs: upload(rp.qs), scales: upload(rp.scales) };
  };
  const kquantBuf = (raw: Uint8Array, blockBytes: number): GPUBuffer =>
    upload(repackKQuant(raw, 0, raw.length / blockBytes, blockBytes));

  // ---- pipeline condivise ----
  // `layout: "auto"` resta il default dei ~20 kernel a binding fisso; il layout
  // esplicito serve alle sole pipeline expert, perche' `hasDynamicOffset` NON e'
  // esprimibile con "auto" (WebGPU deriva il layout dal WGSL, e il WGSL non dice
  // se un uniform e' bindato a offset dinamico).
  const mkPipe = (code: string, layout?: GPUBindGroupLayout) =>
    device.createComputePipeline({
      layout: layout ? device.createPipelineLayout({ bindGroupLayouts: [layout] }) : "auto",
      compute: { module: device.createShaderModule({ code }), entryPoint: "main" },
    });

  // ---- cache expert in modo ARENA (C3a fase 4, strato 1) ----
  // La cache si costruisce PRIMA delle pipeline expert: e' lei a dire quanti
  // buffer ha una classe, e quel numero e' baked nel WGSL (i binding d'arena e
  // gli archi dello switch di `ld4`). Al contrario del regime a sotto-range, qui
  // il bind group non dipende piu' dallo slot: e' UNO per (classe, stadio),
  // costruito al load, e lo slot entra come indirizzo attraverso `Sel`.
  // "shadow" (slice B): il router GPU gira e risolve Sel in una regione OMBRA,
  // ma chi comanda resta la CPU — stesso readback, stesso routerSelect f64,
  // stessi ensure, stessa Sel di produzione. Serve a misurare la fedelta' del
  // router GPU sul corpus vero PRIMA di dargli il volante (slice C).
  // "gpu" (slice C): comanda il router GPU. La CPU non sa piu' chi e' stato
  // scelto, quindi non puo' ne' caricare ne' pinnare niente durante il token: e'
  // ammesso solo se il parco expert del modello e' GIA' tutto in VRAM.
  const shadow = opts.select === "shadow";
  const gpuSel = opts.select === "gpu";
  const optSel = opts.select === "optimistic";
  if (opts.select !== undefined && opts.select !== "cpu" && !shadow && !gpuSel && !optSel) {
    throw new Error(`glmmodel: select "${opts.select}" non esiste — "cpu", "shadow", "gpu" o "optimistic"`);
  }
  // Il router GPU (routerTopK + resolve) gira in shadow, gpu e optimistic:
  // cambia dove scrive, non se gira. In shadow va nella regione ombra e non
  // decide niente; in gpu/optimistic scrive la produzione ed e' l'unico a
  // decidere. `oneSubmit` = la struttura a UN submit per token dello slice C:
  // optimistic la eredita per intero, cambia solo cosa e' ammesso trovare
  // nella slotTable (MISS) e cosa se ne fa il tail (dirty, non throw).
  const routerGpu = shadow || gpuSel || optSel;
  const oneSubmit = gpuSel || optSel;
  // repair+replay attivo di default in optimistic; `false` e' SOLO harness
  // (ktest fase 2: rende osservabile il degrado definito, che il repair
  // altrimenti ripara prima del ritorno).
  const repairOn = opts.optimisticRepair !== false;
  if (opts.checkpointHidden === true && !oneSubmit) {
    throw new Error('glmmodel: checkpointHidden richiede select "gpu" o "optimistic" (submit unico)');
  }
  // C3c fase 4: prefetch in-forward — vive nella sezione sync per-posizione
  // (select "cpu" e "shadow": stesso branch !oneSubmit). Nei modi a submit
  // unico non esiste il punto di consumo (I1-I5 c3b non si toccano).
  const prefetchOn = opts.prefetch === "inforward";
  if (prefetchOn && oneSubmit) {
    throw new Error('glmmodel: prefetch "inforward" vive nel path sync per-posizione (select "cpu"/"shadow")');
  }
  // PRECONDIZIONE DI RESIDENZA TOTALE (design §4 slice C, R6). Si verifica PRIMA
  // di costruire la cache: un errore dopo lascerebbe in giro i buffer d'arena
  // (GB di VRAM) di un modello che non nascera' mai.
  if (gpuSel) {
    const slots = expertSlots(opts.cache);
    const park = modelExpertPark(nLayer);
    const corte = (["q4_0", "q4_1"] as const).filter((c) => park[c] > slots[c]);
    if (corte.length > 0) {
      throw new Error(
        'glmmodel: select "gpu" esige la RESIDENZA TOTALE — ' +
        corte.map((c) => `classe ${c}: ${slots[c]} slot per ${park[c]} expert del modello`).join("; ") +
        ". In modo gpu la CPU non vede piu' la selezione, quindi non puo' ne' caricare ne' pinnare " +
        "durante il token: a residenza parziale uno slot risolto puo' essere evinto DOPO la " +
        "risoluzione, nello stesso token (seam evict-post-resolve), e i kernel expert leggerebbero " +
        "lo slab di un altro expert senza che nessun contatore lo dica. Allargare gli slot " +
        '(residenza totale, fase 4c) oppure usare select "cpu".');
    }
  }
  // PRECONDIZIONE DEL REGIME OTTIMISTICO (C3b spec §2, emendata in it.4 —
  // docket c3b item 6): residenza TOTALE complessiva >= soglia. Sul TOTALE e
  // non per classe: P(dirty) dipende dalla frazione complessiva di selezioni
  // residenti, le classi sono un dettaglio di storage — e il riparto
  // proporzionale rendeva q4_1 un vincolo artificiale (0.88 per classe =
  // 13.5 GiB di slab, oltre il tetto fisico del device a QUALUNQUE host).
  // La soglia di RIFIUTO e' 0.80: WP-0 misura il collasso a <= 50% (100%
  // token sporchi) e piena funzionalita' a 82% (P(dirty) 85%, 9.55 tok/s
  // proiettati — sopra il decode sync attuale); ~0.88 resta il riferimento
  // delle proiezioni "near-total", non il confine del meccanismo.
  const optMinRes = opts.optimisticMinResidency ?? 0.8;
  if (optSel) {
    const slots = expertSlots(opts.cache);
    const park = modelExpertPark(nLayer);
    const parkTot = park.q4_0 + park.q4_1;
    const slotTot = Math.min(slots.q4_0, park.q4_0) + Math.min(slots.q4_1, park.q4_1);
    if (slotTot < Math.ceil(optMinRes * parkTot)) {
      throw new Error(
        `glmmodel: select "optimistic" esige la residenza NEAR-TOTAL — ${slotTot} slot per ` +
        `${parkTot} expert = ${(100 * slotTot / parkTot).toFixed(1)}% < ${optMinRes}. ` +
        "Sotto questa soglia il decode ottimistico collassa (WP-0: a budget 50/25% ogni token e' " +
        "sporco e il replay costa piu' del sync). Il regime di scarsita' e' materia della fase C3c " +
        '(sync+overlap): usare select "cpu" oppure allargare gli slot.');
    }
  }
  const cache = new ExpertCache(device, { ...opts.cache, arena: true, slotTable: routerGpu });
  const arenaOptsOf = (geo: ArenaGeometry): ArenaOpts => {
    const l = geo.layout;
    // Gli offset arrivano da SlabLayout (moe.ts), che li garantisce multipli di
    // 256. Il requisito dei kernel d'arena è più forte di quello della word:
    // `ld4` indicizza `array<vec4<u32>>`, cioè a 16 B, e l'indice si ottiene
    // con `>> 2u` sulla word — un offset multiplo di 4 ma non di 16 verrebbe
    // TRONCATO in silenzio al vec4 precedente, leggendo pesi sfasati invece di
    // fallire. Si asserta qui, sezione per sezione, e non si spera.
    const secs: Array<[string, number]> = [
      ["gateQs", l.gateQs], ["gateScales", l.gateScales], ["upQs", l.upQs], ["upScales", l.upScales],
      ["downQs", l.downQs], ["downScales", l.downScales], ["bytes (passo dello slab)", l.bytes],
    ];
    for (const [nm, off] of secs) {
      if (off % 16 !== 0) {
        throw new Error(
          `glmmodel arena: ${nm} = ${off} B non e' multiplo di 16 (indice vec4 di ld4) — ` +
          "il >> 2u lo troncherebbe senza errore");
      }
    }
    return {
      nBuf: geo.nBuf, slabWords: geo.slabWords, slabsPerBuf: geo.slabsPerBuf,
      qsWords: l.downQs / 4, scalesWords: l.downScales / 4,
      gateQsWords: l.gateQs / 4, gateScWords: l.gateScales / 4,
      upQsWords: l.upQs / 4, upScWords: l.upScales / 4,
    };
  };
  /**
   * Il kernel `gemvAccumFast` esiste solo per i due formati legacy: la classe
   * (ora una stringa, goal fase-D) va ristretta esplicitamente, con errore
   * parlante se un modello introducesse una classe che quel kernel non sa fare.
   */
  const legacyDownKind = (cls: ExpertClass): "q4_0" | "q4_1" => {
    if (cls !== "q4_0" && cls !== "q4_1") {
      throw new Error(`glmmodel: classe ${cls} senza kernel gemvAccumFast (formati legacy attesi)`);
    }
    return cls;
  };
  const mkExpertClass = (cls: ExpertClass) => {
    const geo = cache.arenaGeometry(cls);
    const need = expertArenaBindings(geo.nBuf);
    if (need > device.limits.maxStorageBuffersPerShaderStage) {
      throw new Error(
        `glmmodel arena: la classe ${cls} ha ${geo.nBuf} buffer ⇒ ${need} storage binding, ` +
        `il device ne concede ${device.limits.maxStorageBuffersPerShaderStage} — negoziare ` +
        "maxStorageBuffersPerShaderStage con arenaBuffers (gpulimits/arenaNeeds)");
    }
    const arena = arenaOptsOf(geo);
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
    return {
      geo, bgl,
      gu: mkPipe(pairGemvSiluFastWgsl({ K: G.dModel, N: G.dFfnExpert, arena }), bgl),
      down: mkPipe(gemvAccumFastWgsl({ kind: legacyDownKind(cls), K: G.dFfnExpert, N: G.dModel, arena }), bgl),
    };
  };
  const expert: Record<ExpertClass, ReturnType<typeof mkExpertClass>> = { q4_0: mkExpertClass("q4_0"), q4_1: mkExpertClass("q4_1") };

  const pipes = {
    rmsD: mkPipe(rmsnormWgsl(G.dModel, G.rmsEps)),
    rmsQA: mkPipe(rmsnormWgsl(G.qLora, G.rmsEps)),
    rmsKvA: mkPipe(rmsnormWgsl(G.kvLora, G.rmsEps)),
    gemvQA: mkPipe(gemvQuantWgsl({ kind: "q4_0", K: G.dModel, N: G.qLora, hasBias: false })),
    gemvQB: mkPipe(gemvQuantWgsl({ kind: "q4_0", K: G.qLora, N: G.nHead * HL, hasBias: false })),
    gemvKvA: mkPipe(gemvQuantWgsl({ kind: "q8_0", K: G.dModel, N: G.keyLen, hasBias: false })),
    ropeQ: mkPipe(ropeMlaNormWgsl({ nVec: G.nHead, stride: HL, offset: G.qkNope, ropeDims: G.ropeDims, freqBase: G.ropeFreqBase })),
    ropeKPe: mkPipe(ropeMlaNormWgsl({ nVec: 1, stride: G.keyLen, offset: G.kvLora, ropeDims: G.ropeDims, freqBase: G.ropeFreqBase })),
    kvApp: mkPipe(kvAppendWgsl(G.keyLen)),
    absorbKb: mkPipe(gemvQ8HeadsWgsl({ K: G.qkNope, rowsPerHead: G.kvLora, nHead: G.nHead, xStride: HL, xOffset: 0 })),
    copyCkv: mkPipe(stridedCopyWgsl({ nVec: G.nHead, len: G.kvLora, srcStride: G.kvLora, srcOffset: 0, dstStride: G.keyLen, dstOffset: 0 })),
    copyQRope: mkPipe(stridedCopyWgsl({ nVec: G.nHead, len: G.ropeDims, srcStride: HL, srcOffset: G.qkNope, dstStride: G.keyLen, dstOffset: G.kvLora })),
    // attention MLA split sul contesto (fase 4c): part su sMax workgroup che
    // coprono TUTTE le head, reduce log-sum-exp su nHead. Il monolitico
    // (mlaAttnDecodeWgsl) resta nel repo: lo usano glmforward e i ktest, ed e'
    // il riferimento del test di identita'.
    attnPart: mkPipe(mlaAttnSplitPartWgsl({
      nHead: G.nHead, kvLora: G.kvLora, ropeDims: G.ropeDims, ctxMax,
      scale: 1 / Math.sqrt(G.headLenMla), chunk: MLA_CHUNK_P,
    })),
    attnReduce: mkPipe(mlaAttnSplitReduceWgsl({ nHead: G.nHead, kvLora: G.kvLora, ctxMax, chunk: MLA_CHUNK_P })),
    voutVb: mkPipe(gemvQ8HeadsWgsl({ K: G.kvLora, rowsPerHead: G.headLenMla, nHead: G.nHead, xStride: G.kvLora, xOffset: 0 })),
    gemvO: mkPipe(gemvQuantWgsl({ kind: "q4_0", K: G.nHead * G.headLenMla, N: G.dModel, hasBias: false })),
    // ffn denso blk.0
    gemvDenseGU: mkPipe(gemvQuantWgsl({ kind: "q4_0", K: G.dModel, N: G.dFfnDense, hasBias: false })),
    gemvDenseDown: mkPipe(gemvQuantWgsl({ kind: "q4_1", K: G.dFfnDense, N: G.dModel, hasBias: false })),
    siluDense: mkPipe(siluMulWgsl(G.dFfnDense)),
    // MoE
    router: mkPipe(gemvF32Wgsl({ K: G.dModel, N: G.nExpert })),
    // shexp: famiglia fast portata sui K-quant (it.13). gate+up+silu Q5_K in un
    // dispatch, down Q6_K con la stessa struttura. I gemelli lenti
    // (gemvQ5KWgsl/gemvQ6KWgsl) restano nel repo: li usano i ktest.
    pairSiluShexp: mkPipe(pairGemvSiluQ5KFastWgsl({ K: G.dModel, N: G.dFfnExpert })),
    gemvShexpDown: mkPipe(gemvQ6KFastWgsl({ K: G.dFfnExpert, N: G.dModel })),
    // (la catena expert vive in `expert[cls]`: e' l'unica famiglia a layout
    // esplicito, perche' e' l'unica che binda l'arena a dynamic offset)
    add: mkPipe(addInPlaceWgsl(G.dModel)),
  };

  // ---- pipeline BATCH del prefill (fase 5, piano glmprefill-wiring-plan) ----
  // Ogni variante e' bit-identica al per-riga (ktest it.27-31, 65/65). M e' il
  // tetto del chunk; i chunk parziali dispatchano rows<M (mai righe garbage:
  // kvApp scriverebbe la cache a posizioni stantie).
  const MPF = GLM_PREFILL_M;
  const pipesB = {
    rmsD: mkPipe(rmsnormWgsl(G.dModel, G.rmsEps, true)),
    rmsQA: mkPipe(rmsnormWgsl(G.qLora, G.rmsEps, true)),
    rmsKvA: mkPipe(rmsnormWgsl(G.kvLora, G.rmsEps, true, { x: G.keyLen, out: G.keyLen })),
    gemvQA: mkPipe(gemvQuantWgsl({ kind: "q4_0", K: G.dModel, N: G.qLora, hasBias: false, batch: true })),
    gemvQB: mkPipe(gemvQuantWgsl({ kind: "q4_0", K: G.qLora, N: G.nHead * HL, hasBias: false, batch: true })),
    gemvKvA: mkPipe(gemvQuantWgsl({ kind: "q8_0", K: G.dModel, N: G.keyLen, hasBias: false, batch: true })),
    ropeQ: mkPipe(ropeMlaNormWgsl({ nVec: G.nHead, stride: HL, offset: G.qkNope, ropeDims: G.ropeDims, freqBase: G.ropeFreqBase, batch: true })),
    ropeKPe: mkPipe(ropeMlaNormWgsl({ nVec: 1, stride: G.keyLen, offset: G.kvLora, ropeDims: G.ropeDims, freqBase: G.ropeFreqBase, batch: true })),
    kvApp: mkPipe(kvAppendWgsl(G.keyLen, true)),
    // la copy k_pe (rope→row576) diventa una strided copy batch: 1 dispatch, non M copy
    kpeCopy: mkPipe(stridedCopyWgsl({ nVec: 1, len: G.ropeDims, srcStride: G.keyLen, srcOffset: G.kvLora, dstStride: G.keyLen, dstOffset: G.kvLora, batch: true })),
    absorbKb: mkPipe(gemvQ8HeadsWgsl({ K: G.qkNope, rowsPerHead: G.kvLora, nHead: G.nHead, xStride: HL, xOffset: 0, batch: true })),
    copyCkv: mkPipe(stridedCopyWgsl({ nVec: G.nHead, len: G.kvLora, srcStride: G.kvLora, srcOffset: 0, dstStride: G.keyLen, dstOffset: 0, batch: true })),
    copyQRope: mkPipe(stridedCopyWgsl({ nVec: G.nHead, len: G.ropeDims, srcStride: HL, srcOffset: G.qkNope, dstStride: G.keyLen, dstOffset: G.kvLora, batch: true })),
    attnPart: mkPipe(mlaAttnSplitPartWgsl({
      nHead: G.nHead, kvLora: G.kvLora, ropeDims: G.ropeDims, ctxMax,
      scale: 1 / Math.sqrt(G.headLenMla), chunk: MLA_CHUNK_P, batch: true,
    })),
    attnReduce: mkPipe(mlaAttnSplitReduceWgsl({ nHead: G.nHead, kvLora: G.kvLora, ctxMax, chunk: MLA_CHUNK_P, batch: true })),
    voutVb: mkPipe(gemvQ8HeadsWgsl({ K: G.kvLora, rowsPerHead: G.headLenMla, nHead: G.nHead, xStride: G.kvLora, xOffset: 0, batch: true })),
    gemvO: mkPipe(gemvQuantWgsl({ kind: "q4_0", K: G.nHead * G.headLenMla, N: G.dModel, hasBias: false, batch: true })),
    gemvDenseGU: mkPipe(gemvQuantWgsl({ kind: "q4_0", K: G.dModel, N: G.dFfnDense, hasBias: false, batch: true })),
    gemvDenseDown: mkPipe(gemvQuantWgsl({ kind: "q4_1", K: G.dFfnDense, N: G.dModel, hasBias: false, batch: true })),
    siluDenseM: mkPipe(siluMulWgsl(MPF * G.dFfnDense)),  // elementwise: griglia tagliata a m righe
    router: mkPipe(gemvF32Wgsl({ K: G.dModel, N: G.nExpert, batch: true })),
    pairSiluShexp: mkPipe(pairGemvSiluQ5KFastWgsl({ K: G.dModel, N: G.dFfnExpert, batch: true })),
    gemvShexpDown: mkPipe(gemvQ6KFastWgsl({ K: G.dFfnExpert, N: G.dModel, batch: true })),
    addM: mkPipe(addInPlaceWgsl(MPF * G.dModel)),
    pairGather: mkPipe(pairGemvSiluGatherWgsl({ K: G.dModel, N: G.dFfnExpert })),
    downSlots40: mkPipe(gemvDownSlotsWgsl({ kind: "q4_0", K: G.dFfnExpert, N: G.dModel })),
    downSlots41: mkPipe(gemvDownSlotsWgsl({ kind: "q4_1", K: G.dFfnExpert, N: G.dModel })),
    combine: mkPipe(moeCombineWgsl({ D: G.dModel })),
  };

  // ---- attivazioni condivise ----
  const x = storage(G.dModel * 4);
  const hn = storage(G.dModel * 4);
  const qaB = storage(G.qLora * 4);
  const qanB = storage(G.qLora * 4);
  const qB = storage(G.nHead * HL * 4);
  const kvB = storage(G.keyLen * 4);
  const row576 = storage(G.keyLen * 4);
  const qCkv = storage(G.nHead * G.kvLora * 4);
  const q576 = storage(G.nHead * G.keyLen * 4);
  const attnCkv = storage(G.nHead * G.kvLora * 4);
  // partials dello split: [nHead × sMax × (kvLora+2)], condiviso fra i layer
  // come attnCkv (il pass 2 lo consuma prima che il layer successivo lo riscriva)
  const attnPartials = storage(mlaPartialsLen(G.nHead, G.kvLora, ctxMax) * 4);
  const attnSMaxN = mlaSMax(ctxMax);
  const attnMla = storage(G.nHead * G.headLenMla * 4);
  const tmp = storage(G.dModel * 4);
  const fnB = storage(G.dModel * 4);
  const gateD = storage(G.dFfnDense * 4);
  const upD = storage(G.dFfnDense * 4);
  // gateE tiene gia' silu(gate)·up: sia la catena expert sia lo shexp fondono
  // gate+up+silu in un dispatch, quindi non esiste piu' un buffer `up` separato
  const gateE = storage(G.dFfnExpert * 4);
  const moeOut = storage(G.dModel * 4);
  const logitsB = storage(G.nExpert * 4);
  // regioni gather per expert dell'unione: allineate a 256 B (minStorageBufferOffsetAlignment)
  const GATHER_REGION = 256;
  // I buffer M× del prefill sono LAZY (fix it.35): allocarli al load costava
  // ~275 MB a ctx 6688 (attnPartialsM) e ha mandato in OOM glmroute/glmconf,
  // che non prefillano mai. Si pagano al PRIMO prefillChunk.
  // niente piu' `wExp`: il peso di mixing e' il campo `w` di Sel (§2.1 del
  // design) — la stessa indirezione che porta lo slot porta anche il peso.
  const P = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  // C3c fase 4: col prefetch lo staging raddoppia — [0,256) logits del router
  // del layer corrente, [256,512) logits del TAP (router L+1 su fnB). Stessa
  // copy window, stessa mapAsync: il tap non aggiunge sync.
  const logitsStaging = device.createBuffer({ size: G.nExpert * 4 * (prefetchOn ? 2 : 1), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const logits2B = prefetchOn ? storage(G.nExpert * 4) : null;
  const hiddenStaging = device.createBuffer({ size: G.dModel * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

  const bg = (pipe: GPUComputePipeline, bufs: Array<GPUBuffer | BindRange>, uni?: GPUBuffer) =>
    device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: [
        ...bufs.map((b, i) => ({ binding: i, resource: "buffer" in b ? b : { buffer: b } })),
        ...(uni ? [{ binding: bufs.length, resource: { buffer: P } }] : []),
      ],
    });

  // ---- pesi e bind group per-layer ----
  interface Step { pipe: GPUComputePipeline; bind: GPUBindGroup; wgs: [number, number] }
  const step = (pipe: GPUComputePipeline, bufs: Array<GPUBuffer | BindRange>, wgs: number | [number, number], uni?: boolean): Step =>
    ({ pipe, bind: bg(pipe, bufs, uni ? P : undefined), wgs: typeof wgs === "number" ? [wgs, 1] : wgs });
  const stepB = (pipe: GPUComputePipeline, bufs: Array<GPUBuffer | BindRange>, wgs: number | [number, number], rowsDim: "x" | "y" | "z"): BStep =>
    ({ pipe, bind: bg(pipe, bufs), wgs: typeof wgs === "number" ? [wgs, 1] : wgs, rowsDim });


  interface BStep { pipe: GPUComputePipeline; bind: GPUBindGroup; wgs: [number, number]; rowsDim: "x" | "y" | "z" }

  interface MoeLayerGpu {
    bias: Float32Array;
    biasBuf: GPUBuffer | null; // gli stessi bias in VRAM: li legge il router GPU (shadow)
    cls: ExpertClass;        // classe degli expert del layer (down q4_0 o q4_1)
    moeIdx: number;          // indice del layer fra i soli MoE
    selBase: number;         // prima entry di Sel del layer: moeLayerIdx * nUsed
    routerBind: GPUBindGroup | null; // bind group del router GPU (shadow)
    preRouter: Step[];       // ffn_norm + router (dopo l'attention)
    /** C3c fase 4: GEMV del router del PROSSIMO layer MoE su fnB (il tap).
     *  Assente sull'ultimo layer MoE: niente predizioni oltre il confine di
     *  token (WP-0). Costruito solo con prefetch attivo. */
    tapNext?: { st: Step; layer: number; bias: Float32Array };
    shexp: Step[];           // scrive moeOut
    addMoe: Step;            // x += moeOut
    downLayout: SlabLayout;  // layout slab della classe (sub-range dei gather)
  }
  interface LayerGpu {
    attn: Step[];            // attention completa (residuo incluso), con 1 copy k_pe
    kpeCopy: { srcOff: number; dstOff: number; bytes: number }; // dentro attn dopo rmsKvA
    dense?: Step[];          // blk.0
    moe?: MoeLayerGpu;
    cache: GPUBuffer;        // kv cache 576/token
    // ref dei pesi per la costruzione LAZY dei passi batch del prefill
    bw: LayerBW;
  }
  interface QBb { qs: GPUBuffer; scales: GPUBuffer }
  interface LayerBW {
    attnNorm: GPUBuffer; qANorm: GPUBuffer; kvANorm: GPUBuffer;
    wQA: QBb; wQB: QBb; wKvA: QBb; wKB: QBb; wVB: QBb; wO: QBb; cache: GPUBuffer;
    ffnNorm?: GPUBuffer;
    wGate?: QBb; wUp?: QBb; wDown?: QBb;
    routerW?: GPUBuffer; gateShexp?: GPUBuffer; upShexp?: GPUBuffer; downShexp?: GPUBuffer;
  }

  const layers: LayerGpu[] = [];
  let nMoeSeen = 0;                   // indice del layer MoE, per la base in Sel
  // layer ASSOLUTO di ogni indice MoE: la slotTable e' indicizzata da
  // `expertKey(layer, e)` = layer·nExpert + e, e quel layer e' quello vero (blk.N),
  // non la sua posizione fra i MoE — la cache evince senza guardare i layer.
  const moeLayerAbs: number[] = [];
  const weightBufs: GPUBuffer[] = []; // per destroy
  const track = <T extends GPUBuffer>(b: T): T => { weightBufs.push(b); return b; };

  for (let l = 0; l < nLayer; l++) {
    const nm = (n: string) => `blk.${l}.${n}.weight`;
    const w = (n: string, kind: "q4_0" | "q4_1" | "q8_0") => {
      const q = quantBufs(src.nonExpert(nm(n)), kind);
      track(q.qs); track(q.scales);
      return q;
    };
    const f = (n: string) => track(upload(f32Of(src.nonExpert(nm(n)))));
    const attnNorm = f("attn_norm");
    const qANorm = f("attn_q_a_norm");
    const kvANorm = f("attn_kv_a_norm");
    const ffnNorm = f("ffn_norm");
    const wQA = w("attn_q_a", "q4_0");
    const wQB = w("attn_q_b", "q4_0");
    const wKvA = w("attn_kv_a_mqa", "q8_0");
    const wKB = w("attn_k_b", "q8_0");
    const wVB = w("attn_v_b", "q8_0");
    const wO = w("attn_output", "q4_0");
    const cache = storage(ctxMax * G.keyLen * 4);
    weightBufs.push(cache);

    const attn: Step[] = [
      step(pipes.rmsD, [x, attnNorm, hn], 1),
      step(pipes.gemvQA, [wQA.qs, wQA.scales, hn, qaB], gemvGrid(G.qLora)),
      step(pipes.rmsQA, [qaB, qANorm, qanB], 1),
      step(pipes.gemvQB, [wQB.qs, wQB.scales, qanB, qB], gemvGrid(G.nHead * HL)),
      step(pipes.ropeQ, [qB], Math.ceil((G.nHead * G.ropeDims / 2) / 64), true),
      step(pipes.gemvKvA, [wKvA.qs, wKvA.scales, hn, kvB], gemvGrid(G.keyLen)),
      step(pipes.ropeKPe, [kvB], Math.ceil((G.ropeDims / 2) / 64), true),
      step(pipes.rmsKvA, [kvB, kvANorm, row576], 1),
      // (copy k_pe qui — vedi kpeCopy)
      step(pipes.kvApp, [row576, cache], Math.ceil(G.keyLen / 64), true),
      step(pipes.absorbKb, [wKB.qs, wKB.scales, qB, qCkv], gemvGrid(G.nHead * G.kvLora)),
      step(pipes.copyCkv, [qCkv, q576], Math.ceil((G.nHead * G.kvLora) / 64)),
      step(pipes.copyQRope, [qB, q576], Math.ceil((G.nHead * G.ropeDims) / 64)),
      step(pipes.attnPart, [q576, cache, attnPartials], [attnSMaxN, 1], true),
      step(pipes.attnReduce, [attnPartials, attnCkv], G.nHead, true),
      step(pipes.voutVb, [wVB.qs, wVB.scales, attnCkv, attnMla], gemvGrid(G.nHead * G.headLenMla)),
      step(pipes.gemvO, [wO.qs, wO.scales, attnMla, tmp], gemvGrid(G.dModel)),
      step(pipes.add, [x, tmp], Math.ceil(G.dModel / 64)),
    ];
    const kpeCopy = { srcOff: G.kvLora * 4, dstOff: G.kvLora * 4, bytes: G.ropeDims * 4 };
    // REF pesi per la costruzione LAZY dei passi batch (fix OOM it.35)
    const bwAttn = { attnNorm, qANorm, kvANorm, wQA, wQB, wKvA, wKB, wVB, wO, cache };

    if (l < G.denseLead) {
      const wGate = w("ffn_gate", "q4_0");
      const wUp = w("ffn_up", "q4_0");
      const wDown = w("ffn_down", "q4_1");
      layers.push({
        attn, kpeCopy, cache,
        dense: [
          step(pipes.rmsD, [x, ffnNorm, fnB], 1),
          step(pipes.gemvDenseGU, [wGate.qs, wGate.scales, fnB, gateD], gemvGrid(G.dFfnDense)),
          step(pipes.gemvDenseGU, [wUp.qs, wUp.scales, fnB, upD], gemvGrid(G.dFfnDense)),
          step(pipes.siluDense, [gateD, upD], Math.ceil(G.dFfnDense / 64)),
          step(pipes.gemvDenseDown, [wDown.qs, wDown.scales, gateD, tmp], gemvGrid(G.dModel)),
          step(pipes.add, [x, tmp], Math.ceil(G.dModel / 64)),
        ],
        bw: { ...bwAttn, ffnNorm, wGate, wUp, wDown },
      });
    } else {
      const routerW = track(upload(f32Of(src.nonExpert(nm("ffn_gate_inp")))));
      const bias = f32Of(src.nonExpert(`blk.${l}.exp_probs_b.bias`));
      const gateShexp = track(kquantBuf(src.nonExpert(nm("ffn_gate_shexp")), Q5_K_BLOCK_BYTES));
      const upShexp = track(kquantBuf(src.nonExpert(nm("ffn_up_shexp")), Q5_K_BLOCK_BYTES));
      const downShexp = track(kquantBuf(src.nonExpert(nm("ffn_down_shexp")), Q6_K_BLOCK_BYTES));
      moeLayerAbs.push(l);
      layers.push({
        attn, kpeCopy, cache,
        bw: { ...bwAttn, ffnNorm, routerW, gateShexp, upShexp, downShexp },
        moe: {
          bias,
          biasBuf: routerGpu ? track(upload(bias)) : null,
          cls: ExpertCache.classOf(l),
          moeIdx: nMoeSeen,
          routerBind: null, // riempito dopo selBuf/moeIdxUni (esistono solo dopo)
          selBase: nMoeSeen++ * G.nExpertUsed,
          preRouter: [
            step(pipes.rmsD, [x, ffnNorm, fnB], 1),
            step(pipes.router, [routerW, fnB, logitsB], G.nExpert),
          ],
          shexp: [
            step(pipes.pairSiluShexp, [gateShexp, upShexp, fnB, gateE], gemvGrid(G.dFfnExpert)),
            step(pipes.gemvShexpDown, [downShexp, gateE, moeOut], gemvGrid(G.dModel)),
          ],
          downLayout: downIsQ4_1(l) ? SLAB_DOWN_Q4_1 : SLAB_DOWN_Q4_0,
          addMoe: step(pipes.add, [x, moeOut], Math.ceil(G.dModel / 64)),
        },
      });
    }
  }

  // ---- tap del prefetch in-forward (C3c fase 4): ogni layer MoE aggancia il
  // router del MoE SUCCESSIVO (pesi già in VRAM) sullo stesso fnB. L'ultimo
  // MoE resta senza tap: dentro il token, mai oltre il confine (WP-0).
  if (prefetchOn) {
    const moeIdxs = layers.map((L, i) => (L.moe ? i : -1)).filter((i) => i >= 0);
    for (let j = 0; j + 1 < moeIdxs.length; j++) {
      const cur = layers[moeIdxs[j]].moe!;
      const nxt = layers[moeIdxs[j + 1]];
      cur.tapNext = {
        st: step(pipes.router, [nxt.bw.routerW!, fnB, logits2B!], G.nExpert),
        layer: moeIdxs[j + 1],
        bias: nxt.moe!.bias,
      };
    }
  }

  // ---- Sel + MoeIdx + i 4 bind group STATICI dell'arena ----
  // Reader costruito UNA volta: se la sorgente ha il file slab, la cache prende
  // il percorso senza pack CPU; altrimenti ripiega sui byte GGUF grezzi.
  const expertReader: ExpertReader = src.hasSlabs && src.expertSlab
    ? { raw: (ll, ee) => src.expert(ll, ee), slab: (ll, ee) => src.expertSlab!(ll, ee) }
    : (ll: number, ee: number) => src.expert(ll, ee);
  // ---- residenza TOTALE del modo gpu: tutto il parco caricato QUI, al load ----
  // Meccanismo scelto (il piu' semplice che garantisca zero evict): un `ensure`
  // per ogni expert dei layer MoE, alla costruzione, e un solo flush della
  // slotTable. Niente pin: il pin serve a proteggere dalla vittima, e qui la
  // vittima non esiste — la precondizione ha appena verificato nSlots >= parco,
  // quindi la lista `free` non si esaurisce e il ramo di eviction di `ensure` non
  // e' raggiungibile. E' una proprieta' della geometria, non della policy: si
  // asserta subito dopo, perche' e' l'ipotesi su cui poggia tutto il modo gpu
  // (dopo questo punto la slotTable NON cambia piu' per tutta la vita del
  // modello — se servisse un flush per token, il modo gpu sarebbe sbagliato).
  //
  // SCALA: questo ciclo e' SINCRONO e BLOCCANTE, ed e' dimensionato sul ktest
  // (64 expert, ~350 MB). Sul modello vero sarebbero 2 944 read OPFS in fila piu'
  // altrettante writeBuffer senza mai cedere il controllo: minuti di stallo e
  // pressione sulla staging interna della coda (firma R4 del design). La fase 4c
  // non deve riusare questo ciclo cosi' com'e': le serve il percorso file-slab
  // (`expertSlab`, che salta il pack CPU) con preload chunked/asincrono e la
  // pubblicazione della slotTable per blocchi. Qui il meccanismo e' il piu'
  // semplice che garantisca zero evict, non il piu' veloce.
  const preloadCheck = (): void => {
    const ev = cache.stats().evictions;
    if (ev !== 0) {
      throw new Error(
        `glmmodel: preload con ${ev} evizioni — la precondizione sugli slot ` +
        "e' passata ma la cache ha comunque evinto (riparto degli slot per classe incoerente col parco)");
    }
  };
  if (gpuSel) {
    // Residenza totale a scala ktest: il ciclo sincrono basta (v. nota SCALA).
    for (const l of moeLayerAbs) {
      for (let e = 0; e < G.nExpert; e++) cache.ensure(l, e, expertReader);
    }
    cache.flushSlotTable();
    preloadCheck();
  }
  // In optimistic il preload e' ASINCRONO E CHUNKED (la nota SCALA qui sopra
  // non era decorativa: 2 417 ensure sincroni = ~12 GiB di writeBuffer senza
  // mai drenare la staging interna della coda — firma R4, device perso su
  // modello vero alla prima run di bench it.4). Ogni 64 expert (~340 MB) si
  // attende onSubmittedWorkDone: la staging si svuota e la coda respira.
  // Il parco puo' eccedere gli slot: si riempie FINO ALLA CAPACITA' e mai
  // oltre — zero evict resta la precondizione (un evict scombinerebbe la LRU
  // prima del primo token); gli expert oltre capacita' restano MISS, e' il
  // regime del modo. Ordine (layer, expert) crescente: arbitrario finche' il
  // repair non porta una policy (landmine it.2: se P(dirty) esce dalla
  // proiezione, guardare prima qui). forward/prefillChunk attendono la
  // promise al primo uso.
  let preloadReady: Promise<void> | null = null;
  if (optSel) {
    preloadReady = (async () => {
      let n = 0;
      for (const l of moeLayerAbs) {
        const cls = ExpertCache.classOf(l);
        const cap = cache.arenaGeometry(cls).nSlots;
        for (let e = 0; e < G.nExpert; e++) {
          if (cache.stats().occupied[cls] >= cap) break;
          cache.ensure(l, e, expertReader);
          if (++n % 64 === 0) await device.queue.onSubmittedWorkDone();
        }
      }
      cache.flushSlotTable();
      await device.queue.onSubmittedWorkDone();
      preloadCheck();
    })();
  }
  const awaitPreload = async (): Promise<void> => {
    if (preloadReady) {
      await preloadReady;
      preloadReady = null;
    }
  };
  const nMoeLayer = nMoeSeen;
  const nSel = Math.max(1, nMoeLayer * G.nExpertUsed);
  if (MOE_IDX_STRIDE % device.limits.minUniformBufferOffsetAlignment !== 0) {
    throw new Error(
      `glmmodel arena: stride ${MOE_IDX_STRIDE} non multiplo di ` +
      `minUniformBufferOffsetAlignment ${device.limits.minUniformBufferOffsetAlignment}`);
  }
  // In modo shadow il buffer Sel RADDOPPIA (design §4): [0, nSel) e' la regione
  // di produzione — quella che i kernel expert leggono e che riempie la CPU —,
  // [nSel, 2·nSel) e' l'ombra dove scrive il resolve GPU. Due regioni dello
  // stesso buffer e non due buffer: al tail si copiano insieme, e lo stesso
  // kernel serve lo slice C cambiando solo la entry di uniform che lo indirizza.
  // In modo gpu la regione e' UNA SOLA: il resolve scrive la produzione, e non
  // c'e' niente da mettere in ombra perche' non c'e' piu' una seconda opinione.
  const nSelTot = shadow ? nSel * 2 : nSel;
  const selBuf = device.createBuffer({
    size: nSelTot * SEL_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  // Entry di MoeIdx: nSel di produzione (una per (layer MoE, k)) piu', in modo
  // shadow, una per layer MoE che indirizza il router — `selIdx` gia' spostato
  // nell'ombra. E' cosi' che il kernel di resolve resta identico fra shadow e
  // modo gpu: cambia la entry, non il WGSL. In modo gpu le entry in piu' NON
  // servono: il router scrive la produzione, e la entry (layer, k=0) — che
  // esiste gia' — porta esattamente `selIdx = m·nUsed` e il `tableBase` giusto.
  const nMoeIdx = nSel + (shadow ? nMoeLayer : 0);
  const moeIdxUni = device.createBuffer({
    size: nMoeIdx * MOE_IDX_STRIDE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  // MoeIdx e' STATICA: contenuto noto al load, una scrittura sola. Il dynamic
  // offset seleziona la entry, la entry dice quale Sel leggere. `tableBase` e'
  // la base del layer nella slotTable e la usa il resolve GPU (slice B).
  {
    const u = new Uint32Array(nMoeIdx * (MOE_IDX_STRIDE / 4));
    for (let m = 0; m < nMoeLayer; m++) {
      // base della slotTable: layer ASSOLUTO × nExpert, la chiave di `expertKey`
      const tableBase = moeLayerAbs[m] * G.nExpert;
      for (let k = 0; k < G.nExpertUsed; k++) {
        const selIdx = m * G.nExpertUsed + k;
        const w = selIdx * (MOE_IDX_STRIDE / 4);
        u[w] = selIdx; u[w + 1] = tableBase; u[w + 2] = m; u[w + 3] = 0;
      }
      if (shadow) {
        const w = (nSel + m) * (MOE_IDX_STRIDE / 4);
        u[w] = nSel + m * G.nExpertUsed; u[w + 1] = tableBase; u[w + 2] = m; u[w + 3] = 0;
      }
    }
    device.queue.writeBuffer(moeIdxUni, 0, u as unknown as BufferSource);
  }
  // Sel per-token: 4 entry (64 B) per layer MoE, scritte dopo gli `ensure`.
  const selScratch = new ArrayBuffer(G.nExpertUsed * SEL_BYTES);
  const selU32 = new Uint32Array(selScratch);
  const selF32 = new Float32Array(selScratch);
  const arenaBg = (cls: ExpertClass, xBuf: GPUBuffer, outBuf: GPUBuffer): GPUBindGroup => {
    const e = expert[cls];
    return device.createBindGroup({
      layout: e.bgl,
      entries: [
        ...cache.arenaBuffers(cls).map((b, j) => ({ binding: j, resource: { buffer: b } })),
        { binding: e.geo.nBuf, resource: { buffer: xBuf } },
        { binding: e.geo.nBuf + 1, resource: { buffer: outBuf } },
        { binding: e.geo.nBuf + 2, resource: { buffer: selBuf } },
        // `size` ESPLICITA: con hasDynamicOffset la validazione chiede
        // offset+dynamicOffset+size <= buffer.size, e senza `size` il binding
        // varrebbe l'intero buffer ⇒ qualunque offset dinamico > 0 sarebbe illegale.
        { binding: e.geo.nBuf + 3, resource: { buffer: moeIdxUni, offset: 0, size: MOE_IDX_BYTES } },
      ],
    });
  };
  const expBg: Record<ExpertClass, { gu: GPUBindGroup; down: GPUBindGroup }> = {
    q4_0: { gu: arenaBg("q4_0", fnB, gateE), down: arenaBg("q4_0", gateE, moeOut) },
    q4_1: { gu: arenaBg("q4_1", fnB, gateE), down: arenaBg("q4_1", gateE, moeOut) },
  };

  // ---- router GPU + resolve (slice B in ombra, slice C al comando) ----
  // Un dispatch per layer MoE, in coda al GEMV del router e nello STESSO pass:
  // legge i logits appena scritti, sceglie i top-4 come la CPU e risolve gli
  // slot dalla slotTable. In SHADOW gira PRIMA degli `ensure` del suo layer —
  // e' l'ordine del path attuale, non una scelta: cio' che il layer sta per
  // caricare non e' ancora nella tabella, quindi un expert appena entrato si
  // risolve MISS. Non e' un difetto della risoluzione: e' la residenza parziale
  // vista da GPU, ed e' esattamente la ragione per cui il modo gpu esige la
  // totale — dove la tabella e' completa dal load e ogni risoluzione e' un hit.
  const routerIds = routerGpu ? storage(G.nExpertUsed * 4) : null;
  const routerWts = routerGpu ? storage(G.nExpertUsed * 4) : null;
  // dirtyB (solo optimistic, spec §3b): [0] = primo layer MoE sporco
  // (atomicMin, sentinel 0xffffffff), [1] = conteggio miss. 16 B per
  // allineamento; ri-inizializzato dalla CPU a ogni token (writeBuffer in
  // coda PRIMA del submit — l'ordine di coda garantisce che il reset preceda
  // i resolve del token). La staging viaggia nel tail insieme a Sel/hidden.
  const dirtyB = optSel
    ? device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC })
    : null;
  const dirtyStaging = optSel
    ? device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
    : null;
  const dirtyInit = new Uint32Array([0xffffffff, 0, 0, 0]);
  // hiddenCkpt (spec §3c): nLayer righe da dModel f32 — l'hidden di INGRESSO
  // di ogni layer, copiato nell'encoder del token. 47×2048×4 = 376 KiB sul
  // modello vero. COPY_SRC per il readback dell'harness (readHiddenCkpt) e,
  // in fase 3, per la copy di rientro del replay.
  // In optimistic il checkpoint NON e' opzionale: e' l'input del replay.
  const hiddenCkpt = opts.checkpointHidden === true || optSel
    ? device.createBuffer({ size: nLayer * G.dModel * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC })
    : null;
  let routerGpuPipe: GPUComputePipeline | null = null;
  if (routerGpu) {
    const bglRouter = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        {
          binding: 6, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: MOE_IDX_BYTES },
        },
        ...(optSel ? [{
          binding: 7, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "storage" as const },
        }] : []),
      ],
    });
    routerGpuPipe = mkPipe(routerTopKWgsl({
      nExpert: G.nExpert, nUsed: G.nExpertUsed,
      weightsScale: G.weightsScale, clampMin: WEIGHTS_SUM_CLAMP_MIN,
      resolve: { nExpert: G.nExpert, nUsed: G.nExpertUsed, dirty: optSel },
    }), bglRouter);
    const slotTable = cache.slotTableBuffer();
    for (const L of layers) {
      if (!L.moe) continue;
      L.moe.routerBind = device.createBindGroup({
        layout: bglRouter,
        entries: [
          { binding: 0, resource: { buffer: logitsB } },
          { binding: 1, resource: { buffer: L.moe.biasBuf! } },
          { binding: 2, resource: { buffer: routerIds! } },
          { binding: 3, resource: { buffer: routerWts! } },
          { binding: 4, resource: { buffer: selBuf } },
          { binding: 5, resource: { buffer: slotTable } },
          { binding: 6, resource: { buffer: moeIdxUni, offset: 0, size: MOE_IDX_BYTES } },
          ...(optSel ? [{ binding: 7, resource: { buffer: dirtyB! } }] : []),
        ],
      });
    }
  }
  // Copia di TUTTE le regioni di Sel nel tail: in shadow sono due (5 888 B) —
  // l'ombra per il confronto col router GPU, la produzione per rileggere cio' che
  // i kernel expert hanno visto —, in modo gpu e' una sola (2 944 B) ed e' l'unica
  // fonte di `routing[]`: nessuno sulla CPU ha mai visto la selezione di questo
  // token. La mapAsync parte insieme a quella dell'hidden, quindi non aggiunge
  // round-trip: e' il tail che c'era gia'.
  const selStaging = routerGpu
    ? device.createBuffer({ size: nSelTot * SEL_BYTES, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
    : null;

  // (con l'it.13 anche lo shexp Q5_K/Q6_K e' passato alla famiglia fusa: nel
  // modello non resta nessun dispatch silu separato — solo il denso di blk.0,
  // che usa `pipes.siluDense`)

  // ---- output head (fase 6): rms(output_norm) → gemvQ6K fast [2048→vocab] ----
  const vocab = opts.vocab ?? G.vocab;
  let headSteps: Step[] = [];
  let logitsVocab: GPUBuffer | null = null;
  let vocabStaging: GPUBuffer | null = null;
  if (opts.head) {
    const outNorm = track(upload(f32Of(src.nonExpert("output_norm.weight"))));
    const outW = track(kquantBuf(src.nonExpert("output.weight"), Q6_K_BLOCK_BYTES));
    const headPipe = mkPipe(gemvQ6KFastWgsl({ K: G.dModel, N: vocab }));
    logitsVocab = storage(vocab * 4);
    weightBufs.push(logitsVocab);
    vocabStaging = device.createBuffer({ size: vocab * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    weightBufs.push(vocabStaging);
    headSteps = [
      step(pipes.rmsD, [x, outNorm, fnB], 1),
      step(headPipe, [outW, fnB, logitsVocab], gemvGrid(vocab)),
    ];
  }

  const runSteps = (pass: GPUComputePassEncoder, steps: Step[]): void => {
    for (const s of steps) {
      pass.setPipeline(s.pipe);
      pass.setBindGroup(0, s.bind);
      pass.dispatchWorkgroups(s.wgs[0], s.wgs[1]);
    }
    T.dispatches += steps.length;
  };

  // Runner dei passi BATCH del prefill (fase 5): le righe del chunk vanno
  // nella dimensione che la variante del kernel usa — "y" per gli elementwise
  // a gid (rope/kvAppend/stridedCopy), "z" per i gemv/rms/MLA a wid.
  const runStepsB = (pass: GPUComputePassEncoder, steps: BStep[], rows: number): void => {
    for (const s of steps) {
      pass.setPipeline(s.pipe);
      pass.setBindGroup(0, s.bind);
      if (s.rowsDim === "x") pass.dispatchWorkgroups(rows);
      else if (s.rowsDim === "y") pass.dispatchWorkgroups(s.wgs[0], rows);
      else pass.dispatchWorkgroups(s.wgs[0], s.wgs[1], rows);
    }
    T.dispatches += steps.length;
  };

  // Valore DERIVATO dal piano: attn 17/layer (16 fino alla fase 4b; l'attention
  // split ne aggiunge uno, part+reduce al posto del kernel monolitico) + denso
  // 6 (solo blk.0) + per layer
  // MoE 2 preRouter + 2 shexp (it.13: gate+up+silu Q5_K fusi, era 4) + 4 catene
  // expert da 2 (fase 4b: gate+up+silu fusi) + 1 add = 13; col router+resolve su
  // GPU (shadow e gpu, slice B e C) 1 dispatch in piu' per layer MoE ⇒ 14.
  // Lo slice C NON cambia questo numero: sposta i confini dei SUBMIT (da 1 per
  // layer MoE + 1 a 1 per token) e toglie i readback, non i dispatch — il
  // routerTopK resta, e nessun dispatch sparisce ne' compare.
  // NON include la testa (rms + lm_head = 2), che con readLogits gira a ogni
  // token: e' la ragione per cui questo numero e' sempre stato 2 sotto il vero.
  // Il conteggio reale e' `telemetry().dispatches` (contatore in runSteps e
  // nelle catene expert).
  const nMoe = layers.filter((l) => l.moe).length;
  const nDense = layers.filter((l) => l.dense).length;
  const dispatchesPerTokenPlanned = 17 * nLayer + 6 * nDense + (routerGpu ? 14 : 13) * nMoe;

  const mapLogits = async (): Promise<Float32Array> => {
    await logitsStaging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(logitsStaging.getMappedRange().slice(0));
    logitsStaging.unmap();
    return out;
  };

  // ---- prefill batched: stato LAZY (fix OOM it.35 — si paga al primo uso) ----
  interface PrefillState {
    xM: GPUBuffer; fnBM: GPUBuffer; tmpM: GPUBuffer; sM: GPUBuffer;
    hSlotsM: GPUBuffer; ySlotsM: GPUBuffer; wBufM: GPUBuffer;
    logitsBM: GPUBuffer; logitsMStaging: GPUBuffer; rowPosB: GPUBuffer; gatherM: GPUBuffer;
    logits2BM: GPUBuffer | null; // C3c fase 7: logits del tap batched
    bgAddM: GPUBindGroup; bgSiluDenseM: GPUBindGroup; bgCombineM: GPUBindGroup;
    perLayer: Array<{
      attnB: BStep[]; denseB?: BStep[]; preRouterB?: BStep[]; shexpB?: BStep[];
      /** C3c fase 7: tap batched del prefetch — router del MoE successivo su
       *  fnBM (stesso pattern del decode, fase 4). Solo con prefetch attivo. */
      tapNextB?: { st: BStep[]; layer: number; bias: Float32Array };
    }>;
  }
  let PF: PrefillState | null = null;
  const initPrefill = (): PrefillState => {
    if (PF) return PF;
    const xM = storage(MPF * G.dModel * 4);
    const hnM = storage(MPF * G.dModel * 4);
    const fnBM = storage(MPF * G.dModel * 4);
    const qaBM = storage(MPF * G.qLora * 4);
    const qanBM = storage(MPF * G.qLora * 4);
    const qBM = storage(MPF * G.nHead * HL * 4);
    const kvBM = storage(MPF * G.keyLen * 4);
    const row576M = storage(MPF * G.keyLen * 4);
    const qCkvM = storage(MPF * G.nHead * G.kvLora * 4);
    const q576M = storage(MPF * G.nHead * G.keyLen * 4);
    const attnPartialsM = storage(MPF * mlaPartialsLen(G.nHead, G.kvLora, ctxMax) * 4);
    const attnCkvM = storage(MPF * G.nHead * G.kvLora * 4);
    const attnMlaM = storage(MPF * G.nHead * G.headLenMla * 4);
    const tmpM = storage(MPF * G.dModel * 4);
    const gateDM = storage(MPF * G.dFfnDense * 4);
    const upDM = storage(MPF * G.dFfnDense * 4);
    const gateEM = storage(MPF * G.dFfnExpert * 4);
    const hSlotsM = storage(MPF * G.nExpertUsed * G.dFfnExpert * 4);
    const ySlotsM = storage(MPF * G.nExpertUsed * G.dModel * 4);
    const sM = storage(MPF * G.dModel * 4);
    const wBufM = storage(MPF * G.nExpertUsed * 4);
    const logitsBM = storage(MPF * G.nExpert * 4);
    // C3c fase 7: col prefetch lo staging M raddoppia — [0, MPF·64·4) logits
    // del router corrente, [MPF·64·4, …) logits del TAP (router L+1 su fnBM).
    // Offset FISSO a MPF anche coi chunk parziali: una convenzione sola.
    const logitsMStaging = device.createBuffer({ size: MPF * G.nExpert * 4 * (prefetchOn ? 2 : 1), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const logits2BM = prefetchOn ? storage(MPF * G.nExpert * 4) : null;
    const rowPosB = storage(MPF * 4);
    const gatherM = storage(G.nExpert * GATHER_REGION);
    const perLayer: PrefillState["perLayer"] = layers.map((L) => {
      const w = L.bw;
      const attnB: BStep[] = [
        stepB(pipesB.rmsD, [xM, w.attnNorm, hnM], 1, "x"),
        stepB(pipesB.gemvQA, [w.wQA.qs, w.wQA.scales, hnM, qaBM], gemvGrid(G.qLora), "z"),
        stepB(pipesB.rmsQA, [qaBM, w.qANorm, qanBM], 1, "x"),
        stepB(pipesB.gemvQB, [w.wQB.qs, w.wQB.scales, qanBM, qBM], gemvGrid(G.nHead * HL), "z"),
        stepB(pipesB.ropeQ, [qBM, rowPosB], Math.ceil((G.nHead * G.ropeDims / 2) / 64), "y"),
        stepB(pipesB.gemvKvA, [w.wKvA.qs, w.wKvA.scales, hnM, kvBM], gemvGrid(G.keyLen), "z"),
        stepB(pipesB.ropeKPe, [kvBM, rowPosB], Math.ceil((G.ropeDims / 2) / 64), "y"),
        stepB(pipesB.rmsKvA, [kvBM, w.kvANorm, row576M], 1, "x"),
        stepB(pipesB.kpeCopy, [kvBM, row576M], Math.ceil(G.ropeDims / 64), "y"),
        stepB(pipesB.kvApp, [row576M, w.cache, rowPosB], Math.ceil(G.keyLen / 64), "y"),
        stepB(pipesB.absorbKb, [w.wKB.qs, w.wKB.scales, qBM, qCkvM], gemvGrid(G.nHead * G.kvLora), "z"),
        stepB(pipesB.copyCkv, [qCkvM, q576M], Math.ceil((G.nHead * G.kvLora) / 64), "y"),
        stepB(pipesB.copyQRope, [qBM, q576M], Math.ceil((G.nHead * G.ropeDims) / 64), "y"),
        stepB(pipesB.attnPart, [q576M, w.cache, attnPartialsM, rowPosB], [attnSMaxN, 1], "z"),
        stepB(pipesB.attnReduce, [attnPartialsM, attnCkvM, rowPosB], G.nHead, "z"),
        stepB(pipesB.voutVb, [w.wVB.qs, w.wVB.scales, attnCkvM, attnMlaM], gemvGrid(G.nHead * G.headLenMla), "z"),
        stepB(pipesB.gemvO, [w.wO.qs, w.wO.scales, attnMlaM, tmpM], gemvGrid(G.dModel), "z"),
      ];
      if (L.dense) {
        return { attnB, denseB: [
          stepB(pipesB.rmsD, [xM, w.ffnNorm!, fnBM], 1, "x"),
          stepB(pipesB.gemvDenseGU, [w.wGate!.qs, w.wGate!.scales, fnBM, gateDM], gemvGrid(G.dFfnDense), "z"),
          stepB(pipesB.gemvDenseGU, [w.wUp!.qs, w.wUp!.scales, fnBM, upDM], gemvGrid(G.dFfnDense), "z"),
          stepB(pipesB.gemvDenseDown, [w.wDown!.qs, w.wDown!.scales, gateDM, tmpM], gemvGrid(G.dModel), "z"),
        ] };
      }
      return { attnB, preRouterB: [
        stepB(pipesB.rmsD, [xM, w.ffnNorm!, fnBM], 1, "x"),
        stepB(pipesB.router, [w.routerW!, fnBM, logitsBM], G.nExpert, "z"),
      ], shexpB: [
        stepB(pipesB.pairSiluShexp, [w.gateShexp!, w.upShexp!, fnBM, gateEM], gemvGrid(G.dFfnExpert), "z"),
        stepB(pipesB.gemvShexpDown, [w.downShexp!, gateEM, sM], gemvGrid(G.dModel), "z"),
      ] };
    });
    // C3c fase 7: aggancio del tap batched (come il decode, fase 4 — il
    // router del MoE SUCCESSIVO sullo stesso fnBM; l'ultimo MoE senza tap)
    if (prefetchOn) {
      const moeIdxs = layers.map((L, i) => (L.moe ? i : -1)).filter((i) => i >= 0);
      for (let j = 0; j + 1 < moeIdxs.length; j++) {
        const nxt = layers[moeIdxs[j + 1]];
        perLayer[moeIdxs[j]].tapNextB = {
          st: [stepB(pipesB.router, [nxt.bw.routerW!, fnBM, logits2BM!], G.nExpert, "z")],
          layer: moeIdxs[j + 1],
          bias: nxt.moe!.bias,
        };
      }
    }
    PF = {
      xM, fnBM, tmpM, sM, hSlotsM, ySlotsM, wBufM, logitsBM, logitsMStaging, rowPosB, gatherM, logits2BM,
      bgAddM: bg(pipesB.addM, [xM, tmpM]),
      bgSiluDenseM: bg(pipesB.siluDenseM, [gateDM, upDM]),
      bgCombineM: bg(pipesB.combine, [xM, sM, ySlotsM, wBufM]),
      perLayer,
    };
    return PF;
  };

  return {
    dispatchesPerTokenPlanned,
    cacheStats: () => cache.stats(),

    async prefillChunk(xRows: Float32Array, posStart: number, readLogitsLast = false) {
      // Piano: docs/engine/glmprefill-wiring-plan-2026-08-06.md. Telemetria:
      // contatori sempre-on (dispatches/submits/routerSyncs); il dettaglio ms
      // liv.1 del prefill si aggiunge quando servira' all'attribuzione.
      await awaitPreload();
      const m = xRows.length / G.dModel;
      if (!Number.isInteger(m) || m < 1 || m > MPF) throw new Error(`prefillChunk: righe ${m} (ammesse 1..${MPF})`);
      if (posStart + m > ctxMax) throw new Error("glmmodel: contesto pieno (chunk)");
      if (readLogitsLast && !opts.head) throw new Error("glmmodel: head non abilitata");
      const pf = initPrefill(); // LAZY: primo uso alloca i buffer M× (fix it.35)
      device.queue.writeBuffer(pf.xM, 0, xRows as unknown as BufferSource);
      device.queue.writeBuffer(pf.rowPosB, 0, Uint32Array.from({ length: m }, (_, i) => posStart + i) as unknown as BufferSource);
      device.pushErrorScope("validation");
      device.pushErrorScope("out-of-memory");
      const routing: GlmRouting[][] = Array.from({ length: m }, () => []);
      // prefetch del prefill (fase 7): vive DENTRO il chunk — l'ultimo MoE non
      // ha tap, quindi niente attraversa il confine di chunk né di token
      let pendingPrefetchP: { layer: number; ids: number[] } | null = null;
      let enc = device.createCommandEncoder();
      let pass: GPUComputePassEncoder | null = null;
      const ensureP = (): GPUComputePassEncoder => { if (!pass) pass = enc.beginComputePass(); return pass; };
      const endP = () => { if (pass) { pass.end(); pass = null; } };
      // add elementwise su m righe (il kernel e' dimensionato a M·D; la
      // griglia tagliata copre esattamente m·D — D multiplo di 64)
      const inline = (pipe: GPUComputePipeline, bind: GPUBindGroup, elems: number) => {
        const p2 = ensureP();
        p2.setPipeline(pipe); p2.setBindGroup(0, bind);
        p2.dispatchWorkgroups(Math.ceil(elems / 64));
        T.dispatches++;
      };

      for (const [l, L] of layers.entries()) {
        runStepsB(ensureP(), pf.perLayer[l].attnB, m);
        inline(pipesB.addM, pf.bgAddM, m * G.dModel); // xM += tmpM (residuo attention)
        if (pf.perLayer[l].denseB) {
          runStepsB(ensureP(), pf.perLayer[l].denseB!.slice(0, 3), m);
          inline(pipesB.siluDenseM, pf.bgSiluDenseM, m * G.dFfnDense);
          runStepsB(ensureP(), pf.perLayer[l].denseB!.slice(3), m); // down → tmpM
          inline(pipesB.addM, pf.bgAddM, m * G.dModel);
          continue;
        }
        const mo = L.moe!;
        runStepsB(ensureP(), pf.perLayer[l].preRouterB!, m);
        // C3c fase 7: tap batched — router L+1 su fnBM, stesso pass del router
        const tapB = prefetchOn ? pf.perLayer[l].tapNextB : undefined;
        if (tapB) runStepsB(ensureP(), tapB.st, m);
        endP();
        enc.copyBufferToBuffer(pf.logitsBM, 0, pf.logitsMStaging, 0, m * G.nExpert * 4);
        if (tapB) enc.copyBufferToBuffer(pf.logits2BM!, 0, pf.logitsMStaging, MPF * G.nExpert * 4, m * G.nExpert * 4);
        device.queue.submit([enc.finish()]);
        if (telemOn) { T.submits++; T.routerSyncs++; }
        // ---- prefetch in-forward del prefill (C3c fase 7): consuma l'UNIONE
        // predetta al layer MoE precedente ADESSO, mentre la GPU esegue il
        // submit appena accodato — è l'overlap che il TTFT a freddo paga per
        // stare nel budget 1.25× (ruling item 1a). Ordine di coda: writeBuffer
        // dopo il submit = eseguita dopo i suoi dispatch, niente corruzione.
        if (prefetchOn && pendingPrefetchP) {
          const pfp = pendingPrefetchP; pendingPrefetchP = null;
          const tPf = nowT();
          const pinnedP = new Set<number>();
          for (const e of pfp.ids) pinnedP.add(expertKey(pfp.layer, e));
          for (const e of pfp.ids) {
            const r = cache.ensure(pfp.layer, e, expertReader, pinnedP);
            if (telemOn) { T.prefetchPreds++; if (r.hit) T.prefetchResident++; else T.prefetchFetches++; }
          }
          cache.flushSlotTable();
          if (telemOn) T.prefetchMs += nowT() - tPf;
        }
        // ---- IL sync del chunk: M×64 logit in una mapAsync (46/chunk ≈ 46/m per token) ----
        await pf.logitsMStaging.mapAsync(GPUMapMode.READ);
        const lg = new Float32Array(pf.logitsMStaging.getMappedRange().slice(0));
        pf.logitsMStaging.unmap();
        const sels = [];
        for (let r = 0; r < m; r++) {
          const sel = routerSelect(lg.subarray(r * G.nExpert, (r + 1) * G.nExpert), mo.bias, ROUTER_GLM47);
          routing[r].push({ layer: l, experts: sel.experts, weights: sel.weights });
          sels.push(sel);
        }
        // predizioni per il PROSSIMO layer MoE: unione dei top-4 sulle m righe
        if (tapB) {
          const uni = new Set<number>();
          for (let r = 0; r < m; r++) {
            const p = routerSelect(lg.subarray(MPF * G.nExpert + r * G.nExpert, MPF * G.nExpert + (r + 1) * G.nExpert), tapB.bias, ROUTER_GLM47);
            for (const e of p.experts) uni.add(e);
          }
          pendingPrefetchP = { layer: tapB.layer, ids: [...uni] };
        }
        const plan = planMoeChunk(sels, MPF);
        // ensure dell'UNIONE, tutte le chiavi pinnate (nessuna vittima interna al chunk)
        const pinned = new Set<number>();
        for (const b of plan.experts) pinned.add(expertKey(l, b.expert));
        const slotOf = new Map<number, SlotRef>();
        for (const b of plan.experts) slotOf.set(b.expert, cache.ensure(l, b.expert, expertReader, pinned).slot);
        cache.flushSlotTable();
        device.queue.writeBuffer(pf.wBufM, 0, plan.weights as unknown as BufferSource);
        plan.experts.forEach((b, i) => {
          device.queue.writeBuffer(pf.gatherM, i * GATHER_REGION,
            Uint32Array.from(b.rows, (row, j) => row | (b.slots[j] << 16)) as unknown as BufferSource);
        });
        enc = device.createCommandEncoder();
        const p2 = ensureP();
        runStepsB(p2, pf.perLayer[l].shexpB!, m); // sM = shexp per riga (la base della combine)
        // catena expert sull'unione: bind group a SOTTO-RANGE dello slot
        // (la CPU conosce lo slot: il sync col router c'e' comunque)
        const layout = mo.downLayout;
        const dp = mo.cls === "q4_0" ? pipesB.downSlots40 : pipesB.downSlots41;
        const [gx, gy] = gemvGrid(G.dFfnExpert / 4);
        const [dx, dy] = gemvGrid(G.dModel / 4);
        for (const [i, b] of plan.experts.entries()) {
          const slot = slotOf.get(b.expert)!;
          const sub = (off: number, size: number): BindRange => ({ buffer: slot.buffer, offset: slot.offset + off, size });
          const gsub: BindRange = { buffer: pf.gatherM, offset: i * GATHER_REGION, size: Math.max(4, b.rows.length * 4) };
          p2.setPipeline(pipesB.pairGather);
          p2.setBindGroup(0, bg(pipesB.pairGather, [
            sub(layout.gateQs, layout.qsBytes), sub(layout.gateScales, layout.gateScalesBytes),
            sub(layout.upQs, layout.qsBytes), sub(layout.upScales, layout.gateScalesBytes),
            pf.fnBM, gsub, pf.hSlotsM,
          ]));
          p2.dispatchWorkgroups(gx, gy, b.rows.length);
          p2.setPipeline(dp);
          p2.setBindGroup(0, bg(dp, [
            sub(layout.downQs, layout.qsBytes), sub(layout.downScales, layout.downScalesBytes),
            pf.hSlotsM, gsub, pf.ySlotsM,
          ]));
          p2.dispatchWorkgroups(dx, dy, b.rows.length);
          T.dispatches += 2;
        }
        // combine k-order: xM += sM + Σ w·y — la catena esatta del decode
        p2.setPipeline(pipesB.combine);
        p2.setBindGroup(0, pf.bgCombineM);
        p2.dispatchWorkgroups(Math.ceil(G.dModel / 64), m);
        T.dispatches++;
      }
      // head/hidden sull'ULTIMA riga: copia in `x` e riusa i passi del decode
      endP();
      enc.copyBufferToBuffer(pf.xM, (m - 1) * G.dModel * 4, x, 0, G.dModel * 4);
      if (readLogitsLast) {
        const p3 = enc.beginComputePass();
        runSteps(p3, headSteps);
        p3.end();
      }
      enc.copyBufferToBuffer(x, 0, hiddenStaging, 0, G.dModel * 4);
      if (readLogitsLast) enc.copyBufferToBuffer(logitsVocab!, 0, vocabStaging!, 0, vocab * 4);
      device.queue.submit([enc.finish()]);
      if (telemOn) T.submits++;
      const errOom = await device.popErrorScope();
      const errVal = await device.popErrorScope();
      if (errOom || errVal) throw new Error(`prefillChunk error scope: ${(errOom ?? errVal)!.message.slice(0, 300)}`);
      await hiddenStaging.mapAsync(GPUMapMode.READ);
      const hidden = new Float32Array(hiddenStaging.getMappedRange().slice(0));
      hiddenStaging.unmap();
      let logits: Float32Array | undefined;
      if (readLogitsLast) {
        await vocabStaging!.mapAsync(GPUMapMode.READ);
        logits = new Float32Array(vocabStaging!.getMappedRange().slice(0));
        vocabStaging!.unmap();
      }
      if (telemOn) T.forwards += m;
      return { hidden, logits, routing };
    },

    async forward(xIn: Float32Array, pos: number, readLogits = false) {
      if (pos >= ctxMax) throw new Error("glmmodel: contesto pieno");
      if (readLogits && !opts.head) throw new Error("glmmodel: head non abilitata");
      await awaitPreload();
      device.queue.writeBuffer(x, 0, xIn as unknown as BufferSource);
      device.queue.writeBuffer(P, 0, new Uint32Array([pos, pos, 0, 0]));
      const routing: GlmRouting[] = [];
      // ---- UN GIRO: encode (da startLayer) + submit + readback di coda ----
      // C3b spec §4: il replay e' un SECONDO giro dello stesso token, dopo il
      // repair, con rientro da hiddenCkpt[startLayer] (I4). startLayer = 0 e'
      // il giro normale, e l'unico che i modi non-optimistic conoscano.
      const runPass = async (startLayer: number) => {
        // reset di dirtyB per GIRO: in coda PRIMA del submit, quindi eseguito
        // prima di ogni resolve del giro (ordine di coda). Sentinel 0xffffffff
        // in [0] perche' il kernel fa atomicMin sul layer MoE.
        if (dirtyB) device.queue.writeBuffer(dirtyB, 0, dirtyInit);
        device.pushErrorScope("validation");
        device.pushErrorScope("out-of-memory");

        let enc = device.createCommandEncoder();
        // Rientro del replay (I4): x = hiddenCkpt[startLayer]. La copy precede
        // ogni dispatch del giro nel command buffer, e il checkpoint e' l'input
        // BIT-IDENTICO che il layer aveva visto nel giro ottimistico.
        if (startLayer > 0) {
          enc.copyBufferToBuffer(hiddenCkpt!, startLayer * G.dModel * 4, x, 0, G.dModel * 4);
        }
        let pass: GPUComputePassEncoder | null = null;
        let passIdx = 0;         // indice nel query set, per token
        let tSeg = nowT();       // inizio del segmento di encode CPU corrente
        const passCats: string[] = []; // etichetta per indice di pass (solo byCat)
        let curCat = "";
        // `cat` è l'etichetta della categoria di kernel che sta per essere
        // encodata. Fuori da wantByCat è inerte: i pass restano esattamente
        // quelli di prima, così il bench headline non cambia forma.
        const ensurePass = (cat: string) => {
          if (pass && wantByCat && cat !== curCat) endPass();
          if (pass) return pass;
          curCat = cat;
          if (wantGpuTs) {
            if (passIdx < TSQ_PASSES) {
              pass = enc.beginComputePass({
                timestampWrites: { querySet: querySet!, beginningOfPassWriteIndex: passIdx * 2, endOfPassWriteIndex: passIdx * 2 + 1 },
              });
              passCats[passIdx] = cat;
              passIdx++;
              return pass;
            }
            T.gpuPassOverflow++;
          }
          pass = enc.beginComputePass();
          return pass;
        };
        const endPass = () => { if (pass) { pass.end(); pass = null; } };

        // Stato del prefetch in-forward: vive DENTRO questo forward — una
        // predizione non puo' strutturalmente attraversare il confine di token
        // (l'esclusione del contratto C3c, WP-0).
        let pendingPrefetch: { layer: number; ids: number[] } | null = null;
        let lastPredRecall: { layer: number; p4: Set<number>; p8: Set<number> } | null = null;

        for (const [l, L] of layers.entries()) {
          if (l < startLayer) continue; // replay: i layer puliti non si rigiocano
          // checkpoint dell'hidden di INGRESSO del layer (spec §3c): la copy va
          // fuori dal pass — endPass garantisce anche l'ordine (i dispatch che
          // hanno prodotto questo x vengono prima della copy nel command buffer).
          if (hiddenCkpt) {
            endPass();
            enc.copyBufferToBuffer(x, 0, hiddenCkpt, l * G.dModel * 4, G.dModel * 4);
          }
          // attention (con la copy k_pe dopo rmsKvA: indice 8 = kvApp)
          ensurePass("attn");
          runSteps(pass!, L.attn.slice(0, 8));
          endPass();
          enc.copyBufferToBuffer(kvB, L.kpeCopy.srcOff, row576, L.kpeCopy.dstOff, L.kpeCopy.bytes);
          ensurePass("attn");
          runSteps(pass!, L.attn.slice(8));
          if (L.dense) {
            ensurePass("dense");
            runSteps(pass!, L.dense);
            continue;
          }
          const m = L.moe!;
          ensurePass("router");
          runSteps(pass!, m.preRouter);
          if (routerGpu) {
            // stessa categoria di telemetria del GEMV che lo precede (`router`) e
            // stesso pass: e' selezione, non un blocco nuovo.
            // L'unica differenza fra shadow e gpu e' la ENTRY di MoeIdx che il
            // dynamic offset seleziona — in shadow quella dedicata, che punta
            // all'ombra; in gpu la entry (layer, k=0), che punta alla produzione.
            // Il WGSL e il bind group sono gli stessi.
            pass!.setPipeline(routerGpuPipe!);
            pass!.setBindGroup(0, m.routerBind!, [(oneSubmit ? m.selBase : nSel + m.moeIdx) * MOE_IDX_STRIDE]);
            pass!.dispatchWorkgroups(1);
            T.dispatches++;
          }
          if (!oneSubmit) {
            // tap del prefetch: stesso pass del router (categoria "router")
            if (prefetchOn && m.tapNext) runSteps(pass!, [m.tapNext.st]);
            endPass();
            enc.copyBufferToBuffer(logitsB, 0, logitsStaging, 0, G.nExpert * 4);
            if (prefetchOn && m.tapNext) enc.copyBufferToBuffer(logits2B!, 0, logitsStaging, G.nExpert * 4, G.nExpert * 4);
            device.queue.submit([enc.finish()]);
            // ---- sync GPU→CPU: selezione ----
            if (telemOn) { T.encodeCpuMs += performance.now() - tSeg; T.submits++; T.routerSyncs++; }
            // ---- prefetch in-forward (C3c fase 4): consuma le predizioni del
            // layer MoE precedente ADESSO, mentre la GPU esegue il submit appena
            // accodato — il fetch OPFS+upload cade nel tempo d'attesa del router,
            // non in coda a esso. writeBuffer dopo il submit = eseguito dopo i
            // suoi dispatch (ordine di coda): nessuno slab in volo si corrompe.
            if (prefetchOn && pendingPrefetch) {
              const pf = pendingPrefetch; pendingPrefetch = null;
              const tPf = nowT();
              const pinnedP = new Set<number>();
              for (const e of pf.ids) pinnedP.add(expertKey(pf.layer, e));
              for (const e of pf.ids) {
                const r = cache.ensure(pf.layer, e, expertReader, pinnedP);
                if (telemOn) { T.prefetchPreds++; if (r.hit) T.prefetchResident++; else T.prefetchFetches++; }
              }
              cache.flushSlotTable();
              if (telemOn) T.prefetchMs += nowT() - tPf;
            }
            const tWait = nowT();
            const mapped = await mapLogits();
            const logits = prefetchOn ? mapped.subarray(0, G.nExpert) : mapped;
            if (telemOn) tSeg = performance.now();
            if (telemOn) T.routerWaitMs += tSeg - tWait;
            const sel = routerSelect(logits, m.bias, ROUTER_GLM47);
            // policy tier (C3c fase 5): registra la selezione (no-op in lru)
            cache.noteSelection(l, sel.experts);
            // ---- recall in-engine + prossima predizione (C3c fase 4) ----
            if (prefetchOn) {
              if (telemOn && lastPredRecall && lastPredRecall.layer === l) {
                T.recallPreds++;
                for (const e of sel.experts) {
                  if (lastPredRecall.p4.has(e)) T.recallHits4++;
                  if (lastPredRecall.p8.has(e)) T.recallHits8++;
                }
              }
              lastPredRecall = null;
              if (m.tapNext) {
                const lg2 = mapped.subarray(G.nExpert, 2 * G.nExpert);
                const pred = routerSelect(lg2, m.tapNext.bias, ROUTER_GLM47); // K=4 (spec §3)
                pendingPrefetch = { layer: m.tapNext.layer, ids: Array.from(pred.experts) };
                if (telemOn) {
                  const p8 = routerSelect(lg2, m.tapNext.bias, { ...ROUTER_GLM47, nUsed: 8 }).experts;
                  lastPredRecall = { layer: m.tapNext.layer, p4: new Set(pred.experts), p8: new Set(p8) };
                }
              }
            }
            routing.push({ layer: l, experts: sel.experts, weights: sel.weights });
            const pinned = new Set<number>();
            for (const e of sel.experts) pinned.add(expertKey(l, e));
            const slots: SlotRef[] = [];
            const tEns = nowT();
            for (const e of sel.experts) slots.push(cache.ensure(l, e, expertReader, pinned).slot);
            // slotTable: un flush per layer, DOPO gli ensure (le writeBuffer degli
            // slab sono gia' in coda) e prima di qualunque dispatch che la legga —
            // il primo sara' il router del layer MoE successivo, che sta in un
            // encoder ancora da creare. Da spenta (`select:"cpu"`) e' un no-op.
            // Il suo costo cade in `ensureMs`: e' manutenzione della residenza.
            cache.flushSlotTable();
            if (telemOn) {
              const tE = performance.now();
              T.ensureMs += tE - tEns;
              tSeg += tE - tEns; // ensure NON è encode: scalarlo dal segmento corrente
            }
            // ---- Sel del layer: 4 entry, un writeBuffer da 64 B ----
            // ORDINE OBBLIGATORIO: prima gli `ensure` (che fanno writeBuffer degli
            // slab), poi questa. La coda le esegue in ordine di enqueue, quindi lo
            // slot pubblicato qui e' gia' pieno quando i dispatch lo leggono.
            for (let k = 0; k < G.nExpertUsed; k++) {
              selU32[k * 4] = sel.experts[k];
              selU32[k * 4 + 1] = slots[k].idx;
              selF32[k * 4 + 2] = sel.weights[k]; // gia' ×1.8 (routerSelect)
              selU32[k * 4 + 3] = 0;
            }
            device.queue.writeBuffer(selBuf, m.selBase * SEL_BYTES, selScratch);
            // ---- encode MoE: shexp scrive moeOut, gli expert accumulano ----
            enc = device.createCommandEncoder();
          }
          // In modo gpu qui non e' successo NIENTE fra il router e lo shexp: niente
          // copy dei logits, niente submit, niente mapAsync, niente routerSelect,
          // niente ensure, niente writeBuffer di Sel. L'encode prosegue nello stesso
          // encoder e nello stesso pass, e le entry di Sel che i dispatch qui sotto
          // leggeranno le ha appena scritte il dispatch di resolve: il dato passa
          // per la VRAM e non tocca piu' l'host.
          ensurePass("shexp");
          runSteps(pass!, m.shexp);
          ensurePass("experts");
          // Bind group STATICI: cambia solo il dynamic offset, che dice quale
          // entry di Sel leggere — un valore CPU-noto (il layer e il k), MAI
          // l'expert. E' il punto dello strato 1, e in modo gpu e' cio' che rende
          // possibile l'interruttore: queste righe sono identiche nei tre modi,
          // perche' l'encode non ha mai avuto bisogno di sapere chi fosse scelto.
          const ex = expert[m.cls];
          for (let k = 0; k < G.nExpertUsed; k++) {
            const off = (m.selBase + k) * MOE_IDX_STRIDE;
            pass!.setPipeline(ex.gu);
            pass!.setBindGroup(0, expBg[m.cls].gu, [off]);
            const [gx, gy] = gemvGrid(G.dFfnExpert / 4); // 4 righe per workgroup
            pass!.dispatchWorkgroups(gx, gy);
            pass!.setPipeline(ex.down);
            pass!.setBindGroup(0, expBg[m.cls].down, [off]);
            const [dx, dy] = gemvGrid(G.dModel / 4);
            pass!.dispatchWorkgroups(dx, dy);
          }
          // le catene expert non passano da runSteps (bind group per-slot scelti
          // qui): 2 dispatch per expert (gate+up+silu fusi, down) — erano 4 prima
          // della famiglia fusa portata da Qwen (fase 4b)
          T.dispatches += G.nExpertUsed * 2;
          ensurePass("addMoe");
          runSteps(pass!, [m.addMoe]);
        }
        if (readLogits) {
          ensurePass("head");
          runSteps(pass!, headSteps);
        }
        endPass();
        enc.copyBufferToBuffer(x, 0, hiddenStaging, 0, G.dModel * 4);
        if (readLogits) enc.copyBufferToBuffer(logitsVocab!, 0, vocabStaging!, 0, vocab * 4);
        // Sel intera (produzione, piu' l'ombra in shadow) nel submit finale: e' il
        // solo momento in cui le regioni sono complete per tutti i layer del token.
        if (selStaging) enc.copyBufferToBuffer(selBuf, 0, selStaging, 0, nSelTot * SEL_BYTES);
        if (dirtyStaging) enc.copyBufferToBuffer(dirtyB!, 0, dirtyStaging, 0, 16);
        // resolve dei timestamp del token DENTRO il submit finale; la mapAsync
        // (armTsq) parte dopo — mai prima, altrimenti Dawn droppa il command
        // buffer (known-issue fase A, root-cause in tsq-diag-2026-07-29).
        let tsqStaging: GPUBuffer | null = null;
        if (wantGpuTs && passIdx > 0) {
          enc.resolveQuerySet(querySet!, 0, passIdx * 2, tsqResolve!, 0);
          tsqStaging = device.createBuffer({ label: "glm-tsq-staging", size: passIdx * 2 * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
          enc.copyBufferToBuffer(tsqResolve!, 0, tsqStaging, 0, passIdx * 2 * 8);
        }
        device.queue.submit([enc.finish()]);
        // I1 (spec §1): da qui al readback di coda la slotTable e' INTOCCABILE —
        // il guard vive in ExpertCache e si arma solo in optimistic (nel path
        // sync l'ensure fra submit e mapAsync e' il design, non una violazione).
        // Se un throw scatta fra qui e il rilascio (error scope), inFlight
        // resta true: e' DELIBERATO — quel throw e' fatale per il modello e un
        // I1 spurio su un'istanza morta e' meglio di un rilascio nel finally
        // che mascheri lo stato.
        if (optSel) cache.setInFlight(true);
        if (tsqStaging) armTsq(tsqStaging, passIdx, wantByCat ? passCats.slice(0, passIdx) : undefined);
        if (telemOn) { T.encodeCpuMs += performance.now() - tSeg; T.submits++; }
        const tTail = nowT();

        const errOom = await device.popErrorScope();
        const errVal = await device.popErrorScope();
        if (errOom || errVal) throw new Error(`glmmodel error scope: ${(errOom ?? errVal)!.message.slice(0, 300)}`);
        // Le mapAsync di coda partono INSIEME (stesso submit alle spalle): il
        // costo e' un round-trip host solo, come quando il readback era uno. La
        // regione ombra non "viaggia dentro" la mapAsync dell'hidden — e' una
        // staging propria, mappata in parallelo: dirlo com'e' costa una riga.
        const maps: Promise<undefined>[] = [hiddenStaging.mapAsync(GPUMapMode.READ)];
        if (readLogits) maps.push(vocabStaging!.mapAsync(GPUMapMode.READ));
        if (selStaging) maps.push(selStaging.mapAsync(GPUMapMode.READ));
        if (dirtyStaging) maps.push(dirtyStaging.mapAsync(GPUMapMode.READ));
        await Promise.all(maps);
        if (optSel) cache.setInFlight(false); // confine di token: la tabella torna toccabile
        const hidden = new Float32Array(hiddenStaging.getMappedRange().slice(0));
        hiddenStaging.unmap();
        let logits: Float32Array | undefined;
        if (readLogits) {
          logits = new Float32Array(vocabStaging!.getMappedRange().slice(0));
          vocabStaging!.unmap();
        }
        let selMissTok = 0;
        const selMisses: { layer: number; id: number; k: number }[] = [];
        if (selStaging) {
          const raw = selStaging.getMappedRange().slice(0);
          selStaging.unmap();
          const u = new Uint32Array(raw), f = new Float32Array(raw);
          // In shadow si legge la Sel INTERA, non la sola ombra: la copia costa
          // 5 888 B invece di 2 944 e in cambio si vede anche cio' che la GPU ha
          // letto DAVVERO nella regione di produzione. E' l'unico modo di chiudere
          // R5 dal lato che conta — che la writeBuffer della CPU sia arrivata in
          // VRAM com'era, nell'ordine giusto, per il layer giusto.
          const readSel = (base: number) => {
            const experts = new Int32Array(G.nExpertUsed);
            const weights = new Float64Array(G.nExpertUsed);
            const slots = new Uint32Array(G.nExpertUsed);
            const flags = new Uint32Array(G.nExpertUsed);
            for (let k = 0; k < G.nExpertUsed; k++) {
              experts[k] = u[base + k * 4];
              slots[k] = u[base + k * 4 + 1];
              weights[k] = f[base + k * 4 + 2];
              flags[k] = u[base + k * 4 + 3];
            }
            return { experts, weights, slots, flags };
          };
          const countMiss = (flags: Uint32Array) => {
            for (const fl of flags) if ((fl & 1) !== 0) selMissTok++;
          };
          if (oneSubmit) {
            // Qui `routing` e' ancora VUOTO: in modo gpu/optimistic nessuno sulla
            // CPU ha visto la selezione mentre il token girava. Si ricostruisce ORA,
            // dalla stessa copia di Sel che serviva gia' al confronto — la regione
            // e' una sola, ed e' insieme la decisione del router e cio' che i
            // kernel hanno letto. In optimistic gli stessi flag alimentano la
            // lista dei miss (spec §4: la lista viene da Sel, il flag aggregato
            // da dirtyB — e i due si incrociano piu' sotto).
            for (let mi = 0; mi < nMoeLayer; mi++) {
              const r = readSel(mi * G.nExpertUsed * 4);
              countMiss(r.flags);
              if (optSel) {
                for (let k = 0; k < G.nExpertUsed; k++) {
                  if ((r.flags[k] & 1) !== 0) selMisses.push({ layer: moeLayerAbs[mi], id: r.experts[k], k });
                }
              }
              routing.push({ layer: moeLayerAbs[mi], experts: r.experts, weights: r.weights, vram: r });
            }
          } else {
            for (const [mi, r] of routing.entries()) {
              // `routing` e' nell'ordine di visita dei layer MoE ⇒ l'indice E' m
              r.vram = readSel(mi * G.nExpertUsed * 4);
              r.gpu = readSel((nSel + mi * G.nExpertUsed) * 4);
              countMiss(r.vram.flags); // l'ombra NO: li' i MISS sono attesi (§4)
            }
          }
          T.selMiss += selMissTok;
        }
        if (telemOn) T.tailWaitMs += performance.now() - tTail;
        // R6: in modo gpu un MISS nella Sel di produzione significa che i kernel
        // expert hanno saltato un expert — il token e' INVALIDO e si dice, invece di
        // restituire numeri plausibili. Non e' raggiungibile a residenza totale (la
        // slotTable e' completa dal load e non cambia piu'): se scatta, l'ipotesi da
        // cui dipende il modo gpu e' caduta. E' il seam evict-post-resolve: a
        // residenza parziale uno slot risolto puo' essere evinto DOPO la risoluzione,
        // nello stesso token, e questo flag e' l'unico posto in cui si vedrebbe.
        // Le staging sono gia' smappate e gli error scope gia' drenati: si puo' alzare.
        if (gpuSel && selMissTok > 0) {
          throw new Error(
            `glmmodel select "gpu": ${selMissTok} entry di Sel con flag MISS in questo token — ` +
            "token INVALIDO (i kernel expert hanno letto uno slot mai pubblicato). A residenza totale " +
            "non puo' accadere; a residenza parziale e' il seam evict-post-resolve — uno slot risolto " +
            "evinto DOPO la risoluzione, nello stesso token.");
        }
        // In optimistic il MISS e' il regime, non un errore: si riporta `dirty`
        // e il driver qui sotto ripara e rigioca PRIMA di restituire (I2: un
        // token sporco non si campiona mai).
        // Il flag aggregato del kernel (dirtyB) e la derivazione da Sel devono
        // COINCIDERE: sono due strade indipendenti verso lo stesso fatto, e una
        // divergenza e' un bug strutturale (resolve che scrive Sel e dirtyB in
        // disaccordo), mai un dato da interpretare.
        let dirty: GlmDirtyInfo | undefined;
        if (optSel) {
          const du = new Uint32Array(dirtyStaging!.getMappedRange().slice(0));
          dirtyStaging!.unmap();
          const kFirstMoe = du[0], kCount = du[1];
          const derivedFirst = selMisses.length > 0 ? selMisses[0].layer : -1;
          const kernelFirst = kFirstMoe === 0xffffffff ? -1 : moeLayerAbs[kFirstMoe];
          if (kCount !== selMissTok || kernelFirst !== derivedFirst) {
            throw new Error(
              `glmmodel select "optimistic": dirtyB (${kCount} miss, primo layer ${kernelFirst}) ` +
              `!= derivazione da Sel (${selMissTok} miss, primo layer ${derivedFirst}) — ` +
              "il resolve ha scritto Sel e dirtyB in disaccordo: bug strutturale, token non interpretabile");
          }
          dirty = { firstDirtyLayer: derivedFirst, missCount: selMissTok, misses: selMisses };
        }
        return { hidden, logits, dirty };
      };

      let out = await runPass(0);
      // `forwards` conta i TOKEN, non i giri: il replay non e' un forward in
      // piu' (submits/dispatches invece contano tutto — e' cosi' che il bench
      // vede la tassa). Sempre, anche a telemetria spenta, come prima.
      T.forwards++;
      let dirty = out.dirty;
      // ---- repair + replay ITERATIVO al confine di token (spec §4, I3/I4
      // emendate in it.4 — docket item 7). I4 vale SOLO per il primo layer
      // sporco di ogni round: il suo input e' il checkpoint bit-identico ⇒
      // stessa selezione ⇒ coi miss riparati si risolve pulito. Nei layer A
      // VALLE l'hidden riparato differisce da quello degradato del giro prima
      // ⇒ il router puo' scegliere expert DIVERSI, anche non residenti ⇒ un
      // nuovo miss e' FISIOLOGIA, non bug (scoperto sul modello vero al primo
      // token: il mini-modello ktest ha un solo layer MoE e non ha valle).
      // La convergenza e' per PREFISSO: firstDirty cresce STRETTAMENTE a ogni
      // round (assert — la sua violazione SI' e' un bug strutturale), quindi
      // <= nMoeLayer round. Il costo dei round e' parte della tassa misurata.
      if (dirty && dirty.missCount > 0) {
        T.dirtyTokens++;
        if (repairOn) {
          let cur: GlmDirtyInfo | undefined = out.dirty;
          let lastFirst = -1;
          let rounds = 0;
          while (cur && cur.missCount > 0) {
            if (cur.firstDirtyLayer <= lastFirst) {
              throw new Error(
                `glmmodel select "optimistic": replay round ${rounds + 1} sporco allo STESSO layer ` +
                `${cur.firstDirtyLayer} (precedente ${lastFirst}) — progresso violato: o l'eviction ha ` +
                "toccato uno slot pinnato o l'insert non ha preceduto il flush. Bug strutturale.");
            }
            if (++rounds > nMoeLayer + 1) {
              throw new Error(`glmmodel select "optimistic": ${rounds} round di repair — oltre il cap teorico (${nMoeLayer} layer MoE)`);
            }
            lastFirst = cur.firstDirtyLayer;
            // pin-for-replay del ROUND: gli slot referenziati dalla Sel
            // corrente nei layer >= firstDirty (miss inclusi: stanno li').
            const pinned = new Set<number>();
            for (const r of routing) {
              if (r.layer < cur.firstDirtyLayer) continue;
              for (const e of r.experts) pinned.add(expertKey(r.layer, e));
            }
            const tRep = nowT();
            // fetch dei mancanti; UN flush, DOPO le writeBuffer degli slab
            // (ordine R5: il dato prima della tabella che lo indirizza).
            for (const ms of cur.misses) cache.ensure(ms.layer, ms.id, expertReader, pinned);
            cache.flushSlotTable();
            if (telemOn) T.repairMs += performance.now() - tRep;
            T.replays++;
            T.replayLayers += nLayer - cur.firstDirtyLayer; // frazione rigiocata (spec §5)
            // replay: la routing si ricostruisce dalla Sel del giro nuovo —
            // le entry dei layer non rigiocati restano in VRAM dal giro prima,
            // la copia di coda e' sempre dell'intera Sel.
            routing.length = 0;
            out = await runPass(cur.firstDirtyLayer);
            cur = out.dirty;
          }
          dirty = { ...dirty, repaired: true, repairRounds: rounds };
        }
      }
      return { hidden: out.hidden, logits: out.logits, routing, dirty };
    },
    setTelemetry(on: boolean, gpu = true, byCat = false) {
      telemOn = on;
      wantGpuTs = on && gpu && canGpuTs;
      wantByCat = wantGpuTs && byCat;
    },
    async telemetry(): Promise<GlmTelemetry> {
      // drena i batch in volo (uno per token): il gpuBusy è la somma delle
      // durate dei pass, NON include le bolle fra submit — è esattamente la
      // semantica del gpuBusy Qwen (engine.worker, repliche liv.2 dedicate)
      const batches = await Promise.all(pendingTsq.splice(0));
      for (const b of batches) {
        T.gpuBusyMs += b.ms; T.gpuPasses += b.passes;
        for (const [c, v] of b.byCat) catMs.set(c, (catMs.get(c) ?? 0) + v);
        for (const [c, n] of b.byCatN) catPasses.set(c, (catPasses.get(c) ?? 0) + n);
      }
      return {
        on: telemOn, forwards: T.forwards, encodeCpuMs: T.encodeCpuMs, ensureMs: T.ensureMs,
        routerWaitMs: T.routerWaitMs, tailWaitMs: T.tailWaitMs, routerSyncs: T.routerSyncs,
        submits: T.submits, selMiss: T.selMiss, gpuBusyMs: canGpuTs ? T.gpuBusyMs : null,
        dirtyTokens: T.dirtyTokens, replays: T.replays, replayLayers: T.replayLayers, repairMs: T.repairMs,
        gpuPasses: T.gpuPasses, gpuPassOverflow: T.gpuPassOverflow,
        dispatches: T.dispatches,
        gpuByCatMs: catMs.size ? Object.fromEntries(catMs) : null,
        gpuByCatPasses: catPasses.size ? Object.fromEntries(catPasses) : null,
        // C3c fase 4: null quando il prefetch e' spento (schema unico)
        prefetch: prefetchOn ? {
          preds: T.prefetchPreds, fetches: T.prefetchFetches, resident: T.prefetchResident,
          prefetchMs: T.prefetchMs,
          recallPreds: T.recallPreds, recallHits4: T.recallHits4, recallHits8: T.recallHits8,
        } : null,
      };
    },
    debugMarkMiss(misses: { layer: number; id: number }[]) {
      if (!routerGpu) throw new Error("glmmodel: debugMarkMiss richiede un modo col router GPU");
      cache.debugMarkMiss(misses.map((m) => expertKey(m.layer, m.id)));
    },
    async readHiddenCkpt() {
      if (!hiddenCkpt) throw new Error("glmmodel: readHiddenCkpt richiede opts.checkpointHidden");
      const staging = device.createBuffer({ size: nLayer * G.dModel * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const enc2 = device.createCommandEncoder();
      enc2.copyBufferToBuffer(hiddenCkpt, 0, staging, 0, nLayer * G.dModel * 4);
      device.queue.submit([enc2.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(staging.getMappedRange().slice(0));
      staging.destroy();
      return out;
    },
    destroy() {
      for (const b of [x, hn, qaB, qanB, qB, kvB, row576, qCkv, q576, attnCkv, attnPartials, attnMla, tmp, fnB, gateD, upD, gateE, moeOut, logitsB, selBuf, moeIdxUni, P, logitsStaging, hiddenStaging, ...weightBufs]) b.destroy();
      for (const b of [routerIds, routerWts, selStaging, dirtyB, dirtyStaging, hiddenCkpt, logits2B]) b?.destroy();
      tsqResolve?.destroy();
      querySet?.destroy();
      cache.destroy();
    },
  };
}
