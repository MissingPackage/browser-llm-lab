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
import { routerSelect, WEIGHTS_SUM_CLAMP_MIN } from "./moe";
import { MLA_CHUNK_P, mlaSMax, mlaPartialsLen, mlaSplitWorkgroupStorageBytes } from "./mlasplit";
import {
  pairGemvSiluQ5KFastWorkgroupStorageBytes, gemvQ6KFastWorkgroupStorageBytes,
} from "./kquantfast";
import { ExpertCache, expertKey, type ArenaGeometry, type ExpertClass, type ExpertRawBytes, type ExpertReader, type SlotRef, type BindRange } from "./residency";
import { expertArenaBindings } from "./gpulimits";
import {
  addInPlaceWgsl, gemvF32Wgsl, gemvGrid, gemvQ6KFastWgsl, gemvQ8HeadsWgsl,
  gemvQuantWgsl, kvAppendWgsl, mlaAttnSplitPartWgsl, mlaAttnSplitReduceWgsl,
  ropeMlaNormWgsl, rmsnormWgsl,
  siluMulWgsl, stridedCopyWgsl, pairGemvSiluFastWgsl, pairGemvSiluQ5KFastWgsl,
  gemvAccumFastWgsl, routerTopKWgsl, type ArenaOpts,
} from "./kernels/wgsl";

const HL = G.qkNope + G.ropeDims; // 256

// Indirezione dell'arena expert (C3a fase 4 strato 1, design §2).
// `Sel` = una entry per (layer MoE, k): {id, slot, w già ×1.8, flags}, 16 B.
const SEL_BYTES = 16;
// `MoeIdx` = 16 B utili, ma le entry vanno spaziate a
// minUniformBufferOffsetAlignment perché l'uniform si binda a dynamic offset.
// 256 è il default di spec e il massimo che un device possa chiedere.
const MOE_IDX_BYTES = 16;
const MOE_IDX_STRIDE = 256;

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
  cache: { budgetBytes: number; maxBindingBytes: number; maxBufferBytes: number; slotsOverride?: { q4_0: number; q4_1: number }; timing?: boolean };
  /**
   * Chi riempie `Sel`, cioe' chi decide quali expert bindare (C3a fase 4).
   * - "cpu" (default): la selezione e' il routerSelect f64 dopo il sync per layer.
   * - "shadow" (slice B): il router+resolve gira anche su GPU e scrive Sel in una
   *   regione OMBRA, ma il forward resta bit-per-bit quello di "cpu" — serve a
   *   misurare la fedelta' del router GPU sul corpus vero prima di fidarsene.
   *   Costa 1 dispatch per layer MoE, 12 KB di slotTable e un readback di 5,7 KB
   *   su staging dedicata, mappata INSIEME a quella dell'hidden (un round-trip
   *   host solo, non uno in piu').
   * - "gpu" (slice C, non implementato): un submit per token, niente readback.
   * Chiedere "gpu" oggi e' un errore esplicito, mai una degradazione silenziosa.
   */
  select?: "cpu" | "gpu" | "shadow";
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
   * La regione di PRODUZIONE di Sel riletta dalla VRAM (stessi campi, stesso
   * modo `shadow`): e' cio' che i kernel expert hanno letto davvero. Confrontarla
   * con `experts`/`weights` e con gli slot degli `ensure` e' l'unica verifica
   * diretta che la writeBuffer della CPU sia arrivata dove e quando doveva
   * (rischio R5 del design, lato produzione). Non costa un readback in piu': la
   * copia era gia' dell'intera Sel.
   */
  vram?: { experts: Int32Array; weights: Float64Array; slots: Uint32Array; flags: Uint32Array };
}

// Scomposizione del wall per token (contatori CUMULATIVI: il chiamante prende
// le differenze per finestra, come per cacheStats). Identità di costruzione:
//   wall = encodeCpuMs + ensureMs + routerWaitMs + tailWaitMs + residuo
// dove `residuo` (= sync non attribuito) si ottiene per differenza dal wall
// misurato fuori, e gpuBusyMs è la somma delle durate dei compute pass.
export interface GlmTelemetry {
  on: boolean;
  forwards: number;
  encodeCpuMs: number;    // tempo JS di encode/bind/writeBuffer, await esclusi
  ensureMs: number;       // tempo dentro ExpertCache.ensure (residenza)
  routerWaitMs: number;   // tempo negli await di readback del router (46/token)
  tailWaitMs: number;     // tempo nell'await finale (hidden + logits)
  routerSyncs: number;    // readback router EFFETTIVI (non una costante)
  submits: number;        // submit effettivi
  gpuBusyMs: number | null; // somma delle durate dei pass (solo telemetryGpu)
  gpuPasses: number;
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
  // Dispatch CONTATI a runtime (non la formula del piano). Il contatore gira
  // sempre, anche a telemetria spenta: un incremento intero ogni ~120 µs di
  // lavoro GPU e' sotto il rumore, e avere il numero vero in ogni report vale
  // piu' della purezza. Confronta con `dispatchesPerTokenPlanned`.
  dispatches: number;
}

export interface GlmModel {
  // hidden = stato post-ultimo-layer (readback: harness di conformance, non
  // bench). readLogits richiede opts.head: aggiunge final norm + lm_head e
  // ritorna i logits interi (154.880 × f32 = 620 KB/readback).
  forward(x: Float32Array, pos: number, readLogits?: boolean): Promise<{ hidden: Float32Array; logits?: Float32Array; routing: GlmRouting[] }>;
  // ATTENZIONE alla semantica: questo e' il valore DERIVATO dal piano statico
  // (formula sui conteggi di layer), non un conteggio. Il numero misurato sta
  // in `telemetry().dispatches`. I due divergono: la formula non contiene la
  // testa (rms + lm_head = 2 dispatch), che pero' viene eseguita a ogni token
  // con readLogits. Tenuti entrambi e nominati per quello che sono.
  dispatchesPerTokenPlanned: number;
  cacheStats(): ReturnType<ExpertCache["stats"]>;
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
    routerSyncs: 0, submits: 0, gpuBusyMs: 0, gpuPasses: 0, gpuPassOverflow: 0,
    dispatches: 0,
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
  const shadow = opts.select === "shadow";
  if (opts.select !== undefined && opts.select !== "cpu" && !shadow) {
    throw new Error(
      `glmmodel: select "${opts.select}" non e' implementato (slice C della fase 4) — ` +
      'oggi si accettano "cpu" e "shadow"');
  }
  const cache = new ExpertCache(device, { ...opts.cache, arena: true, slotTable: shadow });
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
      down: mkPipe(gemvAccumFastWgsl({ kind: cls, K: G.dFfnExpert, N: G.dModel, arena }), bgl),
    };
  };
  const expert = { q4_0: mkExpertClass("q4_0"), q4_1: mkExpertClass("q4_1") };

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
  // niente piu' `wExp`: il peso di mixing e' il campo `w` di Sel (§2.1 del
  // design) — la stessa indirezione che porta lo slot porta anche il peso.
  const P = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const logitsStaging = device.createBuffer({ size: G.nExpert * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
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

  interface MoeLayerGpu {
    bias: Float32Array;
    biasBuf: GPUBuffer | null; // gli stessi bias in VRAM: li legge il router GPU (shadow)
    cls: ExpertClass;        // classe degli expert del layer (down q4_0 o q4_1)
    moeIdx: number;          // indice del layer fra i soli MoE
    selBase: number;         // prima entry di Sel del layer: moeLayerIdx * nUsed
    routerBind: GPUBindGroup | null; // bind group del router GPU (shadow)
    preRouter: Step[];       // ffn_norm + router (dopo l'attention)
    shexp: Step[];           // scrive moeOut
    addMoe: Step;            // x += moeOut
  }
  interface LayerGpu {
    attn: Step[];            // attention completa (residuo incluso), con 1 copy k_pe
    kpeCopy: { srcOff: number; dstOff: number; bytes: number }; // dentro attn dopo rmsKvA
    dense?: Step[];          // blk.0
    moe?: MoeLayerGpu;
    cache: GPUBuffer;        // kv cache 576/token
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
        moe: {
          bias,
          biasBuf: shadow ? track(upload(bias)) : null,
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
          addMoe: step(pipes.add, [x, moeOut], Math.ceil(G.dModel / 64)),
        },
      });
    }
  }

  // ---- Sel + MoeIdx + i 4 bind group STATICI dell'arena ----
  // Reader costruito UNA volta: se la sorgente ha il file slab, la cache prende
  // il percorso senza pack CPU; altrimenti ripiega sui byte GGUF grezzi.
  const expertReader: ExpertReader = src.hasSlabs && src.expertSlab
    ? { raw: (ll, ee) => src.expert(ll, ee), slab: (ll, ee) => src.expertSlab!(ll, ee) }
    : (ll: number, ee: number) => src.expert(ll, ee);
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
  // kernel servira' lo slice C cambiando solo la entry di uniform che lo indirizza.
  const nSelTot = shadow ? nSel * 2 : nSel;
  const selBuf = device.createBuffer({
    size: nSelTot * SEL_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  // Entry di MoeIdx: nSel di produzione (una per (layer MoE, k)) piu', in modo
  // shadow, una per layer MoE che indirizza il router — `selIdx` gia' spostato
  // nell'ombra. E' cosi' che il kernel di resolve resta identico fra shadow e
  // slice C: cambia la entry, non il WGSL.
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
  const expBg = {
    q4_0: { gu: arenaBg("q4_0", fnB, gateE), down: arenaBg("q4_0", gateE, moeOut) },
    q4_1: { gu: arenaBg("q4_1", fnB, gateE), down: arenaBg("q4_1", gateE, moeOut) },
  };

  // ---- router GPU + resolve (slice B, in ombra) ----
  // Un dispatch per layer MoE, in coda al GEMV del router e nello STESSO submit:
  // legge i logits appena scritti, sceglie i top-4 come la CPU e risolve gli
  // slot dalla slotTable. Gira PRIMA degli `ensure` del suo layer — e' l'ordine
  // del path attuale, non una scelta: cio' che il layer sta per caricare non e'
  // ancora nella tabella, quindi in shadow un expert appena entrato si risolve
  // MISS. Non e' un difetto della risoluzione: e' la residenza parziale vista da
  // GPU, ed e' esattamente la ragione per cui lo slice C esige la totale.
  const routerIds = shadow ? storage(G.nExpertUsed * 4) : null;
  const routerWts = shadow ? storage(G.nExpertUsed * 4) : null;
  let routerGpuPipe: GPUComputePipeline | null = null;
  if (shadow) {
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
      ],
    });
    routerGpuPipe = mkPipe(routerTopKWgsl({
      nExpert: G.nExpert, nUsed: G.nExpertUsed,
      weightsScale: G.weightsScale, clampMin: WEIGHTS_SUM_CLAMP_MIN,
      resolve: { nExpert: G.nExpert, nUsed: G.nExpertUsed },
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
        ],
      });
    }
  }
  // Copia di ENTRAMBE le regioni di Sel nel tail (5 888 B): l'ombra per il
  // confronto col router GPU, la produzione per rileggere cio' che i kernel
  // expert hanno visto. La mapAsync parte insieme a quella dell'hidden.
  const selStaging = shadow
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

  // Valore DERIVATO dal piano: attn 17/layer (16 fino alla fase 4b; l'attention
  // split ne aggiunge uno, part+reduce al posto del kernel monolitico) + denso
  // 6 (solo blk.0) + per layer
  // MoE 2 preRouter + 2 shexp (it.13: gate+up+silu Q5_K fusi, era 4) + 4 catene
  // expert da 2 (fase 4b: gate+up+silu fusi) + 1 add = 13; in modo shadow
  // (slice B) il router+resolve su GPU ne aggiunge 1 per layer MoE ⇒ 14.
  // NON include la testa (rms + lm_head = 2), che con readLogits gira a ogni
  // token: e' la ragione per cui questo numero e' sempre stato 2 sotto il vero.
  // Il conteggio reale e' `telemetry().dispatches` (contatore in runSteps e
  // nelle catene expert).
  const nMoe = layers.filter((l) => l.moe).length;
  const nDense = layers.filter((l) => l.dense).length;
  const dispatchesPerTokenPlanned = 17 * nLayer + 6 * nDense + (shadow ? 14 : 13) * nMoe;

  const mapLogits = async (): Promise<Float32Array> => {
    await logitsStaging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(logitsStaging.getMappedRange().slice(0));
    logitsStaging.unmap();
    return out;
  };

  return {
    dispatchesPerTokenPlanned,
    cacheStats: () => cache.stats(),
    async forward(xIn: Float32Array, pos: number, readLogits = false) {
      if (pos >= ctxMax) throw new Error("glmmodel: contesto pieno");
      if (readLogits && !opts.head) throw new Error("glmmodel: head non abilitata");
      device.queue.writeBuffer(x, 0, xIn as unknown as BufferSource);
      device.queue.writeBuffer(P, 0, new Uint32Array([pos, pos, 0, 0]));
      device.pushErrorScope("validation");
      device.pushErrorScope("out-of-memory");

      const routing: GlmRouting[] = [];
      let enc = device.createCommandEncoder();
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

      for (const [l, L] of layers.entries()) {
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
        if (shadow) {
          // stessa categoria di telemetria del GEMV che lo precede (`router`) e
          // stesso pass: e' selezione, non un blocco nuovo
          pass!.setPipeline(routerGpuPipe!);
          pass!.setBindGroup(0, m.routerBind!, [(nSel + m.moeIdx) * MOE_IDX_STRIDE]);
          pass!.dispatchWorkgroups(1);
          T.dispatches++;
        }
        endPass();
        enc.copyBufferToBuffer(logitsB, 0, logitsStaging, 0, G.nExpert * 4);
        device.queue.submit([enc.finish()]);
        // ---- sync GPU→CPU: selezione ----
        if (telemOn) { T.encodeCpuMs += performance.now() - tSeg; T.submits++; T.routerSyncs++; }
        const tWait = nowT();
        const logits = await mapLogits();
        if (telemOn) tSeg = performance.now();
        if (telemOn) T.routerWaitMs += tSeg - tWait;
        const sel = routerSelect(logits, m.bias);
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
        ensurePass("shexp");
        runSteps(pass!, m.shexp);
        ensurePass("experts");
        // Bind group STATICI: cambia solo il dynamic offset, che dice quale
        // entry di Sel leggere — un valore CPU-noto (il layer e il k), MAI
        // l'expert. E' il punto dello strato 1: l'encode non dipende piu' dalla
        // selezione, quindi lo slice C potra' toglierne il sync.
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
      // Sel intera (produzione + ombra) nel submit finale: e' l'unico momento in
      // cui le due regioni sono complete per tutti i layer del token.
      if (shadow) enc.copyBufferToBuffer(selBuf, 0, selStaging!, 0, nSelTot * SEL_BYTES);
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
      if (tsqStaging) armTsq(tsqStaging, passIdx, wantByCat ? passCats.slice(0, passIdx) : undefined);
      // `forwards` e `dispatches` sono la stessa coppia: entrambi SEMPRE, anche
      // a telemetria spenta, altrimenti il rapporto dispatch/token e' spazzatura
      // (il gate ktest ha beccato esattamente questo alla prima esecuzione).
      T.forwards++;
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
      if (shadow) maps.push(selStaging!.mapAsync(GPUMapMode.READ));
      await Promise.all(maps);
      const hidden = new Float32Array(hiddenStaging.getMappedRange().slice(0));
      hiddenStaging.unmap();
      let logits: Float32Array | undefined;
      if (readLogits) {
        logits = new Float32Array(vocabStaging!.getMappedRange().slice(0));
        vocabStaging!.unmap();
      }
      if (shadow) {
        const raw = selStaging!.getMappedRange().slice(0);
        selStaging!.unmap();
        const u = new Uint32Array(raw), f = new Float32Array(raw);
        // Si legge la Sel INTERA, non la sola ombra: la copia costa 5 888 B
        // invece di 2 944 e in cambio si vede anche cio' che la GPU ha letto
        // DAVVERO nella regione di produzione. E' l'unico modo di chiudere R5
        // dal lato che conta — che la writeBuffer della CPU sia arrivata in VRAM
        // com'era, nell'ordine giusto, per il layer giusto.
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
        for (const [mi, r] of routing.entries()) {
          // `routing` e' nell'ordine di visita dei layer MoE ⇒ l'indice E' m
          r.vram = readSel(mi * G.nExpertUsed * 4);
          r.gpu = readSel((nSel + mi * G.nExpertUsed) * 4);
        }
      }
      if (telemOn) T.tailWaitMs += performance.now() - tTail;
      return { hidden, logits, routing };
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
        submits: T.submits, gpuBusyMs: canGpuTs ? T.gpuBusyMs : null,
        gpuPasses: T.gpuPasses, gpuPassOverflow: T.gpuPassOverflow,
        dispatches: T.dispatches,
        gpuByCatMs: catMs.size ? Object.fromEntries(catMs) : null,
        gpuByCatPasses: catPasses.size ? Object.fromEntries(catPasses) : null,
      };
    },
    destroy() {
      for (const b of [x, hn, qaB, qanB, qB, kvB, row576, qCkv, q576, attnCkv, attnPartials, attnMla, tmp, fnB, gateD, upD, gateE, moeOut, logitsB, selBuf, moeIdxUni, P, logitsStaging, hiddenStaging, ...weightBufs]) b.destroy();
      for (const b of [routerIds, routerWts, selStaging]) b?.destroy();
      tsqResolve?.destroy();
      querySet?.destroy();
      cache.destroy();
    },
  };
}
