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
// per-layer precostruiti al load; bind group expert cached per-slot (la cache
// LRU ha slot stabili: ~2.5k voci massime per pipeline).
import { GLM47_FLASH as G } from "./shape";
import { repackQ4_0, repackQ4_1, repackQ8_0, repackKQuant, Q5_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES } from "./quant";
import { routerSelect } from "./moe";
import { ExpertCache, expertKey, slotBindRanges, type ExpertRawBytes, type SlotRef, type BindRange } from "./residency";
import {
  addInPlaceWgsl, gemvF32Wgsl, gemvGrid, gemvQ5KWgsl, gemvQ6KWgsl, gemvQ8HeadsWgsl,
  gemvQuantWgsl, kvAppendWgsl, mlaAttnDecodeWgsl, ropeMlaNormWgsl, rmsnormWgsl,
  siluMulWgsl, stridedCopyWgsl,
} from "./kernels/wgsl";

const HL = G.qkNope + G.ropeDims; // 256

// Sorgente pesi: byte GREZZI GGUF. In produzione è OPFS (ExpertOpfsStore +
// GgufExpertIndex); nei test un mock sintetico. `nonExpert` riceve il nome
// completo ("blk.3.ffn_norm.weight", "token_embd.weight", …).
export interface GlmWeightSource {
  nonExpert(name: string): Uint8Array;
  expert(layer: number, expert: number): ExpertRawBytes;
}

export interface GlmModelOpts {
  nLayer?: number;  // default G.nLayer; i test usano 2 (blk.0 denso + blk.1 MoE)
  ctxMax: number;
  cache: { budgetBytes: number; maxBindingBytes: number; maxBufferBytes: number; slotsOverride?: { q4_0: number; q4_1: number }; timing?: boolean };
}

export interface GlmRouting { layer: number; experts: Int32Array; weights: Float64Array }

export interface GlmModel {
  // hidden = stato post-ultimo-layer (readback: harness di conformance, non bench)
  forward(x: Float32Array, pos: number): Promise<{ hidden: Float32Array; routing: GlmRouting[] }>;
  dispatchesPerToken: number;
  cacheStats(): ReturnType<ExpertCache["stats"]>;
  destroy(): void;
}

export function createGlmModel(device: GPUDevice, src: GlmWeightSource, opts: GlmModelOpts): GlmModel {
  const nLayer = opts.nLayer ?? G.nLayer;
  const ctxMax = opts.ctxMax;

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
  const mkPipe = (code: string) =>
    device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
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
    attn: mkPipe(mlaAttnDecodeWgsl({ nHead: G.nHead, kvLora: G.kvLora, ropeDims: G.ropeDims, ctxMax, scale: 1 / Math.sqrt(G.headLenMla) })),
    voutVb: mkPipe(gemvQ8HeadsWgsl({ K: G.kvLora, rowsPerHead: G.headLenMla, nHead: G.nHead, xStride: G.kvLora, xOffset: 0 })),
    gemvO: mkPipe(gemvQuantWgsl({ kind: "q4_0", K: G.nHead * G.headLenMla, N: G.dModel, hasBias: false })),
    // ffn denso blk.0
    gemvDenseGU: mkPipe(gemvQuantWgsl({ kind: "q4_0", K: G.dModel, N: G.dFfnDense, hasBias: false })),
    gemvDenseDown: mkPipe(gemvQuantWgsl({ kind: "q4_1", K: G.dFfnDense, N: G.dModel, hasBias: false })),
    siluDense: mkPipe(siluMulWgsl(G.dFfnDense)),
    // MoE
    router: mkPipe(gemvF32Wgsl({ K: G.dModel, N: G.nExpert })),
    gemvShexpGU: mkPipe(gemvQ5KWgsl({ K: G.dModel, N: G.dFfnExpert })),
    gemvShexpDown: mkPipe(gemvQ6KWgsl({ K: G.dFfnExpert, N: G.dModel })),
    siluExp: mkPipe(siluMulWgsl(G.dFfnExpert)),
    gemvExpGU: mkPipe(gemvQuantWgsl({ kind: "q4_0", K: G.dModel, N: G.dFfnExpert, hasBias: false })),
    gemvExpDown40: mkPipe(gemvQuantWgsl({ kind: "q4_0", K: G.dFfnExpert, N: G.dModel, hasBias: false, scaledAccum: true })),
    gemvExpDown41: mkPipe(gemvQuantWgsl({ kind: "q4_1", K: G.dFfnExpert, N: G.dModel, hasBias: false, scaledAccum: true })),
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
  const attnMla = storage(G.nHead * G.headLenMla * 4);
  const tmp = storage(G.dModel * 4);
  const fnB = storage(G.dModel * 4);
  const gateD = storage(G.dFfnDense * 4);
  const upD = storage(G.dFfnDense * 4);
  const gateE = storage(G.dFfnExpert * 4);
  const upE = storage(G.dFfnExpert * 4);
  const moeOut = storage(G.dModel * 4);
  const logitsB = storage(G.nExpert * 4);
  const wExp = Array.from({ length: G.nExpertUsed }, () => storage(4)); // peso mixing per catena
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
      step(pipes.attn, [q576, cache, attnCkv], G.nHead, true),
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
      layers.push({
        attn, kpeCopy, cache,
        moe: {
          bias,
          preRouter: [
            step(pipes.rmsD, [x, ffnNorm, fnB], 1),
            step(pipes.router, [routerW, fnB, logitsB], G.nExpert),
          ],
          shexp: [
            step(pipes.gemvShexpGU, [gateShexp, fnB, gateE], gemvGrid(G.dFfnExpert)),
            step(pipes.gemvShexpGU, [upShexp, fnB, upE], gemvGrid(G.dFfnExpert)),
            step(pipes.siluExp, [gateE, upE], Math.ceil(G.dFfnExpert / 64)),
            step(pipes.gemvShexpDown, [downShexp, gateE, moeOut], gemvGrid(G.dModel)),
          ],
          addMoe: step(pipes.add, [x, moeOut], Math.ceil(G.dModel / 64)),
        },
      });
    }
  }

  // ---- cache expert + bind group per-slot (cached: slot stabili) ----
  const cache = new ExpertCache(device, opts.cache);
  interface SlotBgs { gate: GPUBindGroup; up: GPUBindGroup; down: GPUBindGroup[] } // down: uno per wExp[k]
  const slotBgCache = new Map<GPUBuffer, Map<number, SlotBgs>>();
  const slotBgs = (slot: SlotRef): SlotBgs => {
    let byOff = slotBgCache.get(slot.buffer);
    if (!byOff) { byOff = new Map(); slotBgCache.set(slot.buffer, byOff); }
    let got = byOff.get(slot.offset);
    if (!got) {
      const r = slotBindRanges(slot);
      const downPipe = slot.cls === "q4_1" ? pipes.gemvExpDown41 : pipes.gemvExpDown40;
      got = {
        gate: bg(pipes.gemvExpGU, [r.gateQs, r.gateScales, fnB, gateE]),
        up: bg(pipes.gemvExpGU, [r.upQs, r.upScales, fnB, upE]),
        down: wExp.map((wb) => bg(downPipe, [r.downQs, r.downScales, gateE, moeOut, wb])),
      };
      byOff.set(slot.offset, got);
    }
    return got;
  };

  const siluExpBind = bg(pipes.siluExp, [gateE, upE]);

  const runSteps = (pass: GPUComputePassEncoder, steps: Step[]): void => {
    for (const s of steps) {
      pass.setPipeline(s.pipe);
      pass.setBindGroup(0, s.bind);
      pass.dispatchWorkgroups(s.wgs[0], s.wgs[1]);
    }
  };

  // dispatch/token: attn 16/layer + denso 6 (solo blk.0) + per layer MoE
  // 2 preRouter + 4 shexp + 4 catene expert da 4 + 1 add = 23
  const nMoe = layers.filter((l) => l.moe).length;
  const nDense = layers.filter((l) => l.dense).length;
  const dispatchesPerToken = 16 * nLayer + 6 * nDense + 23 * nMoe;

  const mapLogits = async (): Promise<Float32Array> => {
    await logitsStaging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(logitsStaging.getMappedRange().slice(0));
    logitsStaging.unmap();
    return out;
  };

  return {
    dispatchesPerToken,
    cacheStats: () => cache.stats(),
    async forward(xIn: Float32Array, pos: number) {
      if (pos >= ctxMax) throw new Error("glmmodel: contesto pieno");
      device.queue.writeBuffer(x, 0, xIn as unknown as BufferSource);
      device.queue.writeBuffer(P, 0, new Uint32Array([pos, pos, 0, 0]));
      device.pushErrorScope("validation");
      device.pushErrorScope("out-of-memory");

      const routing: GlmRouting[] = [];
      let enc = device.createCommandEncoder();
      let pass: GPUComputePassEncoder | null = null;
      const ensurePass = () => (pass ??= enc.beginComputePass());
      const endPass = () => { if (pass) { pass.end(); pass = null; } };

      for (const [l, L] of layers.entries()) {
        // attention (con la copy k_pe dopo rmsKvA: indice 8 = kvApp)
        ensurePass();
        runSteps(pass!, L.attn.slice(0, 8));
        endPass();
        enc.copyBufferToBuffer(kvB, L.kpeCopy.srcOff, row576, L.kpeCopy.dstOff, L.kpeCopy.bytes);
        ensurePass();
        runSteps(pass!, L.attn.slice(8));
        if (L.dense) {
          runSteps(pass!, L.dense);
          continue;
        }
        const m = L.moe!;
        runSteps(pass!, m.preRouter);
        endPass();
        enc.copyBufferToBuffer(logitsB, 0, logitsStaging, 0, G.nExpert * 4);
        device.queue.submit([enc.finish()]);
        // ---- sync GPU→CPU: selezione ----
        const logits = await mapLogits();
        const sel = routerSelect(logits, m.bias);
        routing.push({ layer: l, experts: sel.experts, weights: sel.weights });
        const pinned = new Set<number>();
        for (const e of sel.experts) pinned.add(expertKey(l, e));
        const slots: SlotRef[] = [];
        for (const e of sel.experts) slots.push(cache.ensure(l, e, (ll, ee) => src.expert(ll, ee), pinned).slot);
        for (let k = 0; k < G.nExpertUsed; k++) {
          device.queue.writeBuffer(wExp[k], 0, new Float32Array([sel.weights[k]]));
        }
        // ---- encode MoE: shexp scrive moeOut, gli expert accumulano ----
        enc = device.createCommandEncoder();
        ensurePass();
        runSteps(pass!, m.shexp);
        for (let k = 0; k < G.nExpertUsed; k++) {
          const bgs = slotBgs(slots[k]);
          pass!.setPipeline(pipes.gemvExpGU);
          pass!.setBindGroup(0, bgs.gate);
          const [gx, gy] = gemvGrid(G.dFfnExpert);
          pass!.dispatchWorkgroups(gx, gy);
          pass!.setBindGroup(0, bgs.up);
          pass!.dispatchWorkgroups(gx, gy);
          pass!.setPipeline(pipes.siluExp);
          pass!.setBindGroup(0, siluExpBind);
          pass!.dispatchWorkgroups(Math.ceil(G.dFfnExpert / 64));
          const downPipe = slots[k].cls === "q4_1" ? pipes.gemvExpDown41 : pipes.gemvExpDown40;
          pass!.setPipeline(downPipe);
          pass!.setBindGroup(0, bgs.down[k]);
          const [dx, dy] = gemvGrid(G.dModel);
          pass!.dispatchWorkgroups(dx, dy);
        }
        runSteps(pass!, [m.addMoe]);
      }
      endPass();
      enc.copyBufferToBuffer(x, 0, hiddenStaging, 0, G.dModel * 4);
      device.queue.submit([enc.finish()]);

      const errOom = await device.popErrorScope();
      const errVal = await device.popErrorScope();
      if (errOom || errVal) throw new Error(`glmmodel error scope: ${(errOom ?? errVal)!.message.slice(0, 300)}`);
      await hiddenStaging.mapAsync(GPUMapMode.READ);
      const hidden = new Float32Array(hiddenStaging.getMappedRange().slice(0));
      hiddenStaging.unmap();
      return { hidden, routing };
    },
    destroy() {
      for (const b of [x, hn, qaB, qanB, qB, kvB, row576, qCkv, q576, attnCkv, attnMla, tmp, fnB, gateD, upD, gateE, upE, moeOut, logitsB, ...wExp, P, logitsStaging, hiddenStaging, ...weightBufs]) b.destroy();
      cache.destroy();
    },
  };
}
