// Assemblaggio GPU del layer GLM-4.7-Flash (goal C2 fase 4): blk.0 denso con
// MLA in formulazione ABSORBED (spec §3) — decode, un token per submit.
//
// È il primo orchestratore del path GLM: stessi principi di gpuforward (bind
// group precostruiti, un submit per token, error scope come contratto), scala
// di un layer solo. La fase 6 lo estende al forward completo (47 layer, MoE in
// fase 5); qui è il gate di conformance layer-level contro cpuref f64 naive.
//
// Percorso per token (semantica verificata nell'oracolo, it.4):
//   rms(attn_norm) → q_a → rms(q_a_norm) → q_b → rope NORM su q[192..256)/head
//   kv_a_mqa → rope NORM su kv[512..576) → rms(kv_a_norm) su [0..512)
//   cache[pos] = [c_kv normata | k_pe ruotato] (576)
//   q_ckv/head = wk_b(h)·q_nope (absorbed) → q576 = [q_ckv | q_rope]
//   attn MQA su cache 576 (scale 1/sqrt(256)) → out in spazio c_kv [20×512]
//   → wv_b(h) [512→256] → attn_output → residuo → ffn denso → residuo
import { GLM47_FLASH as G } from "./shape";
import { repackQ4_0, repackQ4_1, repackQ8_0 } from "./quant";
import {
  addInPlaceWgsl, gemvGrid, gemvQ8HeadsWgsl, gemvQuantWgsl, kvAppendWgsl,
  mlaAttnDecodeWgsl, ropeMlaNormWgsl, rmsnormWgsl, siluMulWgsl, stridedCopyWgsl,
} from "./kernels/wgsl";

const HL = G.qkNope + G.ropeDims; // 256: head len di q [nope|rope]

// Byte GREZZI del GGUF per i tensori di blk.0 (dal fixture o dal reader).
export interface GlmLayer0RawWeights {
  attnNorm: Float32Array; qANorm: Float32Array; kvANorm: Float32Array; ffnNorm: Float32Array;
  wQA: Uint8Array;   // Q4_0 [2048→768]
  wQB: Uint8Array;   // Q4_0 [768→5120]
  wKvA: Uint8Array;  // Q8_0 [2048→576]
  wKB: Uint8Array;   // Q8_0 [192,512,20]
  wVB: Uint8Array;   // Q8_0 [512,256,20]
  wO: Uint8Array;    // Q4_0 [5120→2048]
  wGate: Uint8Array; // Q4_0 [2048→10240]
  wUp: Uint8Array;   // Q4_0 [2048→10240]
  wDown: Uint8Array; // Q4_1 [10240→2048]
}

export interface GlmLayer0Gpu {
  // pos === numero di forward già fatti (decode sequenziale); ritorna l'hidden
  // post-layer (2048) — readback a ogni token: harness di conformance, non bench.
  forward(x: Float32Array, pos: number): Promise<Float32Array>;
  dispatchesPerToken: number;
  destroy(): void;
}

export function createGlmLayer0(device: GPUDevice, w: GlmLayer0RawWeights, ctxMax = 64): GlmLayer0Gpu {
  const upload = (data: Uint32Array | Float32Array): GPUBuffer => {
    const b = device.createBuffer({ size: Math.max(16, data.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(b, 0, data as BufferSource);
    return b;
  };
  const storage = (bytesLen: number): GPUBuffer =>
    device.createBuffer({ size: Math.max(16, bytesLen), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const quant = (raw: Uint8Array, repack: (s: Uint8Array, o: number, n: number) => { qs: Uint32Array; scales: Uint32Array }, blockBytes: number) => {
    const rp = repack(raw, 0, raw.length / blockBytes);
    return { qs: upload(rp.qs), scales: upload(rp.scales) };
  };

  const wQA = quant(w.wQA, repackQ4_0, 18);
  const wQB = quant(w.wQB, repackQ4_0, 18);
  const wKvA = quant(w.wKvA, repackQ8_0, 34);
  const wKB = quant(w.wKB, repackQ8_0, 34);
  const wVB = quant(w.wVB, repackQ8_0, 34);
  const wO = quant(w.wO, repackQ4_0, 18);
  const wGate = quant(w.wGate, repackQ4_0, 18);
  const wUp = quant(w.wUp, repackQ4_0, 18);
  const wDown = quant(w.wDown, repackQ4_1, 20);
  const attnNorm = upload(w.attnNorm);
  const qANorm = upload(w.qANorm);
  const kvANorm = upload(w.kvANorm);
  const ffnNorm = upload(w.ffnNorm);

  // attivazioni
  const x = storage(G.dModel * 4);
  const hn = storage(G.dModel * 4);
  const qaB = storage(G.qLora * 4);
  const qanB = storage(G.qLora * 4);
  const qB = storage(G.nHead * HL * 4);
  const kvB = storage(G.keyLen * 4);
  const row576 = storage(G.keyLen * 4);
  const cache = storage(ctxMax * G.keyLen * 4);
  const qCkv = storage(G.nHead * G.kvLora * 4);
  const q576 = storage(G.nHead * G.keyLen * 4);
  const attnCkv = storage(G.nHead * G.kvLora * 4);
  const attnMla = storage(G.nHead * G.headLenMla * 4);
  const fnB = storage(G.dModel * 4);
  const gateB = storage(G.dFfnDense * 4);
  const upB = storage(G.dFfnDense * 4);
  const tmp = storage(G.dModel * 4);
  const P = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const staging = device.createBuffer({ size: G.dModel * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

  const mkPipe = (code: string) =>
    device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
  const pipes = {
    rmsAttn: mkPipe(rmsnormWgsl(G.dModel, G.rmsEps)),
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
    gemvGate: mkPipe(gemvQuantWgsl({ kind: "q4_0", K: G.dModel, N: G.dFfnDense, hasBias: false })),
    gemvDown: mkPipe(gemvQuantWgsl({ kind: "q4_1", K: G.dFfnDense, N: G.dModel, hasBias: false })),
    silu: mkPipe(siluMulWgsl(G.dFfnDense)),
    add: mkPipe(addInPlaceWgsl(G.dModel)),
  };
  const bg = (pipe: GPUComputePipeline, bufs: GPUBuffer[], uni?: GPUBuffer) =>
    device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: [
        ...bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })),
        ...(uni ? [{ binding: bufs.length, resource: { buffer: uni } }] : []),
      ],
    });

  type Step =
    | { kind: "compute"; pipe: GPUComputePipeline; bind: GPUBindGroup; wgs: [number, number] }
    | { kind: "copy"; src: GPUBuffer; srcOff: number; dst: GPUBuffer; dstOff: number; bytes: number };
  const steps: Step[] = [];
  const push = (pipe: GPUComputePipeline, bufs: GPUBuffer[], wgs: number | [number, number], uni?: GPUBuffer) =>
    steps.push({ kind: "compute", pipe, bind: bg(pipe, bufs, uni), wgs: typeof wgs === "number" ? [wgs, 1] : wgs });

  push(pipes.rmsAttn, [x, attnNorm, hn], 1);
  push(pipes.gemvQA, [wQA.qs, wQA.scales, hn, qaB], gemvGrid(G.qLora));
  push(pipes.rmsQA, [qaB, qANorm, qanB], 1);
  push(pipes.gemvQB, [wQB.qs, wQB.scales, qanB, qB], gemvGrid(G.nHead * HL));
  push(pipes.ropeQ, [qB], Math.ceil((G.nHead * G.ropeDims / 2) / 64), P);
  push(pipes.gemvKvA, [wKvA.qs, wKvA.scales, hn, kvB], gemvGrid(G.keyLen));
  push(pipes.ropeKPe, [kvB], Math.ceil((G.ropeDims / 2) / 64), P);
  // kv_a_norm tocca solo [0..512): rms scrive row576[0..512), il k_pe ruotato
  // arriva per copy (offset 512·4 allineato, copyBufferToBuffer senza stride)
  push(pipes.rmsKvA, [kvB, kvANorm, row576], 1);
  steps.push({ kind: "copy", src: kvB, srcOff: G.kvLora * 4, dst: row576, dstOff: G.kvLora * 4, bytes: G.ropeDims * 4 });
  push(pipes.kvApp, [row576, cache], Math.ceil(G.keyLen / 64), P);
  push(pipes.absorbKb, [wKB.qs, wKB.scales, qB, qCkv], gemvGrid(G.nHead * G.kvLora));
  push(pipes.copyCkv, [qCkv, q576], Math.ceil((G.nHead * G.kvLora) / 64));
  push(pipes.copyQRope, [qB, q576], Math.ceil((G.nHead * G.ropeDims) / 64));
  push(pipes.attn, [q576, cache, attnCkv], G.nHead, P);
  push(pipes.voutVb, [wVB.qs, wVB.scales, attnCkv, attnMla], gemvGrid(G.nHead * G.headLenMla));
  push(pipes.gemvO, [wO.qs, wO.scales, attnMla, tmp], gemvGrid(G.dModel));
  push(pipes.add, [x, tmp], Math.ceil(G.dModel / 64));
  push(pipes.rmsAttn, [x, ffnNorm, fnB], 1);
  push(pipes.gemvGate, [wGate.qs, wGate.scales, fnB, gateB], gemvGrid(G.dFfnDense));
  push(pipes.gemvGate, [wUp.qs, wUp.scales, fnB, upB], gemvGrid(G.dFfnDense));
  push(pipes.silu, [gateB, upB], Math.ceil(G.dFfnDense / 64));
  push(pipes.gemvDown, [wDown.qs, wDown.scales, gateB, tmp], gemvGrid(G.dModel));
  push(pipes.add, [x, tmp], Math.ceil(G.dModel / 64));

  return {
    dispatchesPerToken: steps.filter((s) => s.kind === "compute").length,
    async forward(xIn: Float32Array, pos: number): Promise<Float32Array> {
      if (pos >= ctxMax) throw new Error("glmforward: contesto pieno");
      device.queue.writeBuffer(x, 0, xIn as unknown as BufferSource);
      device.queue.writeBuffer(P, 0, new Uint32Array([pos, pos, 0, 0]));
      // error scope come CONTRATTO di ogni percorso di encode nuovo (landmine B1)
      device.pushErrorScope("validation");
      device.pushErrorScope("out-of-memory");
      const enc = device.createCommandEncoder();
      let pass = enc.beginComputePass();
      for (const s of steps) {
        if (s.kind === "copy") {
          pass.end();
          enc.copyBufferToBuffer(s.src, s.srcOff, s.dst, s.dstOff, s.bytes);
          pass = enc.beginComputePass();
          continue;
        }
        pass.setPipeline(s.pipe);
        pass.setBindGroup(0, s.bind);
        pass.dispatchWorkgroups(s.wgs[0], s.wgs[1]);
      }
      pass.end();
      enc.copyBufferToBuffer(x, 0, staging, 0, G.dModel * 4);
      device.queue.submit([enc.finish()]);
      const errOom = await device.popErrorScope();
      const errVal = await device.popErrorScope();
      if (errOom || errVal) throw new Error(`glmforward error scope: ${(errOom ?? errVal)!.message.slice(0, 300)}`);
      await staging.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(staging.getMappedRange().slice(0));
      staging.unmap();
      return out;
    },
    destroy() {
      for (const b of [x, hn, qaB, qanB, qB, kvB, row576, cache, qCkv, q576, attnCkv, attnMla, fnB, gateB, upB, tmp, P, staging]) b.destroy();
    },
  };
}
