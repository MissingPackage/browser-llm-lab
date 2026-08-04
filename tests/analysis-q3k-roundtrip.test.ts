// ANALISI slice 4c-A, gate (1) e (2) del design §9 — gira solo con Q3K_ROUNDTRIP=1
// perché serve il GGUF da 17 GB e la sua ri-quantizzazione Q3_K da 13 GB.
//
// Gate (1) R1: dequant(quantize(x)) del NOSTRO quantizzatore confrontato con
// llama-quantize --allow-requantize --pure del checkout C1, su un tensore expert
// VERO (blk.5.ffn_gate_exps expert 0 = 3 145 728 pesi). Byte-identico se possibile,
// altrimenti RMS relativa dichiarata.
// Gate (2): dequantQ3_K/dequantQ2_K contro dequantize_row_q3_K/q2_K dell'oracolo
// (vettori generati da tools/oracle-moe/kqref, che linka libggml del checkout).
//
// Effetto collaterale voluto: scrive tests/fixtures/glm-q3k-q2k-blocks.json — la
// fixture PINNATA (R1: «si pinna il NOSTRO output come riferimento con test») su
// cui gira il test veloce engine-q3k-quant, che non ha bisogno di nessun file da GB.
import { describe, it, expect } from "vitest";
import { existsSync, openSync, readSync, closeSync, writeFileSync, mkdirSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { parseGguf, type GgufTensorInfo } from "../src/engine/gguf";
import { GLM47_FLASH_SHA256 } from "../src/engine/shape";
import {
  dequantQ4_0, quantizeQ3_K, dequantQ3_K, quantizeQ2_K, dequantQ2_K,
  Q3_K_BLOCK_BYTES, Q2_K_BLOCK_BYTES, QK_K,
} from "../src/engine/quant";

const GGUF_SRC = join(homedir(), ".cache/blab-models/GLM-4.7-Flash-Q4_0.gguf");
// prodotto da: llama-quantize --allow-requantize --pure <src> <dst> Q3_K 24
const GGUF_Q3K = process.env.Q3K_GGUF ?? join(homedir(), ".cache/blab-models/tmp-4cA-q3k-pure.gguf");
const KQREF = join(process.cwd(), "tools/oracle-moe/kqref");
const FIXTURE = join(process.cwd(), "tests/fixtures/glm-q3k-q2k-blocks.json");
const REPORT = join(process.cwd(), "results/engine/q3k-roundtrip-2026-08-04.json");

const TENSOR = "blk.5.ffn_gate_exps.weight"; // Q4_0 puro (blk.5 è fuori dalla classe q4_1)
const EXPERT_WEIGHTS = 1536 * 2048; // 3 145 728 — un tensore expert

// Casi della fixture. UN solo tensore da 8 superblocchi era una fixture CIECA:
// due mutazioni semantiche di makeQkx2Quants (il ciclo `is <= nstep` accorciato a
// `is < nstep`, e il criterio MAD sostituito da quello quadratico) producevano
// gli stessi 8 superblocchi e passavano. Il secondo caso — 64 superblocchi da un
// tensore DIVERSO — le separa entrambe (primi superblocchi divergenti: 0 e 3).
const CASES = [
  { name: "blk.5 gate e0", tensor: TENSOR, expert: 0, blocks: 8 },
  { name: "blk.20 up e7", tensor: "blk.20.ffn_up_exps.weight", expert: 7, blocks: 64 },
];

const b64 = (a: ArrayBufferView): string =>
  Buffer.from(a.buffer, a.byteOffset, a.byteLength).toString("base64");

function headerOf(path: string): { f: ReturnType<typeof parseGguf>; fd: number } {
  const fd = openSync(path, "r");
  const HEADER = 64 * 1024 * 1024;
  const hbuf = Buffer.alloc(HEADER);
  readSync(fd, hbuf, 0, HEADER, 0);
  const f = parseGguf(hbuf.buffer.slice(hbuf.byteOffset, hbuf.byteOffset + HEADER));
  return { f, fd };
}

function readRange(fd: number, off: number, len: number): Uint8Array {
  const b = Buffer.alloc(len);
  const got = readSync(fd, b, 0, len, off);
  if (got !== len) throw new Error(`read corta: ${got}/${len} @${off}`);
  return new Uint8Array(b.buffer, b.byteOffset, len);
}

describe.skipIf(!process.env.Q3K_ROUNDTRIP || !existsSync(GGUF_SRC))(
  "q3k/q2k round-trip vs oracolo (gate 4c-A)", () => {
    it("tensore expert vero: quantize/dequant vs llama-quantize e vs ggml", () => {
      // --- sorgente: tensori expert VERI, Q4_0 → f32
      const { f: fSrc, fd: fdSrc } = headerOf(GGUF_SRC);
      const q40Bytes = (EXPERT_WEIGHTS / 32) * 18;
      const expertF32 = (name: string, expert: number): Float32Array => {
        const info = fSrc.tensors.find((t) => t.name === name) as GgufTensorInfo;
        expect(info, `${name} assente`).toBeTruthy();
        expect(info.dims.slice(0, 2)).toEqual([2048, 1536]);
        const raw = readRange(fdSrc, fSrc.dataOffset + info.offset + expert * q40Bytes, q40Bytes);
        const out = new Float32Array(EXPERT_WEIGHTS);
        dequantQ4_0(raw, 0, EXPERT_WEIGHTS / 32, out);
        return out;
      };
      const x = expertF32(TENSOR, 0);

      const nb = EXPERT_WEIGHTS / QK_K; // 12288 superblocchi
      const ourQ3 = new Uint8Array(nb * Q3_K_BLOCK_BYTES);
      const ourQ2 = new Uint8Array(nb * Q2_K_BLOCK_BYTES);

      // --- costo di quantizzazione (gate 6): mediana di 3 passate sul tensore vero
      const msQ3: number[] = [], msQ2: number[] = [];
      for (let r = 0; r < 3; r++) {
        let t = performance.now();
        quantizeQ3_K(x, 0, nb, ourQ3, 0);
        msQ3.push(performance.now() - t);
        t = performance.now();
        quantizeQ2_K(x, 0, nb, ourQ2, 0);
        msQ2.push(performance.now() - t);
      }
      msQ3.sort((a, b) => a - b); msQ2.sort((a, b) => a - b);
      const quantMsQ3 = msQ3[1], quantMsQ2 = msQ2[1];

      // --- gate (1): confronto byte con llama-quantize --pure
      let vsLlamaQuantize: Record<string, unknown> = { run: false, reason: `${GGUF_Q3K} assente` };
      if (existsSync(GGUF_Q3K)) {
        const { f: fQ3, fd: fdQ3 } = headerOf(GGUF_Q3K);
        const infoQ3 = fQ3.tensors.find((t) => t.name === TENSOR) as GgufTensorInfo;
        expect(infoQ3, `${TENSOR} assente nel GGUF Q3_K`).toBeTruthy();
        expect(infoQ3.type, "il tensore non è stato convertito a Q3_K (ggml_type 11)").toBe(11);
        const ref = readRange(fdQ3, fQ3.dataOffset + infoQ3.offset, nb * Q3_K_BLOCK_BYTES);
        closeSync(fdQ3);
        let diffBytes = 0, firstDiff = -1;
        for (let i = 0; i < ref.length; i++) {
          if (ref[i] !== ourQ3[i]) { diffBytes++; if (firstDiff < 0) firstDiff = i; }
        }
        // se non è byte-identico: RMS relativa sui valori dequantizzati (R1)
        let rmsRel = 0;
        if (diffBytes) {
          const a = new Float32Array(EXPERT_WEIGHTS), b = new Float32Array(EXPERT_WEIGHTS);
          dequantQ3_K(ourQ3, 0, nb, a); dequantQ3_K(ref, 0, nb, b);
          let se = 0, ss = 0;
          for (let i = 0; i < EXPERT_WEIGHTS; i++) { const d = a[i] - b[i]; se += d * d; ss += b[i] * b[i]; }
          rmsRel = Math.sqrt(se / EXPERT_WEIGHTS) / Math.sqrt(ss / EXPERT_WEIGHTS);
        }
        vsLlamaQuantize = {
          run: true, tensor: TENSOR, bytes: ref.length, diffBytes, firstDiff, rmsRel,
          byteIdentical: diffBytes === 0,
        };
        expect(diffBytes, `byte diversi da llama-quantize (primo @${firstDiff})`).toBe(0);
      }

      // --- errore della doppia quantizzazione Q4_0→f32→Q3_K/Q2_K (caveat R8)
      const deqQ3 = new Float32Array(EXPERT_WEIGHTS);
      const deqQ2 = new Float32Array(EXPERT_WEIGHTS);
      dequantQ3_K(ourQ3, 0, nb, deqQ3);
      dequantQ2_K(ourQ2, 0, nb, deqQ2);
      const relErr = (y: Float32Array): number => {
        let se = 0, ss = 0;
        for (let i = 0; i < EXPERT_WEIGHTS; i++) { const d = y[i] - x[i]; se += d * d; ss += x[i] * x[i]; }
        return Math.sqrt(se / ss);
      };
      const rmsRelQ3 = relErr(deqQ3), rmsRelQ2 = relErr(deqQ2);

      // --- gate (2): vettori di riferimento dall'oracolo, per ogni caso
      const dir = join(tmpdir(), "blab-q3k");
      mkdirSync(dir, { recursive: true });
      const diffB = (a: Uint8Array, b: Uint8Array): number => {
        let n = 0; for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) n++; return n;
      };
      const diffF = (a: Float32Array, b: Float32Array): number => {
        let n = 0; for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) n++; return n;
      };
      // Riferimento dell'oracolo su `blocks` superblocchi di dati f32 arbitrari.
      const oracle = (src: Float32Array, blocks: number): {
        q3: Uint8Array; d3: Float32Array; q2: Uint8Array; d2: Float32Array;
      } => {
        const sample = src.subarray(0, blocks * QK_K);
        const fin = join(dir, "in.f32"), fout = join(dir, "out.bin");
        writeFileSync(fin, Buffer.from(sample.buffer, sample.byteOffset, sample.byteLength));
        execFileSync(KQREF, [fin, fout], { stdio: "pipe" });
        const fdo = openSync(fout, "r");
        const buf = readRange(fdo, 0,
          blocks * (Q3_K_BLOCK_BYTES + Q2_K_BLOCK_BYTES + 2 * QK_K * 4));
        closeSync(fdo);
        let p = 0;
        const q3 = buf.slice(p, p += blocks * Q3_K_BLOCK_BYTES);
        const d3 = new Float32Array(buf.slice(p, p += blocks * QK_K * 4).buffer);
        const q2 = buf.slice(p, p += blocks * Q2_K_BLOCK_BYTES);
        const d2 = new Float32Array(buf.slice(p, p += blocks * QK_K * 4).buffer);
        return { q3, d3, q2, d2 };
      };

      // campione ampio sul primo tensore (256 superblocchi): il gate (2)
      const SAMPLE_BLOCKS = 256;
      const wide = oracle(x, SAMPLE_BLOCKS);
      const ourD3 = new Float32Array(SAMPLE_BLOCKS * QK_K);
      const ourD2 = new Float32Array(SAMPLE_BLOCKS * QK_K);
      dequantQ3_K(ourQ3, 0, SAMPLE_BLOCKS, ourD3);
      dequantQ2_K(ourQ2, 0, SAMPLE_BLOCKS, ourD2);
      const vsGgml = {
        sampleBlocks: SAMPLE_BLOCKS, tensor: TENSOR,
        q3QuantDiffBytes: diffB(wide.q3, ourQ3.subarray(0, wide.q3.length)),
        q2QuantDiffBytes: diffB(wide.q2, ourQ2.subarray(0, wide.q2.length)),
        q3DequantDiffFloats: diffF(wide.d3, ourD3),
        q2DequantDiffFloats: diffF(wide.d2, ourD2),
      };
      expect(vsGgml.q3QuantDiffBytes, "quantizeQ3_K ≠ quantize_q3_K").toBe(0);
      expect(vsGgml.q2QuantDiffBytes, "quantizeQ2_K ≠ quantize_q2_K").toBe(0);
      expect(vsGgml.q3DequantDiffFloats, "dequantQ3_K ≠ dequantize_row_q3_K").toBe(0);
      expect(vsGgml.q2DequantDiffFloats, "dequantQ2_K ≠ dequantize_row_q2_K").toBe(0);

      // --- fixture pinnata (tutti i riferimenti vengono dall'ORACOLO, non da noi)
      const cases = CASES.map((c) => {
        const src = c.tensor === TENSOR && c.expert === 0 ? x : expertF32(c.tensor, c.expert);
        const ref = oracle(src, c.blocks);
        // il nostro output DEVE già coincidere: la fixture non pinna un bug
        const q3 = new Uint8Array(c.blocks * Q3_K_BLOCK_BYTES);
        const q2 = new Uint8Array(c.blocks * Q2_K_BLOCK_BYTES);
        quantizeQ3_K(src, 0, c.blocks, q3, 0);
        quantizeQ2_K(src, 0, c.blocks, q2, 0);
        expect(diffB(ref.q3, q3), `${c.name}: Q3_K ≠ oracolo`).toBe(0);
        expect(diffB(ref.q2, q2), `${c.name}: Q2_K ≠ oracolo`).toBe(0);
        return {
          name: c.name, tensor: c.tensor, expert: c.expert, blocks: c.blocks,
          weights: c.blocks * QK_K,
          srcF32: b64(src.subarray(0, c.blocks * QK_K)),
          q3k: b64(ref.q3), q3kDequant: b64(ref.d3),
          q2k: b64(ref.q2), q2kDequant: b64(ref.d2),
        };
      });
      closeSync(fdSrc);
      writeFileSync(FIXTURE, JSON.stringify({
        kind: "glm-q3k-q2k-blocks", schemaVersion: 2,
        source: { gguf: "GLM-4.7-Flash-Q4_0.gguf", sha256: GLM47_FLASH_SHA256 },
        oracle: { impl: "tools/oracle-moe/kqref", llamaCppCommit: "5f55650a78f92aff4d48d671423e888fac0469ff" },
        note: "due tensori DIVERSI: con un solo caso da 8 superblocchi la fixture era cieca "
          + "a due mutazioni semantiche di makeQkx2Quants (is<=nstep→is<nstep, MAD→quadratico)",
        cases,
      }, null, 1));

      const report = {
        kind: "q3k-roundtrip", schemaVersion: 1, date: new Date().toISOString(),
        tensor: TENSOR, weights: EXPERT_WEIGHTS, superblocks: nb,
        vsLlamaQuantize, vsGgml,
        doubleQuantization: {
          note: "Q4_0 → f32 → Q3_K/Q2_K: errore COMPOSTO, non confrontabile con le tabelle pubbliche (design R8)",
          rmsRelQ3, rmsRelQ2,
        },
        quantizeCost: {
          msPerTensorQ3K: quantMsQ3, msPerTensorQ2K: quantMsQ2,
          runsQ3K: msQ3, runsQ2K: msQ2,
          weightsPerTensor: EXPERT_WEIGHTS,
          projectionPmax1024: {
            tensorsPerExpert: 3, experts: 1024,
            secondsQ3K: (quantMsQ3 * 3 * 1024) / 1000,
            secondsQ2K: (quantMsQ2 * 3 * 1024) / 1000,
            note: "single-thread; l'import di §6 può usare un pool di worker (non cambia i byte)",
          },
        },
      };
      // Il report R1 si SOVRASCRIVE solo se il gate è stato davvero rieseguito:
      // senza il GGUF Q3_K (13 GB, cancellato dopo la chiusura del gate) una
      // riesecuzione lo azzererebbe, cancellando l'evidenza invece di rifarla.
      if (vsLlamaQuantize.run) writeFileSync(REPORT, JSON.stringify(report, null, 1));
      // eslint-disable-next-line no-console
      console.log(vsLlamaQuantize.run
        ? JSON.stringify(report, null, 1)
        : `R1 NON rieseguito (${GGUF_Q3K} assente): ${REPORT} lasciato intatto. `
          + `Per rifarlo: llama-quantize --allow-requantize --pure <src> ${GGUF_Q3K} Q3_K 24`);
    }, 60 * 60_000);
  });
