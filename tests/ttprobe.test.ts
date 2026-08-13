// Test del banco delle sonde di engine-ttft riga 1 (src/microbench/tt*.ts).
//
// Non misura niente: verifica le due cose che, sbagliate, avrebbero fatto
// buttare una run GPU — il conto del workgroup storage LETTO dal testo (che e'
// il done-when d) e l'invarianza strutturale delle varianti generate.
import { describe, expect, it } from "vitest";
import { attnDecodeWgsl, gemvQuantWgsl } from "../src/engine/kernels/wgsl";
import {
  gemmDenseF32Wgsl, gemmQ4MultiRowRegsWgsl, gemmQ4MultiRowSharedWgsl,
  gemmQ4MultiRowSplitKWgsl, workgroupStorageBytes,
} from "../src/microbench/ttGemm";
import { TT_ATTN_SHAPE, attnPrefillGrid, attnPrefillStreamWgsl } from "../src/microbench/ttAttn";

describe("workgroupStorageBytes: LETTO dal testo, non dedotto", () => {
  it("conta il legacy dell'attenzione come 4*ctxMax + 256 (la formula di gpulimits)", () => {
    const code = attnDecodeWgsl({ ...TT_ATTN_SHAPE, batch: true });
    expect(code).toContain(`var<workgroup> scores: array<f32, ${TT_ATTN_SHAPE.ctxMax}>;`);
    expect(workgroupStorageBytes(code)).toBe(4 * TT_ATTN_SHAPE.ctxMax + 256);
  });

  it("conta la forma attuale del gemv come i soli partial[64]", () => {
    expect(workgroupStorageBytes(gemvQuantWgsl({ kind: "q4_0", K: 2560, N: 9216, hasBias: false, batch: true }))).toBe(256);
  });

  it("conta la GEMM densa come due tile da 64x8 f32, costante nelle shape", () => {
    expect(workgroupStorageBytes(gemmDenseF32Wgsl(4096, 4096, 4096))).toBe(4096);
    expect(workgroupStorageBytes(gemmDenseF32Wgsl(6336, 2560, 9216))).toBe(4096);
  });

  it("allinea i vec4 a 16 B", () => {
    expect(workgroupStorageBytes("var<workgroup> a: array<f32, 1>;\nvar<workgroup> b: vec4<f32>;")).toBe(32);
  });
});

describe("moltiplicatore multi-riga: shared scala con M, non coi pesi", () => {
  const K = 2560;
  const N = 9216;

  it("'regs' tiene in workgroup memory le sole attivazioni M x 64", () => {
    for (const M of [1, 8, 16, 32]) {
      expect(workgroupStorageBytes(gemmQ4MultiRowRegsWgsl({ K, N, M }))).toBe(M * 256);
    }
  });

  it("'regs' a M=16 sta sotto il pavimento di spec di 16.384 B", () => {
    expect(workgroupStorageBytes(gemmQ4MultiRowRegsWgsl({ K, N, M: 16 }))).toBeLessThanOrEqual(16384);
  });

  it("'shared' paga i pesi in workgroup memory (4.096 B costanti) piu' le attivazioni", () => {
    for (const M of [8, 16, 32]) {
      expect(workgroupStorageBytes(gemmQ4MultiRowSharedWgsl({ K, N, M }))).toBe(4096 + 128 + M * 256);
    }
  });

  it("'splitk' ha lo stesso shared di 'regs' (e' 'regs' con K spezzato)", () => {
    expect(workgroupStorageBytes(gemmQ4MultiRowSplitKWgsl({ K, N, M: 16, splits: 4 })))
      .toBe(workgroupStorageBytes(gemmQ4MultiRowRegsWgsl({ K, N, M: 16 })));
  });

  it("il corpo di dequantizzazione e' quello misurato in fase 0, non una parente", () => {
    // stesso pairing lo/hi sfalsato di 4 vec4 di src/microbench/kdGemv.ts
    for (const code of [gemmQ4MultiRowRegsWgsl({ K, N, M: 16 }), gemmQ4MultiRowSplitKWgsl({ K, N, M: 16, splits: 4 })]) {
      expect(code).toContain("bd = bd + dot(lo[wi], xs[xo + wi]) + dot(hi[wi], xs[xo + 4u + wi]);");
    }
  });

  it("rifiuta le shape che non sa fare invece di generare un kernel non misurato", () => {
    expect(() => gemmQ4MultiRowRegsWgsl({ K: 96, N: 64, M: 16 })).toThrow(/multiplo di 64/);
    expect(() => gemmQ4MultiRowSplitKWgsl({ K: 2560, N: 9216, M: 16, splits: 3 })).toThrow(/fette/);
    expect(() => gemmDenseF32Wgsl(4096, 4096, 100)).toThrow(/allineata/);
  });
});

describe("attenzione a chunk del prefill: costante in ctxMax", () => {
  it("nessuna variante streaming dichiara scores[ctxMax]", () => {
    for (const o of [{ headsPerWg: 1, rowsPerWg: 1 }, { headsPerWg: 4, rowsPerWg: 1 }, { headsPerWg: 4, rowsPerWg: 2 }] as const) {
      const code = attnPrefillStreamWgsl(o);
      expect(code).not.toContain(`array<f32, ${TT_ATTN_SHAPE.ctxMax}>`);
    }
  });

  it("il workgroup storage cresce con headsPerWg x rowsPerWg e sta sotto i 16.384", () => {
    const cases: Array<[{ headsPerWg: 1 | 4; rowsPerWg: 1 | 2 }, number]> = [
      [{ headsPerWg: 1, rowsPerWg: 1 }, 1536],
      [{ headsPerWg: 4, rowsPerWg: 1 }, 6144],
      [{ headsPerWg: 4, rowsPerWg: 2 }, 12288],
    ];
    for (const [o, want] of cases) {
      expect(workgroupStorageBytes(attnPrefillStreamWgsl(o))).toBe(want);
      expect(want).toBeLessThanOrEqual(16384);
    }
  });

  it("la griglia copre tutte le righe del chunk e tutte le head", () => {
    expect(attnPrefillGrid({ headsPerWg: 1, rowsPerWg: 1 }, 16)).toEqual([16, 1, 16]);
    expect(attnPrefillGrid({ headsPerWg: 4, rowsPerWg: 1 }, 16)).toEqual([4, 1, 16]);
    expect(attnPrefillGrid({ headsPerWg: 4, rowsPerWg: 2 }, 16)).toEqual([4, 1, 8]);
  });

  it("ogni riga del chunk porta il SUO nPast (la causalita' arriva da rowPast)", () => {
    const code = attnPrefillStreamWgsl({ headsPerWg: 4, rowsPerWg: 2 });
    expect(code).toContain("let n0 = rowPast[rBase + 0u] + 1u;");
    expect(code).toContain("let n1 = rowPast[rBase + 1u] + 1u;");
    expect(code).toContain("let nMax = max(n0, n1);");
  });
});
