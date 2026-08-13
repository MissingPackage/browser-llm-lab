import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import {
  attnDecodeWgsl, attnDecodeRefWgsl, attnDecodeCombineWgsl, attnDecodeWorkgroupStorageBytes,
} from "../src/engine/kernels/wgsl";
import { q35AttnSplitPlan } from "../src/engine/q35attnsplit";

interface LegacyShape { nHead: number; nKvHead: number; headDim: number; ctxMax: number }
interface LegacyCase { opts: LegacyShape; batch: string; ref: string }
// Generato UNA VOLTA da `git show HEAD:src/engine/kernels/wgsl.ts` (vedi campo
// `source`): e' il kernel di IERI congelato su disco. Non si rigenera — se un
// giorno il ramo batch cambia, questo test deve diventare rosso.
const golden = JSON.parse(
  readFileSync(join(process.cwd(), "tests/fixtures/attn-decode-legacy.golden.json"), "utf8"),
) as LegacyCase & { source: string; ktest: LegacyCase };

// Il decode Qwen passa da "scores[ctxMax] in workgroup memory" a split +
// softmax in streaming. Qui si prova cio' che si puo' provare SENZA GPU: che il
// testo emesso ha la forma congelata (binding, dispatch, niente array in ctxMax),
// che il fabbisogno di workgroup storage e' COSTANTE in ctxMax, e che il ramo
// prefill (batch) e il riferimento del ktest sono ancora, byte per byte, il
// kernel di ieri. La matematica dei kernel la verificano ktest e conformance.

const ENGINE = { nHead: 16, nKvHead: 4, headDim: 256, ctxMax: 6400 };
const CTX_CASES = [525, 4096, 8192, 16384, 131072];

/** binding dichiarati, nell'ordine testuale di apparizione */
const bindings = (src: string): { binding: number; space: string; name: string; type: string }[] => {
  const out: { binding: number; space: string; name: string; type: string }[] = [];
  const re = /@group\(0\)\s*@binding\((\d+)\)\s*var<([^>]*)>\s*(\w+)\s*:\s*([^;]+);/g;
  for (let m = re.exec(src); m !== null; m = re.exec(src)) {
    out.push({ binding: Number(m[1]), space: m[2].replace(/\s+/g, " ").trim(), name: m[3], type: m[4].trim() });
  }
  return out;
};

describe("attnDecodeWgsl non-batch — split + softmax in streaming", () => {
  it("[a] niente array di workgroup dimensionato su ctxMax", () => {
    const src = attnDecodeWgsl(ENGINE);
    expect(src).not.toContain("scores: array<f32");
    // e piu' in generale: nessun array<..., ctxMax> in workgroup storage
    expect(src).not.toContain(`, ${ENGINE.ctxMax}>`);
  });

  it("[d] binding nell'ordine congelato 0..5", () => {
    const b = bindings(attnDecodeWgsl(ENGINE));
    expect(b.map((x) => x.binding)).toEqual([0, 1, 2, 3, 4, 5]);
    expect(b.map((x) => x.name)).toEqual(["q", "kCache", "vCache", "partOut", "partMS", "P"]);
    expect(b.map((x) => x.space)).toEqual([
      "storage, read", "storage, read", "storage, read",
      "storage, read_write", "storage, read_write", "uniform",
    ]);
    expect(b[0].type).toBe("array<vec4<f32>>");
    expect(b[1].type).toBe("array<vec4<f32>>");
    expect(b[2].type).toBe("array<vec4<f32>>");
    expect(b[3].type).toBe("array<vec4<f32>>");
    expect(b[4].type).toBe("array<f32>");
    expect(b[5].type).toBe("TokParams");
  });

  it("dispatch e geometria: workgroup_size 64, chunk su wid.y, plan condiviso col CPU-side", () => {
    const src = attnDecodeWgsl(ENGINE);
    const plan = q35AttnSplitPlan(ENGINE.ctxMax);
    expect(src).toContain("@compute @workgroup_size(64)");
    expect(src).toContain("let chunk = wid.y;");
    expect(src).toContain(`let pStart = chunk * ${plan.chunkLen}u;`);
    expect(src).toContain(`let pEnd = min(P.nPast + 1u, pStart + ${plan.chunkLen}u);`);
    // indicizzazione dei parziali: riga = h*splits + chunk
    expect(src).toContain(`let row = h * ${plan.splits}u + chunk;`);
    expect(src).toContain("partOut[row * HD4 + t] = acc;");
    expect(src).toContain("partMS[row * 2u] = m;");
    expect(src).toContain("partMS[row * 2u + 1u] = s;");
    // chunk vuoto: neutro nel combine
    expect(src).toContain("-3.0e38");
  });

  it("[e] vincoli di codegen: headDim non multiplo di 4, o HD4 > 64, sono errori", () => {
    expect(() => attnDecodeWgsl({ ...ENGINE, headDim: 6 })).toThrow();
    expect(() => attnDecodeWgsl({ ...ENGINE, headDim: 512 })).toThrow();
    expect(() => attnDecodeCombineWgsl({ nHead: 16, headDim: 6, ctxMax: 6400 })).toThrow();
    expect(() => attnDecodeCombineWgsl({ nHead: 16, headDim: 512, ctxMax: 6400 })).toThrow();
    // headDim 32 (shape del ktest) resta legale: HD4 = 8 < 64
    expect(() => attnDecodeWgsl({ nHead: 4, nKvHead: 2, headDim: 32, ctxMax: 64 })).not.toThrow();
  });

  it("con HD4 < 64 i thread in eccesso non scrivono ma attraversano le barriere", () => {
    const src = attnDecodeWgsl({ nHead: 4, nKvHead: 2, headDim: 32, ctxMax: 64 });
    expect(src).toContain("const HD4 = 8u;");
    // le barriere non stanno MAI dentro un if su t: control flow uniforme
    for (const line of src.split("\n")) {
      if (line.includes("workgroupBarrier()")) {
        expect(line.trim().startsWith("workgroupBarrier();"), `barriera non uniforme: ${line}`).toBe(true);
      }
    }
    // i guard esistono: le scritture su qsh/partOut sono condizionate a t < HD4
    expect(src).toContain("t < HD4");
  });
});

describe("attnDecodeCombineWgsl — riduzione log-sum-exp dei parziali", () => {
  it("[d] binding nell'ordine congelato 0..2, nessun uniform", () => {
    const b = bindings(attnDecodeCombineWgsl({ nHead: 16, headDim: 256, ctxMax: 6400 }));
    expect(b.map((x) => x.binding)).toEqual([0, 1, 2]);
    expect(b.map((x) => x.name)).toEqual(["partOut", "partMS", "out"]);
    expect(b.map((x) => x.space)).toEqual(["storage, read", "storage, read", "storage, read_write"]);
    expect(b.some((x) => x.space === "uniform")).toBe(false);
  });

  it("gli splits combinati sono quelli del piano CPU-side", () => {
    for (const ctxMax of CTX_CASES) {
      const src = attnDecodeCombineWgsl({ nHead: 16, headDim: 256, ctxMax });
      expect(src).toContain(`const S = ${q35AttnSplitPlan(ctxMax).splits}u;`);
    }
    expect(attnDecodeCombineWgsl({ nHead: 16, headDim: 256, ctxMax: 6400 }))
      .toContain("@compute @workgroup_size(64)");
  });
});

describe("attnDecodeWorkgroupStorageBytes — costante in ctxMax", () => {
  it("[b] stesso valore a 4096 e 131072, e sempre sotto il default di spec", () => {
    expect(attnDecodeWorkgroupStorageBytes(4096)).toBe(attnDecodeWorkgroupStorageBytes(131072));
    for (const ctxMax of CTX_CASES) {
      expect(attnDecodeWorkgroupStorageBytes(ctxMax), `ctxMax ${ctxMax}`).toBeLessThan(16384);
    }
  });

  it("[b] la forma e' 16·ceil(headDim/4) + 512, headDim default 256 (famiglia Qwen)", () => {
    expect(attnDecodeWorkgroupStorageBytes(6400)).toBe(16 * 64 + 512);
    expect(attnDecodeWorkgroupStorageBytes(6400, 256)).toBe(attnDecodeWorkgroupStorageBytes(6400));
    expect(attnDecodeWorkgroupStorageBytes(64, 32)).toBe(16 * 8 + 512);
  });
});

describe("prefill e riferimento del ktest — il kernel di ieri, byte per byte", () => {
  it("[c] attnDecodeWgsl({batch:true}) === golden.batch", () => {
    expect(attnDecodeWgsl({ ...golden.opts, batch: true })).toBe(golden.batch);
    expect(golden.batch).toContain("scores: array<f32"); // il golden e' davvero il vecchio kernel
  });

  it("[c] attnDecodeRefWgsl() === golden.ref", () => {
    expect(attnDecodeRefWgsl(golden.opts)).toBe(golden.ref);
    expect(golden.ref).toContain("scores: array<f32");
  });

  it("[c] anche sulla shape del ktest (headDim 32) i due rami sono invariati", () => {
    expect(attnDecodeWgsl({ ...golden.ktest.opts, batch: true })).toBe(golden.ktest.batch);
    expect(attnDecodeRefWgsl(golden.ktest.opts)).toBe(golden.ktest.ref);
  });

  it("batch e ref escono dallo STESSO template: differiscono solo per il binding 4 e l'offset di riga", () => {
    const b = bindings(attnDecodeWgsl({ ...golden.opts, batch: true }));
    const r = bindings(attnDecodeRefWgsl(golden.opts));
    expect(b.map((x) => x.name)).toEqual(["q", "kCache", "vCache", "out", "rowPast"]);
    expect(r.map((x) => x.name)).toEqual(["q", "kCache", "vCache", "out", "P"]);
    // stesso numero di righe: e' un template solo, non due copie divergenti
    expect(golden.batch.split("\n").length).toBe(golden.ref.split("\n").length);
  });
});
