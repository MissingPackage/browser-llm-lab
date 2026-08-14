// Test del banco delle sonde di engine-ttft riga 1 (src/microbench/tt*.ts).
//
// Non misura niente: verifica le due cose che, sbagliate, avrebbero fatto
// buttare una run GPU — il conto del workgroup storage LETTO dal testo (che e'
// il done-when d) e l'invarianza strutturale delle varianti generate.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { attnDecodeLegacyBatchWgsl, attnDecodeWgsl, gemvQuantWgsl } from "../src/engine/kernels/wgsl";
import {
  gemmDenseF32Wgsl, gemmQ4MultiRowRegsWgsl, gemmQ4MultiRowSharedWgsl,
  gemmQ4MultiRowSplitKWgsl, workgroupStorageBytes,
} from "../src/microbench/ttGemm";
import { TT_ATTN_SHAPE, attnPrefillGrid, attnPrefillStreamWgsl } from "../src/microbench/ttAttn";

describe("workgroupStorageBytes: LETTO dal testo, non dedotto", () => {
  it("conta il legacy dell'attenzione come 4*ctxMax + 256 (la formula di gpulimits)", () => {
    // il LEGACY, che dal task T1-kernel-batch-streaming non e' piu' cio' che
    // `attnDecodeWgsl({batch:true})` emette: e' il fallback dichiarato, e resta
    // il termine di paragone della sonda (la baseline che le candidate battono)
    const code = attnDecodeLegacyBatchWgsl(TT_ATTN_SHAPE);
    expect(code).toContain(`var<workgroup> scores: array<f32, ${TT_ATTN_SHAPE.ctxMax}>;`);
    expect(workgroupStorageBytes(code)).toBe(4 * TT_ATTN_SHAPE.ctxMax + 256);
  });

  it("il ramo batch di oggi non e' piu' quel legacy: 1.536 B costanti in ctxMax", () => {
    const code = attnDecodeWgsl({ ...TT_ATTN_SHAPE, batch: true });
    expect(code).not.toContain("scores: array<f32");
    expect(workgroupStorageBytes(code)).toBe(1_536);
    expect(workgroupStorageBytes(attnDecodeWgsl({ ...TT_ATTN_SHAPE, ctxMax: 131_072, batch: true })))
      .toBe(1_536);
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

// ---------------------------------------------------------------------------
// LA BASELINE DELLA SONDA DEVE RESTARE IL LEGACY.
//
// `src/microbench/tt*.ts` importa deliberatamente la forma di PRODUZIONE invece
// di imitarla (ttAttn.ts, riga 4: «la forma attuale non vive qui»). Ottima
// scelta finche' quella forma e' la baseline — ma dal task
// T1-kernel-batch-streaming `attnDecodeWgsl({ batch: true })` E' la candidata
// vincente, non piu' la baseline. Se un sito della sonda continua a chiamarlo
// per fabbricare il braccio `legacy`, la sonda confronta streaming contro
// streaming e non se ne accorge nessuno: il numero esce lo stesso, e' solo
// falso.
//
// Questi test sono l'unico sensore di quel guasto: la sonda gira solo su GPU,
// e il suo esito e' un JSON che nessun gate rilegge.
// ---------------------------------------------------------------------------
const microbenchSrc = (f: string): string =>
  readFileSync(join(process.cwd(), "src/microbench", f), "utf8");
/**
 * Solo il CODICE: i commenti sono liberi di raccontare cosa faceva la baseline
 * prima (ed e' bene che lo raccontino) — cio' che non deve piu' esistere e' una
 * CHIAMATA.
 */
const codeOnly = (src: string): string =>
  src.replace(/\/\*[\s\S]*?\*\//g, " ").replace(/^\s*\/\/.*$/gm, "");

describe("sonda della riga 1: il braccio `legacy` fabbrica DAVVERO il legacy", () => {
  const FILES = ["ttAttn.ts", "ttRunner.ts"];

  it("nessun file della sonda costruisce piu' un kernel con attnDecodeWgsl(batch: true)", () => {
    for (const f of FILES) {
      const src = microbenchSrc(f);
      const hits = [...codeOnly(src).matchAll(/attnDecodeWgsl\([^)]*batch:\s*true/g)];
      expect(hits.length, `${f}: ${hits.length} chiamate a attnDecodeWgsl con batch: true`).toBe(0);
      // e nemmeno lo NOMINA come "forma attuale": il nome privato del template
      // legacy e la riga del vecchio switch sono due bugie che restano nel JSON
      // (qui il commento NON e' scusato: e' proprio quel testo che finisce nel
      // campo `ctx` dell'artefatto, e che un lettore prende per buono)
      expect(src, `${f} nomina ancora attnDecodeLegacyWgsl / wgsl.ts:530`)
        .not.toMatch(/attnDecodeLegacyWgsl|wgsl\.ts:530/);
    }
  });

  it("ttRunner costruisce il codice della baseline da attnDecodeLegacyBatchWgsl, in ENTRAMBI i siti", () => {
    const src = microbenchSrc("ttRunner.ts");
    // due siti: lo sweep del limite negoziabile e il banco dell'attenzione
    const legacyDecls = [...src.matchAll(/const legacyCode = (\w+)\(/g)].map((m) => m[1]);
    expect(legacyDecls.length, "siti che fabbricano il codice della baseline").toBe(2);
    expect(new Set(legacyDecls)).toEqual(new Set(["attnDecodeLegacyBatchWgsl"]));
  });

  it("cio' che il braccio `legacy` promette nel JSON e' cio' che il suo codice contiene", () => {
    const src = microbenchSrc("ttRunner.ts");
    // la stringa `ctx` finisce nell'artefatto come DESCRIZIONE della baseline:
    // se promette scores[ctxMax] e letture scalari, il codice deve averli
    const ctx = /id: "legacy",[\s\S]{0,400}?ctx: `([^`]*)`/.exec(src)?.[1];
    expect(ctx, "braccio `legacy` con la sua stringa ctx").toBeTruthy();
    expect(ctx).toContain("scores[ctxMax=");
    const code = attnDecodeLegacyBatchWgsl(TT_ATTN_SHAPE);
    expect(code).toContain(`var<workgroup> scores: array<f32, ${TT_ATTN_SHAPE.ctxMax}>;`);
    // ...e la forma di oggi NON li ha: e' precisamente cio' che la sonda misura
    expect(attnDecodeWgsl({ ...TT_ATTN_SHAPE, batch: true })).not.toContain("scores: array<f32");
  });

  it("lo sweep del limite negoziabile misura ancora una pressione VERA sul workgroup storage", () => {
    // `requestedLimits` esiste per vedere se il tetto concesso morde: se la
    // baseline scende a 1.536 B, ogni limite dello sweep e' soddisfatto e
    // l'esperimento non misura piu' niente
    const src = microbenchSrc("ttRunner.ts");
    const lim = /requestedLimits: \[([\d, ]+)\]/.exec(src)?.[1].split(",").map((s) => Number(s.trim()));
    expect(lim, "requestedLimits dello sweep").toBeTruthy();
    const need = workgroupStorageBytes(attnDecodeLegacyBatchWgsl(TT_ATTN_SHAPE));
    expect(need).toBe(4 * TT_ATTN_SHAPE.ctxMax + 256);
    expect(Math.min(...lim!), "il limite piu' basso deve NON bastare alla baseline").toBeLessThan(need);
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
