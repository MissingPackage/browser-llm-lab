// NUCLEO PARAMETRICO DEI GEMV K-QUANT (spec 2026-08-17-q2k-q3k-kernels, T2/T3).
//
// Perche' questo file esiste. I gemv K-quant erano TRE funzioni scritte a mano;
// aggiungere Q2_K e Q3_K a mano sarebbe stata la quarta e la quinta copia, e il
// ruling di riuso del progetto dice che la ripetizione si fattorizza. La
// fattorizzazione pero' tocca kernel che girano in produzione, quindi il gate
// non e' "i test passano": e' che il TESTO WGSL dei tre formati esistenti sia
// identico CARATTERE PER CARATTERE a quello di prima. Le stringhe di
// riferimento in `tests/fixtures/kquant-core/` sono state catturate eseguendo i
// generatori PRIMA di toccare `wgsl.ts`: sono il reperto, non un'aspettativa
// riscritta a posteriori.
//
// Niente GPU qui: il testo emesso e' una proprieta' del generatore, e si
// verifica dove costa zero.
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  gemvQ2KWgsl, gemvQ3KWgsl, gemvQ4KWgsl, gemvQ5KWgsl, gemvQ6KWgsl,
  gemvKQuantWgsl, KQUANT_GEMV_DESC, kquantWorkSplit,
  type KQuantGemvKind,
  PREFILL_GEMM_KINDS, prefillGemmWiring,
  prefillGemmQ2KSplitKIdotWgsl, prefillGemmQ2KSplitKWgsl,
  prefillGemmQ3KSplitKIdotWgsl, prefillGemmQ3KSplitKWgsl,
} from "../src/engine/kernels/wgsl";
import {
  dequantQ2_K, dequantQ3_K, repackKQuant, Q2_K_BLOCK_BYTES, Q3_K_BLOCK_BYTES,
} from "../src/engine/quant";

/** Gli stessi parametri d'arena usati dallo script di cattura delle fixture. */
const ARENA = { nBuf: 3, slabWords: 4096, slabsPerBuf: 8, tensorWords: 1152 };

/** Le shape di produzione: gate/up degli expert, down degli expert, attn del 35B. */
const SHAPES = [
  { K: 2048, N: 512 },
  { K: 512, N: 2048 },
  { K: 2048, N: 4096 },
];

type Opts = { K: number; N: number; arena?: typeof ARENA; accum?: boolean; batch?: boolean };
const VARIANTS: Record<string, (s: { K: number; N: number }) => Opts> = {
  plain: (s) => ({ ...s }),
  batch: (s) => ({ ...s, batch: true }),
  arena: (s) => ({ ...s, arena: ARENA }),
  "arena-accum": (s) => ({ ...s, arena: ARENA, accum: true }),
};

/**
 * Le varianti LEGALI per formato. Il q5_K ne ha due e non e' una svista: la sua
 * firma congelata e' `{K, N, batch?}` — non ha mai avuto il regime d'arena,
 * perche' non e' un formato di expert.
 */
const LEGAL: Record<string, string[]> = {
  q4_K: ["plain", "batch", "arena", "arena-accum"],
  q5_K: ["plain", "batch"],
  q6_K: ["plain", "batch", "arena", "arena-accum"],
};

const GEN: Record<string, (o: Opts) => string> = {
  q2_K: gemvQ2KWgsl as (o: Opts) => string,
  q3_K: gemvQ3KWgsl as (o: Opts) => string,
  q4_K: gemvQ4KWgsl as (o: Opts) => string,
  q5_K: gemvQ5KWgsl as (o: Opts) => string,
  q6_K: gemvQ6KWgsl as (o: Opts) => string,
};

describe("[a] identita' bit per bit dei tre gemv K-quant migrati", () => {
  const cases: { fmt: string; variant: string; K: number; N: number }[] = [];
  for (const fmt of ["q4_K", "q5_K", "q6_K"]) {
    for (const variant of LEGAL[fmt]) {
      for (const s of SHAPES) cases.push({ fmt, variant, K: s.K, N: s.N });
    }
  }

  it.each(cases)(
    "$fmt $variant K=$K N=$N: WGSL identico alla stringa catturata prima della fattorizzazione",
    ({ fmt, variant, K, N }) => {
      const atteso = readFileSync(
        `tests/fixtures/kquant-core/${fmt}-${variant}-K${K}-N${N}.wgsl`, "utf8",
      );
      expect(GEN[fmt](VARIANTS[variant]({ K, N }))).toBe(atteso);
    },
  );

  it("le fixture ci sono tutte (una cattura mancante renderebbe il gate vuoto)", () => {
    expect(cases.length).toBe(30);
  });
});

describe("[b] un solo generatore: i cinque wrapper delegano al nucleo", () => {
  const cases: { fmt: KQuantGemvKind; variant: string; K: number; N: number }[] = [];
  for (const fmt of ["q2_K", "q3_K", "q4_K", "q5_K", "q6_K"] as KQuantGemvKind[]) {
    // q2_K/q3_K/q4_K/q6_K hanno tutte e quattro le varianti; il q5_K due.
    for (const variant of LEGAL[fmt] ?? LEGAL.q4_K) {
      for (const s of SHAPES) cases.push({ fmt, variant, K: s.K, N: s.N });
    }
  }

  it.each(cases)(
    "$fmt $variant K=$K N=$N: gemvQxxWgsl(o) === gemvKQuantWgsl(desc, o)",
    ({ fmt, variant, K, N }) => {
      const o = VARIANTS[variant]({ K, N });
      expect(GEN[fmt](o)).toBe(gemvKQuantWgsl(KQUANT_GEMV_DESC[fmt], o));
    },
  );
});

describe("[c] i descrittori dichiarano il formato, e il formato non si indovina", () => {
  it("cinque voci, una per formato K-quant leggibile dal motore", () => {
    expect(Object.keys(KQUANT_GEMV_DESC).sort())
      .toEqual(["q2_K", "q3_K", "q4_K", "q5_K", "q6_K"]);
  });

  it.each([
    { fmt: "q4_K" as const, bytes: 144 },
    { fmt: "q5_K" as const, bytes: 176 },
    { fmt: "q6_K" as const, bytes: 210 },
    { fmt: "q2_K" as const, bytes: 84 },
    { fmt: "q3_K" as const, bytes: 110 },
  ])("$fmt: $bytes byte per superblocco", ({ fmt, bytes }) => {
    expect(KQUANT_GEMV_DESC[fmt].blockBytes).toBe(bytes);
  });

  it("dmin c'e' sui formati AFFINI (q2_K, q4_K, q5_K) e non sui SIMMETRICI (q3_K, q6_K)", () => {
    expect(KQUANT_GEMV_DESC.q2_K.dmin).toBe(true);
    expect(KQUANT_GEMV_DESC.q4_K.dmin).toBe(true);
    expect(KQUANT_GEMV_DESC.q5_K.dmin).toBe(true);
    expect(KQUANT_GEMV_DESC.q3_K.dmin).toBe(false);
    expect(KQUANT_GEMV_DESC.q6_K.dmin).toBe(false);
  });

  it("la scala del q6_K sta IN CODA, a byte 208 del superblocco", () => {
    expect(KQUANT_GEMV_DESC.q6_K.scaleByteOffset).toBe(208);
  });
});

describe("[d] il prefill multi-riga conosce i due formati nuovi, e non li instrada", () => {
  it("l'elenco cresce IN CODA, nell'ordine che l'interfaccia congela", () => {
    expect([...PREFILL_GEMM_KINDS])
      .toEqual(["q4_0", "q5_K", "q4_1", "q4_K", "q6_K", "q8_0", "q2_K", "q3_K"]);
  });

  it("q2_K e q3_K sono NON cablati: il cablaggio e' un altro task, e si misura su GPU", () => {
    expect(prefillGemmWiring("q2_K").wired).toBe(false);
    expect(prefillGemmWiring("q3_K").wired).toBe(false);
  });

  it.each([
    { name: "q2_K idot", gen: prefillGemmQ2KSplitKIdotWgsl, kind: "q2_K" as const },
    { name: "q2_K f32", gen: prefillGemmQ2KSplitKWgsl, kind: "q2_K" as const },
    { name: "q3_K idot", gen: prefillGemmQ3KSplitKIdotWgsl, kind: "q3_K" as const },
    { name: "q3_K f32", gen: prefillGemmQ3KSplitKWgsl, kind: "q3_K" as const },
  ])("$name: genera WGSL non vuoto su {K:2048,N:512,M:16,splits:4}", ({ gen, kind }) => {
    const src = gen({ kind, K: 2048, N: 512, M: 16, splits: 4 });
    expect(src.length).toBeGreaterThan(0);
    expect(src).toContain("@compute @workgroup_size(64)");
  });
});

// ---------------------------------------------------------------------------
// [e] L'ARITMETICA DEI DUE FORMATI NUOVI, CONTRO IL RIFERIMENTO CPU.
//
// Perche' esiste, oltre al done-when. I kernel q2_K/q3_K sono nati «per
// analogia» dai gemelli q4_K/q6_K, e l'analogia e' esattamente il modo in cui
// un offset sbagliato passa inosservato: il WGSL compila, il modello gira e i
// numeri sono plausibili. Qui l'indirizzamento del corpo emesso viene SIMULATO
// in JS e confrontato con `dequantQ2_K`/`dequantQ3_K`, che sono gia' verificati
// byte-identici a `llama-quantize`.
//
// COSA QUESTO NON E'. Non e' conformita' su GPU — quella e' un task a se' e
// gira sul device. E' la verifica che gli INDICI e i CAMPI DI BIT del testo che
// stiamo emettendo sono quelli del formato. La simulazione ricopia il corpo
// riga per riga, quindi potrebbe divergere dal generatore: per questo ogni caso
// ANCORA le espressioni chiave al WGSL emesso — se qualcuno sposta un offset
// nel kernel, la riga d'ancoraggio cade prima della simulazione.
// ---------------------------------------------------------------------------
describe("[e] q2_K e q3_K: gli indici del kernel sono quelli di dequantQ2_K/dequantQ3_K", () => {
  /** PRNG deterministico: un fallimento dev'essere riproducibile. */
  const prng = (seed: number) => {
    let s = seed;
    return (): number => ((s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
  };

  const byteOf = (w: Uint32Array, base: number, i: number): number =>
    (w[base + (i >> 2)] >>> ((i & 3) * 8)) & 0xff;

  /** f16 -> f32, come `unpack2x16float` sulla meta' `half` della parola. */
  const f16At = (w: Uint32Array, word: number, half: 0 | 1): number => {
    const bits = (w[word] >>> (half * 16)) & 0xffff;
    const s = bits >> 15 ? -1 : 1, e = (bits >> 10) & 0x1f, m = bits & 0x3ff;
    if (e === 0) return s * m * 2 ** -24;
    if (e === 31) return m ? NaN : s * Infinity;
    return s * (1 + m / 1024) * 2 ** (e - 15);
  };

  /** superblocchi casuali, con una `d`/`dmin` f16 di esponente sano */
  const superblocchi = (fmt: "q2_K" | "q3_K", sbPerRow: number, seed: number) => {
    const bb = fmt === "q2_K" ? Q2_K_BLOCK_BYTES : Q3_K_BLOCK_BYTES;
    const r = prng(seed);
    const raw = new Uint8Array(sbPerRow * bb);
    for (let i = 0; i < raw.length; i++) raw[i] = Math.floor(r() * 256);
    for (let b = 0; b < sbPerRow; b++) {
      const o = b * bb;
      if (fmt === "q2_K") {
        raw[o + 80] = 0x34; raw[o + 81] = 0x2c;   // d
        raw[o + 82] = 0x11; raw[o + 83] = 0x2b;   // dmin
      } else {
        raw[o + 108] = 0x34; raw[o + 109] = 0x2c; // d
      }
    }
    const atteso = new Float32Array(sbPerRow * 256);
    if (fmt === "q2_K") dequantQ2_K(raw, 0, sbPerRow, atteso);
    else dequantQ3_K(raw, 0, sbPerRow, atteso);
    return { packed: repackKQuant(raw, 0, sbPerRow, bb), atteso, words: Math.ceil(bb / 4) };
  };

  /** la scala a 6 bit del q3_K, come `q3scale6` nel WGSL */
  const q3sc = (p: Uint32Array, wb: number, j: number): number => {
    const low = j < 8 ? byteOf(p, wb, 96 + j) & 0xf : byteOf(p, wb, 96 + j - 8) >> 4;
    const high = (byteOf(p, wb, 104 + (j & 3)) >> (2 * (j >> 2))) & 3;
    return (low | (high << 4)) - 32;
  };

  it.each([256, 512, 2048])("q2_K K=%i: il corpo del gemv ricostruisce i pesi del dequant", (K) => {
    const sbPerRow = K / 256;
    const { packed, atteso, words } = superblocchi("q2_K", sbPerRow, 12_345);
    const src = gemvQ2KWgsl({ K, N: 64 });
    expect(src).toContain("let dm = unpack2x16float(blkw(wb + 20u)); // (d, dmin) a byte offset 80");
    expect(src).toContain("let qsO = 16u + n * 32u;");
    expect(src).toContain("let sc = sbyte(wb, scO + 2u * j + hi);");

    const { lpu, chunks, unitsPerSb, units } = kquantWorkSplit(sbPerRow, 2);
    const ricostruito = new Float64Array(K);
    for (let u = 0; u < units; u++) {
      const sb = Math.floor(u / unitsPerSb);
      const rem = u % unitsPerSb;
      const n = Math.floor(rem / chunks);
      const lo = (rem % chunks) * lpu;
      const wb = sb * words;
      const d = f16At(packed, wb + 20, 0), dmin = f16At(packed, wb + 20, 1);
      for (let l = lo; l < lo + lpu; l++) {
        const hi = l >> 4;
        const qb = byteOf(packed, wb, 16 + n * 32 + l);
        for (let j = 0; j < 4; j++) {
          const sc = byteOf(packed, wb, n * 8 + 2 * j + hi);
          ricostruito[sb * 256 + n * 128 + j * 32 + l] =
            d * (sc & 0xf) * ((qb >> (2 * j)) & 3) - dmin * (sc >> 4);
        }
      }
    }
    for (let i = 0; i < K; i++) expect(ricostruito[i], `peso ${i}`).toBeCloseTo(atteso[i], 6);
  });

  it.each([256, 512, 2048])("q3_K K=%i: il corpo del gemv ricostruisce i pesi del dequant", (K) => {
    const sbPerRow = K / 256;
    const { packed, atteso, words } = superblocchi("q3_K", sbPerRow, 999);
    const src = gemvQ3KWgsl({ K, N: 64 });
    expect(src).toContain("let d = unpack2x16float(blkw(wb + 27u)).x; // d f16 a byte offset 108");
    expect(src).toContain("let qsO = 32u + n * 32u;");
    expect(src).toContain("let hm = sbyte(wb, l);");

    const { lpu, chunks, unitsPerSb, units } = kquantWorkSplit(sbPerRow, 2);
    const ricostruito = new Float64Array(K);
    for (let u = 0; u < units; u++) {
      const sb = Math.floor(u / unitsPerSb);
      const rem = u % unitsPerSb;
      const n = Math.floor(rem / chunks);
      const lo = (rem % chunks) * lpu;
      const wb = sb * words;
      const d = f16At(packed, wb + 27, 0);
      for (let l = lo; l < lo + lpu; l++) {
        const hi = l >> 4;
        const qb = byteOf(packed, wb, 32 + n * 32 + l);
        const hm = byteOf(packed, wb, l);
        for (let j = 0; j < 4; j++) {
          const sc = q3sc(packed, wb, n * 8 + 2 * j + hi);
          // il bit SPENTO della hmask e' quello che sottrae 4 (dequantQ3_K)
          const q = ((qb >> (2 * j)) & 3) - ((hm & (1 << (n * 4 + j))) !== 0 ? 0 : 4);
          ricostruito[sb * 256 + n * 128 + j * 32 + l] = d * sc * q;
        }
      }
    }
    for (let i = 0; i < K; i++) expect(ricostruito[i], `peso ${i}`).toBeCloseTo(atteso[i], 6);
  });

  it("q2_K prefill: l'unpack impacchettato della via intera da' gli STESSI pesi", () => {
    // la via intera estrae quattro pesi per parola con un solo `and`
    // impacchettato: e' il punto in cui un campo di bit sbagliato non si vede.
    const { packed, atteso } = superblocchi("q2_K", 1, 7);
    const src = prefillGemmQ2KSplitKIdotWgsl({ kind: "q2_K", K: 256, N: 64, M: 16, splits: 1 });
    expect(src).toContain("w8[ii] = (blocks[wb + 4u + n * 8u + ii] >> sh) & 0x03030303u;");
    const d = f16At(packed, 20, 0), dmin = f16At(packed, 20, 1);
    for (let n = 0; n < 2; n++) {
      for (let c = 0; c < 4; c++) {
        const blk = n * 4 + c;
        const scA = byteOf(packed, 0, n * 8 + 2 * c);
        const scB = byteOf(packed, 0, n * 8 + 2 * c + 1);
        for (let ii = 0; ii < 8; ii++) {
          const w8 = (packed[4 + n * 8 + ii] >>> (2 * c)) & 0x03030303;
          for (let t = 0; t < 4; t++) {
            const lvl = (w8 >>> (t * 8)) & 0xff;
            const sc = ii < 4 ? scA : scB;   // le due meta' hanno scale diverse
            expect(d * (sc & 0xf) * lvl - dmin * (sc >> 4), `blk${blk} ii${ii} t${t}`)
              .toBeCloseTo(atteso[blk * 32 + ii * 4 + t], 6);
          }
        }
      }
    }
  });

  it("q2_K/q3_K prefill via f32: il fallback legge gli STESSI pesi della via intera", () => {
    // Il fallback e' la via che gira quando `dot4I8Packed` non c'e': se
    // sbagliasse un offset, sarebbe un percorso di produzione che nessun altro
    // caso guarda — e che produce numeri, non errori.
    const q2 = superblocchi("q2_K", 1, 21);
    const src2 = prefillGemmQ2KSplitKWgsl({ kind: "q2_K", K: 256, N: 64, M: 16, splits: 1 });
    expect(src2).toContain("q[l] = f32((sbyte(wb, qsO + l) >> sh) & 3u);");
    const d2 = f16At(q2.packed, 20, 0), dmin2 = f16At(q2.packed, 20, 1);
    for (let blk = 0; blk < 8; blk++) {
      const n = blk >> 2, c = blk & 3;
      const scA = byteOf(q2.packed, 0, n * 8 + 2 * c);
      const scB = byteOf(q2.packed, 0, n * 8 + 2 * c + 1);
      for (let l = 0; l < 32; l++) {
        const lvl = (byteOf(q2.packed, 0, 16 + n * 32 + l) >> (2 * c)) & 3;
        const sc = l < 16 ? scA : scB;
        expect(d2 * (sc & 0xf) * lvl - dmin2 * (sc >> 4), `q2 blk${blk} l${l}`)
          .toBeCloseTo(q2.atteso[blk * 32 + l], 6);
      }
    }

    const q3 = superblocchi("q3_K", 1, 22);
    const src3 = prefillGemmQ3KSplitKWgsl({ kind: "q3_K", K: 256, N: 64, M: 16, splits: 1 });
    expect(src3).toContain("q[l] = f32((sbyte(wb, qsO + l) >> sh) & 3u) - select(4.0, 0.0, (hm & bit) != 0u);");
    const d3 = f16At(q3.packed, 27, 0);
    for (let blk = 0; blk < 8; blk++) {
      const n = blk >> 2, c = blk & 3;
      const scA = q3sc(q3.packed, 0, n * 8 + 2 * c);
      const scB = q3sc(q3.packed, 0, n * 8 + 2 * c + 1);
      for (let l = 0; l < 32; l++) {
        const hm = byteOf(q3.packed, 0, l);
        const q = ((byteOf(q3.packed, 0, 32 + n * 32 + l) >> (2 * c)) & 3)
          - ((hm & (1 << (n * 4 + c))) !== 0 ? 0 : 4);
        expect(d3 * (l < 16 ? scA : scB) * q, `q3 blk${blk} l${l}`)
          .toBeCloseTo(q3.atteso[blk * 32 + l], 6);
      }
    }
  });

  it("q3_K prefill: pesi senza segno + offset -4 fuori dal prodotto = quelli del dequant", () => {
    const { packed, atteso } = superblocchi("q3_K", 1, 4242);
    const src = prefillGemmQ3KSplitKIdotWgsl({ kind: "q3_K", K: 256, N: 64, M: 16, splits: 1 });
    expect(src).toContain("w8[ii] = ((qsw >> sh) & 0x03030303u) | (((hmw >> hb) & 0x01010101u) << 2u);");
    const d = f16At(packed, 27, 0);
    for (let n = 0; n < 2; n++) {
      for (let c = 0; c < 4; c++) {
        const blk = n * 4 + c;
        const scA = q3sc(packed, 0, n * 8 + 2 * c);
        const scB = q3sc(packed, 0, n * 8 + 2 * c + 1);
        for (let ii = 0; ii < 8; ii++) {
          const qsw = packed[8 + n * 8 + ii];
          const hmw = packed[ii];
          const w8 = ((qsw >>> (2 * c)) & 0x03030303) | (((hmw >>> (n * 4 + c)) & 0x01010101) << 2);
          for (let t = 0; t < 4; t++) {
            const lvl = (w8 >>> (t * 8)) & 0xff;
            const sc = ii < 4 ? scA : scB;
            // l'offset -4 esce dal prodotto scalare, esattamente come nel kernel
            expect(d * sc * (lvl - 4), `blk${blk} ii${ii} t${t}`)
              .toBeCloseTo(atteso[blk * 32 + ii * 4 + t], 6);
          }
        }
      }
    }
  });
});
