// PORT del moltiplicatore multi-riga di prefill dal banco (src/microbench/ttGemm.ts)
// al motore (src/engine/kernels/wgsl.ts).
//
// La tesi di questo file e' UNA: il kernel che il motore manda al device e'
// *lo stesso testo* che la riga 1 ha misurato — non "una versione equivalente".
// Perche' la misura (1,745x della via intera su splitk f32, it.6) e' una
// proprieta' del TESTO, non dell'intenzione: cambia una riga del ciclo caldo e
// il numero misurato non parla piu' del codice in produzione.
//
// Quindi qui non si verifica la matematica (la verificano ktest/conformance sul
// device): si verifica che il port non abbia riscritto niente, che ogni riga
// divergente sia DICHIARATA con la sua ragione, e che i contorni non misurati
// vengano rifiutati invece che inventati.
import { describe, expect, it } from "vitest";
import {
  gemmQ4MultiRowSplitKIdotWgsl, gemmQ4MultiRowSplitKWgsl,
  quantXQ8Wgsl, splitKCombineWgsl, workgroupStorageBytes,
} from "../src/microbench/ttGemm";
import {
  kquantQ5KMultiRowSplitKIdotWgsl, kquantQ5KMultiRowSplitKWgsl, kquantSplitsFor,
} from "../src/microbench/ttKQuant";
import {
  prefillGemmQ4SplitKIdotWgsl, prefillGemmQ4SplitKWgsl,
  prefillGemmQ5KSplitKIdotWgsl, prefillGemmQ5KSplitKWgsl,
  prefillQuantXQ8Wgsl, prefillSplitKCombineWgsl,
  prefillGemmGrid, prefillQuantXGrid, prefillCombineGrid,
  prefillGemmSplitsFor, prefillPartialFloats, prefillGemmWorkgroupStorageBytes,
  PREFILL_GEMM_PORT_DIFFS, PREFILL_GEMM_KINDS,
  type PrefillGemmOpts, type PrefillGemmKind,
} from "../src/engine/kernels/wgsl";

/** Le DUE shape misurate in riga 1 (qkv e down del 4B a chunk 16). */
const MEASURED: PrefillGemmOpts[] = [
  { kind: "q4_0", K: 2560, N: 9216, M: 16, splits: 4 },
  { kind: "q4_0", K: 9216, N: 2560, M: 16, splits: 4 },
];

/**
 * La shape MISURATA in fase 0 di engine-kquant per il Q5_K: `blk.*.ssm_out` del
 * 4B (K=4096 = 16 superblocchi da 256, N=2560), a chunk 16 e 4 fette.
 * 1,2700 → 0,0452 ms = 28,10x sulla legacy
 * (results/microbench/kquant-fase0-4090-linux-2026-08-14T19-29-20-014Z.json).
 */
const MEASURED_Q5K: PrefillGemmOpts[] = [
  { kind: "q5_K", K: 4096, N: 2560, M: 16, splits: 4 },
];

/**
 * Le righe del CICLO CALDO come stanno nel banco. Non sono ricopiate a mano:
 * il test pretende che siano nel testo del banco (altrimenti e' il banco ad
 * essere cambiato e questa lista va rifatta) e POI nel testo di produzione.
 */
const HOT_IDOT = [
  "idot = idot + dot4I8Packed(lo[wi], xs[xo + wi]);",
  "idot = idot + dot4I8Packed(hi[wi], xs[xo + 4u + wi]);",
  "acc[m] = acc[m] + f32(idot) * sc * xss[m * 2u + bi];",
];
const HOT_F32 = "bd = bd + dot(lo[wi], xs[xo + wi]) + dot(hi[wi], xs[xo + 4u + wi]);";

/**
 * Le righe del ciclo caldo del Q5_K. Le tre che contano sono il piano del 5o
 * bit sommato IMPACCHETTATO (+16 per byte invece di 32 test), i due
 * `dot4I8Packed` sui quartetti e il termine `dmin*mn_j*Sigma(x)` che il q4_0
 * non ha.
 */
const HOT_Q5K_IDOT = [
  "lo[ii] = (word & 0x0f0f0f0fu) + (((qhw >> is) & 0x01010101u) << 4u);",
  "hi[ii] = ((word >> 4u) & 0x0f0f0f0fu) + (((qhw >> (is + 1u)) & 0x01010101u) << 4u);",
  "i1 = i1 + dot4I8Packed(lo[ii], xs[bLo + ii]);",
  "i2 = i2 + dot4I8Packed(hi[ii], xs[bHi + ii]);",
  "xsum[idx] = f32(sq) * xss[idx];",
];
const HOT_Q5K_F32 = [
  "if ((qh & (1u << is)) != 0u) { a = a + 16.0; }",
  "acc[m] = acc[m] + dsc * qx - dmn * sx;",
];

/** diff riga-per-riga (LCS) fra il testo di banco e quello di produzione */
function diffLines(bench: string[], prod: string[]): { tag: "-" | "+"; line: string }[] {
  const n = bench.length, m = prod.length;
  const lcs: number[][] = Array.from({ length: n + 1 }, () => new Array<number>(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      lcs[i][j] = bench[i] === prod[j] ? lcs[i + 1][j + 1] + 1 : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
    }
  }
  const out: { tag: "-" | "+"; line: string }[] = [];
  let i = 0, j = 0;
  while (i < n && j < m) {
    if (bench[i] === prod[j]) { i++; j++; }
    else if (lcs[i + 1][j] >= lcs[i][j + 1]) out.push({ tag: "-", line: bench[i++] });
    else out.push({ tag: "+", line: prod[j++] });
  }
  while (i < n) out.push({ tag: "-", line: bench[i++] });
  while (j < m) out.push({ tag: "+", line: prod[j++] });
  return out;
}

/** i quattro kernel del port, appaiati col loro gemello di banco */
function pairs(o: PrefillGemmOpts): { name: string; bench: string; prod: string }[] {
  const { K, N, M, splits } = o;
  return [
    { name: "splitk-idot", bench: gemmQ4MultiRowSplitKIdotWgsl({ K, N, M, splits }), prod: prefillGemmQ4SplitKIdotWgsl(o) },
    { name: "splitk-f32", bench: gemmQ4MultiRowSplitKWgsl({ K, N, M, splits }), prod: prefillGemmQ4SplitKWgsl(o) },
    { name: "quantX", bench: quantXQ8Wgsl({ K, N, M }), prod: prefillQuantXQ8Wgsl({ K, M }) },
    { name: "combine", bench: splitKCombineWgsl({ K, N, M, splits }), prod: prefillSplitKCombineWgsl({ N, M, splits }) },
  ];
}

/**
 * Le DUE forme Q5_K del port, appaiate col loro gemello di banco
 * (`src/microbench/ttKQuant.ts`). Il quantizzatore delle attivazioni e la
 * combine NON compaiono: sono gli STESSI del q4_0 — i sotto-blocchi K-quant
 * sono anch'essi da 32 e la via intera Q5_K riusa `prefillQuantXQ8Wgsl` tale e
 * quale (correzione C0-2 della fase 0), quindi sono gia' appaiati sopra.
 */
function q5kPairs(o: PrefillGemmOpts): { name: string; bench: string; prod: string }[] {
  const { K, N, M, splits } = o;
  const b = { family: "q5_K" as const, K, N, M, splits };
  return [
    { name: "q5k-idot", bench: kquantQ5KMultiRowSplitKIdotWgsl(b), prod: prefillGemmQ5KSplitKIdotWgsl(o) },
    { name: "q5k-f32", bench: kquantQ5KMultiRowSplitKWgsl(b), prod: prefillGemmQ5KSplitKWgsl(o) },
  ];
}

describe("[a] semantica misurata preservata: il ciclo caldo e' quello del banco", () => {
  for (const o of MEASURED) {
    const tag = `K${o.K} N${o.N} M${o.M} splits${o.splits}`;

    it(`${tag}: le tre righe della via intera stanno nel banco e in produzione`, () => {
      const bench = gemmQ4MultiRowSplitKIdotWgsl(o);
      const prod = prefillGemmQ4SplitKIdotWgsl(o);
      for (const line of HOT_IDOT) {
        expect(bench, "riga sparita dal BANCO: la lista va rifatta").toContain(line);
        expect(prod).toContain(line);
      }
    });

    it(`${tag}: la riga della via f32 sta nel banco e in produzione`, () => {
      expect(gemmQ4MultiRowSplitKWgsl(o), "riga sparita dal BANCO").toContain(HOT_F32);
      expect(prefillGemmQ4SplitKWgsl(o)).toContain(HOT_F32);
    });

    it(`${tag}: quantX e combine sono il testo del banco`, () => {
      const { K, N, M, splits } = o;
      expect(prefillQuantXQ8Wgsl({ K, M })).toBe(quantXQ8Wgsl({ K, N, M }));
      expect(prefillSplitKCombineWgsl({ N, M, splits })).toBe(splitKCombineWgsl({ K, N, M, splits }));
    });
  }
});

// ---------------------------------------------------------------------------
// [a]-port Q5_K — la stessa tesi, sull'altra famiglia: il testo che il motore
// manda al device e' quello che la fase 0 di engine-kquant ha misurato (28,10x
// a M=16), riga per riga. Qui il confronto e' TOTALE (`toBe` sull'intero
// testo), non solo sulle righe calde: e' l'asserzione che non lascia margine.
// ---------------------------------------------------------------------------
describe("[a]-port Q5_K: il testo e' quello del banco, riga per riga", () => {
  for (const o of MEASURED_Q5K) {
    const tag = `K${o.K} N${o.N} M${o.M} splits${o.splits}`;

    for (const { name, bench, prod } of q5kPairs(o)) {
      it(`${tag} ${name}: identico al banco riga per riga`, () => {
        const b = bench.split("\n");
        const p = prod.split("\n");
        expect(p.length, "numero di righe").toBe(b.length);
        for (let i = 0; i < b.length; i++) {
          expect(p[i], `${name}: riga ${i + 1} divergente dal banco`).toBe(b[i]);
        }
        expect(prod).toBe(bench);
      });
    }

    it(`${tag}: le righe calde della via intera stanno nel banco e in produzione`, () => {
      const bench = kquantQ5KMultiRowSplitKIdotWgsl({ family: "q5_K", K: o.K, N: o.N, M: o.M, splits: o.splits });
      const prod = prefillGemmQ5KSplitKIdotWgsl(o);
      for (const line of HOT_Q5K_IDOT) {
        expect(bench, "riga sparita dal BANCO: la lista va rifatta").toContain(line);
        expect(prod).toContain(line);
      }
    });

    it(`${tag}: le righe calde della via f32 stanno nel banco e in produzione`, () => {
      const bench = kquantQ5KMultiRowSplitKWgsl({ family: "q5_K", K: o.K, N: o.N, M: o.M, splits: o.splits });
      const prod = prefillGemmQ5KSplitKWgsl(o);
      for (const line of HOT_Q5K_F32) {
        expect(bench, "riga sparita dal BANCO").toContain(line);
        expect(prod).toContain(line);
      }
    });

    it(`${tag}: niente enable packed_4x8 (language feature, non estensione)`, () => {
      for (const code of [prefillGemmQ5KSplitKIdotWgsl(o), prefillGemmQ5KSplitKWgsl(o)]) {
        expect(code).not.toContain("enable packed_4x8");
      }
      expect(prefillGemmQ5KSplitKIdotWgsl(o)).toContain("dot4I8Packed");
      // la via f32 e' il FALLBACK: niente prodotto scalare intero li' dentro
      expect(prefillGemmQ5KSplitKWgsl(o)).not.toContain("dot4I8Packed");
    });

    it(`${tag}: binding CONGELATI, letti dal testo generato`, () => {
      expect(storageBindings(prefillGemmQ5KSplitKIdotWgsl(o))).toEqual([
        "0:read:blocks:array<u32>",
        "1:read:xq:array<u32>",
        "2:read_write:part:array<f32>",
        "3:read:xsc:array<f32>",
      ]);
      expect(storageBindings(prefillGemmQ5KSplitKWgsl(o))).toEqual([
        "0:read:blocks:array<u32>",
        "1:read:x:array<f32>",
        "2:read_write:part:array<f32>",
      ]);
      // la via f32 legge le attivazioni DENSE row-major M x K
      expect(prefillGemmQ5KSplitKWgsl(o)).toContain(`const K_DIM = ${o.K}u;`);
      expect(prefillGemmQ5KSplitKWgsl(o)).toContain("xs[idx] = x[m * K_DIM + base + (idx % 32u)];");
      // e la via intera rilegge xq/xsc dove `prefillQuantXQ8Wgsl` li ha scritti
      expect(prefillGemmQ5KSplitKIdotWgsl(o)).toContain("xs[idx] = xq[((m * SBPR + sb) * 8u + jj) * 8u + ii];");
      expect(prefillGemmQ5KSplitKIdotWgsl(o)).toContain("xss[idx] = xsc[(m * SBPR + sb) * 8u + jj];");
      // part[(s*M + m)*N + r]: la fetta scrive dove la combine (condivisa col
      // q4_0) legge
      const write = "for (var m = 0u; m < M_ROWS; m = m + 1u) { part[(s * M_ROWS + m) * N_ROWS + r] = acc[m]; }";
      for (const code of [prefillGemmQ5KSplitKIdotWgsl(o), prefillGemmQ5KSplitKWgsl(o)]) {
        expect(code).toContain(write);
        expect(code).toContain(`const SBPR = ${o.K / 256}u;`);
        expect(code).toContain(`const PER = ${o.K / 256 / o.splits}u;`);
        expect(code).toContain("let r = wid.x * 64u + t;");
        expect(code).toContain("let s = wid.y;");
        expect(code).toContain("@compute @workgroup_size(64)");
      }
    });

    it(`${tag}: griglia e parziali sono le stesse formule del q4_0`, () => {
      expect(prefillGemmGrid(o)).toEqual([Math.ceil(o.N / 64), o.splits, 1]);
      expect(prefillPartialFloats(o)).toBe(o.splits * o.M * o.N);
      expect(prefillPartialFloats(o)).toBe(4 * 16 * 2560);
    });
  }

  it("i kind della via veloce sono esattamente q4_0 e q5_K", () => {
    expect([...PREFILL_GEMM_KINDS]).toEqual(["q4_0", "q5_K"]);
  });
});

describe("[b] ogni divergenza dal banco e' dichiarata in PREFILL_GEMM_PORT_DIFFS", () => {
  // Il diff e' il sensore di tutto questo blocco: se non vedesse le differenze,
  // "nessuna riga non coperta" sarebbe vero e vuoto. Qui si prova il sensore.
  it("auto-prova del sensore: il diff riga-per-riga vede davvero le differenze", () => {
    expect(diffLines(["a", "b", "c"], ["a", "x", "c"]))
      .toEqual([{ tag: "-", line: "b" }, { tag: "+", line: "x" }]);
    expect(diffLines(["a", "b"], ["a", "b", "c"])).toEqual([{ tag: "+", line: "c" }]);
    expect(diffLines(["a", "b"], ["a", "b"])).toEqual([]);
  });

  it("ogni ragione e' una ragione (>= 30 caratteri)", () => {
    for (const [k, why] of Object.entries(PREFILL_GEMM_PORT_DIFFS)) {
      expect(why.length, `${k}: ragione troppo corta ("${why}")`).toBeGreaterThanOrEqual(30);
    }
  });

  const seen = new Set<string>();
  const ALL: { o: PrefillGemmOpts; ps: { name: string; bench: string; prod: string }[] }[] = [
    ...MEASURED.map((o) => ({ o, ps: pairs(o) })),
    ...MEASURED_Q5K.map((o) => ({ o, ps: q5kPairs(o) })),
  ];
  for (const { o, ps } of ALL) {
    for (const { name, bench, prod } of ps) {
      it(`K${o.K} N${o.N} ${name}: nessuna riga divergente non coperta`, () => {
        for (const d of diffLines(bench.split("\n"), prod.split("\n"))) {
          // La chiave e' la riga COME STA — rientro compreso, righe bianche
          // comprese. Trimmare e saltare le righe vuote lasciava scoperte
          // indentazione e spaziatura, che nel testo di un kernel sono
          // differenze come tutte le altre.
          const key = d.line;
          seen.add(key);
          expect(
            Object.prototype.hasOwnProperty.call(PREFILL_GEMM_PORT_DIFFS, key),
            `${name}: riga "${d.tag}${key}" divergente dal banco e NON dichiarata in PREFILL_GEMM_PORT_DIFFS`,
          ).toBe(true);
        }
      });
    }
  }

  // Bidirezionale: una chiave che non corrisponde piu' a nessuna divergenza e'
  // una ragione per un port che non esiste, e mente sul rapporto col banco.
  it("nessuna chiave morta (ogni chiave e' una divergenza vera)", () => {
    for (const k of Object.keys(PREFILL_GEMM_PORT_DIFFS)) {
      expect(seen.has(k), `chiave morta in PREFILL_GEMM_PORT_DIFFS: "${k}"`).toBe(true);
    }
  });

  // Il record VUOTO dice una cosa precisa: il port non ha riscritto niente.
  // Senza questa asserzione, "vuoto" sarebbe indistinguibile da "dimenticato".
  it("record vuoto => i sei kernel sono byte-per-byte quelli del banco", () => {
    if (Object.keys(PREFILL_GEMM_PORT_DIFFS).length !== 0) return;
    for (const { o, ps } of ALL) {
      for (const { name, bench, prod } of ps) {
        expect(prod, `${name} K${o.K} N${o.N}`).toBe(bench);
      }
    }
  });
});

// ---------------------------------------------------------------------------
// [g] L'INTERFACCIA CONGELATA, ASSERITA A MANO — non per transitivita'.
//
// Il diff contro il banco NON e' un guardiano dell'ordine dei binding ne' del
// layout dei buffer, per due ragioni verificate: l'unica asserzione forte
// (`prod === bench`) si AUTO-DISABILITA alla prima chiave dichiarata, e il
// gate che resta controlla solo CHE una riga divergente sia dichiarata, mai
// COSA dichiara — qualunque stringa di 30 caratteri lo soddisfa. Con quel solo
// gate si puo' rompere `part[(s*M+m)*N+r]` (che la combine e i consumatori a
// valle danno per assunto) e restare verdi.
//
// Qui l'ordine, i tipi e i nomi dei binding e le formule di indirizzamento
// sono pinnati direttamente sul testo generato, INDIPENDENTEMENTE da
// PREFILL_GEMM_PORT_DIFFS. Se una riga di questo blocco va aggiornata, e' un
// cambio di contratto verso i consumatori, non un dettaglio di port.
// ---------------------------------------------------------------------------

/** `indice:accesso:nome:tipo` per ogni `var<storage>`, NELL'ORDINE del testo. */
function storageBindings(code: string): string[] {
  const re = /@group\(0\) @binding\((\d+)\) var<storage, (read|read_write)> (\w+): ([^;]+);/g;
  return [...code.matchAll(re)].map((m) => `${m[1]}:${m[2]}:${m[3]}:${m[4]}`);
}

describe("[g] ordine dei binding e layout dei buffer: congelati per t5/t3", () => {
  const o = MEASURED[0];
  const { K, N, M, splits } = o;

  const EXPECTED: Record<string, { code: string; bindings: string[] }> = {
    "splitk-idot": {
      code: prefillGemmQ4SplitKIdotWgsl(o),
      bindings: [
        "0:read:qs4:array<vec4<u32>>",
        "1:read:scales:array<u32>",
        "2:read:xq:array<u32>",
        "3:read_write:part:array<f32>",
        "4:read:xsc:array<f32>",
      ],
    },
    "splitk-f32": {
      code: prefillGemmQ4SplitKWgsl(o),
      bindings: [
        "0:read:qs4:array<vec4<u32>>",
        "1:read:scales:array<u32>",
        "2:read:x4:array<vec4<f32>>",
        "3:read_write:part:array<f32>",
      ],
    },
    quantX: {
      code: prefillQuantXQ8Wgsl({ K, M }),
      bindings: [
        "0:read:x4:array<vec4<f32>>",
        "1:read_write:xq:array<u32>",
        "2:read_write:xsc:array<f32>",
      ],
    },
    combine: {
      code: prefillSplitKCombineWgsl({ N, M, splits }),
      bindings: [
        "0:read:part:array<f32>",
        "1:read_write:y:array<f32>",
      ],
    },
  };

  for (const [name, { code, bindings }] of Object.entries(EXPECTED)) {
    it(`${name}: binding esattamente [${bindings.map((b) => b.split(":")[2]).join(", ")}]`, () => {
      expect(storageBindings(code)).toEqual(bindings);
      // nessun binding sfuggito alla regex (una dichiarazione in altra forma
      // sarebbe un binding fantasma per chi costruisce il bind group)
      expect((code.match(/var<storage/g) ?? []).length).toBe(bindings.length);
    });
  }

  it("auto-prova del sensore: storageBindings vede ordine, nome e accesso", () => {
    expect(storageBindings(
      "@group(0) @binding(1) var<storage, read_write> y: array<f32>;\n" +
      "@group(0) @binding(0) var<storage, read> part: array<f32>;",
    )).toEqual(["1:read_write:y:array<f32>", "0:read:part:array<f32>"]);
  });

  it("part[(s*M + m)*N + r]: la fetta scrive dove la combine legge", () => {
    const write = `for (var m = 0u; m < M_ROWS; m = m + 1u) { part[(s * M_ROWS + m) * N_ROWS + r] = acc[m]; }`;
    for (const code of [EXPECTED["splitk-idot"].code, EXPECTED["splitk-f32"].code]) {
      expect(code).toContain(write);
      expect(code).toContain(`const M_ROWS = ${M}u;`);
      expect(code).toContain(`const N_ROWS = ${N}u;`);
      // la riga di uscita e' wid.x*64 + lid.x, la fetta e' wid.y: e' cio' che
      // rende [ceil(N/64), splits, 1] la griglia giusta e non una convenzione
      expect(code).toContain("let r = wid.x * 64u + t;");
      expect(code).toContain("let s = wid.y;");
      expect(code).toContain("@compute @workgroup_size(64)");
    }
  });

  it("y[m*N + r] = somma delle S fette, un thread per uscita", () => {
    const code = EXPECTED.combine.code;
    expect(code).toContain(`const TOTAL = ${M * N}u;`);
    expect(code).toContain(`const S = ${splits}u;`);
    // i = m*N + r, quindi part[s*TOTAL + i] = part[(s*M + m)*N + r]: e' lo
    // stesso indirizzo che scrivono le due vie del moltiplicatore
    expect(code).toContain("let i = gid.x;");
    expect(code).toContain("for (var s = 0u; s < S; s = s + 1u) { v = v + part[s * TOTAL + i]; }");
    expect(code).toContain("y[i] = v;");
  });

  it("le attivazioni si leggono dove quantX le ha scritte", () => {
    const q = EXPECTED.quantX.code;
    expect(q).toContain(`const BLOCKS = ${(M * K) / 32}u;`);
    expect(q).toContain(`const BPR = ${K / 32}u;`);
    expect(q).toContain(`const K4 = ${K / 4}u;`);
    expect(q).toContain("let base = m * K4 + blk * 8u;");   // 8 vec4 = 1 blocco
    expect(q).toContain("xq[b * 8u + i] =");                 // 8 u32 per blocco
    expect(q).toContain("xsc[b] = sc;");                     // 1 scala per blocco
    // e il moltiplicatore intero li rilegge con lo STESSO indirizzo (b = m*BPR + blk)
    expect(EXPECTED["splitk-idot"].code).toContain("xs[idx] = xq[(m * BPR + b0) * 8u + qi];");
    expect(EXPECTED["splitk-idot"].code).toContain("xss[idx] = xsc[m * BPR + b0 + (idx % 2u)];");
    // la via f32 legge invece le attivazioni dense
    expect(EXPECTED["splitk-f32"].code).toContain("xs[idx] = x4[m * K4 + b0 * 8u + qi];");
  });

  it("i pesi: gb = r*BPR + b0 + bi, UNA scala f16 per blocco", () => {
    for (const code of [EXPECTED["splitk-idot"].code, EXPECTED["splitk-f32"].code]) {
      expect(code).toContain(`const BPR = ${K / 32}u;`);
      expect(code).toContain(`const PER = ${K / 32 / splits}u;`);
      expect(code).toContain("let gb = r * BPR + b0 + bi;");
      expect(code).toContain("let sc = unpack2x16float(scales[gb >> 1u])[gb & 1u];");
      expect(code).toContain("let w = qs4[gb];");
    }
  });
});

describe("[c] dot4I8Packed e' una language feature, non un'estensione", () => {
  // Scriverlo e' costato una run in it.5: Tint risponde «expected extension».
  for (const o of MEASURED) {
    it(`K${o.K} N${o.N}: niente enable packed_4x8 nel testo generato`, () => {
      for (const code of [prefillGemmQ4SplitKIdotWgsl(o), prefillGemmQ4SplitKWgsl(o),
        prefillQuantXQ8Wgsl({ K: o.K, M: o.M }), prefillSplitKCombineWgsl(o)]) {
        expect(code).not.toContain("enable packed_4x8");
      }
      expect(prefillGemmQ4SplitKIdotWgsl(o)).toContain("dot4I8Packed");
    });
  }
});

describe("[d] workgroup storage: LETTO dal testo, non dedotto", () => {
  const o = MEASURED[0];

  it("via intera 1.152 B e via f32 4.096 B a M=16", () => {
    expect(workgroupStorageBytes(prefillGemmQ4SplitKIdotWgsl(o))).toBe(1152);
    expect(workgroupStorageBytes(prefillGemmQ4SplitKWgsl(o))).toBe(4096);
  });

  // Il numero serve a NEGOZIARE `requiredLimits`, quindi deve essere quello
  // della via che si accende davvero: chiedere 4.096 per la sola pipeline
  // intera vuol dire chiedere 3,55x lo storage che serve. La via si sceglie a
  // runtime sulla language feature, percio' la funzione la prende come
  // argomento e resta conservativa (il peggiore) quando non gliela si dice.
  it("prefillGemmWorkgroupStorageBytes rende i DUE numeri, non solo il peggiore", () => {
    expect(prefillGemmWorkgroupStorageBytes(o, "idot")).toBe(1152);
    expect(prefillGemmWorkgroupStorageBytes(o, "f32")).toBe(4096);
    expect(prefillGemmWorkgroupStorageBytes(o)).toBe(4096); // default = il peggiore
  });

  it("per ogni via il numero e' quello LETTO dal testo, e sta sotto il limite", () => {
    for (const M of [8, 16, 32]) {
      const oo: PrefillGemmOpts = { ...o, M };
      const idot = workgroupStorageBytes(prefillGemmQ4SplitKIdotWgsl(oo));
      const f32 = workgroupStorageBytes(prefillGemmQ4SplitKWgsl(oo));
      expect(prefillGemmWorkgroupStorageBytes(oo, "idot"), `M=${M}`).toBe(idot);
      expect(prefillGemmWorkgroupStorageBytes(oo, "f32"), `M=${M}`).toBe(f32);
      expect(prefillGemmWorkgroupStorageBytes(oo), `M=${M}`).toBe(Math.max(idot, f32));
      // il default di spec WebGPU, che il motore non vuole superare
      expect(idot, `M=${M}`).toBeLessThanOrEqual(16384);
      expect(f32, `M=${M}`).toBeLessThanOrEqual(16384);
      expect(prefillGemmWorkgroupStorageBytes(oo), `M=${M}`).toBeLessThanOrEqual(16384);
    }
  });

  // -------------------------------------------------------------------------
  // [b]-storage Q5_K. La via intera tiene TRE array in memoria di gruppo (le
  // attivazioni impacchettate `xs`, le loro scale `xss` e la somma per
  // sotto-blocco `xsum`) = 320·M B; la via f32 ne tiene UNO solo, e di UN
  // sotto-blocco per volta invece che del superblocco intero = 128·M B. Il
  // superblocco intero sarebbe M·256 f32 = 16.384 B a M=16, cioe' esattamente
  // il minimo garantito da WebGPU per un solo array: il tile piccolo e' la
  // ragione per cui questa forma gira anche dove il tetto e' quello di spec.
  // -------------------------------------------------------------------------
  const q5 = MEASURED_Q5K[0];

  it("Q5_K: 5.120 B via intera e 2.048 B via f32 a M=16", () => {
    expect(prefillGemmWorkgroupStorageBytes(q5, "idot")).toBe(5120);
    expect(prefillGemmWorkgroupStorageBytes(q5, "f32")).toBe(2048);
    // senza `via` = il peggiore dei due, che qui e' la via INTERA (al contrario
    // del q4_0, dove il peggiore e' la f32)
    expect(prefillGemmWorkgroupStorageBytes(q5)).toBe(5120);
  });

  it("Q5_K: i due numeri sono quelli LETTI dal testo, e stanno sotto il tetto di spec", () => {
    for (let M = 1; M <= 16; M++) {
      const oo: PrefillGemmOpts = { ...q5, M };
      const idot = workgroupStorageBytes(prefillGemmQ5KSplitKIdotWgsl(oo));
      const f32 = workgroupStorageBytes(prefillGemmQ5KSplitKWgsl(oo));
      expect(prefillGemmWorkgroupStorageBytes(oo, "idot"), `M=${M}`).toBe(idot);
      expect(prefillGemmWorkgroupStorageBytes(oo, "f32"), `M=${M}`).toBe(f32);
      expect(prefillGemmWorkgroupStorageBytes(oo), `M=${M}`).toBe(Math.max(idot, f32));
      expect(idot, `M=${M}`).toBeLessThanOrEqual(16384);
      expect(f32, `M=${M}`).toBeLessThanOrEqual(16384);
      expect(prefillGemmWorkgroupStorageBytes(oo), `M=${M}`).toBeLessThanOrEqual(16384);
    }
  });
});

describe("[e] griglie di dispatch", () => {
  it("gemm = [ceil(N/64), splits, 1]", () => {
    expect(prefillGemmGrid({ kind: "q4_0", K: 2560, N: 9216, M: 16, splits: 4 })).toEqual([144, 4, 1]);
    expect(prefillGemmGrid({ kind: "q4_0", K: 9216, N: 2560, M: 16, splits: 4 })).toEqual([40, 4, 1]);
  });

  it("combine = [ceil(M*N/64), 1, 1]", () => {
    expect(prefillCombineGrid({ N: 9216, M: 16 })).toEqual([Math.ceil((16 * 9216) / 64), 1, 1]);
  });

  it("quantX = [ceil(M*K/32/64), 1, 1]", () => {
    expect(prefillQuantXGrid({ K: 2560, M: 16 })).toEqual([Math.ceil((16 * 80) / 64), 1, 1]);
  });

  it("parziali = splits*M*N f32", () => {
    expect(prefillPartialFloats({ kind: "q4_0", K: 2560, N: 9216, M: 16, splits: 4 })).toBe(4 * 16 * 9216);
  });

  it("splitsFor rende le 4 fette MISURATE sulle shape di riga 1", () => {
    expect(prefillGemmSplitsFor(2560, 9216)).toBe(4);
    expect(prefillGemmSplitsFor(9216, 2560)).toBe(4);
  });

  // QWEN25_05B ha dModel = 896 (src/engine/shape.ts): 28 blocchi, che in 4
  // fette da BK=2 non ci stanno. Rifiutare avrebbe tolto il prefill veloce a
  // una shape GIA' IN ALBERO. Il ripiego e' 1 = nessuno split-K, cioe' la forma
  // da cui lo split-K partiva: mai piu' veloce del misurato, sempre corretta.
  // Non 2 e non 3: quelli sarebbero numeri inventati.
  it("shape reale non divisibile in 4: ripiego DICHIARATO a 1 fetta, non rifiuto", () => {
    for (const K of [896, 2752, 4736]) {
      expect(prefillGemmSplitsFor(K, 4864), `K=${K}`).toBe(1);
    }
    // dFfn = 4864 -> 152 blocchi, che invece in 4 fette ci stanno: il ripiego
    // scatta per shape, non per modello
    expect(prefillGemmSplitsFor(4864, 896)).toBe(4);
    // e la forma a 1 fetta e' generabile davvero (il ripiego non e' un numero
    // che poi fa esplodere il generatore due righe dopo)
    const one: PrefillGemmOpts = { kind: "q4_0", K: 896, N: 4864, M: 16, splits: 1 };
    expect(prefillGemmQ4SplitKIdotWgsl(one)).toContain("const PER = 28u;");
    expect(prefillGemmGrid(one)).toEqual([Math.ceil(4864 / 64), 1, 1]);
    expect(prefillPartialFloats(one)).toBe(16 * 4864);
    // ...e la scala q4_0 e' INVARIATA anche senza terzo argomento: il default
    // e' q4_0, quindi nessun chiamante fuori da questa riga cambia numero.
    expect(prefillGemmSplitsFor(2560, 9216)).toBe(4);
    expect(prefillGemmSplitsFor(9216, 2560)).toBe(4);
    expect(prefillGemmSplitsFor(896, 4864)).toBe(1);
    expect(prefillGemmSplitsFor(4864, 896)).toBe(4);
  });

  // -------------------------------------------------------------------------
  // [c]-splits Q5_K. L'unita' indivisibile non e' il blocco da 32: e' il
  // SUPERBLOCCO da 256, perche' le scale a 6 bit sono condivise dagli otto
  // sotto-blocchi e una fetta che ne tagliasse meta' dovrebbe rileggere
  // l'header comunque — falsificando il conto dei byte su cui poggia il goal.
  // La scala e' quella di `kquantSplitsFor`, e il test la confronta CONTRO
  // quella funzione, non contro i numeri riscritti a mano.
  // -------------------------------------------------------------------------
  it("Q5_K: 4 fette se le unita' per riga si dividono per 4, 2 per 2, altrimenti 1", () => {
    expect(prefillGemmSplitsFor(4096, 2560, "q5_K")).toBe(4);   // ssm_out del 4B: 16 superblocchi
    expect(prefillGemmSplitsFor(512, 2048, "q5_K")).toBe(2);    // 2 superblocchi
    expect(prefillGemmSplitsFor(256, 2048, "q5_K")).toBe(1);    // 1 superblocco
    // e sono gli STESSI numeri della funzione di banco, non una seconda scala
    for (const K of [4096, 2560, 1024, 768, 512, 256]) {
      expect(prefillGemmSplitsFor(K, 2560, "q5_K"), `K=${K}`).toBe(kquantSplitsFor("q5_K", K));
    }
  });

  it("Q5_K: K non multiplo di 256 -> throw che NOMINA 256", () => {
    for (const K of [4096 + 32, 2592, 128]) {
      expect(() => prefillGemmSplitsFor(K, 2560, "q5_K"), `K=${K}`).toThrow(/256/);
    }
    // il terzo argomento e' OPZIONALE e il default e' q4_0: 2688 e' multiplo di
    // 64 ma NON di 256, quindi sulla scala q4_0 non solleva e su quella q5_K si'
    expect(() => prefillGemmSplitsFor(2688, 2560)).not.toThrow();
    expect(() => prefillGemmSplitsFor(2688, 2560, "q5_K")).toThrow(/256/);
  });
});

describe("[f] rifiuto invece di kernel non misurato", () => {
  const bad = (patch: Partial<PrefillGemmOpts>): PrefillGemmOpts =>
    ({ ...MEASURED[0], ...patch });

  it("K non multiplo di 64 -> throw", () => {
    expect(() => prefillGemmQ4SplitKIdotWgsl(bad({ K: 2592 }))).toThrow(/64/);
    expect(() => prefillGemmQ4SplitKWgsl(bad({ K: 2592 }))).toThrow(/64/);
    expect(() => prefillQuantXQ8Wgsl({ K: 2592, M: 16 })).toThrow(/64/);
    expect(() => prefillGemmSplitsFor(2592, 9216)).toThrow(/64/);
  });

  it("blocchi non divisibili per splits*2 -> throw", () => {
    // K=2560 -> 80 blocchi: 80 % (6*2) != 0
    expect(() => prefillGemmQ4SplitKIdotWgsl(bad({ splits: 6 }))).toThrow(/fette/);
    expect(() => prefillGemmQ4SplitKWgsl(bad({ splits: 6 }))).toThrow(/fette/);
    expect(() => prefillGemmGrid(bad({ splits: 6 }))).toThrow(/fette/);
    expect(() => prefillPartialFloats(bad({ splits: 6 }))).toThrow(/fette/);
    // `prefillGemmSplitsFor` non ha questo modo di fallire: SCEGLIE lui le
    // fette, e quando le 4 misurate non dividono ripiega su 1 (vedi [e]).
  });

  it("kind fuori dai due supportati -> throw che NOMINA il kind e i kind supportati", () => {
    for (const kind of ["q8_0", "q4_K", "q6_K", "f32"]) {
      const k = { ...MEASURED[0], kind } as unknown as PrefillGemmOpts;
      for (const gen of [prefillGemmQ4SplitKIdotWgsl, prefillGemmQ4SplitKWgsl,
        prefillGemmQ5KSplitKIdotWgsl, prefillGemmQ5KSplitKWgsl]) {
        // il kind RIFIUTATO e' nominato...
        expect(() => gen(k), kind).toThrow(new RegExp(kind));
        // ...e anche quelli supportati, cosi' il messaggio dice dove si puo' andare
        for (const ok of PREFILL_GEMM_KINDS) {
          expect(() => gen(k), `${kind} deve nominare ${ok}`).toThrow(new RegExp(ok));
        }
      }
    }
  });

  it("Q5_K: prima il FORMATO poi la GEOMETRIA, e il generatore non inventa", () => {
    const q5 = MEASURED_Q5K[0];
    // K non multiplo di 256 -> geometria, e il messaggio nomina 256
    for (const gen of [prefillGemmQ5KSplitKIdotWgsl, prefillGemmQ5KSplitKWgsl]) {
      expect(() => gen({ ...q5, K: 4128 })).toThrow(/256/);
      // superblocchi non divisibili per le fette
      expect(() => gen({ ...q5, splits: 3 })).toThrow(/fette/);
    }
    // e un kind non supportato viene rifiutato PRIMA della geometria: il
    // messaggio parla del formato, non del K
    const k = { ...q5, kind: "q4_K", K: 4128 } as unknown as PrefillGemmOpts;
    expect(() => prefillGemmQ5KSplitKIdotWgsl(k)).toThrow(/q4_K/);
    let msg = "";
    try { prefillGemmQ5KSplitKIdotWgsl(k); } catch (e) { msg = (e as Error).message; }
    expect(msg).not.toContain("4128");
  });

  it("il predicato sta in UNA sede sola: tutte le funzioni rifiutano allo stesso modo", () => {
    // Sei generatori/funzioni, un solo `prefillGemmCheck`. Se una di queste
    // avesse il suo predicato, un kind nuovo entrerebbe da una porta e non
    // dall'altra.
    const kinds: PrefillGemmKind[] = [...PREFILL_GEMM_KINDS];
    expect(kinds.length).toBe(2);
    const k = { ...MEASURED[0], kind: "q6_K" } as unknown as PrefillGemmOpts;
    for (const f of [prefillGemmGrid, prefillPartialFloats, prefillGemmWorkgroupStorageBytes]) {
      expect(() => f(k)).toThrow(/q6_K/);
    }
  });
});
