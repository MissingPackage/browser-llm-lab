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
  prefillGemmQ4SplitKIdotWgsl, prefillGemmQ4SplitKWgsl,
  prefillQuantXQ8Wgsl, prefillSplitKCombineWgsl,
  prefillGemmGrid, prefillQuantXGrid, prefillCombineGrid,
  prefillGemmSplitsFor, prefillPartialFloats, prefillGemmWorkgroupStorageBytes,
  PREFILL_GEMM_PORT_DIFFS, type PrefillGemmOpts,
} from "../src/engine/kernels/wgsl";

/** Le DUE shape misurate in riga 1 (qkv e down del 4B a chunk 16). */
const MEASURED: PrefillGemmOpts[] = [
  { kind: "q4_0", K: 2560, N: 9216, M: 16, splits: 4 },
  { kind: "q4_0", K: 9216, N: 2560, M: 16, splits: 4 },
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
  for (const o of MEASURED) {
    for (const { name, bench, prod } of pairs(o)) {
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
  it("record vuoto => i quattro kernel sono byte-per-byte quelli del banco", () => {
    if (Object.keys(PREFILL_GEMM_PORT_DIFFS).length !== 0) return;
    for (const o of MEASURED) {
      for (const { name, bench, prod } of pairs(o)) {
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

  it("kind != q4_0 -> throw che NOMINA la ragione", () => {
    const k8 = { ...MEASURED[0], kind: "q8_0" } as unknown as PrefillGemmOpts;
    for (const gen of [prefillGemmQ4SplitKIdotWgsl, prefillGemmQ4SplitKWgsl]) {
      expect(() => gen(k8)).toThrow(/q4_0-only/);
    }
  });
});
