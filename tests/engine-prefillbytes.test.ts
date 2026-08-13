import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import {
  weightBytesPerRow, dispatchWeightBytes, createPrefillWeightMeter,
  type PrefillQuantKind,
} from "../src/engine/prefillbytes";
import {
  Q4_0_BLOCK_BYTES, Q4_1_BLOCK_BYTES, Q8_0_BLOCK_BYTES,
  Q4_K_BLOCK_BYTES, Q5_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES,
} from "../src/engine/quant";
import { planPrefill } from "../src/engine/prefillplan";

// CONTATORE DEL TRAFFICO PESI del prefill (t7 dell'ondata TTFT). Il numero che
// interessa al goal non e' "quanti byte pesa il modello" ma QUANTI BYTE DI PESI
// ATTRAVERSANO LA MEMORIA per token di prefill: e' la voce che la forma legacy
// (M fette su wid.z, riuso ZERO per costruzione) paga M volte e che il
// multi-riga paga UNA volta per chunk. Modulo PURO: nessuna GPU, nessuna
// allocazione device — gira in CI su qualunque runner (test D qui sotto lo
// sorveglia).

// Shape FFN del 4B (q35shape): dModel 2560 → dFfn 9216.
const K_FFN = 2560;
const N_FFN = 9216;

describe("(a) byte di pesi per riga: il blocco quantizzato, non una stima", () => {
  it("q4_0 = (K/32)*18 — 16 B di nibble + 2 B di scala f16 per blocco di 32", () => {
    expect(weightBytesPerRow("q4_0", 2560)).toBe((2560 / 32) * 18);
    expect(weightBytesPerRow("q4_0", 2560)).toBe(1440);
    expect(weightBytesPerRow("q4_0", 9216)).toBe(5184);
  });

  it("i byte/blocco sono quelli di quant.ts, allineati alla parola come in VRAM", () => {
    // CONTROLLO INCROCIATO, non contro noi stessi: il repack K-quant scrive con
    // `stride = ceil(blockBytes/4)*4` (quant.ts) e il kernel indicizza con
    // quello stride, quindi il q6_K attraversa 212 B per superblocco, non 210.
    const stride = (b: number): number => Math.ceil(b / 4) * 4;
    const perBlock = (kind: PrefillQuantKind, weights: number): number =>
      weightBytesPerRow(kind, weights * 4) / 4;
    expect(perBlock("q4_0", 32)).toBe(Q4_0_BLOCK_BYTES);         // 18, repack 16+2
    expect(perBlock("q4_1", 32)).toBe(Q4_1_BLOCK_BYTES);         // 20, repack 16+4
    expect(perBlock("q8_0", 32)).toBe(Q8_0_BLOCK_BYTES);         // 34, repack 32+2
    expect(perBlock("q4_K", 256)).toBe(stride(Q4_K_BLOCK_BYTES)); // 144, gia' allineato
    expect(perBlock("q5_K", 256)).toBe(stride(Q5_K_BLOCK_BYTES)); // 176, gia' allineato
    expect(perBlock("q6_K", 256)).toBe(stride(Q6_K_BLOCK_BYTES)); // 210 → 212 (2 B di pad)
    expect(perBlock("q6_K", 256)).toBe(212);
    expect(Q6_K_BLOCK_BYTES).toBe(210);   // il pad e' una differenza, non un refuso
    expect(weightBytesPerRow("f32", 2560)).toBe(2560 * 4);
  });

  it("copre TUTTI i kind che il consumatore puo' spingere (ssm_out q5_K, down q4_K)", () => {
    // q35gpumodel.loadW ha un ramo per F32, Q4_0/Q4_1, Q8_0 e Q4_K/Q5_K/Q6_K, e
    // il gemello a M righe (`pushB`) esiste per tutti: un kind mancante qui
    // significherebbe dispatch non contabilizzati, cioe' il buco che il meter
    // deve rendere visibile.
    const kinds: PrefillQuantKind[] = ["q4_0", "q4_1", "q8_0", "q4_K", "q5_K", "q6_K", "f32"];
    for (const k of kinds) expect(weightBytesPerRow(k, 2560)).toBeGreaterThan(0);
  });

  it("K non multiplo del blocco e' un errore, non un arrotondamento silenzioso", () => {
    expect(() => weightBytesPerRow("q4_0", 30)).toThrow(/K/);
    expect(() => weightBytesPerRow("q6_K", 2560 + 32)).toThrow(/K/);
    expect(() => weightBytesPerRow("q4_0", 0)).toThrow(/K/);
  });
});

describe("(b) legacy vs multirow: il rapporto E' M, per costruzione", () => {
  const shape = { kind: "q4_0" as PrefillQuantKind, K: K_FFN, N: N_FFN };

  it("legacy = M * N * bytesPerRow (M fette su wid.z: riuso dei pesi ZERO)", () => {
    for (const M of [1, 8, 16]) {
      expect(dispatchWeightBytes({ form: "legacy", ...shape, M }))
        .toBe(M * N_FFN * weightBytesPerRow("q4_0", K_FFN));
    }
  });

  it("multirow = N * bytesPerRow: una passata sui pesi per chunk, M non entra", () => {
    for (const M of [1, 8, 16, 64]) {
      expect(dispatchWeightBytes({ form: "multirow", ...shape, M }))
        .toBe(N_FFN * weightBytesPerRow("q4_0", K_FFN));
    }
  });

  it("il rapporto per la FFN del 4B (K2560 N9216) vale ESATTAMENTE M", () => {
    const ratio = (M: number): number =>
      dispatchWeightBytes({ form: "legacy", ...shape, M }) /
      dispatchWeightBytes({ form: "multirow", ...shape, M });
    expect(ratio(1)).toBe(1);
    expect(ratio(8)).toBe(8);
    expect(ratio(16)).toBe(16);
    // la barra del done-when del goal (>= 8x a M >= 16) e' gia' vera qui, prima
    // della run: la formula la garantisce, la run la conferma sul piano vero
    expect(ratio(16)).toBeGreaterThanOrEqual(8);
    expect(ratio(32)).toBeGreaterThanOrEqual(8);
  });

  it("i valori assoluti della FFN 4B a M=16 (numero, non proporzione)", () => {
    const multirow = dispatchWeightBytes({ form: "multirow", ...shape, M: 16 });
    expect(multirow).toBe(9216 * 1440);            // 13.271.040 B per chunk
    expect(dispatchWeightBytes({ form: "legacy", ...shape, M: 16 })).toBe(16 * multirow);
  });

  it("argomenti non validi falliscono invece di produrre un numero finto", () => {
    expect(() => dispatchWeightBytes({ form: "legacy", kind: "q4_0", K: K_FFN, N: 0, M: 1 })).toThrow(/N/);
    expect(() => dispatchWeightBytes({ form: "legacy", kind: "q4_0", K: K_FFN, N: N_FFN, M: 0 })).toThrow(/M/);
    expect(() => dispatchWeightBytes({ form: "legacy", kind: "q4_0", K: K_FFN, N: 1.5, M: 1 })).toThrow(/N/);
  });
});

describe("(b-bis) M e' l'estensione z del dispatch, non le righe utili del chunk", () => {
  // Il kernel batch non ha early-out per riga non valida (wgsl: `let xRB =
  // wid.z * K`, nessuna guardia), e il motore passa sempre `[gx, gy, M_MAX]`:
  // su un chunk parziale le fette in eccesso rileggono comunque la matrice.
  const M_MAX = 8;

  it("un prefill da 20 token paga 3 chunk PIENI, non 8+8+4", () => {
    const chunks = planPrefill(20, 0, M_MAX);
    expect(chunks.map((c) => c.rows)).toEqual([8, 8, 4]);   // l'ultimo e' parziale

    const perDispatch = (M: number): number =>
      dispatchWeightBytes({ form: "legacy", kind: "q4_0", K: K_FFN, N: N_FFN, M });
    const reale = chunks.reduce((s) => s + perDispatch(M_MAX), 0);
    const sbagliato = chunks.reduce((s, c) => s + perDispatch(c.rows), 0);

    expect(reale).toBe(3 * perDispatch(M_MAX));
    expect(sbagliato).toBeLessThan(reale);                 // chunk.rows sottostima
    expect(sbagliato / reale).toBeCloseTo(20 / 24, 12);
  });

  it("la sottostima colpisce SOLO legacy, quindi falsa anche il rapporto", () => {
    const leg = (M: number): number => dispatchWeightBytes({ form: "legacy", kind: "q4_0", K: K_FFN, N: N_FFN, M });
    const mul = (M: number): number => dispatchWeightBytes({ form: "multirow", kind: "q4_0", K: K_FFN, N: N_FFN, M });
    expect(mul(4)).toBe(mul(8));                            // M non entra: invariato
    expect(leg(4) / mul(4)).toBe(4);                        // rapporto apparente
    expect(leg(8) / mul(8)).toBe(8);                        // rapporto vero (z = M_MAX)
  });
});

describe("(c) il meter accumula sulla SEQUENZA dei dispatch, non su una formula", () => {
  it("perToken(M) = totalBytes/M e byForm separa i due contributi", () => {
    const m = createPrefillWeightMeter();
    expect(m.totalBytes()).toBe(0);
    expect(m.byForm()).toEqual({ legacy: 0, multirow: 0 });

    // sequenza mista come quella che il piano gemello spinge: FFN e attenzione
    // in multi-riga, i pesi non-q4_0 (K-quant, f32) restano legacy
    m.add({ form: "multirow", kind: "q4_0", K: K_FFN, N: N_FFN, M: 16 });   // gate
    m.add({ form: "multirow", kind: "q4_0", K: K_FFN, N: N_FFN, M: 16 });   // up
    m.add({ form: "multirow", kind: "q4_0", K: N_FFN, N: K_FFN, M: 16 });   // down
    m.add({ form: "legacy", kind: "q5_K", K: K_FFN, N: 512, M: 16 });       // ssm_out 4B
    m.add({ form: "legacy", kind: "q6_K", K: K_FFN, N: 512, M: 16 });       // resta legacy
    m.add({ form: "legacy", kind: "f32", K: K_FFN, N: 64, M: 16 });         // resta legacy

    const multirow = 2 * N_FFN * weightBytesPerRow("q4_0", K_FFN)
      + K_FFN * weightBytesPerRow("q4_0", N_FFN);
    const legacy = 16 * 512 * weightBytesPerRow("q5_K", K_FFN)
      + 16 * 512 * weightBytesPerRow("q6_K", K_FFN)
      + 16 * 64 * weightBytesPerRow("f32", K_FFN);

    expect(m.byForm()).toEqual({ legacy, multirow });
    expect(m.totalBytes()).toBe(legacy + multirow);
    expect(m.perToken(16)).toBe((legacy + multirow) / 16);
    expect(m.byForm().legacy + m.byForm().multirow).toBe(m.totalBytes());
  });

  it("su PIU' chunk perToken vuole i token totali, non l'M del chunk", () => {
    // il meter accumula l'intero piano: 24 token con M_MAX=8 ⇒ 3 dispatch nello
    // stesso totale. perToken(8) darebbe un numero 3 volte piu' grande.
    const M_MAX = 8, nTokens = 24;
    const m = createPrefillWeightMeter();
    const chunks = planPrefill(nTokens, 0, M_MAX);
    expect(chunks).toHaveLength(3);
    for (const _c of chunks) m.add({ form: "legacy", kind: "q4_0", K: K_FFN, N: N_FFN, M: M_MAX });

    const unChunk = dispatchWeightBytes({ form: "legacy", kind: "q4_0", K: K_FFN, N: N_FFN, M: M_MAX });
    expect(m.totalBytes()).toBe(3 * unChunk);
    expect(m.perToken(nTokens)).toBe(3 * unChunk / 24);
    expect(m.perToken(nTokens)).toBe(unChunk / M_MAX);      // per token, invariante ai chunk
    expect(m.perToken(M_MAX)).toBe(3 * m.perToken(nTokens)); // l'errore che il doc previene
  });

  it("perToken rifiuta M non valido e reset riazzera tutto", () => {
    const m = createPrefillWeightMeter();
    m.add({ form: "legacy", kind: "q4_0", K: K_FFN, N: N_FFN, M: 4 });
    expect(m.totalBytes()).toBeGreaterThan(0);
    expect(() => m.perToken(0)).toThrow(/M/);
    m.reset();
    expect(m.totalBytes()).toBe(0);
    expect(m.byForm()).toEqual({ legacy: 0, multirow: 0 });
    expect(m.perToken(16)).toBe(0);
  });

  it("byForm() restituisce una COPIA: mutarla non falsifica il meter", () => {
    const m = createPrefillWeightMeter();
    m.add({ form: "multirow", kind: "q4_0", K: K_FFN, N: N_FFN, M: 8 });
    const snap = m.byForm();
    snap.multirow = 0;
    expect(m.byForm().multirow).toBe(N_FFN * weightBytesPerRow("q4_0", K_FFN));
  });

  it("due meter sono indipendenti (nessuno stato globale condiviso)", () => {
    const a = createPrefillWeightMeter(), b = createPrefillWeightMeter();
    a.add({ form: "legacy", kind: "q4_0", K: K_FFN, N: N_FFN, M: 1 });
    expect(b.totalBytes()).toBe(0);
  });

  it("il meter misura il RIUSO: stessa sequenza di shape, forme diverse", () => {
    // e' il confronto che il done-when 2 chiede — misurato sulla sequenza dei
    // dispatch, non dedotto da una formula scritta a parte
    const M = 16;
    const shapes = [
      { kind: "q4_0" as PrefillQuantKind, K: K_FFN, N: N_FFN },
      { kind: "q4_0" as PrefillQuantKind, K: K_FFN, N: N_FFN },
      { kind: "q4_0" as PrefillQuantKind, K: N_FFN, N: K_FFN },
    ];
    const leg = createPrefillWeightMeter(), mul = createPrefillWeightMeter();
    for (const s of shapes) {
      leg.add({ form: "legacy", ...s, M });
      mul.add({ form: "multirow", ...s, M });
    }
    expect(leg.totalBytes() / mul.totalBytes()).toBe(M);
    expect(leg.perToken(M)).toBe(mul.totalBytes());   // legacy per token = tutta la passata
  });
});

describe("(d) il modulo resta PURO: niente GPU, testabile in CI senza device", () => {
  const src = readFileSync(join(__dirname, "../src/engine/prefillbytes.ts"), "utf8");

  it("nessun createBuffer (resta fuori dall'allowlist di engine-one-mechanism)", () => {
    expect(/\bcreateBuffer\b/.test(src)).toBe(false);
  });

  it("nessun import dai kernel WGSL", () => {
    expect(/from\s*["'][^"']*kernels\/wgsl/.test(src)).toBe(false);
    expect(/import\s*\(\s*["'][^"']*kernels\/wgsl/.test(src)).toBe(false);
  });

  it("nessun import affatto: il contatore non trascina il repack", () => {
    expect(/^\s*import\s/m.test(src)).toBe(false);
  });

  it("la sorgente e' stata letta davvero (guard: senza questo, un path sbagliato sarebbe verde)", () => {
    expect(src).toContain("weightBytesPerRow");
    expect(src.length).toBeGreaterThan(200);
  });
});
