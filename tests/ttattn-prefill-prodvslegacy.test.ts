// IL PRIMA E IL DOPO DELLA SONDA `attn-prefill-chunk` (AC2 della riga 3 di
// engine-ttft): la lista delle varianti deve contenere DUE bracci confrontabili
// — `legacy`, la forma che il prefill del 4B usava fino al task
// T1-kernel-batch-streaming, e `prod`, la forma che il motore usa DA OGGI.
//
// Perche' un test di sorgente e non una misura: la sonda gira solo su GPU e il
// suo esito e' un JSON che nessun gate rilegge. Se il braccio `legacy` smettesse
// di fabbricare il legacy (o se `prod` non fosse il kernel di produzione), il
// numero uscirebbe lo stesso — solo falso. Questo file e' l'unico sensore che
// vede la differenza SENZA una GPU: verifica che i due bracci siano davvero due
// kernel diversi, e che siano quei due.
//
// EREDITA' — l'ultimo describe («la lista dice il vero su se stessa») raccoglie
// i tre guardiani che stavano in tests/ttprobe.test.ts finche' la lista viveva
// dentro ttRunner.ts: nessuna chiamata a `attnDecodeWgsl({batch:true})` che
// fabbrichi la baseline, i siti `const legacyCode =` tutti su
// `attnDecodeLegacyBatchWgsl`, e la promessa della stringa `ctx` verificata
// contro il codice. Sono qui perche' la lista e' qui.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { attnDecodeWgsl, attnDecodeLegacyBatchWgsl } from "../src/engine/kernels/wgsl";
import { workgroupStorageBytes } from "../src/microbench/ttGemm";
import { TT_ATTN_SHAPE, prefillAttnVariants } from "../src/microbench/ttAttn";
import { ATTN_CTX, ATTN_M } from "../src/microbench/ttRunner";

const variants = () => prefillAttnVariants();
const byId = (id: string) => {
  const v = variants().find((x) => x.id === id);
  expect(v, `variante "${id}" nella lista di prefillAttnVariants()`).toBeTruthy();
  return v!;
};

describe("prefillAttnVariants: il prima e il dopo sono nella stessa lista", () => {
  it("(a) contiene sia `legacy` sia `prod`", () => {
    const ids = variants().map((v) => v.id);
    expect(ids, `ids: ${ids.join(", ")}`).toContain("legacy");
    expect(ids).toContain("prod");
    // e non sono la stessa cosa scritta due volte
    expect(byId("prod").code).not.toBe(byId("legacy").code);
  });

  it("(a bis) i due bracci si dispacciano sulla stessa griglia [nHead, M, 1]", () => {
    for (const id of ["legacy", "prod"]) {
      expect(byId(id).grid, `griglia di ${id}`).toEqual([TT_ATTN_SHAPE.nHead, ATTN_M, 1]);
    }
  });

  it("(b) `legacy` E' il kernel di ieri (scores[ctxMax] in workgroup memory)", () => {
    expect(byId("legacy").code).toContain(`array<f32, ${TT_ATTN_SHAPE.ctxMax}>`);
    expect(byId("legacy").code).toContain(`array<f32, 6400>`);
    // ...e viene da li' byte per byte, non da una copia locale
    expect(byId("legacy").code).toBe(attnDecodeLegacyBatchWgsl(TT_ATTN_SHAPE));
  });

  it("(b) `prod` NON lo contiene, ed E' il kernel di produzione di oggi", () => {
    expect(byId("prod").code).not.toContain(`array<f32, ${TT_ATTN_SHAPE.ctxMax}>`);
    expect(byId("prod").code).not.toContain(`array<f32, 6400>`);
    expect(byId("prod").code).toBe(attnDecodeWgsl({ ...TT_ATTN_SHAPE, batch: true }));
  });

  it("(c) il workgroup storage di `prod` e' COSTANTE in ctxMax e sta sotto i 16.384 B", () => {
    // `wgs` e' il braccio com'e' misurato: ctxMax 6.400, quello del 4B
    const wgs = workgroupStorageBytes(byId("prod").code);
    expect(wgs).toBeLessThanOrEqual(16_384);
    // la costanza si misura su ctxMax DIVERSI da quello del braccio, altrimenti
    // si riafferma la stessa stringa: 12.224 (un'altra taglia) e 131.072 (20x)
    for (const ctxMax of [12_224, 131_072]) {
      expect(workgroupStorageBytes(attnDecodeWgsl({ ...TT_ATTN_SHAPE, ctxMax, batch: true })), `prod a ctxMax ${ctxMax}`)
        .toBe(wgs);
    }
  });

  it("(c) quello di `legacy` e' 4*ctxMax + 256 — ed e' il debito che il dopo chiude", () => {
    expect(workgroupStorageBytes(byId("legacy").code)).toBe(4 * TT_ATTN_SHAPE.ctxMax + 256);
    for (const ctxMax of [12_224, 131_072]) {
      expect(workgroupStorageBytes(attnDecodeLegacyBatchWgsl({ ...TT_ATTN_SHAPE, ctxMax })), `legacy a ctxMax ${ctxMax}`)
        .toBe(4 * ctxMax + 256);
    }
  });
});

describe("i due contesti che AC2 pretende", () => {
  it("(d) ATTN_CTX = [388, 6333] e ATTN_M = 16", () => {
    expect(ATTN_CTX).toEqual([388, 6333]);
    expect(ATTN_M).toBe(16);
    // 6333 e' il contesto lungo che AC2 chiede (>= 6000): non un numero tondo
    expect(Math.max(...ATTN_CTX)).toBeGreaterThanOrEqual(6_000);
  });
});

const microbenchSrc = (f: string): string =>
  readFileSync(join(process.cwd(), "src/microbench", f), "utf8");
/** solo il CODICE: i commenti raccontano la storia, le chiamate la fanno */
const codeOf = (src: string): string =>
  src.replace(/\/\*[\s\S]*?\*\//g, " ").replace(/^\s*\/\/.*$/gm, "");

describe("il runner consuma la lista, e non ne tiene una sua", () => {
  const runnerSrc = microbenchSrc("ttRunner.ts");
  const codeOnly = codeOf(runnerSrc);

  it("(e) ttRunner importa prefillAttnVariants da ./ttAttn", () => {
    expect(codeOnly).toMatch(/import \{[^}]*prefillAttnVariants[^}]*\} from "\.\/ttAttn"/);
    expect(codeOnly).toContain("prefillAttnVariants()");
  });

  it("(e) ttRunner non fabbrica piu' nessun kernel con attnDecodeWgsl(batch: true)", () => {
    const hits = [...codeOnly.matchAll(/attnDecodeWgsl\([^)]*batch:\s*true/g)];
    expect(hits.length, `${hits.length} chiamate a attnDecodeWgsl con batch: true`).toBe(0);
  });

  it("(e) sweepPlan.attnLegacy punta al legacy, non alla forma nuova", () => {
    // il sito dello sweep del limite negoziabile: se misurasse la forma di oggi
    // chiamandola «legacy», il prima/dopo mentirebbe due volte
    const decls = [...codeOnly.matchAll(/const legacyCode = (\w+)\(/g)].map((m) => m[1]);
    expect(decls, `siti che fabbricano il codice della baseline: ${decls.join(", ")}`).toContain("attnDecodeLegacyBatchWgsl");
    expect(new Set(decls)).toEqual(new Set(["attnDecodeLegacyBatchWgsl"]));
    // la virgola distingue il LETTERALE (`wgsl: legacyCode,`) dalla
    // dichiarazione di tipo del piano (`wgsl: string;`)
    const attnLegacy = /attnLegacy: \{[\s\S]{0,200}?wgsl: (\w+),/.exec(codeOnly)?.[1];
    expect(attnLegacy, "sweepPlan.attnLegacy.wgsl").toBe("legacyCode");
  });

  it("(e) lo sweep del tetto negoziabile misura ancora una pressione VERA", () => {
    // guardiano ereditato da ttprobe: se la baseline dello sweep scendesse a
    // 1.536 B, ogni limite richiesto sarebbe soddisfatto e la sonda (d) non
    // misurerebbe piu' niente
    const lim = /requestedLimits: \[([\d, ]+)\]/.exec(runnerSrc)?.[1].split(",").map((s) => Number(s.trim()));
    expect(lim, "requestedLimits dello sweep").toBeTruthy();
    const need = workgroupStorageBytes(attnDecodeLegacyBatchWgsl(TT_ATTN_SHAPE));
    expect(Math.min(...lim!), "il limite piu' basso deve NON bastare alla baseline").toBeLessThan(need);
  });
});

// ---------------------------------------------------------------------------
// LA SONDA GIRA SOLO SU GPU, e il suo esito e' un JSON che nessun gate rilegge:
// se il braccio `legacy` smettesse di fabbricare il legacy, il confronto sarebbe
// streaming contro streaming e il numero uscirebbe lo stesso, solo falso. Questo
// blocco e' quel sensore, ora che la lista si e' spostata in ttAttn.ts: fino a
// oggi lo teneva tests/ttprobe.test.ts puntando a ttRunner.ts.
// ---------------------------------------------------------------------------
describe("la lista dice il vero su se stessa", () => {
  const attnCode = codeOf(microbenchSrc("ttAttn.ts"));
  const runnerCode = codeOf(microbenchSrc("ttRunner.ts"));

  it("l'unica chiamata a attnDecodeWgsl(batch: true) della sonda fabbrica `prod`", () => {
    const hits = [...attnCode.matchAll(/attnDecodeWgsl\([^)]*batch:\s*true/g)];
    expect(hits.length, `ttAttn.ts: ${hits.length} chiamate a attnDecodeWgsl con batch: true`).toBe(1);
    // ...e sta nel record `prod`, non in quello `legacy`
    const prodRecord = /id: "prod", code: ([^,]+),/.exec(attnCode)?.[1];
    expect(prodRecord, "record `prod`").toMatch(/attnDecodeWgsl\(/);
    const legacyRecord = /id: "legacy", code: ([^,]+),/.exec(attnCode)?.[1];
    expect(legacyRecord, "record `legacy`").toBe("attnDecodeLegacyBatchWgsl(TT_ATTN_SHAPE)");
  });

  it("nessuno dei due file nomina piu' il template privato ne' la riga del vecchio switch", () => {
    for (const [f, src] of [["ttAttn.ts", microbenchSrc("ttAttn.ts")], ["ttRunner.ts", microbenchSrc("ttRunner.ts")]] as const) {
      expect(src, `${f} nomina ancora attnDecodeLegacyWgsl / wgsl.ts:530`)
        .not.toMatch(/attnDecodeLegacyWgsl|wgsl\.ts:530/);
    }
    expect(runnerCode).not.toContain("attnPrefillStreamWgsl(");
  });

  it("cio' che i due bracci promettono nel JSON e' cio' che il loro codice contiene", () => {
    // la stringa `ctx` finisce nell'artefatto come DESCRIZIONE del braccio: se
    // promette scores[ctxMax], il codice deve averlo — e se promette che non ce
    // l'ha, non deve averlo
    const legacy = byId("legacy");
    expect(legacy.ctx).toContain("scores[ctxMax=");
    expect(legacy.code).toContain(`var<workgroup> scores: array<f32, ${TT_ATTN_SHAPE.ctxMax}>;`);
    const prod = byId("prod");
    expect(prod.ctx).toContain("niente scores[ctxMax]");
    expect(prod.code).not.toContain("scores: array<f32");
  });

  it("nessuna stringa `ctx` spaccia per verita' i millisecondi di UNA run", () => {
    // le `ctx` finiscono nell'artefatto: un numero misurato scritto li' dentro
    // resta congelato alla run che lo produsse (e le run del 13-08 sullo stesso
    // 4090 danno gqa 2,065-2,360 ms e stream 1,819-1,877 ms). Il VERDETTO si
    // scrive; la cifra si legge dal JSON.
    for (const v of variants()) {
      expect(v.ctx, `ctx di ${v.id}: cifra in ms congelata`).not.toMatch(/\d[.,]\d+\s*ms/);
    }
  });
});
