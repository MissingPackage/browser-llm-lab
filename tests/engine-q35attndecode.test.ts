import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import {
  attnDecodeWgsl, attnDecodeRefWgsl, attnDecodeCombineWgsl, attnDecodeWorkgroupStorageBytes,
  attnDecodeLegacyBatchWgsl,
} from "../src/engine/kernels/wgsl";
import { q35AttnSplitPlan } from "../src/engine/q35attnsplit";
import { ATTN_CHUNK_REL_TOL, ATTN_CHUNK_ABS_TOL } from "../src/engine/attnchunktol";

interface LegacyShape { nHead: number; nKvHead: number; headDim: number; ctxMax: number }
interface LegacyCase { opts: LegacyShape; batch: string; ref: string; batchStream: string }
// Generato UNA VOLTA da `git show HEAD:src/engine/kernels/wgsl.ts` (vedi campo
// `source`): e' il kernel di IERI congelato su disco. Non si rigenera — se un
// giorno il ramo batch cambia, questo test deve diventare rosso.
//
// E' successo, DI PROPOSITO (T1-kernel-batch-streaming): il ramo batch di
// `attnDecodeWgsl` non instrada piu' al legacy. Le chiavi `batch` e `ref` NON
// sono state rigenerate — restano il byte per byte di ieri — ma `batch` ora e'
// prodotta dal nuovo export `attnDecodeLegacyBatchWgsl`, che e' il fallback
// dichiarato e il termine di paragone del debito in gpulimits. La chiave NUOVA
// `batchStream` congela il testo streaming che il ramo batch emette da oggi.
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

/**
 * Byte di workgroup storage LETTI dal testo del kernel: `array<f32, N>` conta
 * 4·N, `array<vec4<f32>, N>` conta 16·N. Ogni `var<workgroup>` deve essere
 * riconosciuta — se domani una dichiarazione cambia tipo (u32, matrici, un
 * singolo vec4 fuori array) la somma non deve poter calare in silenzio: si
 * asserisce che il numero di dichiarazioni CONTATE sia quello delle
 * dichiarazioni PRESENTI.
 */
const workgroupF32Bytes = (src: string): number => {
  const found = [...src.matchAll(/var<workgroup>\s+\w+\s*:\s*array<(vec4<f32>|f32),\s*(\d+)>/g)];
  const all = [...src.matchAll(/var<workgroup>/g)];
  expect(found.length, "var<workgroup> non riconosciute dal contatore").toBe(all.length);
  return found.reduce((s, m) => s + Number(m[2]) * (m[1] === "f32" ? 4 : 16), 0);
};

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

// ---------------------------------------------------------------------------
// RAMO BATCH (attenzione a chunk del prefill) — T1-kernel-batch-streaming.
//
// `attnDecodeWgsl({ batch: true })` non instrada piu' al legacy: emette la
// forma con softmax in STREAMING a tile di 64 posizioni e letture vec4. Cio'
// che NON cambia (l'assemblatore di q35gpumodel non si tocca): i binding
// [q, kCache, vCache, out, rowPast], `@workgroup_size(64)`, la griglia
// [nHead, M, 1] con wid.x = head e wid.y = riga del chunk, `nPast` per riga da
// `rowPast[wid.y]`, la causalita' dalla sola `rowPast` (nessuna maschera).
//
// VINCOLO NEGATIVO (docket item 21): NON si fondono le teste del gruppo GQA e
// NON si legge la KV una volta per gruppo. Misurato: 2,0879 contro 1,8207 ms a
// ctx 6333 — taglia il traffico 4x ma scende da 256 a 64 workgroup su 76 SM.
// Un workgroup per (head, riga) resta la forma giusta, ed e' cio' che il test
// dei binding e della griglia inchioda.
// ---------------------------------------------------------------------------
describe("attnDecodeWgsl batch — softmax in streaming, un workgroup per (head, riga)", () => {
  const BATCH = { ...ENGINE, batch: true } as const;

  it("[a] niente array di workgroup in ctxMax, e le letture sono vec4", () => {
    const src = attnDecodeWgsl(BATCH);
    expect(src).not.toContain(`array<f32, ${ENGINE.ctxMax}>`);
    expect(src).not.toContain("scores: array<f32");
    expect(src).toContain("vec4<f32>");
  });

  it("[b] il workgroup storage LETTO dal testo e' la formula, e non dipende da ctxMax", () => {
    for (const ctxMax of CTX_CASES) {
      expect(workgroupF32Bytes(attnDecodeWgsl({ ...ENGINE, ctxMax, batch: true })), `ctxMax ${ctxMax}`)
        .toBe(attnDecodeWorkgroupStorageBytes(ctxMax));
    }
    // costante in ctxMax, alle due estremita' del done-when
    expect(workgroupF32Bytes(attnDecodeWgsl({ ...ENGINE, ctxMax: 4_096, batch: true })))
      .toBe(workgroupF32Bytes(attnDecodeWgsl({ ...ENGINE, ctxMax: 131_072, batch: true })));
    // ...e la formula vale ora per ENTRAMBI i rami: nessuna seconda aritmetica
    expect(workgroupF32Bytes(attnDecodeWgsl(ENGINE)))
      .toBe(workgroupF32Bytes(attnDecodeWgsl(BATCH)));
  });

  it("[c] binding congelati [q, kCache, vCache, out, rowPast] e griglia (head, riga)", () => {
    const src = attnDecodeWgsl(BATCH);
    const b = bindings(src);
    expect(b.map((x) => x.binding)).toEqual([0, 1, 2, 3, 4]);
    expect(b.map((x) => x.name)).toEqual(["q", "kCache", "vCache", "out", "rowPast"]);
    expect(b.map((x) => x.space)).toEqual([
      "storage, read", "storage, read", "storage, read", "storage, read_write", "storage, read",
    ]);
    expect(b.slice(0, 4).map((x) => x.type)).toEqual(Array(4).fill("array<vec4<f32>>"));
    expect(b[4].type).toBe("array<u32>");
    expect(src).toContain("@compute @workgroup_size(64)");
    expect(src).toContain("let h = wid.x;");
    expect(src).toContain("let row = wid.y;");
    // nPast PER RIGA dalla sola rowPast: nessuna maschera, nessun uniform
    expect(src).toContain("let n = rowPast[row] + 1u;");
    expect(src).not.toContain("var<uniform>");
  });

  it("[e] anche sulla shape del ktest (headDim 32) il ramo batch e' costante in ctxMax", () => {
    const KT = { nHead: 4, nKvHead: 2, headDim: 32 };
    expect(() => attnDecodeWgsl({ ...KT, ctxMax: 64, batch: true })).not.toThrow();
    const src = attnDecodeWgsl({ ...KT, ctxMax: 64, batch: true });
    expect(src).toContain("const HD4 = 8u;");
    expect(workgroupF32Bytes(src)).toBe(attnDecodeWorkgroupStorageBytes(64, 32));
    expect(workgroupF32Bytes(attnDecodeWgsl({ ...KT, ctxMax: 131_072, batch: true })))
      .toBe(workgroupF32Bytes(src));
    // stessa postura di attnDecodeStreamWgsl con HD4 < 64: guardie sugli
    // accessi, barriere in control flow uniforme
    expect(src).toContain("t < HD4");
    for (const line of src.split("\n")) {
      if (line.includes("workgroupBarrier()")) {
        expect(line.trim().startsWith("workgroupBarrier();"), `barriera non uniforme: ${line}`).toBe(true);
      }
    }
  });

  it("[e] il testo del ramo batch e' congelato dal golden (chiave batchStream)", () => {
    expect(attnDecodeWgsl({ ...golden.opts, batch: true })).toBe(golden.batchStream);
    expect(attnDecodeWgsl({ ...golden.ktest.opts, batch: true })).toBe(golden.ktest.batchStream);
  });
});

describe("prefill e riferimento del ktest — il kernel di ieri, byte per byte", () => {
  it("[d] attnDecodeLegacyBatchWgsl() === golden.batch", () => {
    expect(attnDecodeLegacyBatchWgsl(golden.opts)).toBe(golden.batch);
    expect(golden.batch).toContain("scores: array<f32"); // il golden e' davvero il vecchio kernel
    // il fallback dichiarato NON e' piu' cio' che `attnDecodeWgsl` instrada
    expect(attnDecodeWgsl({ ...golden.opts, batch: true })).not.toBe(golden.batch);
  });

  it("[d] attnDecodeRefWgsl() === golden.ref", () => {
    expect(attnDecodeRefWgsl(golden.opts)).toBe(golden.ref);
    expect(golden.ref).toContain("scores: array<f32");
  });

  it("[d] anche sulla shape del ktest (headDim 32) i due rami legacy sono invariati", () => {
    expect(attnDecodeLegacyBatchWgsl(golden.ktest.opts)).toBe(golden.ktest.batch);
    expect(attnDecodeRefWgsl(golden.ktest.opts)).toBe(golden.ktest.ref);
  });

  it("batch e ref escono dallo STESSO template: differiscono solo per il binding 4 e l'offset di riga", () => {
    const b = bindings(attnDecodeLegacyBatchWgsl(golden.opts));
    const r = bindings(attnDecodeRefWgsl(golden.opts));
    expect(b.map((x) => x.name)).toEqual(["q", "kCache", "vCache", "out", "rowPast"]);
    expect(r.map((x) => x.name)).toEqual(["q", "kCache", "vCache", "out", "P"]);
    // stesso numero di righe: e' un template solo, non due copie divergenti
    expect(golden.batch.split("\n").length).toBe(golden.ref.split("\n").length);
  });
});

// ---------------------------------------------------------------------------
// L'UNICA VERIFICA GPU DEL RAMO BATCH: il banco `dense-batch-attn-chunk` del
// ktest. Non gira qui (serve una GPU), quindi cio' che si puo' verificare da
// vitest e' la sua FORMA — ed e' la forma che la riscrittura ha reso sbagliata
// in due modi.
//
// (1) IL METODO DI CONFRONTO. Il banco confrontava il ramo batch col
// riferimento per-riga BIT A BIT (`Object.is`), e poteva farlo perche' i due
// erano gemelli strutturali: stesso template, stesso ordine di accumulo. La
// forma in streaming cambia l'ordine (rescale online + `dot()` su vec4), quindi
// il confronto bit a bit e' rosso PER COSTRUZIONE. La riga 3 (d) di PHASES del
// goal engine-ttft lo prevede alla lettera: «tolleranza dichiarata PRIMA».
// Lasciarlo bit a bit non e' un test severo: e' un gate permanente che non
// distingue piu' fra un errore e la riscrittura che si e' appena approvata.
//
// (2) LA COPERTURA. Il banco gira a `basePast` 9 con M righe, cioe' n = 10..12
// a fronte di tile da 64 posizioni: UN tile solo. Tutto il codice nuovo di
// questa riscrittura — il rescale online FRA tile (`rs = exp(m - nm)`,
// `s = s * rs + red[0]`, `acc = acc * rs`) — non verrebbe eseguito mai, ne'
// prima ne' dopo il passaggio a tolleranza. Al primo giro `rs` vale 0 per
// costruzione (m parte da NEG): il primo comportamento non banale compare a
// n >= 65. Serve un caso con n > 64, o la verifica GPU passa senza toccare cio'
// che e' cambiato.
// ---------------------------------------------------------------------------
describe("ktest `dense-batch-attn-chunk`: confronto a tolleranza e almeno due tile", () => {
  const KT = readFileSync(join(process.cwd(), "src/engine/ktest/ktest.worker.ts"), "utf8");
  const start = KT.indexOf("{ // attnDecode a chunk");
  const bank = KT.slice(start, KT.indexOf("{ // deltanet conv+core", start));

  it("il banco esiste ed e' ancora quello che dispatcha il ramo batch", () => {
    expect(start, "blocco `{ // attnDecode a chunk` in ktest.worker.ts").toBeGreaterThan(0);
    expect(bank).toContain("attnDecodeWgsl(");
    expect(bank).toContain("batch: true");
    expect(bank).toContain("attnDecodeRefWgsl(");
  });

  it("NON confronta piu' bit a bit: l'ordine di accumulo e' cambiato per costruzione", () => {
    expect(bank, "bitCmp sul ramo batch in streaming: rosso per costruzione")
      .not.toMatch(/bitCmp\(\s*["'`]dense-batch-attn-chunk/);
  });

  it("la tolleranza e' DICHIARATA in una sede unica, e da li' ARRIVA al comparatore", () => {
    // Prima questo controllo leggeva i due numeri dal SORGENTE del banco con una
    // regex. Era piu' debole di quanto sembrasse: verificava che nel testo ci
    // fosse una costante, non che il banco usasse QUELLA. Ora i valori arrivano
    // dall'import — se qualcuno ne scrive una copia locale, il `not.toMatch`
    // qui sotto lo vede.
    expect(ATTN_CHUNK_REL_TOL, "banda relativa: >= 2,5x il caso peggiore simulato (4,26e-6), sotto un bug strutturale")
      .toBeLessThan(1e-3);
    expect(ATTN_CHUNK_ABS_TOL, "banda assoluta: per le componenti vicine a zero").toBeLessThan(1e-3);
    expect(ATTN_CHUNK_REL_TOL).toBeGreaterThan(4.26e-6);
    expect(ATTN_CHUNK_ABS_TOL).toBeGreaterThan(2.98e-8);

    expect(bank, "le due tolleranze devono ARRIVARE al comparatore").toMatch(/ATTN_CHUNK_REL_TOL,\s*ATTN_CHUNK_ABS_TOL/);
    expect(bank, "e NON essere ridichiarate in loco: una soglia, un posto (docket item 23)")
      .not.toMatch(/const\s+ATTN_CHUNK_(REL|ABS)_TOL\s*=/);
  });

  it("la sede unica e' importata dal worker, non ricopiata", () => {
    const KTALL = readFileSync(join(process.cwd(), "src/engine/ktest/ktest.worker.ts"), "utf8");
    expect(KTALL, "import da ../attnchunktol").toMatch(
      /import\s*\{[^}]*ATTN_CHUNK_REL_TOL[^}]*ATTN_CHUNK_ABS_TOL[^}]*\}\s*from\s*["'`]\.\.\/attnchunktol["'`]/);
  });

  it("copre il rescale online FRA tile: almeno un caso con n > 64", () => {
    const cases = [...bank.matchAll(/\{\s*ctxMax:\s*(\d+),\s*basePast:\s*(\d+)/g)]
      .map((m) => ({ ctxMax: Number(m[1]), basePast: Number(m[2]) }));
    expect(cases.length, "lista dei casi del banco, leggibile dal sorgente").toBeGreaterThanOrEqual(2);
    // n della riga 0 = basePast + 1; il tile e' da 64 posizioni
    const multiTile = cases.filter((c) => c.basePast + 1 > 64);
    expect(multiTile.length, `casi con n > 64 fra ${JSON.stringify(cases)}`).toBeGreaterThanOrEqual(1);
    // e la cache deve contenerli davvero
    for (const c of cases) expect(c.ctxMax, `ctxMax ${c.ctxMax} < n`).toBeGreaterThan(c.basePast + 2);
    // il caso corto resta: e' l'unico che prova il tile parziale singolo
    expect(cases.some((c) => c.basePast + 1 <= 64), "caso a UN tile ancora presente").toBe(true);
  });
});
