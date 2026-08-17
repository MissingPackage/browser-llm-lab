// Il banco della fase 0 di `engine-kquant`, verificato SENZA GPU.
//
// Cosa protegge, in ordine di quanto costerebbe scoprirlo tardi:
//  [a] il braccio di paragone e' davvero il kernel di PRODUZIONE, byte per
//      byte — se un giorno qualcuno lo sostituisse con una copia locale, la
//      fase 0 misurerebbe la copia e il rapporto della regola di stop
//      parlerebbe di un motore che non esiste;
//  [b] C0-4: lo split-K non si applica dove non puo' — la shape del down-proj
//      degli expert del 35B ha DUE superblocchi per riga, e una cella a 4 fette
//      sarebbe degenere per costruzione (il precedente che il protocollo cita:
//      una matrice di misure con celle impossibili, eseguite per conformita');
//  [c] la geometria dei formati combacia con `quant.ts`, non con se stessa;
//  [d] nessun kernel candidato sfora il minimo di spec WebGPU di memoria di
//      gruppo (16.384 B): e' la portabilita' che il goal accanto (0.5B) sta per
//      pretendere, e costa zero garantirla adesso invece che riscoprirla dopo;
//  [e] la via intera non scrive `enable packed_4x8_integer_dot_product` — e'
//      una language feature, non un'estensione, e scriverla fa fallire la
//      compilazione (costo' una run intera in it.5 del goal precedente).
import { describe, it, expect } from "vitest";
import {
  KQUANT_GEOM, KQUANT_SHAPES, KQUANT_SPLIT_K, KQUANT_WIRED, kquantSplitsFor, kquantVariants,
  kquantQ5KMultiRowSplitKIdotWgsl, kquantQ41MultiRowSplitKIdotWgsl,
} from "../src/microbench/ttKQuant";
import { gemvQ5KWgsl, gemvQuantWgsl } from "../src/engine/kernels/wgsl";
import { Q5_K_BLOCK_BYTES } from "../src/engine/quant";
import { weightBytesPerRow } from "../src/engine/prefillbytes";
import { workgroupStorageBytes } from "../src/microbench/ttGemm";

/** Il minimo che WebGPU garantisce su qualunque device conforme. */
const WEBGPU_GUARANTEED_WG_STORAGE = 16_384;

describe("[a] il braccio di paragone e' il kernel di produzione, importato", () => {
  it("q5_K: il legacy e' `gemvQ5KWgsl({batch: true})` byte per byte", () => {
    const v = kquantVariants({ family: "q5_K", K: 4096, N: 2560, M: 16 });
    expect(v[0].legacy).toBe(true);
    expect(v[0].id).toBe("base-batch-z");
    expect(v[0].code).toBe(gemvQ5KWgsl({ K: 4096, N: 2560, batch: true }));
  });

  it("q4_1: il legacy e' `gemvQuantWgsl({kind: q4_1, batch: true})` byte per byte", () => {
    const v = kquantVariants({ family: "q4_1", K: 9216, N: 2560, M: 16 });
    expect(v[0].legacy).toBe(true);
    expect(v[0].code).toBe(gemvQuantWgsl({ kind: "q4_1", K: 9216, N: 2560, hasBias: false, batch: true }));
  });

  it("q8_0: il paragone del DECODE e' `gemvQuantWgsl` SENZA batch, byte per byte", () => {
    // stessa protezione del [a] sopra, ma per l'altro regime: il decode emette
    // `gemv` (q35gpumodel.ts:862) e non la forma a M righe su wid.z
    const v = kquantVariants({ family: "q8_0", K: 2048, N: 8192, M: 1 });
    const dec = v.find((x) => x.regime === "decode")!;
    expect(dec.id).toBe("base-decode");
    expect(dec.code).toBe(gemvQuantWgsl({ kind: "q8_0", K: 2048, N: 8192, hasBias: false }));
    // e NON deve coincidere col paragone del prefill: sono due kernel diversi,
    // ed e' l'intero motivo per cui esistono due bracci
    expect(dec.code).not.toBe(v[0].code);
  });

  it("UN denominatore PER REGIME, mai due nello stesso — e il prefill e' il primo", () => {
    // La regola vecchia era «esattamente un legacy, ed e' il primo». Da it.21 i
    // paragoni sono due, uno per regime: il rapporto della regola di stop resta
    // calcolabile solo se dentro OGNI regime il denominatore e' unico.
    for (const s of KQUANT_SHAPES) {
      for (const M of s.Ms) {
        const v = kquantVariants({ family: s.family, K: s.K, N: s.N, M });
        const where = `${s.family}@M${M}`;
        expect(v.filter((x) => x.regime === "prefill").length, where).toBe(1);
        expect(v[0].regime, where).toBe("prefill");
        // il decode esiste SOLO a M=1: a M>1 non c'e' decode da misurare, e un
        // braccio senza batch con M righe sarebbe una forma che nessuno emette
        expect(v.filter((x) => x.regime === "decode").length, where).toBe(M === 1 ? 1 : 0);
        // ogni braccio di paragone dichiara il suo regime, ogni candidato no:
        // un legacy senza regime tornerebbe a essere un denominatore ambiguo
        for (const a of v) expect(a.legacy, `${where}/${a.id}`).toBe(a.regime !== undefined);
        // senza denominatore la regola di stop non e' calcolabile
        expect(v.length).toBeGreaterThanOrEqual(2);
      }
    }
  });
});

describe("[b] C0-4: lo split-K non si applica dove la shape non lo consente", () => {
  it("4B ssm_out (K=4096, 16 superblocchi per riga) → 4 fette", () => {
    expect(kquantSplitsFor("q5_K", 4096)).toBe(KQUANT_SPLIT_K);
  });

  it("35B expert down (K=512, DUE superblocchi per riga) → 2 fette, mai 4", () => {
    // La cella a 4 fette non e' "un po' peggio": non esiste. Emetterla
    // produrrebbe un kernel che rifiuta e una riga di matrice vuota.
    expect(kquantSplitsFor("q5_K", 512)).toBe(2);
    expect(kquantSplitsFor("q5_K", 512)).not.toBe(KQUANT_SPLIT_K);
  });

  it("una shape con un solo superblocco per riga → nessuno split-K", () => {
    expect(kquantSplitsFor("q5_K", 256)).toBe(1);
  });

  it("K non multiplo dell'unita' del formato e' un errore, non un arrotondamento", () => {
    expect(() => kquantSplitsFor("q5_K", 4000)).toThrow(/non multiplo di 256/);
    expect(() => kquantSplitsFor("q4_1", 40)).toThrow(/non multiplo di 32/);
  });

  it("le fette dichiarate dalla lista sono quelle che `kquantSplitsFor` calcola", () => {
    for (const s of KQUANT_SHAPES) {
      const want = kquantSplitsFor(s.family, s.K);
      for (const v of kquantVariants({ family: s.family, K: s.K, N: s.N, M: 16 })) {
        if (!v.legacy) expect(v.splits, `${s.family}/${v.id}`).toBe(want);
      }
    }
  });
});

describe("[b2] cosa si cabla e cosa si misura soltanto — la decisione, resa meccanica", () => {
  it("OGNI famiglia ha entrambe le vie — e' un CONSTRAINT del contratto, non una scelta", () => {
    // GOAL.md §CONSTRAINTS: «Ogni via intera nuova va accompagnata dal suo
    // fallback f32 DICHIARATO, come la q4_0». In it.2 avevo saltato i tre
    // fallback delle famiglie non cablate con una decisione mia; il
    // verificatore l'ha bocciata perche' restringere un vincolo del contratto
    // non e' autorita' dell'agente. Questo test e' la forma meccanica del
    // vincolo: adesso non si puo' piu' saltare per distrazione.
    for (const s of KQUANT_SHAPES) {
      for (const M of s.Ms) {
        const ids = kquantVariants({ family: s.family, K: s.K, N: s.N, M }).map((v) => v.id);
        expect(ids, `${s.family}@M${M}`).toContain("splitk-idot");
        expect(ids, `${s.family}@M${M}`).toContain("splitk-f32");
      }
    }
  });

  it("il done-when copre M = 1, 8, 16 su TUTTE le famiglie", () => {
    // Anche questa era stata ristretta a M=16 sulle tre ereditate, e anche
    // questa e' testo del contratto («a M = 1, 8, 16»).
    //
    // CONTENIMENTO, NON UGUAGLIANZA (corretto il 2026-08-17). Il ratchet esiste
    // per impedire che qualcuno RESTRINGA la griglia — e' quello che era gia'
    // successo (ristretta a M=16) ed e' cio' che il contratto vieta. Ma
    // `toEqual` vietava anche di ALLARGARLA, che il contratto non dice da
    // nessuna parte: aggiungere M=2 e M=4 non toglie niente al done-when, lo
    // soddisfa e misura in piu'. Un ratchet che blocca i miglioramenti oltre ai
    // peggioramenti e' un ratchet mal scritto, e questo lo era.
    for (const s of KQUANT_SHAPES) {
      for (const m of [1, 8, 16]) expect(s.Ms, `${s.family} K${s.K} deve ancora coprire M=${m}`).toContain(m);
    }
  });

  it("le famiglie cablate sono ESATTAMENTE quelle che il 4B ha sul percorso vecchio", () => {
    // 24 ssm_out Q5_K + 4 ffn_down Q4_1: e' l'inventario pinnato del 4B, e i 48
    // siti Q8_0 restano esclusi coi numeri (N=32, 0,204% dei byte).
    expect([...KQUANT_WIRED].sort()).toEqual(["q4_1", "q5_K"]);
  });

  it("q6_K: 212 byte sul device, non i 210 del file — il pad e' del kernel", () => {
    expect(KQUANT_GEOM.q6_K.deviceBytes).toBe(KQUANT_GEOM.q6_K.words * 4);
    expect(weightBytesPerRow("q6_K", 512)).toBe((512 / 256) * 212);
  });
});

describe("[c] la geometria dei formati viene da quant.ts, non da se stessa", () => {
  it("q5_K: 256 pesi in 176 B = 44 parole", () => {
    expect(KQUANT_GEOM.q5_K.deviceBytes).toBe(Q5_K_BLOCK_BYTES);
    expect(KQUANT_GEOM.q5_K.words * 4).toBe(Q5_K_BLOCK_BYTES);
    // e i byte per riga combaciano col meter che il goal usa per la copertura
    expect(weightBytesPerRow("q5_K", 4096)).toBe((4096 / 256) * KQUANT_GEOM.q5_K.deviceBytes);
  });

  it("q4_1: 32 pesi in 20 B device (16 di nibble + d,m)", () => {
    expect(weightBytesPerRow("q4_1", 9216)).toBe((9216 / 32) * KQUANT_GEOM.q4_1.deviceBytes);
  });
});

describe("[d] nessun candidato sfora il minimo di spec WebGPU", () => {
  it("tutte le forme multi-riga stanno sotto 16.384 B di memoria di gruppo", () => {
    for (const s of KQUANT_SHAPES) {
      for (const M of s.Ms) {
        for (const v of kquantVariants({ family: s.family, K: s.K, N: s.N, M })) {
          if (v.legacy) continue;
          const b = workgroupStorageBytes(v.code);
          expect(b, `${s.family}/${v.id}@M${M} chiede ${b} B`).toBeLessThanOrEqual(WEBGPU_GUARANTEED_WG_STORAGE);
        }
      }
    }
  });

  it("il fabbisogno cresce con M e non con K: e' la proprieta' che lo tiene basso", () => {
    const at = (M: number): number =>
      workgroupStorageBytes(kquantQ5KMultiRowSplitKIdotWgsl({ family: "q5_K", K: 4096, N: 2560, M, splits: 4 }));
    expect(at(16)).toBeGreaterThan(at(8));
    const wideK = workgroupStorageBytes(
      kquantQ5KMultiRowSplitKIdotWgsl({ family: "q5_K", K: 8192, N: 2560, M: 16, splits: 4 }));
    expect(wideK).toBe(at(16));
  });
});

describe("[e] la via intera non scrive `enable`", () => {
  it("nessun kernel candidato contiene una direttiva enable", () => {
    for (const s of KQUANT_SHAPES) {
      for (const v of kquantVariants({ family: s.family, K: s.K, N: s.N, M: 16 })) {
        expect(v.code, `${s.family}/${v.id}`).not.toMatch(/^\s*enable\s/m);
      }
    }
  });

  it("solo i bracci `idot` usano dot4I8Packed", () => {
    for (const s of KQUANT_SHAPES) {
      for (const v of kquantVariants({ family: s.family, K: s.K, N: s.N, M: 16 })) {
        expect(v.code.includes("dot4I8Packed"), `${s.family}/${v.id}`).toBe(v.idot);
      }
    }
  });

  it("il termine Sigma(x) esiste in ENTRAMBE le vie intere: senza, il formato e' sbagliato", () => {
    // Q5_K sottrae dmin*mn*Sigma(x), q4_1 somma m*Sigma(x). E' la correzione
    // C0-3: e' l'unico pezzo di aritmetica che la q4_0 non ha, quindi e'
    // l'unico che un port distratto dimenticherebbe.
    const q5 = kquantQ5KMultiRowSplitKIdotWgsl({ family: "q5_K", K: 4096, N: 2560, M: 16, splits: 4 });
    const q41 = kquantQ41MultiRowSplitKIdotWgsl({ family: "q4_1", K: 9216, N: 2560, M: 16, splits: 4 });
    for (const [name, code] of [["q5_K", q5], ["q4_1", q41]] as const) {
      expect(code, name).toContain("xsum");
      expect(code, name).toContain("dot4I8Packed(0x01010101u");
    }
    expect(q5).toMatch(/- min1 \* xsum|- *min1 \* xsum/);
    expect(q41).toContain("+ dm.y * xsum");
  });
});
