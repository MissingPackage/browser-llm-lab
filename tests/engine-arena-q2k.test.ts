import { describe, it, expect, beforeAll } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { GGML_TYPE, tensorByteSize, type GgufFile, type GgufTensorInfo } from "../src/engine/gguf";
import { q35ShapeFromGguf } from "../src/engine/q35shape";
import { q35ExpertTensor, q35MoeConfig } from "../src/engine/q35expertstore";
import { packExpertSlab } from "../src/engine/moe";
import {
  ExpertCache, arenaNeeds, expertSlots, minSlotsOf, moeParkOf, slabInHand, slotTableEntriesOf,
  type MoeModelConfig,
} from "../src/engine/residency";
import { mkMockDevice } from "./helpers/mock-gpu-device";

// LE CLASSI D'ARENA DEL `bartowski Q2_K` (spec 2026-08-17 §1 e §4 T6).
//
// PERCHE' QUESTO TEST ESISTE. Il 35B e' residency-bound: col quant UD-Q4_K_S il
// parco expert e' 17,074 GiB contro un'arena da 11,17 (65% residente) e la chat
// fa 11,5 tok/s contro i 40,06 misurati a zero miss. Col `Q2_K` di bartowski il
// parco scende a 10,391 GiB e ci sta INTERO. Quel «ci sta» e' una proprieta' di
// GEOMETRIA — byte per superblocco, padding a word, quali layer sono in quale
// classe, quali expert l'arena non deve tenere — e una geometria che regredisce
// non si vede a occhio: si vede come un tok/s piu' basso, settimane dopo, e
// nessuno la collega al commit che l'ha rotta. Qui i numeri misurati il
// 2026-08-17 diventano un gate.
//
// COSA E' E COSA NON E'. E' un test di PINNING: fissa numeri che oggi sono gia'
// giusti, e il suo valore e' tutto nel futuro. Non c'e' un rosso da mostrare in
// cima — non descrive un difetto — e infatti la sua tenuta e' stata provata
// mutando il codice di produzione (passo del superblocco Q3_K, riparto del
// budget) e verificando che diventasse rosso.
//
// SENZA GPU E SENZA RETE. L'header vero e' gia' un artefatto in repo
// (`results/eval/q35-header-dump-bartowski-2026-08-17.json`, prodotto da
// `scripts/q35-header-dump.mjs` leggendo i primi 64 MB del file via Range): da
// li' arrivano i metadata e l'inventario dei tensori expert, e il test ci
// ricostruisce sopra un header sintetico che il caso (0) RICONFRONTA con
// l'artefatto tensore per tensore. Un header inventato che non tornasse coi
// byte veri fallirebbe li', prima di poter mentire sul parco.

const GiB = 2 ** 30;
const DUMP = "results/eval/q35-header-dump-bartowski-2026-08-17.json";

/** Una riga del dump: aggregati per (categoria di tensore × tipo ggml). */
interface DumpEntry {
  file: string;
  bytes: number;
  nTensors: number;
  meta: Record<string, unknown>;
  typeHistogram: Record<string, { n: number; bytes: number; lettoDalMotore: boolean }>;
  expertParkBytes: number;
  expertParkGiB: number;
}

const dump = (): DumpEntry => {
  const all = JSON.parse(readFileSync(join(process.cwd(), DUMP), "utf8")) as DumpEntry[];
  const e = all.find((x) => /-Q2_K\.gguf$/.test(x.file));
  if (!e) throw new Error(`${DUMP}: manca la riga del file Q2_K`);
  return e;
};

/**
 * I metadata del dump, rimessi nella forma che il parser del motore produce.
 * Il dump COLLASSA gli array a `chiave#count` (un vocabolario da 248 320 voci in
 * un artefatto di review sarebbe illeggibile): qui si riespandono a array della
 * lunghezza giusta, perche' `q35ShapeFromGguf` legge `tokens.length` per il
 * vocabolario. I `#chars` (chat template) si scartano: nessuna shape li guarda.
 */
const metadataFromDump = (meta: Record<string, unknown>): Record<string, unknown> => {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(meta)) {
    if (k.endsWith("#count")) out[k.slice(0, -"#count".length)] = new Array(v as number).fill("");
    else if (k.endsWith("#chars")) continue;
    else out[k] = v;
  }
  return out;
};

/** Le tre voci expert dell'istogramma del dump, con i nomi che ci hanno. */
const HIST_KEY = {
  // `*` nel nome = tipo che il motore non sapeva leggere QUANDO il dump e'
  // stato preso; oggi lo legge, ed e' esattamente il lavoro di questa spec.
  q2k: "expert:Q2_K *", q3k: "expert:Q3_K *", q8: "expert:Q8_0",
} as const;

const histOf = (e: DumpEntry, k: string): { n: number; bytes: number } => {
  const h = e.typeHistogram[k];
  if (!h) throw new Error(`dump: istogramma senza "${k}"`);
  return { n: h.n, bytes: h.bytes }; // `lettoDalMotore` non e' geometria
};

/**
 * L'inventario dei tensori expert del file, ricostruito dal dump. FUNZIONE
 * PURA: nessuna asserzione qui dentro: gira al `describe`, e un `expect` a
 * tempo di collezione uscirebbe come errore di suite invece che come test
 * rosso. Il confronto con l'artefatto e' il caso (0).
 *
 * COSA DICE IL DUMP, alla lettera: `expert:Q2_K` 100 tensori / 8.808.038.400 B,
 * `expert:Q3_K` 20 / 2.306.867.200, `expert:Q8_0` 3 / 855.638.016. Sono 123 =
 * 41 blocchi × 3 (gate, up, down), cioe' 40 layer del modello piu' il blocco MTP
 * (`nextn_predict_layers` = 1), i cui tre expert sono i Q8_0.
 *
 * COSA IL DUMP NON DICE, e come si chiude il buco. L'istogramma e' aggregato:
 * non dice QUALI dei 120 tensori dei 40 layer veri sono Q3_K. L'unica
 * assegnazione compatibile con un motore che carica questo file e' «gate e up
 * uniformi, 20 `down` in Q3_K»: `q35MoeConfig` esige gate/up dello stesso
 * formato su tutti i layer (altrimenti la classe non puo' essere il solo formato
 * del down, e lancia), ed e' anche quello che fa `llama-quantize`, che alza il
 * formato del solo `ffn_down` sui layer piu' sensibili. QUALI 20 layer non
 * cambia nessuno dei numeri sotto — il parco, le classi e i byte dipendono dai
 * CONTEGGI per classe, non dagli indici — e infatti il test non ne pinna
 * l'identita': pinna che siano 20.
 */
function expertHeaderFromDump(e: DumpEntry): {
  tensors: GgufTensorInfo[]; nQ3kDown: number; nModel: number; nMtp: number;
} {
  const meta = e.meta;
  const num = (k: string): number => {
    const v = meta[k];
    if (typeof v !== "number") throw new Error(`dump: metadata ${k} mancante`);
    return v;
  };
  const nLayerFile = num("qwen35moe.block_count");          // 41 = 40 + testa MTP
  const nMtp = num("qwen35moe.nextn_predict_layers");       // 1
  const nModel = nLayerFile - nMtp;                          // 40
  const d = num("qwen35moe.embedding_length");               // 2048
  const dFfn = num("qwen35moe.expert_feed_forward_length");  // 512
  const nE = num("qwen35moe.expert_count");                  // 256
  const nQ3k = histOf(e, HIST_KEY.q3k).n;                    // 20

  const tensors: GgufTensorInfo[] = [];
  const push = (name: string, dims: number[], type: number): void => {
    tensors.push({ name, dims, type, offset: 0 });
  };
  for (let l = 0; l < nModel; l++) {
    // ordine GGUF: dims[0] e' la dimensione piu' interna (ne[0] di ggml)
    push(q35ExpertTensor(l, "gate"), [d, dFfn, nE], GGML_TYPE.Q2_K);
    push(q35ExpertTensor(l, "up"), [d, dFfn, nE], GGML_TYPE.Q2_K);
    push(q35ExpertTensor(l, "down"), [dFfn, d, nE], l < nQ3k ? GGML_TYPE.Q3_K : GGML_TYPE.Q2_K);
  }
  for (let m = 0; m < nMtp; m++) {
    for (const w of ["gate", "up", "down"] as const) {
      push(q35ExpertTensor(nModel + m, w), [w === "down" ? dFfn : d, w === "down" ? d : dFfn, nE], GGML_TYPE.Q8_0);
    }
  }
  return { tensors, nQ3kDown: nQ3k, nModel, nMtp };
}

/** La config di residenza dedotta dall'header sintetico (nessun altro input). */
function cfgFrom(tensors: GgufTensorInfo[], meta: Record<string, unknown>): MoeModelConfig {
  const byName = new Map(tensors.map((t) => [t.name, t]));
  const f: GgufFile = {
    version: 3, alignment: 32, metadata: metadataFromDump(meta), tensors, dataOffset: 0,
  };
  const shape = q35ShapeFromGguf(f);
  return q35MoeConfig(shape, (n) => {
    const t = byName.get(n);
    if (!t) throw new Error(`header sintetico: tensore ${n} assente`);
    return t;
  });
}

/** Byte del parco = Σ (expert della classe × taglia dello slab della classe). */
const parkBytesOf = (cfg: MoeModelConfig): number => {
  const park = moeParkOf(cfg);
  let b = 0;
  for (const c of cfg.classes) b += park[c] * cfg.layout(c).bytes;
  return b;
};

describe("arena Q2_K/Q3_K — geometria del parco (header 2026-08-17, senza GPU)", () => {
  const e = dump();
  const { tensors, nQ3kDown, nModel, nMtp } = expertHeaderFromDump(e);
  const cfg = cfgFrom(tensors, e.meta);

  it("(0) l'header sintetico riproduce l'artefatto tensore per tensore", () => {
    // Tutto quello che viene dopo si regge su questa ricostruzione: se non
    // tornasse coi byte veri, i numeri del parco sarebbero un'invenzione
    // interna coerente con se stessa. Il peso lo da' `tensorByteSize` DEL
    // MOTORE, non un'aritmetica scritta qui.
    const got: Record<number, { n: number; bytes: number }> = {};
    for (const t of tensors) {
      const g = got[t.type] ?? (got[t.type] = { n: 0, bytes: 0 });
      g.n++;
      g.bytes += tensorByteSize(t);
    }
    expect(got[GGML_TYPE.Q2_K]).toEqual(histOf(e, HIST_KEY.q2k));
    expect(got[GGML_TYPE.Q3_K]).toEqual(histOf(e, HIST_KEY.q3k));
    expect(got[GGML_TYPE.Q8_0]).toEqual(histOf(e, HIST_KEY.q8));
    const somma = Object.values(got).reduce((a, g) => a + g.bytes, 0);
    expect(somma, "parco expert del file (MTP compreso)").toBe(e.expertParkBytes);

    // e la ripartizione assunta (gate/up uniformi, i Q3_K tutti nei `down`)
    // regge solo se i conti del dump la ammettono
    const n = (k: string): number => histOf(e, k).n;
    expect(n(HIST_KEY.q2k) + n(HIST_KEY.q3k) + n(HIST_KEY.q8), "3 tensori per blocco").toBe(3 * (nModel + nMtp));
    expect(n(HIST_KEY.q8), "gli expert del blocco MTP sono i tre Q8_0").toBe(3 * nMtp);
    expect(nQ3kDown, "un solo `down` per layer puo' essere Q3_K").toBeLessThanOrEqual(nModel);
  });

  it("(a) due classi, dedotte dal formato del `down`", () => {
    expect(cfg.classes.length).toBe(2);
    expect([...cfg.classes].sort()).toEqual(["q2k", "q3k"]);
    // e la classe la decide il layer: i primi `nQ3kDown` hanno il down in Q3_K
    expect(cfg.classOf(0)).toBe("q3k");
    expect(cfg.classOf(nQ3kDown - 1)).toBe("q3k");
    expect(cfg.classOf(nQ3kDown)).toBe("q2k");
    expect(cfg.classOf(cfg.nLayer - 1)).toBe("q2k");
  });

  it("(b) i due layout di slab misurano 1.146.880 B (q3k) e 1.032.192 B (q2k)", () => {
    const q2k = cfg.layout("q2k"), q3k = cfg.layout("q3k");
    expect(q2k.bytes).toBe(1_032_192);
    expect(q3k.bytes).toBe(1_146_880);
    // DA DOVE VENGONO, cioe' cosa si rompe se il numero cambia: 1.048.576 pesi
    // per tensore (2048 × 512) = 4096 superblocchi da 256, per tre tensori.
    // Q2_K 84 B/superblocco → 344.064 B; Q3_K 110 B, paddati a word dal repack
    // (110 non e' multiplo di 4) → 112 B → 458.752 B.
    for (const t of [q2k.gate, q2k.up, q2k.down, q3k.gate, q3k.up, q3k.down]) {
      expect(t.nBlocks).toBe(4096);
    }
    expect(q2k.down.dataBytes).toBe(4096 * 84);
    expect(q3k.down.dataBytes).toBe(4096 * 112);
    expect(q2k.bytes).toBe(3 * 344_064);
    expect(q3k.bytes).toBe(2 * 344_064 + 458_752);
    // K-quant: un solo segmento per tensore, nessuna scala separata
    for (const t of [q2k.down, q3k.down]) expect(t.scales).toBeNull();
    // gli offset restano multipli di 256 (minStorageBufferOffsetAlignment): e'
    // il vincolo che rende bindabile un sotto-range di slab
    for (const L of [q2k, q3k]) {
      for (const o of [L.gate.data, L.up.data, L.down.data, L.bytes]) expect(o % 256).toBe(0);
    }
  });

  it("(b-bis) il packer mette i superblocchi Q3_K a passo 112", () => {
    // COSA GATEA QUESTO CASO: il PASSO. Se il packer impacchettasse il down a
    // 110 — o se la geometria del formato tornasse a dire 110 — non fallirebbe
    // niente a valle: uscirebbero pesi plausibili e sbagliati, cioe' qualita'
    // che cala senza un errore. Provato mutando `srcBytesPerBlock` di q3_K in
    // moe.ts: questo caso diventa rosso.
    //
    // COSA NON GATEA, e dove sta invece: i 2 byte di coda risultano zero anche
    // se nessuno li scrive, perche' `packExpertSlab` alloca un `Uint8Array`
    // fresco. Il contratto vero — `repackKQuantInto` azzera la coda ANCHE su un
    // dst gia' sporco, che e' cio' che serve a chi riusa un buffer — e' pinnato
    // da tests/quant-repack-fast.test.ts («repackKQuantInto non ha
    // precondizioni su dst», oggi su blockBytes 144 e 210; 110 percorre lo
    // stesso ramo, `stride = ceil(bb/4)*4`), e li' va tenuto: duplicarlo qui
    // darebbe due copie dello stesso invariante e nessuna copertura in piu'.
    const q3k = cfg.layout("q3k");
    const pat = (n: number, seme: number): Uint8Array => {
      const a = new Uint8Array(n);
      for (let i = 0; i < n; i++) a[i] = (i * 31 + seme * 17 + 1) & 0xff || 0xa5; // mai 0
      return a;
    };
    const gate = pat(4096 * 84, 1), up = pat(4096 * 84, 2), down = pat(4096 * 110, 3);
    const slab = packExpertSlab(gate, up, down, q3k);
    expect(slab.length).toBe(1_146_880);
    // i due tensori a passo 84 (multiplo di 4) sono una copia diretta
    expect(slab.subarray(q3k.gate.data, q3k.gate.data + q3k.gate.dataBytes)).toEqual(gate);
    expect(slab.subarray(q3k.up.data, q3k.up.data + q3k.up.dataBytes)).toEqual(up);
    // il down va riscritto a passo 112: ogni superblocco all'indirizzo giusto
    const atteso = new Uint8Array(4096 * 112);
    for (let b = 0; b < 4096; b++) atteso.set(down.subarray(b * 110, b * 110 + 110), b * 112);
    expect(slab.subarray(q3k.down.data, q3k.down.data + q3k.down.dataBytes)).toEqual(atteso);
  });

  it("(c) parco 5120 slab per classe = 11.156.848.640 B = 10,391 GiB", () => {
    const park = moeParkOf(cfg);
    expect(park).toEqual({ q2k: 5120, q3k: 5120 });
    expect(cfg.nLayer).toBe(40);
    expect(cfg.nExpert).toBe(256);
    expect(cfg.nExpertUsed).toBe(8);
    const bytes = parkBytesOf(cfg);
    expect(bytes).toBe(11_156_848_640);
    expect(bytes / GiB).toBeCloseTo(10.391, 3);

    // IL BLOCCO MTP NON E' NEL PARCO, ed e' meta' della correzione del
    // 2026-08-17: `block_count` vale 41 ma i layer del modello sono 40, e i tre
    // expert Q8_0 di blk.40 (855.638.016 B) l'arena non li tiene. L'altra meta'
    // e' che il parco si conta in SLAB IMPACCHETTATI, non in byte GGUF: la
    // differenza e' esattamente il padding a word dei superblocchi Q3_K
    // (2 B × 4096 blocchi × 5120 slab della classe q3k).
    const rawModelPark = e.expertParkBytes - histOf(e, HIST_KEY.q8).bytes;
    expect(bytes - rawModelPark).toBe(2 * 4096 * 5120);

    // la slotTable indirizza esattamente il parco, niente di piu'
    expect(slotTableEntriesOf(cfg)).toBe(park.q2k + park.q3k);
  });

  it("(d) a budget 11,17 GiB la residenza e' totale: zero slot mancanti", () => {
    const budgetBytes = Math.floor(11.17 * GiB);
    const slots = expertSlots({ budgetBytes, cfg });
    const park = moeParkOf(cfg);
    // 100% del parco: nessuno slot mancante in nessuna classe
    for (const c of cfg.classes) expect(slots[c]).toBe(park[c]);
    let mancanti = 0;
    for (const c of cfg.classes) mancanti += Math.max(0, park[c] - slots[c]);
    expect(mancanti).toBe(0);
    // e il minimo bindabile (top-K per layer MoE) e' ampiamente coperto
    const min = minSlotsOf(cfg);
    for (const c of cfg.classes) expect(slots[c]).toBeGreaterThanOrEqual(min[c]);

    // il piano ci sta nel budget, con il margine misurato (~0,78 GiB)
    const used = parkBytesOf(cfg);
    expect(used).toBeLessThanOrEqual(budgetBytes);
    expect((budgetBytes - used) / GiB).toBeCloseTo(0.779, 3);

    // …e un budget appena sotto il parco NON basta: il gate sopra misura una
    // soglia vera, non un'asserzione che passerebbe con qualsiasi budget.
    const stretto = expertSlots({ budgetBytes: used - 1, cfg });
    let mancantiStretto = 0;
    for (const c of cfg.classes) mancantiStretto += park[c] - stretto[c];
    expect(mancantiStretto).toBeGreaterThan(0);

    // DOV'E' IL CONFINE VERO, che non coincide col parco — nota, non gate. Il
    // riparto del budget fra classi oggi e' proporzionale al NUMERO di expert e
    // non ai byte (`expertSlots`): due classi da 5120 slab prendono meta'
    // budget a testa, e a vincolare e' la q3k, che ha gli slab dell'11% piu'
    // grandi. Con quel riparto la residenza totale chiede 10,938 GiB invece dei
    // 10,391 del parco, cioe' il margine SPENDIBILE e' 0,23 GiB e non 0,78. Il
    // numero e' calcolato qui sotto e non asserito contro `expertSlots` di
    // proposito: il giorno in cui il riparto passera' ai byte, questo commento
    // va aggiornato, ma il test non deve diventare rosso per un miglioramento.
    const soglia = 2 * park.q3k * cfg.layout("q3k").bytes;
    expect(soglia).toBe(11_744_051_200);
    expect(soglia / GiB).toBeCloseTo(10.9375, 4);
    expect(soglia, "anche il riparto piu' sfavorevole sta nell'arena").toBeLessThanOrEqual(budgetBytes);

    // il piano d'arena che i kernel poi consumano (limiti negoziati sul 4090:
    // maxBufferSize 4 GiB−4, maxStorageBufferBindingSize 2 GiB−4)
    const need = arenaNeeds({
      budgetBytes, cfg, maxBufferBytes: 4_294_967_292, maxBindingBytes: 2_147_483_644,
    });
    expect(need.arenaWindowBytes).toBeLessThanOrEqual(2_147_483_644);
    let capienza = 0;
    for (const c of cfg.classes) {
      const perBuf = Math.floor(need.arenaWindowBytes / cfg.layout(c).bytes);
      expect(need.arenaBuffers * perBuf).toBeGreaterThanOrEqual(park[c]);
      capienza += park[c] * cfg.layout(c).bytes;
    }
    expect(capienza).toBe(11_156_848_640);
  });

  describe("(d-bis) zero miss strutturali, con la cache VERA", () => {
    beforeAll(() => {
      (globalThis as Record<string, unknown>).GPUBufferUsage ??= { STORAGE: 0x80, COPY_DST: 8, COPY_SRC: 4 };
    });

    it("due passate su tutto il parco: 10.240 miss, poi 10.240 hit e zero eviction", () => {
      // La cache VERA (`ExpertCache`) su un GPUDevice finto, non una
      // simulazione di LRU scritta qui: e' l'unico modo in cui «zero miss
      // strutturali» dice qualcosa. Un errore di riparto, di conteggio slot o
      // di mappa layer→classe si vede qui e non nell'aritmetica di (d).
      //
      // I byte non si toccano: `slabInHand` consegna uno slab gia'
      // impacchettato (uno per classe, riusato), quindi le 20.480 `ensure`
      // costano mappe e contatori. Impacchettarne 10.240 veri sarebbero 11 GiB
      // di memcpy per verificare un conto di slot.
      const budgetBytes = Math.floor(11.17 * GiB);
      const { device } = mkMockDevice();
      const cache = new ExpertCache(device, {
        budgetBytes, cfg, arena: true,
        maxBufferBytes: 4_294_967_292, maxBindingBytes: 2_147_483_644,
      });
      const slabDiClasse: Record<string, ReturnType<typeof slabInHand>> = {};
      for (const c of cfg.classes) slabDiClasse[c] = slabInHand(new Uint8Array(cfg.layout(c).bytes));

      for (const passata of [0, 1]) {
        for (let l = 0; l < cfg.nLayer; l++) {
          const src = slabDiClasse[cfg.classOf(l)];
          for (let x = 0; x < cfg.nExpert; x++) {
            const { hit } = cache.ensure(l, x, src);
            expect(hit, `passata ${passata}, blk.${l} expert ${x}`).toBe(passata === 1);
          }
        }
      }
      const st = cache.stats();
      const park = moeParkOf(cfg);
      expect(st.misses).toBe(park.q2k + park.q3k); // la prima passata carica il parco…
      expect(st.hits).toBe(park.q2k + park.q3k);   // …la seconda lo trova tutto residente
      expect(st.evictions).toBe(0);                // e non sfratta niente: e' la tesi
      expect(st.retention).toBe(1);
      expect(st.occupied).toEqual(park);
      expect(st.slots).toEqual(park);
    });
  });
});

describe("arena Q2_K/Q3_K — la config e' DEDOTTA, non cablata", () => {
  const e = dump();
  const { tensors } = expertHeaderFromDump(e);
  /** lo stesso header con i tensori expert riscritti nei tipi dati. */
  const conTipi = (tipoDi: (name: string, i: number) => number): MoeModelConfig =>
    cfgFrom(tensors.map((t, i) => ({ ...t, type: tipoDi(t.name, i) })), e.meta);

  it("(e) cambiando il tipo degli expert cambiano classi, byte e verdetto d'arena", () => {
    // TUTTO Q4_K (com'e' il quant UD-Q4_K_S in produzione oggi): una sola
    // classe, slab da 1.769.472 B, e il parco NON ci sta piu' nell'arena.
    const q4k = conTipi(() => GGML_TYPE.Q4_K);
    expect([...q4k.classes]).toEqual(["q4k"]);
    expect(q4k.layout("q4k").bytes).toBe(1_769_472); // 3 × 4096 × 144
    const parcoQ4k = parkBytesOf(q4k);
    expect(parcoQ4k / GiB).toBeCloseTo(16.875, 3);
    const budget = Math.floor(11.17 * GiB);
    expect(expertSlots({ budgetBytes: budget, cfg: q4k }).q4k)
      .toBeLessThan(moeParkOf(q4k).q4k); // residenza parziale: e' il caso 65%

    // TRE classi: il `down` in tre formati diversi ⇒ tre layout, senza toccare
    // una riga di produzione.
    const layerDi = (name: string): number => Number(/^blk\.(\d+)\./.exec(name)?.[1] ?? -1);
    const tre = conTipi((name) => (/_down_exps/.test(name)
      ? [GGML_TYPE.Q2_K, GGML_TYPE.Q3_K, GGML_TYPE.Q6_K][layerDi(name) % 3]
      : GGML_TYPE.Q2_K));
    expect([...tre.classes].sort()).toEqual(["q2k", "q3k", "q6k"]);
    expect(tre.layout("q6k").bytes).toBe(2 * 344_064 + 4096 * 212); // Q6_K 210 B → 212 paddati

    // e gate/up disallineati NON diventano una classe silenziosamente sbagliata
    expect(() => conTipi((name) => (/blk\.7\.ffn_gate_exps/.test(name) ? GGML_TYPE.Q4_K : GGML_TYPE.Q2_K)))
      .toThrow(/gate\/up/);
  });

  it("(e-bis) nel sorgente di produzione non c'e' nessun formato cablato", () => {
    // La prova di sopra mostra che la config SEGUE l'header; questa mostra che
    // non puo' fare altro: dentro `q35MoeConfig` non compare il nome di nessun
    // formato. Se un giorno qualcuno ci scrivesse «i q3k sono i layer 0..19»,
    // sarebbe vero per QUESTO file e falso per il prossimo — e non darebbe un
    // errore, darebbe slab letti con la geometria di un'altra classe.
    const src = readFileSync(join(process.cwd(), "src/engine/q35expertstore.ts"), "utf8");
    const i = src.indexOf("export function q35MoeConfig");
    expect(i).toBeGreaterThan(0);
    const body = src.slice(i, src.indexOf("\n}\n", i));
    expect(body).not.toMatch(/[Qq][2-8]_[K0-9]|q[2-8]k/);
  });
});
