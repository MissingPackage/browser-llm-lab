// MICRO-BENCH del costo di pack per miss (goal fase-D fase 2, done-when:
// "micro-bench del costo/miss PRIMA e DOPO nello stesso JSON").
//
// Le due forme convivono qui: PRIMA = il repack scalare (`|=` per byte) più
// l'array temporaneo ricopiato nello slab; DOPO = `packExpertSlab`, che
// scrive dritto nello slab con una memcpy. La forma "prima" NON è archeologia
// ricostruita a memoria: è la stessa che `repackKQuantInto` esegue tuttora
// sui big-endian, ricopiata qui come oracolo.
//
// Sotto `npm test` gira solo la parte di CORRETTEZZA (le due forme producono
// gli stessi byte), che è veloce e deterministica. Con `PACK_BENCH=1` misura
// e, se c'è `PACK_BENCH_OUT`, scrive il JSON — così la suite non sporca mai
// l'albero di lavoro.
import { describe, expect, it } from "vitest";
import { writeFileSync } from "node:fs";
import { mkSlabLayout, packExpertSlab, type SlabLayout } from "../src/engine/moe";

// geometria REALE del 35B-A3B: d=2048, dE=512 ⇒ 1.048.576 elementi/expert
const ELEMS = 2048 * 512;
const NB = ELEMS / 256;
const SRC = { q4_K: 144, q6_K: 210 } as const;

const layoutFor = (down: "q4_K" | "q6_K"): SlabLayout => mkSlabLayout(
  down === "q6_K" ? "q6k" : "q4k",
  { kind: "q4_K", elems: ELEMS }, { kind: "q4_K", elems: ELEMS }, { kind: down, elems: ELEMS },
);

// --- FORMA "PRIMA": scalare + temporaneo (la versione di q1) ---
const repackScalare = (src: Uint8Array, off: number, nBlocks: number, bb: number): Uint32Array => {
  const wpb = Math.ceil(bb / 4);
  const out = new Uint32Array(nBlocks * wpb);
  for (let b = 0; b < nBlocks; b++) {
    const o = off + b * bb;
    for (let j = 0; j < bb; j++) out[b * wpb + (j >> 2)] |= src[o + j] << ((j & 3) * 8);
  }
  return out;
};
const packPrima = (g: Uint8Array, u: Uint8Array, d: Uint8Array, L: SlabLayout): Uint8Array => {
  const out = new Uint8Array(L.bytes);
  for (const [raw, t] of [[g, L.gate], [u, L.up], [d, L.down]] as const) {
    const w = repackScalare(raw, 0, t.nBlocks, SRC[t.kind as "q4_K" | "q6_K"]);
    out.set(new Uint8Array(w.buffer, 0, w.byteLength), t.data);
  }
  return out;
};

const synth = (n: number, s0: number): Uint8Array => {
  const o = new Uint8Array(n); let s = s0 >>> 0;
  for (let i = 0; i < n; i++) { s = (s * 1664525 + 1013904223) >>> 0; o[i] = s >>> 24; }
  return o;
};

const misura = (f: () => void, n: number): number => {
  for (let i = 0; i < 3; i++) f();                 // warmup
  const t0 = performance.now();
  for (let i = 0; i < n; i++) f();
  return (performance.now() - t0) / n;
};

describe("costo di pack per miss: prima vs dopo", () => {
  const righe: Record<string, unknown>[] = [];

  it.each(["q4_K", "q6_K"] as const)("classe down %s", (down) => {
    const L = layoutFor(down);
    const g = synth(NB * SRC.q4_K, 1), u = synth(NB * SRC.q4_K, 2), d = synth(NB * SRC[down], 3);

    // CORRETTEZZA (sempre): le due forme devono dare lo stesso slab.
    const a = packPrima(g, u, d, L);
    const b = packExpertSlab(g, u, d, L);
    expect(b.length).toBe(a.length);
    const wa = new Uint32Array(a.buffer, 0, a.length / 4);
    const wb = new Uint32Array(b.buffer, b.byteOffset, b.length / 4);
    let primo = -1;
    for (let i = 0; i < wa.length; i++) if (wa[i] !== wb[i]) { primo = i; break; }
    expect(primo, `prima parola diversa all'offset ${primo * 4}`).toBe(-1);

    if (!process.env.PACK_BENCH) return;
    const N = 20;
    const msPrima = misura(() => packPrima(g, u, d, L), N);
    const msDopo = misura(() => packExpertSlab(g, u, d, L), N);
    const riga = {
      classe: down, slabBytes: L.bytes, ripetizioni: N,
      msPrima: +msPrima.toFixed(3), msDopo: +msDopo.toFixed(3),
      fattore: +(msPrima / msDopo).toFixed(2),
      mbsPrima: Math.round(L.bytes / 1e6 / (msPrima / 1000)),
      mbsDopo: Math.round(L.bytes / 1e6 / (msDopo / 1000)),
    };
    righe.push(riga);
    console.log(`PACK ${down}: ${riga.msPrima} → ${riga.msDopo} ms/miss (${riga.fattore}x, ${riga.mbsPrima} → ${riga.mbsDopo} MB/s)`);
    const out = process.env.PACK_BENCH_OUT;
    if (out) {
      writeFileSync(out, `${JSON.stringify({
        schemaVersion: 1, kind: "q35-pack-microbench", goal: "engine-fase-d fase 2",
        geometria: { dModel: 2048, dFfnExpert: 512, elemsPerExpert: ELEMS, superblocchi: NB },
        note: "PRIMA = repack scalare (|= per byte) + array temporaneo ricopiato nello slab; DOPO = repackKQuantInto diretto. Stessi byte in uscita (asserito nel test).",
        righe,
      }, null, 1)}\n`);
    }
  });
});
