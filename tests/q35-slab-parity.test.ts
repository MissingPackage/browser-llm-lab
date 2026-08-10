// PARITÀ BIT-A-BIT della migrazione q35 → meccanica unica (goal fase-D it.7).
//
// La fase 1 sostituisce l'arena expert scritta a mano in `q35gpumodel.ts` con
// la `ExpertCache` di GLM. La domanda che decide se la migrazione è sicura non
// è "gira?", è: **finiscono in VRAM gli stessi byte agli stessi indirizzi?**
// Qui si risponde senza GPU e in millisecondi, riproducendo l'aritmetica
// VECCHIA (quella cancellata dal file, ricopiata qui come oracolo) e
// confrontandola con `mkSlabLayout` + `packExpertSlab`.
//
// Se un giorno il layout cambiasse per una buona ragione, questo test va
// aggiornato CON un riferimento nuovo: è il punto in cui ci si accorge che i
// golden del 35B vanno rigenerati, invece di scoprirlo da una regressione di
// qualità.
import { describe, expect, it } from "vitest";
import { mkSlabLayout, packExpertSlab } from "../src/engine/moe";
import { maxBindRangeOf } from "../src/engine/residency";
import { repackKQuant } from "../src/engine/quant";

// Geometria REALE di Qwen3.6-35B-A3B-UD-Q4_K_S, letta dal GGUF:
// d=2048, dE=512 ⇒ 1.048.576 elementi per expert su gate/up/down.
// down: Q4_K su 37 layer, Q6_K su 3 (34, 38, 39) — il mix UD.
const ELEMS = 2048 * 512;
const NB = ELEMS / 256; // 4096 superblocchi

// --- ORACOLO: l'aritmetica che `q35gpumodel.ts` faceva a mano (q1 it.17).
const SRC_BYTES = { q4_K: 144, q6_K: 210 } as const;
// `repackKQuant` padda i superblocchi a parola: Q6_K 210 → 212, Q4_K resta 144.
const RP_BYTES = { q4_K: 144, q6_K: 212 } as const;
const gateRp = (ELEMS / 256) * RP_BYTES.q4_K;
const oldSlotBytes = (down: "q4_K" | "q6_K"): number => 2 * gateRp + (ELEMS / 256) * RP_BYTES[down];
const oldRpU8 = (raw: Uint8Array, bb: number): Uint8Array => {
  const w = repackKQuant(raw, 0, raw.length / bb, bb);
  return new Uint8Array(w.buffer, 0, w.byteLength);
};

const layoutFor = (down: "q4_K" | "q6_K") => mkSlabLayout(
  down === "q6_K" ? "q6k" : "q4k",
  { kind: "q4_K", elems: ELEMS }, { kind: "q4_K", elems: ELEMS }, { kind: down, elems: ELEMS },
);

// byte grezzi deterministici (nessun Math.random: il test deve essere ripetibile)
const synth = (n: number, seed: number): Uint8Array => {
  const out = new Uint8Array(n);
  let s = seed >>> 0;
  for (let i = 0; i < n; i++) { s = (s * 1664525 + 1013904223) >>> 0; out[i] = s >>> 24; }
  return out;
};

describe("q35 slab: la meccanica unica produce gli stessi byte dell'arena a mano", () => {
  it.each(["q4_K", "q6_K"] as const)("classe down %s: offset e taglia identici all'aritmetica vecchia", (down) => {
    const L = layoutFor(down);
    expect(L.gate.data).toBe(0);
    expect(L.up.data).toBe(gateRp);
    expect(L.down.data).toBe(2 * gateRp);
    expect(L.bytes).toBe(oldSlotBytes(down));
    // i K-quant hanno UN segmento: nessuna scala separata da bindare
    expect(L.gate.scales).toBeNull();
    expect(L.down.scales).toBeNull();
    expect(L.down.dataBytes).toBe(NB * RP_BYTES[down]);
  });

  it("i numeri sono quelli MISURATI sul GGUF 35B (non solo autocoerenti)", () => {
    expect(gateRp).toBe(589_824);
    expect(layoutFor("q4_K").bytes).toBe(1_769_472);
    expect(layoutFor("q6_K").down.dataBytes).toBe(868_352);
    expect(layoutFor("q6_K").bytes).toBe(2_048_000);
    // tutti gli offset allineati a 256: i bind group li usano come offset di
    // sotto-range, e WebGPU esige minStorageBufferOffsetAlignment.
    for (const d of ["q4_K", "q6_K"] as const) {
      const L = layoutFor(d);
      for (const o of [L.gate.data, L.up.data, L.down.data, L.bytes]) expect(o % 256).toBe(0);
    }
  });

  it("il sotto-range più grande è il DOWN sui Q6_K (il controllo sui campi compat lo mancava)", () => {
    const L = layoutFor("q6_K");
    expect(maxBindRangeOf(L)).toBe(868_352);
    expect(maxBindRangeOf(L)).toBeGreaterThan(L.qsBytes); // qsBytes = il GATE: 589.824
  });

  it.each(["q4_K", "q6_K"] as const)("classe down %s: packExpertSlab == i tre repack a mano, BYTE PER BYTE", (down) => {
    const L = layoutFor(down);
    const gRaw = synth(NB * SRC_BYTES.q4_K, 1);
    const uRaw = synth(NB * SRC_BYTES.q4_K, 2);
    const dRaw = synth(NB * SRC_BYTES[down], 3);

    // vecchio: tre writeBuffer a base, base+gateRp, base+2*gateRp
    const atteso = new Uint8Array(oldSlotBytes(down));
    atteso.set(oldRpU8(gRaw, SRC_BYTES.q4_K), 0);
    atteso.set(oldRpU8(uRaw, SRC_BYTES.q4_K), gateRp);
    atteso.set(oldRpU8(dRaw, SRC_BYTES[down]), 2 * gateRp);

    const ottenuto = packExpertSlab(gRaw, uRaw, dRaw, L);
    expect(ottenuto.length).toBe(atteso.length);
    // confronto su interi a 32 bit: 1,7 MB byte per byte in vitest è lento
    const a = new Uint32Array(atteso.buffer, 0, atteso.length / 4);
    const b = new Uint32Array(ottenuto.buffer, ottenuto.byteOffset, ottenuto.length / 4);
    let primoDiverso = -1;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) { primoDiverso = i; break; }
    expect(primoDiverso, `prima parola diversa all'offset ${primoDiverso * 4}`).toBe(-1);
  });
});
