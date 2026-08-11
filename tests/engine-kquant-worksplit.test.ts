// Spartizione del lavoro nei kernel K-quant (goal fase-D fase 4-bis, it.22).
//
// Il kernel non si puo' provare qui — serve una GPU — ma l'ARITMETICA con cui
// spartisce il lavoro sì, ed è dove stanno gli errori che poi diventano numeri
// plausibili e sbagliati: un'unità che copre due volte gli stessi elementi
// gonfia il prodotto scalare, una che ne salta uno lo taglia, e in entrambi i
// casi il modello continua a girare. Qui si verifica la BIIEZIONE: le unità
// coprono ogni (superblocco, gruppo, l) esattamente una volta.
import { describe, expect, it } from "vitest";
import { kquantWorkSplit } from "../src/engine/kernels/wgsl";

/** Le due forme che il 35B usa davvero (it.19/it.21). */
const Q35 = [
  { name: "gate/up q4_K (K=2048)", sbPerRow: 2048 / 256, groups: 4 },
  { name: "down q4_K (K=512)", sbPerRow: 512 / 256, groups: 4 },
  { name: "down q6_K (K=512)", sbPerRow: 512 / 256, groups: 2 },
];

describe("kquantWorkSplit — biiezione della spartizione", () => {
  // K arbitrari (multipli di 256) x le due geometrie di gruppo
  const cases: { sbPerRow: number; groups: number }[] = [];
  for (const sb of [1, 2, 3, 4, 6, 8, 16, 32, 64]) for (const g of [2, 4]) cases.push({ sbPerRow: sb, groups: g });

  it.each(cases)("sbPerRow=$sbPerRow groups=$groups: ogni (sb, gruppo, l) coperto una volta sola", ({ sbPerRow, groups }) => {
    const { lpu, chunks, unitsPerSb, units } = kquantWorkSplit(sbPerRow, groups);
    expect(32 % lpu).toBe(0);            // i pezzi coprono il gruppo esattamente
    expect(chunks * lpu).toBe(32);
    expect(unitsPerSb).toBe(groups * chunks);
    expect(units).toBe(sbPerRow * unitsPerSb);

    // simulazione ESATTA di ciò che fa il WGSL: u → (sb, gruppo, [lo, lo+lpu))
    const seen = new Set<string>();
    for (let u = 0; u < units; u++) {
      const sb = Math.floor(u / unitsPerSb);
      const rem = u % unitsPerSb;
      const grp = Math.floor(rem / chunks);
      const lo = (rem % chunks) * lpu;
      expect(sb).toBeLessThan(sbPerRow);
      expect(grp).toBeLessThan(groups);
      for (let l = lo; l < lo + lpu; l++) {
        expect(l).toBeLessThan(32);
        const key = `${sb}:${grp}:${l}`;
        expect(seen.has(key)).toBe(false); // niente doppio conteggio
        seen.add(key);
      }
    }
    expect(seen.size).toBe(sbPerRow * groups * 32); // copertura totale
  });
});

describe("kquantWorkSplit — le lane che il 35B accende davvero", () => {
  it.each(Q35)("$name: 64 unità = un thread ciascuno, contro le $sbPerRow di prima", ({ sbPerRow, groups }) => {
    const { units } = kquantWorkSplit(sbPerRow, groups);
    // il regime PRECEDENTE spartiva i superblocchi: sbPerRow lane attive su 64
    expect(sbPerRow).toBeLessThanOrEqual(8); // era 8 (gate/up) o 2 (down)
    expect(units).toBe(64);                  // ora sono 64
  });

  it("righe grandi: le unità superano i 64 thread e il loop a stride le copre", () => {
    const { units } = kquantWorkSplit(64, 4); // K = 16384
    expect(units).toBeGreaterThan(64);
    expect(units % 64).toBe(0);
  });

  it("riga minima (K=256): non si scende sotto un valore di l per unità", () => {
    const { lpu, units } = kquantWorkSplit(1, 2);
    expect(lpu).toBeGreaterThanOrEqual(1);
    expect(units).toBeLessThanOrEqual(64);
  });
});
