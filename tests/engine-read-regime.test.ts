// Il regime di lettura si dichiara (goal engine-velocita-decode, it.20).
//
// IL CASO CHE HA FATTO NASCERE QUESTO CODICE, coi numeri veri. Il riferimento
// `bench-glm-4090-b12-riga6-2026-08-15.json` dichiarava 15,330 tok/s di decode
// sul GLM ed era un gate di merge («GLM b12 entro ±5%»). Rimisurato il
// 2026-08-16 lo stesso bench dava 11,35 tok/s, −26%. Fra le due date:
//
//   - il path di lettura e' BYTE-IDENTICO (nessun commit su glmsource/glmmodel/
//     residency/expertstore fra il commit del riferimento e HEAD);
//   - `bytesRead`, `misses` ed `evictions` coincidono CIFRA PER CIFRA;
//   - a muoversi e' solo `readMs`.
//
// La diagnosi sta dentro l'artefatto del riferimento: la sua passata di WARM-UP
// legge a ~3 GiB/s, le sue REPLICHE a 9,55 GiB/s — lo stesso file, a minuti di
// distanza, tre volte piu' veloce. Nessun disco fa questo: erano byte serviti
// dalla page cache del sistema operativo. Il riferimento non misurava il
// motore, misurava la RAM libera di quel pomeriggio, e nessun campo lo diceva.
//
// Questi casi sono la giurisprudenza: se qualcuno alza il tetto o cambia la
// soglia, deve far fallire (o aggiornare con la sua ragione) i numeri veri.
import { describe, expect, it } from "vitest";
import { readBandwidth, OPFS_DEVICE_CEILING_GIBS } from "../src/engine/residency";

const GIB = 1 << 30;

describe("readBandwidth: dal disco o dalla page cache", () => {
  it("le REPLICHE del riferimento 2026-08-15 sono os-cache (9,55 GiB/s)", () => {
    // 3 repliche x 64 token x 4,78125 miss x 5.308.416 byte, readMs sommato
    const bytes = 3 * 64 * 4.78125 * 5_308_416;
    const ms = 3 * 64 * 2.4746875;
    const r = readBandwidth(bytes, ms);
    expect(r.regime).toBe("os-cache");
    expect(r.gibs).toBeGreaterThan(9);
    expect(r.gibs).toBeLessThan(10);
  });

  it("le stesse repliche rimisurate il 2026-08-16 sono disk (1,35 GiB/s)", () => {
    const bytes = 3 * 64 * 4.78125 * 5_308_416;
    const ms = 3 * 64 * 17.509453;
    const r = readBandwidth(bytes, ms);
    expect(r.regime).toBe("disk");
    expect(r.gibs).toBeGreaterThan(1.3);
    expect(r.gibs).toBeLessThan(1.4);
  });

  it("le passate di WARM-UP di entrambe le date sono disk — e sono la provenienza del tetto", () => {
    // e' il confronto interno che smaschera il riferimento: stesso file, stessa
    // run, warm-up a ~3 GiB/s e repliche a 9,55
    const rifPrefill = readBandwidth(20_512_309_248, 6414.145);
    const rifDecode = readBandwidth(1_624_375_296, 460.495);
    const oggiPrefill = readBandwidth(20_512_309_248, 8277.205);
    const oggiDecode = readBandwidth(1_624_375_296, 1117.155);
    for (const r of [rifPrefill, rifDecode, oggiPrefill, oggiDecode]) {
      expect(r.regime).toBe("disk");
      expect(r.gibs!).toBeLessThan(OPFS_DEVICE_CEILING_GIBS);
    }
    // e il tetto dichiarato sta sopra la piu' veloce delle quattro, con margine
    expect(Math.max(...[rifPrefill, rifDecode, oggiPrefill, oggiDecode].map((r) => r.gibs!)))
      .toBeLessThan(OPFS_DEVICE_CEILING_GIBS);
  });

  it("finestra senza letture: `non-misurato`, e mai NaN nel JSON", () => {
    for (const [b, m] of [[0, 12], [1024, 0], [0, 0], [-1, 5]] as const) {
      const r = readBandwidth(b, m);
      expect(r.regime).toBe("non-misurato");
      expect(r.gibs).toBeNull();
      expect(JSON.stringify(r)).not.toContain("NaN");
    }
  });

  it("la soglia e' esclusiva: esattamente al tetto si dichiara disk", () => {
    const bytes = OPFS_DEVICE_CEILING_GIBS * GIB;
    expect(readBandwidth(bytes, 1000).regime).toBe("disk");
    expect(readBandwidth(bytes * 1.001, 1000).regime).toBe("os-cache");
  });
});
