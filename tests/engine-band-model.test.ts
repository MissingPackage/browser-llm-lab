import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import {
  syncMsPerToken, syncToksPerSec, fetchesPerToken, coldTtftMs, missMs,
  BAND_WARM_BPS, BAND_COLD_BPS, BAND_COLD_SEQ_BPS, BASE_SYNC_MS_PER_TOKEN,
} from "../src/engine/bandmodel";

// Modello di banda (C3c fase 6, spec §5): il gate ±15% sui 3 punti MISURATI è
// un TEST — resta un vincolo meccanico di non-regressione, non una frase di
// journal. I coefficienti del modello vengono da artefatti dichiarati nel
// modulo (BASE: bench sync c3b; C/STEP: fit sui punti 1472/736 — il punto b12
// resta fuori dal fit ed è la predizione vera).

const TOL = 0.15;
const bench = (f: string) => {
  const j = JSON.parse(readFileSync(f, "utf8"));
  return {
    decodeToksPerSec: j.gates.decodeMedian as number,
    hAll: j.telemetry.decode.hitRate as number,
    missesPerToken: j.telemetry.decode.missesPerToken as number,
  };
};
const POINTS = [
  { name: "b12 (PREDETTO: fuori dal fit)", file: "results/engine/bench-glm-4090-sync-tier-b12-2026-08-09.json" },
  { name: "b1472 (nel fit)", file: "results/engine/bench-glm-4090-sync-tier-b1472-2026-08-09.json" },
  { name: "b736 (nel fit)", file: "results/engine/bench-glm-4090-sync-tier-b736-2026-08-09.json" },
];

describe("modello di banda — ±15% sui punti misurati (done-when fase 6)", () => {
  for (const p of POINTS) {
    it(`${p.name}: |pred − mis|/mis ≤ 15%`, () => {
      const m = bench(p.file);
      const pred = syncToksPerSec(m.hAll, BAND_WARM_BPS);
      const err = Math.abs(pred - m.decodeToksPerSec) / m.decodeToksPerSec;
      expect(err).toBeLessThanOrEqual(TOL);
    });
    it(`${p.name}: fetchesPerToken(h) riproduce il contatore del bench`, () => {
      const m = bench(p.file);
      // le richieste raddoppiano col prefetch: (1−h)×368 ≈ misses/token misurati
      expect(Math.abs(fetchesPerToken(m.hAll) - m.missesPerToken) / m.missesPerToken).toBeLessThan(0.02);
    });
  }

  it("monotonia: banda più bassa ⇒ mai più veloce; h più alto ⇒ mai più lento", () => {
    for (const h of [0.98, 0.9, 0.71]) {
      expect(syncMsPerToken(h, BAND_COLD_BPS)).toBeGreaterThan(syncMsPerToken(h, BAND_WARM_BPS));
    }
    expect(syncMsPerToken(0.98, BAND_WARM_BPS)).toBeLessThan(syncMsPerToken(0.9, BAND_WARM_BPS));
    expect(syncMsPerToken(0.9, BAND_WARM_BPS)).toBeLessThan(syncMsPerToken(0.71, BAND_WARM_BPS));
  });

  it("h=1 ⇒ costo strutturale puro; input fuori [0,1] ⇒ throw", () => {
    expect(syncMsPerToken(1, BAND_WARM_BPS)).toBe(BASE_SYNC_MS_PER_TOKEN);
    expect(() => syncMsPerToken(1.2, BAND_WARM_BPS)).toThrow();
  });

  it("TTFT freddo: warm + nonExpert/banda + premio freddo − overlap (mai sotto il warm+nonExpert)", () => {
    const base = coldTtftMs({ warmTtftMs: 12600, uniqueBytesRead: 15.7e9 });
    // premio ≈ 15.7e9×(1/3.51e9 − 1/11.41e9) ≈ 3.10 s; nonExpert ≈ 0.386 s
    expect(base).toBeGreaterThan(12600 + 3000);
    expect(base).toBeLessThan(12600 + 4500);
    // overlap enorme: il premio si azzera ma nonExpert resta
    const over = coldTtftMs({ warmTtftMs: 12600, uniqueBytesRead: 15.7e9, overlapMs: 1e9 });
    expect(over).toBeCloseTo(12600 + (1354078720 / BAND_COLD_SEQ_BPS) * 1000, 0);
  });

  it("missMs cresce passando dalla banda warm alla fredda del WP fase 1", () => {
    expect(missMs(BAND_COLD_BPS)).toBeGreaterThan(missMs(BAND_WARM_BPS));
  });
});
