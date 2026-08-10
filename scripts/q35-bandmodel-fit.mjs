// Fit DICHIARATO del modello di banda q35-MoE sui 3 punti tier misurati
// (q1 fase 8): decodeMs(tier) = base + missPerToken(tier) × costoMiss.
// missPerToken = (1 − hitRate) × 320 selezioni/token (hit CUMULATIVO
// prefill+decode del bench: approssimazione DICHIARATA — il decode a cache
// calda ha hit più alto del cumulato). NON tocca bandmodel.ts GLM (il test
// permanente GLM resta): questo è il fit q35, risultato nuovo.
// Uso: node scripts/q35-bandmodel-fit.mjs
import { readFileSync, writeFileSync } from "node:fs";

const tiers = [4, 8, 11].map((a) => {
  const j = JSON.parse(readFileSync(`results/engine/q35-bench-35b-arena${a}-2026-08-10.json`, "utf8"));
  const m = j.moe;
  const hit = m.hits / (m.hits + m.misses);
  return {
    arenaGiB: a,
    decodeMsP50: j.decode.msPerTokenP50,
    hitRate: +hit.toFixed(4),
    missPerToken: +((1 - hit) * 320).toFixed(2),
    residency: +(
      (m.nSlots.q4k * m.slotBytes.q4k + m.nSlots.q6k * m.slotBytes.q6k) /
      (m.parkSlots.q4k * m.slotBytes.q4k + m.parkSlots.q6k * m.slotBytes.q6k)
    ).toFixed(4),
    costPerMissObs: +(m.uploadedBytes / m.misses / 1e6).toFixed(3),
  };
});

// LSQ lineare y = base + slope·x
const xs = tiers.map((t) => t.missPerToken);
const ys = tiers.map((t) => t.decodeMsP50);
const n = xs.length;
const sx = xs.reduce((a, b) => a + b, 0), sy = ys.reduce((a, b) => a + b, 0);
const sxx = xs.reduce((a, b) => a + b * b, 0), sxy = xs.reduce((a, b, i) => a + b * ys[i], 0);
const slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
const base = (sy - slope * sx) / n;
const residuals = tiers.map((t) => {
  const pred = base + slope * t.missPerToken;
  return { arenaGiB: t.arenaGiB, measuredMs: t.decodeMsP50, predictedMs: +pred.toFixed(1), relErr: +((pred - t.decodeMsP50) / t.decodeMsP50).toFixed(4) };
});

const out = {
  schemaVersion: 1,
  kind: "q35-bandmodel-fit",
  date: "2026-08-10",
  model: "Qwen3.6-35B-A3B-UD-Q4_K_S.gguf",
  declared:
    "fit su 3 punti (gradi di liberta' 1): la forma lineare base+miss×costo e' il modello DICHIARATO, non validato out-of-sample come il bandmodel GLM (che ha il punto fuori-fit); hit CUMULATIVO prefill+decode usato come proxy del decode (approssimazione dichiarata); regime correttezza-prima (miss = fetch Range + repack JS + writeBuffer, sequenziale, MAI overlappato) — con la meccanica C3c (prefetch/slab pre-repacked) il costoMiss cambia natura.",
  points: tiers,
  fit: {
    formula: "decodeMsPerToken = base + missPerToken × costoMissMs",
    baseMs: +base.toFixed(1),
    costoMissMs: +slope.toFixed(2),
    note: "costoMissMs ≈ fetch(~1.79 MB da localhost) + repack JS K-quant + writeBuffer; base = forward a cache calda (40 sync router inclusi)",
  },
  residuals,
  glmReference: { file: "src/engine/bandmodel.ts", note: "il modello GLM (base 167.43, F(h)×(bytes/banda+1.96)) resta INTATTO col suo test permanente; forma diversa perche' misura un regime diverso (slab pre-repacked, prefetch, no repack JS on-miss)" },
};
writeFileSync("results/engine/q35-bandmodel-fit-2026-08-10.json", JSON.stringify(out, null, 1));
console.log(`OK: base ${out.fit.baseMs} ms + ${out.fit.costoMissMs} ms/miss; residui:`, residuals.map((r) => `${r.arenaGiB}GiB ${(r.relErr * 100).toFixed(1)}%`).join(" "));
