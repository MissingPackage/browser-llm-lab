// Genera/rigenera il fixture del campione DeltaNet (fase 3 q1).
// Uso: npx vite-node scripts/q35-deltanet-fixture-gen.mjs
// Rigenerarlo è un ATTO ESPLICITO: cambia il riferimento che il test
// anti-drift e il ktest WGSL bersagliano — va motivato nel commit.
import { writeFileSync } from "node:fs";

const { runSample, SAMPLE_DIMS, SAMPLE_T } = await import("../tests/helpers/q35-deltanet-sample.ts");

const { outputs } = runSample();
writeFileSync(
  "tests/fixtures/q35-deltanet-sample.json",
  JSON.stringify(
    {
      note: "Campione DeltaNet cpuref-f64 (q1 fase 3, spec §4). Semantica llama.cpp b10333 (delta-net-base AR). Seed pesi 20260810, input 42.",
      dims: SAMPLE_DIMS,
      T: SAMPLE_T,
      outputs,
    },
    null,
    1,
  ),
);
console.log(`OK: fixture scritto (${SAMPLE_T} token, d=${SAMPLE_DIMS.d})`);
