// Re-export del campione DeltaNet: la definizione vive in src/engine/
// q35sample.ts perché anche il ktest.worker (browser) la importa — src non
// importa da tests/. Questo file resta per stabilità dei path nei test node.
export { sampleLcg as lcg, SAMPLE_DIMS, SAMPLE_T, sampleWeights, sampleInputs, runSample } from "../../src/engine/q35sample";
