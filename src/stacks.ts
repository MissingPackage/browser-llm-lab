import type { StackId } from "./schema";

// Quant a cui il Transformers.js adapter è strutturalmente vincolato: il dtype ONNX
// è hardcoded nell'adapter (vedi src/adapters/transformersjs.ts), e a differenza del
// modelId MLC di WebLLM (che incorpora il quant, es. "...-q4f32_1-MLC"), il modelId
// ONNX non porta alcuna traccia del dtype — quindi un `quant` sbagliato per questo
// stack non è verificabile a posteriori. Fonte di verità unica, importata sia qui
// sia dall'adapter, così le due copie non possono divergere.
export const TRANSFORMERSJS_DTYPE = "q4";

// Quant a cui uno stack è strutturalmente vincolato (indipendentemente da cosa chiede
// la UI/driver). Uno stack assente da questa mappa ha quant libero.
export const STACK_FIXED_QUANT: Partial<Record<StackId, string>> = {
  transformersjs: TRANSFORMERSJS_DTYPE,
};
