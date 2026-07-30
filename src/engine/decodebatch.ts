// Piano del decode loop multi-step (spec B2 §Decode loop multi-step) — puro
// CPU-side, testabile in CI senza GPU. K forward per submit con feedback del
// token on-GPU e readback 1/K; il percorso per-token (forwardToken) resta
// l'oracolo di identità.

export const DECODE_K_MAX = 8; // slot uniform preallocati (stride 256, come il prefill)
export const DECODE_SLOT_STRIDE = 256; // byte per slot (copy-safe, pattern pcSlots B1)

export interface DecodeStep { pos: number; slotOffset: number }

// piano del batch: k step a posizioni consecutive da posStart (pos === kvLen del
// primo step, contratto hard come forwardToken)
export function planDecodeBatch(posStart: number, k: number, ctxMax: number): DecodeStep[] {
  if (!Number.isInteger(k) || k < 1 || k > DECODE_K_MAX) throw new Error(`decodeBatch: k fuori range (${k})`);
  if (posStart + k > ctxMax) throw new Error(`decodeBatch: oltre ctxMax (${posStart}+${k} > ${ctxMax})`);
  return Array.from({ length: k }, (_, i) => ({ pos: posStart + i, slotOffset: i * DECODE_SLOT_STRIDE }));
}

// semantica EOS mid-batch (spec: i token oltre EOS si eliminano con crop, che il
// rollback B1 rende esatto): kept = fino all'EOS INCLUSO; cropTo = posizione a
// cui riportare kvLen (posStart + kept.length: le righe KV oltre sono garbage
// mai letto, sovrascritte al prossimo forward per posizione)
export function trimAtEos(
  ids: number[], posStart: number, eosId: number,
): { kept: number[]; stop: boolean; cropTo: number } {
  const at = ids.indexOf(eosId);
  const kept = at === -1 ? ids : ids.slice(0, at + 1);
  return { kept, stop: at !== -1, cropTo: posStart + kept.length };
}
