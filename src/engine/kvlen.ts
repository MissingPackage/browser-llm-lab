// Length pointer della KV cache (spec B1 §Crop) — logica pura, testabile in CI
// senza GPU (convenzione fase A). Il pointer è l'UNICA verità sulla posizione:
// contratto hard (postura ds4), ogni divergenza è throw, mai best-effort.
//
// - assertNext(pos, count): valida PRIMA di encodare (pos === len, capacità);
// - advance(count): incrementa DOPO il forward riuscito;
// - crop(toLen): rollback zero-GPU (le righe oltre len sono garbage mai letto:
//   l'attention legge solo [0, pos] e le righe si sovrascrivono per posizione);
// - reset() = crop(0).

export interface KvLen {
  readonly len: number;
  assertNext(pos: number, count?: number): void;
  advance(count?: number): void;
  crop(toLen: number): void;
  reset(): void;
}

export function createKvLen(ctxMax: number): KvLen {
  if (!Number.isInteger(ctxMax) || ctxMax < 1) throw new Error(`kvLen: ctxMax non valido (${ctxMax})`);
  let len = 0;
  return {
    get len() { return len; },
    assertNext(pos: number, count = 1): void {
      if (!Number.isInteger(pos) || !Number.isInteger(count) || count < 1) {
        throw new Error(`kvLen: argomenti non validi (pos=${pos}, count=${count})`);
      }
      if (pos !== len) throw new Error(`kvLen: pos ${pos} !== kvLen ${len} (posizioni libere non ammesse: usare crop/reset)`);
      if (len + count > ctxMax) throw new Error("contesto pieno");
    },
    advance(count = 1): void {
      if (!Number.isInteger(count) || count < 1 || len + count > ctxMax) {
        throw new Error(`kvLen: advance non valido (count=${count}, len=${len})`);
      }
      len += count;
    },
    crop(toLen: number): void {
      if (!Number.isInteger(toLen) || toLen < 0 || toLen > len) {
        throw new Error(`kvLen: crop non valido (toLen=${toLen}, kvLen=${len})`);
      }
      len = toLen;
    },
    reset(): void { len = 0; },
  };
}
