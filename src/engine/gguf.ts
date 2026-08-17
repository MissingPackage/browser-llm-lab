// Parser GGUF v3 — subset per il motore (fase A: Qwen2.5-0.5B Q4_0).
//
// Puro: lavora su ArrayBuffer, zero I/O, testabile in CI senza il file da 350 MB
// (fixture sintetica in tests/engine-gguf.test.ts). Postura ds4: questo modulo
// LEGGE e basta; la validazione contro la shape attesa vive in shape.ts e muore
// con throw su qualunque mismatch.
//
// Formato (spec ggml, GGUF v3, little-endian):
//   magic "GGUF" u32 · version u32 · n_tensors u64 · n_kv u64
//   kv: [nome string][tipo u32][valore]  ripetuto n_kv volte
//   tensor info: [nome string][n_dims u32][dims u64×n][tipo u32][offset u64]
//   padding ad `general.alignment` (default 32) · dati tensori (offset relativi)

export const GGUF_MAGIC = 0x46554747; // "GGUF" LE

// Tipi di valore dei metadata (enum gguf_metadata_value_type)
export const GGUF_KV_TYPE = {
  UINT8: 0, INT8: 1, UINT16: 2, INT16: 3, UINT32: 4, INT32: 5,
  FLOAT32: 6, BOOL: 7, STRING: 8, ARRAY: 9, UINT64: 10, INT64: 11, FLOAT64: 12,
} as const;

// Tipi tensore (enum ggml_type) — solo quelli che i modelli del motore usano.
// Q8_0 c'è perché il GGUF ufficiale "q4_0" di Qwen2.5 tiene output.weight in Q8_0
// (verificato sul file, 2026-07-29): il file detta il subset, non il contrario.
// Q4_1/Q5_K/Q6_K: quant mista del GGUF unsloth di GLM-4.7-Flash (verificato
// per-layer sul file, 2026-07-31 — spec C2 §1): down_exps blk.1-4 e ffn_down
// denso Q4_1, shexp Q5_K/Q6_K, output Q6_K.
// Q4_K: expert del GGUF UD-Q4_K_S di Qwen3.6-35B-A3B (q1 fase 2 — inventario
// header 2026-08-10: 117 tensori expert Q4_K + 3 Q6_K; spec q1 §2-3).
// Q2_K/Q3_K: expert del GGUF `bartowski Q2_K` dello stesso modello (dump
// 2026-08-17: 100 tensori expert Q2_K + 20 Q3_K + 3 Q8_0). Servono perche' il
// parco expert scende da 17,07 a 11,15 GiB — sotto l'arena, cioe' residenza
// totale — al costo misurato di +0,13 bit/token.
export const GGML_TYPE = {
  F32: 0, F16: 1, Q4_0: 2, Q4_1: 3, Q8_0: 8,
  Q2_K: 10, Q3_K: 11, Q4_K: 12, Q5_K: 13, Q6_K: 14,
} as const;
export type GgmlTypeId = number;

export interface GgufTensorInfo {
  name: string;
  dims: number[]; // ordine GGUF: dims[0] = dimensione più interna (ne[0] di ggml)
  type: GgmlTypeId;
  offset: number; // relativo all'inizio della sezione dati, già allineato
}

export interface GgufFile {
  version: number;
  alignment: number;
  metadata: Record<string, unknown>;
  tensors: GgufTensorInfo[];
  dataOffset: number; // offset assoluto della sezione dati nel buffer
}

class Reader {
  private v: DataView;
  pos = 0;
  constructor(buf: ArrayBuffer) {
    this.v = new DataView(buf);
  }
  u8(): number { return this.v.getUint8(this.pos++); }
  u32(): number { const x = this.v.getUint32(this.pos, true); this.pos += 4; return x; }
  i32(): number { const x = this.v.getInt32(this.pos, true); this.pos += 4; return x; }
  f32(): number { const x = this.v.getFloat32(this.pos, true); this.pos += 4; return x; }
  f64(): number { const x = this.v.getFloat64(this.pos, true); this.pos += 8; return x; }
  u16(): number { const x = this.v.getUint16(this.pos, true); this.pos += 2; return x; }
  i16(): number { const x = this.v.getInt16(this.pos, true); this.pos += 2; return x; }
  i8(): number { return this.v.getInt8(this.pos++); }
  // u64 come Number: i file di fase A stanno sotto 2^53 byte con margine enorme;
  // un valore oltre Number.MAX_SAFE_INTEGER è un file corrotto, non un caso d'uso.
  u64(): number {
    const x = this.v.getBigUint64(this.pos, true); this.pos += 8;
    if (x > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error(`gguf: u64 fuori range: ${x}`);
    return Number(x);
  }
  i64(): number {
    const x = this.v.getBigInt64(this.pos, true); this.pos += 8;
    if (x > BigInt(Number.MAX_SAFE_INTEGER) || x < -BigInt(Number.MAX_SAFE_INTEGER)) {
      throw new Error(`gguf: i64 fuori range: ${x}`);
    }
    return Number(x);
  }
  str(): string {
    const len = this.u64();
    const bytes = new Uint8Array(this.v.buffer, this.pos, len);
    this.pos += len;
    return new TextDecoder().decode(bytes);
  }
}

function readKvValue(r: Reader, type: number): unknown {
  switch (type) {
    case GGUF_KV_TYPE.UINT8: return r.u8();
    case GGUF_KV_TYPE.INT8: return r.i8();
    case GGUF_KV_TYPE.UINT16: return r.u16();
    case GGUF_KV_TYPE.INT16: return r.i16();
    case GGUF_KV_TYPE.UINT32: return r.u32();
    case GGUF_KV_TYPE.INT32: return r.i32();
    case GGUF_KV_TYPE.FLOAT32: return r.f32();
    case GGUF_KV_TYPE.BOOL: return r.u8() !== 0;
    case GGUF_KV_TYPE.STRING: return r.str();
    case GGUF_KV_TYPE.UINT64: return r.u64();
    case GGUF_KV_TYPE.INT64: return r.i64();
    case GGUF_KV_TYPE.FLOAT64: return r.f64();
    case GGUF_KV_TYPE.ARRAY: {
      const elemType = r.u32();
      const n = r.u64();
      // Gli array giganti del tokenizer (152k stringhe) servono solo quando servirà
      // il tokenizer: si leggono comunque (il cursore deve avanzare) ma restano array JS.
      const out: unknown[] = new Array(n);
      for (let i = 0; i < n; i++) out[i] = readKvValue(r, elemType);
      return out;
    }
    default:
      throw new Error(`gguf: kv type sconosciuto ${type} @${r.pos}`);
  }
}

export function parseGguf(buf: ArrayBuffer): GgufFile {
  const r = new Reader(buf);
  if (r.u32() !== GGUF_MAGIC) throw new Error("gguf: magic mancante");
  const version = r.u32();
  if (version !== 3) throw new Error(`gguf: version ${version} non supportata (solo v3)`);
  const nTensors = r.u64();
  const nKv = r.u64();

  const metadata: Record<string, unknown> = {};
  for (let i = 0; i < nKv; i++) {
    const key = r.str();
    const type = r.u32();
    metadata[key] = readKvValue(r, type);
  }

  const tensors: GgufTensorInfo[] = [];
  for (let i = 0; i < nTensors; i++) {
    const name = r.str();
    const nDims = r.u32();
    if (nDims < 1 || nDims > 4) throw new Error(`gguf: n_dims ${nDims} per ${name}`);
    const dims: number[] = [];
    for (let d = 0; d < nDims; d++) dims.push(r.u64());
    const type = r.u32();
    const offset = r.u64();
    tensors.push({ name, dims, type, offset });
  }

  const alignment = Number(metadata["general.alignment"] ?? 32);
  const dataOffset = Math.ceil(r.pos / alignment) * alignment;
  return { version, alignment, metadata, tensors, dataOffset };
}

// Byte occupati da un tensore nel file (per validazione bounds e per lo slicing
// in fase di load). Righe = dims[1..], elementi per riga = dims[0].
export function tensorByteSize(t: GgufTensorInfo): number {
  const elems = t.dims.reduce((a, b) => a * b, 1);
  switch (t.type) {
    case GGML_TYPE.F32: return elems * 4;
    case GGML_TYPE.F16: return elems * 2;
    case GGML_TYPE.Q4_0: {
      if (t.dims[0] % 32 !== 0) throw new Error(`gguf: ${t.name} Q4_0 con ne[0]=${t.dims[0]} non multiplo di 32`);
      return (elems / 32) * 18; // blocco Q4_0: 2 B scala f16 + 16 B nibbles
    }
    case GGML_TYPE.Q4_1: {
      if (t.dims[0] % 32 !== 0) throw new Error(`gguf: ${t.name} Q4_1 con ne[0]=${t.dims[0]} non multiplo di 32`);
      return (elems / 32) * 20; // blocco Q4_1: 2 B scala f16 + 2 B min f16 + 16 B nibbles
    }
    case GGML_TYPE.Q8_0: {
      if (t.dims[0] % 32 !== 0) throw new Error(`gguf: ${t.name} Q8_0 con ne[0]=${t.dims[0]} non multiplo di 32`);
      return (elems / 32) * 34; // blocco Q8_0: 2 B scala f16 + 32 B int8
    }
    case GGML_TYPE.Q2_K: {
      if (t.dims[0] % 256 !== 0) throw new Error(`gguf: ${t.name} Q2_K con ne[0]=${t.dims[0]} non multiplo di 256`);
      return (elems / 256) * 84; // superblocco: 16 B scale+min 4-bit + 64 B qs + d f16 + dmin f16
    }
    case GGML_TYPE.Q3_K: {
      if (t.dims[0] % 256 !== 0) throw new Error(`gguf: ${t.name} Q3_K con ne[0]=${t.dims[0]} non multiplo di 256`);
      return (elems / 256) * 110; // superblocco: 32 B hmask + 64 B qs + 12 B scale 6-bit + d f16
    }
    case GGML_TYPE.Q4_K: {
      if (t.dims[0] % 256 !== 0) throw new Error(`gguf: ${t.name} Q4_K con ne[0]=${t.dims[0]} non multiplo di 256`);
      return (elems / 256) * 144; // superblocco: d f16 + dmin f16 + 12 B scale 6-bit + 128 B qs
    }
    case GGML_TYPE.Q5_K: {
      if (t.dims[0] % 256 !== 0) throw new Error(`gguf: ${t.name} Q5_K con ne[0]=${t.dims[0]} non multiplo di 256`);
      return (elems / 256) * 176; // superblocco: d f16 + dmin f16 + 12 B scale 6-bit + 32 B qh + 128 B qs
    }
    case GGML_TYPE.Q6_K: {
      if (t.dims[0] % 256 !== 0) throw new Error(`gguf: ${t.name} Q6_K con ne[0]=${t.dims[0]} non multiplo di 256`);
      return (elems / 256) * 210; // superblocco: 128 B ql + 64 B qh + 16 B scale int8 + d f16
    }
    default:
      throw new Error(`gguf: tipo tensore ${t.type} non supportato dal motore (${t.name})`);
  }
}
