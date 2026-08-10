// Tokenizer famiglia Qwen 3.5/3.6 — BPE byte-level GPT-2 style, pre `qwen35`.
//
// Primo tokenizer IN-ENGINE (il metodo GLM consuma id pre-tokenizzati
// dall'oracolo; nel browser reale l'input utente va tokenizzato qui). Gate di
// fedeltà: id IDENTICI a llama-tokenize b10333 col protocollo fissato in fase
// 2 (`--ids --no-bos --no-parse-special`) sul corpus committato — vedi
// tests/engine-q35-tokenizer.test.ts vs tests/fixtures/q35-tok-oracle.json.
//
// Regex pre `qwen35` copiata VERBATIM da llama.cpp llama-vocab.cpp
// (LLAMA_VOCAB_PRE_TYPE_QWEN35, verificata su master 2026-08-10): variante
// qwen2 + \p{M} (combining marks nelle classi lettera). JS la esprime 1:1
// con /u (unicode property escapes).
//
// Ambito: encode/decode del TESTO. Semantica special ESATTA di llama.cpp
// (llama-vocab.cpp:3171, verificata su master 2026-08-10): con
// parse_special=false si saltano SOLO i token CONTROL/UNKNOWN — i token
// USER_DEFINED (nel vocab famiglia: <tool_call>, <tool_response>, <think> e
// chiusure, type 4) si matchano SEMPRE nel partizionamento pre-BPE. Il
// protocollo del fixture (--no-parse-special --no-escape) corrisponde a
// encode(text) con parseSpecial=false. Niente BOS automatico (--no-bos); il
// chat-template vive a un livello sopra, quando servirà.

const PRE_QWEN35 =
  /(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;

// Mappa byte→carattere di GPT-2 (byte "stampabili" identità, gli altri
// rimappati da U+0100 in su): i token del GGUF sono stringhe in QUESTO
// alfabeto, quindi la mappa è il ponte fra UTF-8 reale e vocabolario.
function buildByteMaps(): { enc: string[]; dec: Map<string, number> } {
  const enc = new Array<string>(256);
  const dec = new Map<string, number>();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    const printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
    const cp = printable ? b : 256 + n++;
    enc[b] = String.fromCharCode(cp);
    dec.set(enc[b], b);
  }
  return { enc, dec };
}

// token_type GGUF (enum gguf TokenType)
const TOKEN_TYPE = { NORMAL: 1, UNKNOWN: 2, CONTROL: 3, USER_DEFINED: 4 } as const;

export class Q35Tokenizer {
  private vocab = new Map<string, number>();
  private tokens: string[];
  private ranks = new Map<string, number>();
  private byteEnc: string[];
  private byteDec: Map<string, number>;
  private cache = new Map<string, number[]>();
  private userDefined: { id: number; text: string }[] = [];
  private control: { id: number; text: string }[] = [];

  /** tokens/merges/types = array dei metadata GGUF `tokenizer.ggml.{tokens,merges,token_type}`. */
  constructor(tokens: string[], merges: string[], types: number[]) {
    if (types.length !== tokens.length) throw new Error("q35tokenizer: token_type length ≠ tokens");
    this.tokens = tokens;
    for (let i = 0; i < tokens.length; i++) this.vocab.set(tokens[i], i);
    // rank = posizione nella lista merges; chiave "A B" (nell'alfabeto
    // byte-encoded non esistono spazi letterali, lo spazio è Ġ).
    for (let i = 0; i < merges.length; i++) this.ranks.set(merges[i], i);
    // special in ordine di id ascendente = ordine di partizionamento llama.cpp
    for (let i = 0; i < tokens.length; i++) {
      if (types[i] === TOKEN_TYPE.USER_DEFINED) this.userDefined.push({ id: i, text: tokens[i] });
      else if (types[i] === TOKEN_TYPE.CONTROL) this.control.push({ id: i, text: tokens[i] });
    }
    const m = buildByteMaps();
    this.byteEnc = m.enc;
    this.byteDec = m.dec;
  }

  encode(text: string, parseSpecial = false): number[] {
    // partizionamento pre-BPE sugli special (llama.cpp st_partition): un
    // token alla volta in ordine di id, ogni occorrenza spacca il frammento
    type Frag = { raw: string } | { id: number };
    let frags: Frag[] = [{ raw: text }];
    const specials = parseSpecial ? [...this.userDefined, ...this.control] : this.userDefined;
    for (const st of specials) {
      const next: Frag[] = [];
      for (const fr of frags) {
        if (!("raw" in fr) || fr.raw.length === 0) {
          next.push(fr);
          continue;
        }
        let rest = fr.raw;
        let at = rest.indexOf(st.text);
        while (at >= 0) {
          if (at > 0) next.push({ raw: rest.slice(0, at) });
          next.push({ id: st.id });
          rest = rest.slice(at + st.text.length);
          at = rest.indexOf(st.text);
        }
        if (rest.length > 0) next.push({ raw: rest });
      }
      frags = next;
    }
    const out: number[] = [];
    for (const fr of frags) {
      if ("id" in fr) out.push(fr.id);
      else for (const m of fr.raw.matchAll(PRE_QWEN35)) out.push(...this.encodeChunk(m[0]));
    }
    return out;
  }

  private encodeChunk(chunk: string): number[] {
    const hit = this.cache.get(chunk);
    if (hit) return hit;
    const bytes = new TextEncoder().encode(chunk);
    let word: string[] = new Array(bytes.length);
    for (let i = 0; i < bytes.length; i++) word[i] = this.byteEnc[bytes[i]];
    // merge iterativo: sempre la coppia col rank minimo (BPE canonico)
    while (word.length > 1) {
      let best = -1;
      let bestRank = Infinity;
      for (let i = 0; i < word.length - 1; i++) {
        const r = this.ranks.get(`${word[i]} ${word[i + 1]}`);
        if (r !== undefined && r < bestRank) {
          bestRank = r;
          best = i;
        }
      }
      if (best < 0) break;
      word = [...word.slice(0, best), word[best] + word[best + 1], ...word.slice(best + 2)];
    }
    const ids = word.map((s) => {
      const id = this.vocab.get(s);
      if (id === undefined) throw new Error(`q35tokenizer: simbolo fuori vocabolario: ${JSON.stringify(s)}`);
      return id;
    });
    if (this.cache.size < 32768) this.cache.set(chunk, ids);
    return ids;
  }

  decode(ids: number[]): string {
    const bytes: number[] = [];
    for (const id of ids) {
      const tok = this.tokens[id];
      if (tok === undefined) throw new Error(`q35tokenizer: id fuori vocabolario: ${id}`);
      for (const ch of tok) {
        const b = this.byteDec.get(ch);
        // carattere fuori alfabeto byte (special token testuale): passa in UTF-8
        if (b === undefined) bytes.push(...new TextEncoder().encode(ch));
        else bytes.push(b);
      }
    }
    return new TextDecoder().decode(new Uint8Array(bytes));
  }
}

/** Costruisce il tokenizer dai metadata GGUF già parsati (gguf.ts). */
export function q35TokenizerFromMetadata(metadata: Record<string, unknown>): Q35Tokenizer {
  const model = metadata["tokenizer.ggml.model"];
  const pre = metadata["tokenizer.ggml.pre"];
  if (model !== "gpt2" || pre !== "qwen35") {
    throw new Error(`q35tokenizer: tokenizer ${String(model)}/${String(pre)} non supportato (atteso gpt2/qwen35)`);
  }
  const tokens = metadata["tokenizer.ggml.tokens"];
  const merges = metadata["tokenizer.ggml.merges"];
  const types = metadata["tokenizer.ggml.token_type"];
  if (!Array.isArray(tokens) || !Array.isArray(merges) || !Array.isArray(types)) {
    throw new Error("q35tokenizer: tokens/merges/token_type mancanti nei metadata");
  }
  return new Q35Tokenizer(tokens as string[], merges as string[], types as number[]);
}
