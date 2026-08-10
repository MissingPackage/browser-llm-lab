import { describe, it, expect } from "vitest";
import { closeSync, existsSync, openSync, readFileSync, readSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { parseGguf } from "../src/engine/gguf";
import { q35TokenizerFromMetadata } from "../src/engine/q35tokenizer";

// Conformance tokenizer di fase 2 (q1, spec §5 punto 1): id IDENTICI a
// llama-tokenize b10333 (protocollo --ids --no-bos --no-parse-special) su
// TUTTO il corpus committato (01-08 regime GLM + 09-12 edge). Il fixture è
// generato con cross-check vocab 4B==35B (scripts/q35-tok-oracle-gen.mjs):
// qui basta il 4B. Gate SECCO: ogni divergenza è FAIL, niente near-tie.
const MODEL = join(homedir(), ".cache/blab-models/q35/Qwen3.5-4B-Q4_0.gguf");
const FIXTURE = join(process.cwd(), "tests/fixtures/q35-tok-oracle.json");
const HEADER_BYTES = 64 * 1024 * 1024;

const have = existsSync(MODEL);

describe.skipIf(!have)("q35 tokenizer vs oracolo llama.cpp", () => {
  it("id identici sul corpus completo (12 file, gate secco)", () => {
    const fd = openSync(MODEL, "r");
    const buf = Buffer.alloc(HEADER_BYTES);
    const n = readSync(fd, buf, 0, HEADER_BYTES, 0);
    closeSync(fd);
    const f = parseGguf(buf.buffer.slice(buf.byteOffset, buf.byteOffset + n) as ArrayBuffer);
    const tok = q35TokenizerFromMetadata(f.metadata);

    const fixture = JSON.parse(readFileSync(FIXTURE, "utf8")) as {
      files: Record<string, number[]>;
    };
    const names = Object.keys(fixture.files);
    expect(names.length).toBe(12);

    let tot = 0;
    for (const name of names) {
      const dir = /^(09|10|11|12)-/.test(name) ? "corpus-tok" : "corpus";
      const text = readFileSync(join(process.cwd(), "tools/oracle-moe", dir, name), "utf8");
      const ids = tok.encode(text);
      expect(ids, name).toEqual(fixture.files[name]);
      // e il decode è l'inverso esatto (byte-level BPE è lossless)
      expect(tok.decode(ids), `decode ${name}`).toBe(text);
      tot += ids.length;
    }
    expect(tot).toBe(27714);
  });
});
