import { describe, it, expect } from "vitest";
import { closeSync, existsSync, openSync, readSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { parseGguf } from "../src/engine/gguf";
import { validateQwen35 } from "../src/engine/q35shape";

// Reader della testa MTP (goal fase-D, fase 7). Due pretese, e la seconda conta
// quanto la prima:
//   1. il GGUF `*-MTP-GGUF` si valida SENZA errori e la testa si vede;
//   2. il GGUF BASE si valida ESATTAMENTE come prima — `mtpLayers` 0 e nessun
//      tensore in piu' atteso. Il reader non deve aver reso opzionale niente
//      per i file che la testa non ce l'hanno.
//
// Perche' un test sul file VERO e non su un fixture: la forma della testa e'
// una proprieta' del file pubblicato, non del nostro codice, ed e' esattamente
// la cosa che un fixture scritto a mano confermerebbe per costruzione. Il pin
// (repo + sha) sta in scripts/q35-manifest.json.
const MTP = join(homedir(), ".cache/blab-models/q35/Qwen3.5-4B-MTP-Q4_0.gguf");
const BASE = join(homedir(), ".cache/blab-models/q35/Qwen3.5-4B-Q4_0.gguf");
const HEADER_BYTES = 64 * 1024 * 1024;

function header(path: string): ArrayBuffer {
  const fd = openSync(path, "r");
  const buf = Buffer.alloc(HEADER_BYTES);
  const n = readSync(fd, buf, 0, HEADER_BYTES, 0);
  closeSync(fd);
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + n) as ArrayBuffer;
}

describe.skipIf(!existsSync(MTP) || !existsSync(BASE))("testa MTP (NextN) del 4B", () => {
  it("il file MTP si valida e la testa e' un blocco full + 4 tensori suoi", () => {
    const f = parseGguf(header(MTP));
    const { shape, byName } = validateQwen35(f);

    expect(shape.mtpLayers).toBe(1);
    expect(shape.nLayer).toBe(32);
    const b = `blk.${shape.nLayer}.`;

    // I 4 pezzi che esistono SOLO nella testa. eh_proj proietta [emb; hidden]
    // concatenati, quindi la prima dimensione e' 2*dModel: se un giorno il
    // formato cambiasse in "somma invece di concatenazione", questa riga
    // fallirebbe invece di far girare il draft su pesi mal interpretati.
    expect(byName.get(`${b}nextn.eh_proj.weight`)?.dims).toEqual([2 * shape.dModel, shape.dModel]);
    for (const n of ["enorm", "hnorm", "shared_head_norm"]) {
      expect(byName.get(`${b}nextn.${n}.weight`)?.dims).toEqual([shape.dModel]);
    }

    // La testa e' FULL-attention anche se la regola dell'interval direbbe di no
    // (32 % 4 === 0). E' il motivo per cui `validateQwen35` non la passa a
    // `q35IsFullAttn`: se lo facesse, cercherebbe attn_qkv/ssm_* e fallirebbe.
    expect(shape.nLayer % shape.fullInterval).not.toBe(shape.fullInterval - 1);
    expect(byName.has(`${b}attn_q.weight`)).toBe(true);
    expect(byName.has(`${b}attn_qkv.weight`)).toBe(false);

    // 441 = 426 del base + 15 della testa (11 di blocco + 4 suoi)
    expect(f.tensors.length).toBe(441);
  });

  it("il file BASE resta invariato: nessuna testa, stessa validazione di prima", () => {
    const f = parseGguf(header(BASE));
    const { shape, byName } = validateQwen35(f);
    expect(shape.mtpLayers).toBe(0);
    expect(f.tensors.length).toBe(426);
    expect(byName.has(`blk.${shape.nLayer}.attn_q.weight`)).toBe(false);
  });
});
