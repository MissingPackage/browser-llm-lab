import { describe, it, expect } from "vitest";
import { closeSync, existsSync, fstatSync, openSync, readFileSync, readSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { Q35CpuRefModel } from "../src/engine/q35cpurefmodel";

// readFileSync ha un cap a 2 GiB: il 4B (2.58 GB) si legge a chunk in un
// ArrayBuffer unico.
function readLargeFile(path: string): ArrayBuffer {
  const fd = openSync(path, "r");
  const size = fstatSync(fd).size;
  const ab = new ArrayBuffer(size);
  const CHUNK = 1 << 30;
  for (let off = 0; off < size; off += CHUNK) {
    const len = Math.min(CHUNK, size - off);
    readSync(fd, new Uint8Array(ab, off, len), 0, len, off);
  }
  closeSync(fd);
  return ab;
}

// Differential test e2e del cpuref 4B vs ORACOLO llama.cpp (fase 4 q1,
// spec §5 punto 2): teacher-forced sui token del golden smoke (chat
// template incluso: i token vengono dal JSON, la semantica template resta
// nell'oracolo), argmax == oracolo su OGNI posizione generata. Se questo
// passa, la comprensione del modello (mrope text-only, q+gate fusi,
// QK-norm, GQA, ibrido 3:1, DeltaNet) è giusta e ogni divergenza GPU a
// valle è un bug di kernel — il pattern GLM.
//
// LENTO PER DESIGN (~5-10 min: f64 streaming per-layer su ~40 posizioni,
// head 2.5 GB): gira SOLO con Q35_E2E=1 (il verifier lo esegue; la suite
// no — il costo raddoppierebbe il gate permanente).
const MODEL = join(homedir(), ".cache/blab-models/q35/Qwen3.5-4B-Q4_0.gguf");
const GOLDEN = join(process.cwd(), "results/engine/golden/q35/golden-q35-4b-smoke-2026-08-10.json");
const run = process.env.Q35_E2E === "1" && existsSync(MODEL) && existsSync(GOLDEN);

describe.skipIf(!run)("cpuref 4B e2e vs oracolo (teacher-forced)", () => {
  it("argmax == oracolo su tutte le posizioni generate del golden smoke", () => {
    const golden = JSON.parse(readFileSync(GOLDEN, "utf8")) as {
      modelSha256: string;
      prompts: { promptTokens: number[]; generated: number[]; positions: { argmax: number; top: [number, number][] }[] }[];
    };
    const p = golden.prompts[0];
    const P = p.promptTokens.length;
    // teacher-forced: prompt + tutti i generati tranne l'ultimo (l'ultima
    // posizione predice l'ultimo generato)
    const tokens = [...p.promptTokens, ...p.generated.slice(0, -1)];
    const model = new Q35CpuRefModel(readLargeFile(MODEL));
    const t0 = performance.now();
    const { argmax } = model.forward(tokens);
    const mins = ((performance.now() - t0) / 60000).toFixed(1);
    const details: string[] = [];
    let ok = 0;
    for (let i = 0; i < p.positions.length; i++) {
      const got = argmax[P - 1 + i];
      const want = p.positions[i].argmax;
      const margin = (p.positions[i].top[0]?.[1] ?? 0) - (p.positions[i].top[1]?.[1] ?? 0);
      details.push(`pos ${i}: cpuref ${got} vs oracolo ${want} (margine oracolo ${margin.toFixed(2)})`);
      if (got === want) ok++;
    }
    // gate SECCO: tutte le posizioni (i margini dell'oracolo nel report
    // dicono se un'eventuale divergenza è near-tie — regola C2)
    expect(ok, `${ok}/${p.positions.length} in ${mins} min\n${details.join("\n")}`).toBe(p.positions.length);
  }, 1_800_000);
});
