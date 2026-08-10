import { describe, it, expect } from "vitest";
import { closeSync, existsSync, fstatSync, openSync, readFileSync, readSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { Q35CpuRefModel, type Q35ByteSource } from "../src/engine/q35cpurefmodel";

// Differential e2e del cpuref 35B-A3B (MoE) vs ORACOLO llama.cpp — fase 7
// slice 2 (q1): teacher-forced sul golden smoke 35B, argmax == oracolo su
// ogni posizione generata. Con questo il TERZO modello della famiglia (e il
// primo MoE qwen35moe) ha la comprensione provata: da qui ogni divergenza
// GPU è bug di kernel/paging. Il file (20.9 GB) NON si carica in RAM:
// Q35ByteSource da fd con pread (reader lazy, it.15).
// LENTO (~5-10 min in f64): gira SOLO con Q35_E2E=1.
const MODEL = join(homedir(), ".cache/blab-models/q35/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf");
const GOLDEN = join(process.cwd(), "results/engine/golden/q35/golden-q35-35b-smoke-2026-08-10.json");
const run = process.env.Q35_E2E === "1" && existsSync(MODEL) && existsSync(GOLDEN);

function fdSource(path: string): Q35ByteSource {
  const fd = openSync(path, "r");
  const size = fstatSync(fd).size;
  return {
    size,
    slice(off, len) {
      const out = new Uint8Array(len);
      let got = 0;
      while (got < len) {
        const n = readSync(fd, out, got, len - got, off + got);
        if (n <= 0) throw new Error(`fdSource: read corto a ${off + got}`);
        got += n;
      }
      return out;
    },
  };
  void closeSync; // fd vive quanto il processo di test
}

describe.skipIf(!run)("cpuref 35B-A3B (MoE) e2e vs oracolo", () => {
  it("argmax == oracolo su tutte le posizioni generate del golden smoke", () => {
    const golden = JSON.parse(readFileSync(GOLDEN, "utf8")) as {
      prompts: { promptTokens: number[]; generated: number[]; positions: { argmax: number; top: [number, number][] }[] }[];
    };
    const p = golden.prompts[0];
    const nPos = Math.min(p.positions.length, p.generated.length);
    const P = p.promptTokens.length;
    const tokens = [...p.promptTokens, ...p.generated.slice(0, nPos - 1)];
    const model = new Q35CpuRefModel(fdSource(MODEL));
    expect(model.shape.arch).toBe("qwen35moe");
    const t0 = performance.now();
    const { argmax } = model.forward(tokens);
    const mins = ((performance.now() - t0) / 60000).toFixed(1);
    const details: string[] = [];
    let ok = 0;
    for (let i = 0; i < nPos; i++) {
      const got = argmax[P - 1 + i];
      const want = p.positions[i].argmax;
      const margin = (p.positions[i].top[0]?.[1] ?? 0) - (p.positions[i].top[1]?.[1] ?? 0);
      details.push(`pos ${i}: cpuref ${got} vs oracolo ${want} (margine ${margin.toFixed(2)})`);
      if (got === want) ok++;
    }
    expect(ok, `${ok}/${nPos} in ${mins} min\n${details.join("\n")}`).toBe(nPos);
  }, 3_600_000);
});
