import { describe, it, expect } from "vitest";
import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { loadCpuRef, CpuRefSession, argmax } from "../src/engine/cpuref";

// Differential test: forward CPU di riferimento vs golden llama.cpp. È il gate della
// comprensione del modello (RoPE NEOX, GQA, bias, ordine delle op): se questo passa,
// ogni divergenza dei kernel WGSL è un bug di kernel, non di modello.
//
// Lento per design (~2 min: dequant 2.5 GB + 30-37 token di prefill × 4 prompt su
// CPU): gira solo in locale col modello scaricato; CI lo salta.
const MODEL = `${homedir()}/.cache/blab-models/qwen2.5-0.5b-instruct-q4_0.gguf`;
const GOLDEN = "results/engine/golden/golden-qwen25-05b-q4_0.json";
const N = 4; // token verificati per prompt (teacher-forced sul golden)

describe.skipIf(!existsSync(MODEL) || !existsSync(GOLDEN))("cpuref vs oracolo", () => {
  it("top-1 identico al golden sui primi token di ogni prompt", { timeout: 600_000 }, () => {
    const golden = JSON.parse(readFileSync(GOLDEN, "utf8")) as {
      prompts: { id: string; promptTokens: number[]; positions: { argmax: number; top: [number, number][] }[] }[];
    };
    const raw = readFileSync(MODEL);
    const tensors = loadCpuRef(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));

    let maxDlogit = 0;
    for (const p of golden.prompts) {
      const sess = new CpuRefSession(tensors, p.promptTokens.length + N + 1);
      let logits!: Float32Array;
      for (const tok of p.promptTokens) logits = sess.forward(tok);
      for (let i = 0; i < N; i++) {
        const gold = p.positions[i];
        expect(argmax(logits), `${p.id} pos ${i}`).toBe(gold.argmax);
        for (const [tid, glogit] of gold.top) {
          maxDlogit = Math.max(maxDlogit, Math.abs(logits[tid] - glogit));
        }
        logits = sess.forward(gold.argmax); // teacher-forcing: confronto per-posizione pulito
      }
    }
    // Riportato, non gated (spec §Soglie): stessa dequant esatta, diverso ordine di
    // riduzione f32 — atteso O(1e-2) sui logit. Il gate è il top-1.
    console.log(`[cpuref] max|Δlogit| sui top-32 golden: ${maxDlogit.toFixed(4)}`);
    expect(maxDlogit).toBeLessThan(1); // sanity larga: ordini di grandezza, non rumore
  });
});
