import { describe, it, expect } from "vitest";
import { existsSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
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
// Token verificati per prompt (teacher-forced sul golden). CPUREF_N=128 = protocollo
// completo (~13 min): usato per calibrare il noise floor del confronto con l'oracolo.
const N = Number(process.env.CPUREF_N ?? 4);

describe.skipIf(!existsSync(MODEL) || !existsSync(GOLDEN))("cpuref vs oracolo", () => {
  it("top-1 identico al golden sui primi token di ogni prompt", { timeout: 600_000 }, () => {
    const golden = JSON.parse(readFileSync(GOLDEN, "utf8")) as {
      prompts: { id: string; promptTokens: number[]; positions: { argmax: number; top: [number, number][] }[] }[];
    };
    const raw = readFileSync(MODEL);
    const tensors = loadCpuRef(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));

    let maxDlogit = 0;
    let agree = 0, total = 0;
    const mismatches: string[] = [];
    const dump: { id: string; argmax: number[] }[] = [];
    for (const p of golden.prompts) {
      const sess = new CpuRefSession(tensors, p.promptTokens.length + N + 1);
      let logits!: Float32Array;
      for (const tok of p.promptTokens) logits = sess.forward(tok);
      const chain: number[] = [];
      for (let i = 0; i < N; i++) {
        const gold = p.positions[i];
        const got = argmax(logits);
        chain.push(got);
        total++;
        if (got === gold.argmax) agree++;
        else mismatches.push(`${p.id} pos${i}: ${got}≠${gold.argmax}`);
        for (const [tid, glogit] of gold.top) {
          maxDlogit = Math.max(maxDlogit, Math.abs(logits[tid] - glogit));
        }
        logits = sess.forward(gold.argmax); // teacher-forcing: confronto per-posizione pulito
      }
      dump.push({ id: p.id, argmax: chain });
    }
    // CPUREF_DUMP=1 (con N=128): salva gli argmax f64 come secondo golden del gate
    // doppio di conformance (ruling docket 3, opzione A).
    if (process.env.CPUREF_DUMP === "1" && N >= 128) {
      mkdirSync("results/engine/golden", { recursive: true });
      writeFileSync("results/engine/golden/cpuref-argmax-qwen25-05b-q4_0.json", JSON.stringify({
        schemaVersion: 1, kind: "cpuref-argmax", oracle: "cpuref-f64 (teacher-forced sul golden llama.cpp)",
        model: (golden as unknown as { model?: string }).model ?? "qwen2.5-0.5b-instruct-q4_0.gguf",
        prompts: dump,
      }, null, 1));
      console.log("[cpuref] dump argmax scritto in results/engine/golden/");
    }
    console.log(`[cpuref] top-1: ${agree}/${total} (${((agree / total) * 100).toFixed(2)}%)`);
    if (mismatches.length) console.log(`[cpuref] mismatch: ${mismatches.join("; ")}`);
    // Smoke (N piccolo): parità piena attesa. Calibrazione (N=128): il numero stampato
    // È il noise floor del confronto con l'oracolo (llama.cpp CPU quantizza le
    // attivazioni a Q8: algoritmo diverso ⇒ near-tie che flippano anche in f64).
    if (N <= 8) expect(agree).toBe(total);
    else expect(agree / total).toBeGreaterThan(0.95);
    // Riportato, non gated (spec §Soglie). Calibrato 2026-07-29 sul protocollo
    // completo: maxΔ = 1.12 — NON è rumore nostro, è la distanza fra la matematica
    // esatta e il dot Q8-quantizzato dell'oracolo CPU (llama.cpp vec_dot q4_0×q8_0).
    console.log(`[cpuref] max|Δlogit| sui top-32 golden: ${maxDlogit.toFixed(4)}`);
    expect(maxDlogit).toBeLessThan(2.5); // sanity: ordini di grandezza, non rumore
  });
});
