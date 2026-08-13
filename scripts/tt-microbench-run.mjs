// Driver delle sonde della riga 1 di engine-ttft: apre ttbench.html su Chrome
// headed (landmine SwiftShader: launch() semplice non espone navigator.gpu),
// aspetta done, esegue la SPAZZATA DEL TETTO NEGOZIABILE (sonda d) e scrive il
// run file in results/microbench/.
//
// Uso: BASE_URL=http://localhost:5199 node scripts/tt-microbench-run.mjs \
//        [--label 4090-linux] [--host quiescent] [--ops 64]
//
// ATTENZIONE: due runner playwright sullo stesso profilo si bloccano a vicenda —
// i bench browser vanno eseguiti UNO ALLA VOLTA. Il default di BASE_URL e' 5173:
// leggere i parametri PRIMA di spenderci GPU.
import { chromium } from "playwright";
import { mkdirSync, writeFileSync } from "node:fs";

const arg = (name, dflt) => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : dflt;
};
const LABEL = arg("label", "4090-linux");
const HOST = arg("host", "quiescent");
const SWEEP_OPS = Number(arg("ops", "64"));
const PROFILE = process.env.E2E_PROFILE ?? "/tmp/blab-e2e-profile";
const BASE_URL = process.env.BASE_URL ?? "http://localhost:5173";
const args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPUService", "--ignore-gpu-blocklist"];

const browser = await chromium.launchPersistentContext(PROFILE, { headless: false, channel: "chrome", args });
const page = browser.pages()[0] ?? (await browser.newPage());
page.on("pageerror", (e) => console.log("[tt][pageerror]", e.message.slice(0, 400)));
page.on("console", (m) => { if (m.type() === "error") console.log("[tt][console]", m.text().slice(0, 400)); });
await page.goto(`${BASE_URL}/ttbench.html?label=${encodeURIComponent(LABEL)}&host=${encodeURIComponent(HOST)}`, { waitUntil: "load" });

let last = "";
const iv = setInterval(async () => {
  try {
    const s = await page.evaluate(() => document.querySelector("#status")?.textContent ?? "");
    if (s && s !== last) { last = s; console.log("[tt]", s); }
  } catch { /* pagina in navigazione */ }
}, 2000);

let status = "TIMEOUT";
try {
  await page.waitForFunction(() => {
    const s = document.querySelector("#status")?.textContent ?? "";
    return s === "done" || s.startsWith("ERROR");
  }, null, { timeout: 1_800_000, polling: 1000 });
  status = await page.evaluate(() => document.querySelector("#status").textContent);
} catch (e) {
  console.error("[tt] attesa fallita:", String(e).slice(0, 300));
}
clearInterval(iv);

const report = await page.evaluate(() => window.__report ?? null);
const plan = await page.evaluate(() => window.__sweepPlan ?? null);

// -------------------------------------------------------------------------
// SONDA (d): spazzata del tetto negoziabile. Vive qui e non in src/ perche'
// richiede device DISTINTI con `requiredLimits` espliciti, mentre il punto unico
// di creazione device di src/engine/gpudevice.ts negozia sempre
// `min(adapter, requisito derivato)` e non scenderebbe mai sotto i 30.848 B che
// il motore chiede oggi (tests/gpudevice.test.ts vieta altri creatori in src/).
// Stessa carve-out gia' motivata per scripts/webgpu-limits.mjs.
// -------------------------------------------------------------------------
let sweep = null;
if (status === "done" && plan) {
  console.log("[tt] spazzata dei limiti concessi:", plan.requestedLimits.join(", "));
  sweep = await page.evaluate(async ({ plan, ops }) => {
    const adapter = await navigator.gpu.requestAdapter();
    const points = [];
    const live = [];
    for (const L of plan.requestedLimits) {
      const pt = {
        requestedWorkgroupStorage: L, grantedWorkgroupStorage: null, deviceCreated: false,
        pipelineCreated: false, error: null, msPerOpP50: null, samples: [],
        note: "", legacyAttnPipelineCreated: null, legacyAttnError: null,
      };
      let device = null;
      try {
        device = await adapter.requestDevice({
          requiredLimits: { maxComputeWorkgroupStorageSize: L },
        });
        pt.deviceCreated = true;
        pt.grantedWorkgroupStorage = device.limits.maxComputeWorkgroupStorageSize;
      } catch (e) {
        pt.error = `requestDevice: ${String(e).slice(0, 300)}`;
        points.push(pt);
        continue;
      }
      device.addEventListener("uncapturederror", (e) => console.error("[tt-sweep][gpu-error]", e.error.message.slice(0, 300)));

      // 1) la forma vincente compila e gira a questo tetto?
      const mk = async (code, label) => {
        device.pushErrorScope("validation");
        const module = device.createShaderModule({ code, label });
        const info = await module.getCompilationInfo();
        const errs = info.messages.filter((m) => m.type === "error");
        if (errs.length) {
          await device.popErrorScope();
          return { pipeline: null, error: errs.map((e) => `${e.lineNum}: ${e.message}`).join(" | ").slice(0, 300) };
        }
        let pipeline = null;
        let err = null;
        try {
          pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
        } catch (e) { err = String(e).slice(0, 300); }
        const scoped = await device.popErrorScope();
        if (scoped) return { pipeline: null, error: scoped.message.slice(0, 300) };
        if (err) return { pipeline: null, error: err };
        return { pipeline, error: null };
      };

      const legacy = await mk(plan.attnLegacy.wgsl, "attn-legacy");
      pt.legacyAttnPipelineCreated = legacy.pipeline !== null;
      pt.legacyAttnError = legacy.error;

      const g = plan.gemm;
      const r = await mk(g.wgsl, "gemm-regs");
      if (!r.pipeline) {
        pt.error = `pipeline: ${r.error}`;
        pt.note = `la forma '${g.variant}' (${g.workgroupStorageBytes} B di workgroup storage) non crea la pipeline a ${L} B concessi`;
        points.push(pt);
        device.destroy();
        continue;
      }
      pt.pipelineCreated = true;
      const U = GPUBufferUsage.STORAGE;
      const qs = device.createBuffer({ size: g.qsBytes, usage: U });
      const sc = device.createBuffer({ size: g.scalesBytes, usage: U });
      const x = device.createBuffer({ size: g.xBytes, usage: U });
      const y = device.createBuffer({ size: g.yBytes, usage: U });
      const bg = device.createBindGroup({
        layout: r.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: qs } }, { binding: 1, resource: { buffer: sc } },
          { binding: 2, resource: { buffer: x } }, { binding: 3, resource: { buffer: y } },
        ],
      });
      pt.note = `forma '${g.variant}' K${g.K}xN${g.N} M${g.M}, ${g.workgroupStorageBytes} B di workgroup storage, ${ops} dispatch per campione, timing CPU (onSubmittedWorkDone)`;
      live.push({ pt, device, pipeline: r.pipeline, bg, gx: g.gx, gy: g.gy });
      points.push(pt);
    }

    // esecuzione INTERLEAVATA fra i device: la deriva DVFS non deve coincidere
    // con l'ordine dei tetti concessi.
    const WARM = 3;
    const REPS = 10;
    for (let rep = 0; rep < WARM + REPS; rep++) {
      for (const l of live) {
        const enc = l.device.createCommandEncoder();
        const pass = enc.beginComputePass();
        for (let i = 0; i < ops; i++) {
          pass.setPipeline(l.pipeline);
          pass.setBindGroup(0, l.bg);
          pass.dispatchWorkgroups(l.gx, l.gy, 1);
        }
        pass.end();
        const t0 = performance.now();
        l.device.queue.submit([enc.finish()]);
        await l.device.queue.onSubmittedWorkDone();
        const ms = (performance.now() - t0) / ops;
        if (rep >= WARM) l.pt.samples.push(ms);
      }
    }
    for (const l of live) {
      const s = [...l.pt.samples].sort((a, b) => a - b);
      l.pt.msPerOpP50 = s[Math.floor(s.length / 2)];
      l.device.destroy();
    }
    return {
      variant: plan.gemm.variant,
      shape: { K: plan.gemm.K, N: plan.gemm.N },
      M: plan.gemm.M,
      workgroupStorageBytes: plan.gemm.workgroupStorageBytes,
      legacyAttnWorkgroupStorageBytes: plan.attnLegacy.workgroupStorageBytes,
      legacyAttnCtxMax: plan.attnLegacy.ctxMax,
      points,
    };
  }, { plan, ops: SWEEP_OPS });
}

await browser.close();

if (status !== "done" || !report) {
  console.error("[tt] FALLITO:", status);
  process.exit(2);
}

report.limitSweep = sweep ? [sweep] : null;
mkdirSync("results/microbench", { recursive: true });
const ts = report.ts.replace(/[:.]/g, "-");
const path = `results/microbench/ttft-riga1-${LABEL}-${ts}.json`;
writeFileSync(path, JSON.stringify(report, null, 2));
console.log("[tt] scritto", path);

for (const c of report.cells) {
  const shape = Object.entries(c.shape).map(([k, v]) => `${k}=${v}`).join(",");
  console.log(
    `[tt] ${c.kernel.padEnd(20)} ${c.variant.padEnd(20)} M=${String(c.M).padEnd(5)} ${shape.padEnd(34)} ` +
    `p50 ${c.msPerOp.p50.toFixed(4)} ms` +
    (c.tokensPerSecond ? `  ${c.tokensPerSecond.toFixed(1)} tok/s` : "") +
    (c.tflops ? `  ${c.tflops.toFixed(2)} TFLOP/s` : "") +
    (c.weightBytesPerToken ? `  ${(c.weightBytesPerToken / 1e6).toFixed(2)} MB peso/token` : "") +
    `  wgStorage ${c.workgroupStorageBytes} B`,
  );
}
for (const s of report.skipped) console.log(`[tt] SKIP ${s.kernel} ${s.variant} ${JSON.stringify(s.shape)} — ${s.reason}`);
if (sweep) {
  console.log(`[tt] sweep '${sweep.variant}' (${sweep.workgroupStorageBytes} B) — legacy attn chiede ${sweep.legacyAttnWorkgroupStorageBytes} B a ctxMax ${sweep.legacyAttnCtxMax}`);
  for (const p of sweep.points) {
    console.log(
      `[tt] limite chiesto ${String(p.requestedWorkgroupStorage).padStart(6)} concesso ${String(p.grantedWorkgroupStorage).padStart(6)}` +
      `  pipeline ${p.pipelineCreated ? "OK " : "NO "}  legacy-attn ${p.legacyAttnPipelineCreated ? "OK " : "NO "}` +
      `  p50 ${p.msPerOpP50 === null ? "n/d" : p.msPerOpP50.toFixed(4) + " ms"}` +
      (p.error ? `  err ${p.error}` : "") +
      (p.legacyAttnError ? `  legacyErr ${String(p.legacyAttnError).slice(0, 120)}` : ""),
    );
  }
}
