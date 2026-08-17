// Driver delle sonde della riga 1 di engine-ttft: apre ttbench.html su Chrome
// headed (landmine SwiftShader: launch() semplice non espone navigator.gpu),
// aspetta done, esegue la SPAZZATA DEL TETTO NEGOZIABILE (sonda d) e scrive il
// run file in results/microbench/.
//
// Uso: BASE_URL=http://localhost:5199 node scripts/tt-microbench-run.mjs \
//        [--label 4090-linux] [--host quiescent] [--ops 64] [--tag ttft-riga1]
// `--tag` decide sia il nome del file sia il `kind` dentro il JSON: chi misura
// la fase 0 di un altro goal passa il proprio (es. --tag kquant-fase0).
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
/** Prefisso del file E valore di `kind`: v. il commento accanto alla scrittura. */
const TAG = arg("tag", "ttft-riga1");
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
  console.log("[tt] spazzata dei limiti concessi:", plan.requestedLimits.join(", "),
    "— forme:", plan.forms.map((f) => `${f.variant} (${f.workgroupStorageBytes} B)`).join(", "));
  sweep = await page.evaluate(async ({ plan, ops }) => {
    const mk = async (device, code, label) => {
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

    // una TACCA = un device. `requestDevice` CONSUMA l'adapter (Dawn:
    // Adapter.cpp:328), quindi ogni tacca chiede il suo adapter: un adapter solo
    // fa fallire tutte le tacche dopo la prima, ed e' esattamente cosi' che la
    // prima esecuzione di questa sonda e' andata persa.
    const sweeps = plan.forms.map((f) => ({
      variant: f.variant, shape: { K: plan.K, N: plan.N }, M: plan.M,
      workgroupStorageBytes: f.workgroupStorageBytes,
      legacyAttnWorkgroupStorageBytes: plan.attnLegacy.workgroupStorageBytes,
      legacyAttnCtxMax: plan.attnLegacy.ctxMax,
      points: [],
    }));
    const live = [];
    for (const L of plan.requestedLimits) {
      const adapter = await navigator.gpu.requestAdapter();
      let device = null;
      let devErr = null;
      let granted = null;
      try {
        device = await adapter.requestDevice({ requiredLimits: { maxComputeWorkgroupStorageSize: L } });
        granted = device.limits.maxComputeWorkgroupStorageSize;
        device.addEventListener("uncapturederror", (e) => console.error("[tt-sweep][gpu-error]", e.error.message.slice(0, 300)));
      } catch (e) { devErr = `requestDevice: ${String(e).slice(0, 300)}`; }

      let legacy = { pipeline: null, error: devErr };
      if (device) legacy = await mk(device, plan.attnLegacy.wgsl, "attn-legacy");

      for (let fi = 0; fi < plan.forms.length; fi++) {
        const f = plan.forms[fi];
        const pt = {
          requestedWorkgroupStorage: L, grantedWorkgroupStorage: granted, deviceCreated: device !== null,
          pipelineCreated: false, error: devErr, msPerOpP50: null, samples: [],
          legacyAttnPipelineCreated: device ? legacy.pipeline !== null : null,
          legacyAttnError: legacy.error,
          note: `forma '${f.variant}' K${plan.K}xN${plan.N} M${plan.M}, ${f.workgroupStorageBytes} B di workgroup storage, ${ops} dispatch per campione, timing CPU (onSubmittedWorkDone)`,
        };
        sweeps[fi].points.push(pt);
        if (!device) continue;
        const r = await mk(device, f.wgsl, `gemm-${f.variant}`);
        if (!r.pipeline) {
          pt.error = `pipeline: ${r.error}`;
          continue;
        }
        let combine = null;
        if (f.combineWgsl) {
          combine = await mk(device, f.combineWgsl, `combine-${f.variant}`);
          if (!combine.pipeline) { pt.error = `combine: ${combine.error}`; continue; }
        }
        pt.pipelineCreated = true;
        const U = GPUBufferUsage.STORAGE;
        const qs = device.createBuffer({ size: plan.qsBytes, usage: U });
        const sc = device.createBuffer({ size: plan.scalesBytes, usage: U });
        const x = device.createBuffer({ size: plan.xBytes, usage: U });
        const y = device.createBuffer({ size: plan.yBytes, usage: U });
        const part = combine ? device.createBuffer({ size: f.partBytes, usage: U }) : null;
        const bg = device.createBindGroup({
          layout: r.pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: qs } }, { binding: 1, resource: { buffer: sc } },
            { binding: 2, resource: { buffer: x } }, { binding: 3, resource: { buffer: part ?? y } },
          ],
        });
        const cbg = combine ? device.createBindGroup({
          layout: combine.pipeline.getBindGroupLayout(0),
          entries: [{ binding: 0, resource: { buffer: part } }, { binding: 1, resource: { buffer: y } }],
        }) : null;
        live.push({
          pt, device, pipeline: r.pipeline, bg, gx: f.gx, gy: f.gy,
          combine: combine ? combine.pipeline : null, cbg, cgx: f.combineGx,
        });
      }
    }

    // esecuzione INTERLEAVATA fra device e forme: la deriva DVFS non deve
    // coincidere con l'ordine dei tetti concessi.
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
          if (l.combine) {
            pass.setPipeline(l.combine);
            pass.setBindGroup(0, l.cbg);
            pass.dispatchWorkgroups(l.cgx, 1, 1);
          }
        }
        pass.end();
        const t0 = performance.now();
        l.device.queue.submit([enc.finish()]);
        await l.device.queue.onSubmittedWorkDone();
        const ms = (performance.now() - t0) / ops;
        if (rep >= WARM) l.pt.samples.push(ms);
      }
    }
    const seen = new Set();
    for (const l of live) {
      const s = [...l.pt.samples].sort((a, b) => a - b);
      l.pt.msPerOpP50 = s[Math.floor(s.length / 2)];
      if (!seen.has(l.device)) { seen.add(l.device); }
    }
    for (const d of seen) d.destroy();
    return sweeps;
  }, { plan, ops: SWEEP_OPS });
}

await browser.close();

if (status !== "done" || !report) {
  console.error("[tt] FALLITO:", status);
  process.exit(2);
}

report.limitSweep = sweep ?? null;
// IL NOME E IL `kind` DEVONO DIRE LA STESSA COSA, e devono dire cosa c'e'
// dentro. Il banco e' nato per la riga 1 di `engine-ttft` e da allora ha
// acquisito le celle della fase 0 di `engine-kquant`: un file che si chiama
// `ttft-riga1` e porta anche quelle mente per omissione, ed e' esattamente il
// modo in cui questo repo si e' gia' fatto male una volta (landmine: «prima di
// credere a un artefatto, leggi il suo kind, non il suo nome di file»).
// `--tag` sposta ENTRAMBI, cosi' non possono divergere.
report.kind = `microbench-${TAG}`;
// `goal` e `prereg` viaggiano col `kind`: un artefatto che dichiara il kind
// giusto e la provenienza di un altro goal e' peggio di uno sbagliato in modo
// coerente, perche' passa il filtro e mente nel contenuto (trovato in it.2).
const PROV = {
  "ttft-riga1": {
    goal: "engine-ttft riga 1 (sonde e varianti del prefill)",
    prereg: "docs/deep-dive/ttft-riga1-prereg-2026-08-13.md",
  },
  "kquant-fase0": {
    goal: "engine-kquant riga 1 (fase 0: le famiglie non-q4_0)",
    prereg: "docs/deep-dive/kquant-fase0-prereg-2026-08-14.md",
  },
  "velocita-decode-2d": {
    goal: "engine-velocita-decode riga 2d (la rotta split-K vale il decode?)",
    prereg: "docs/deep-dive/velocita-decode-2d-prereg-2026-08-16.md",
  },
  "costm-decode": {
    goal: "engine-velocita-decode, spike (1) di tre (dove sta il ginocchio di cost(M)?)",
    prereg: "docs/deep-dive/costm-decode-prereg-2026-08-17.md",
  },
}[TAG];
if (!PROV) {
  console.error(`[tt] tag sconosciuto "${TAG}": aggiungilo a PROV con il suo goal e la sua pre-registrazione, invece di scrivere un artefatto senza provenienza`);
  process.exit(2);
}
report.goal = PROV.goal;
report.prereg = PROV.prereg;
mkdirSync("results/microbench", { recursive: true });
const ts = report.ts.replace(/[:.]/g, "-");
const path = `results/microbench/${TAG}-${LABEL}-${ts}.json`;
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
  for (const sw of sweep) {
    console.log(`[tt] sweep '${sw.variant}' (${sw.workgroupStorageBytes} B) — legacy attn chiede ${sw.legacyAttnWorkgroupStorageBytes} B a ctxMax ${sw.legacyAttnCtxMax}`);
    for (const p of sw.points) {
      console.log(
        `[tt]   chiesto ${String(p.requestedWorkgroupStorage).padStart(6)} concesso ${String(p.grantedWorkgroupStorage).padStart(6)}` +
        `  pipeline ${p.pipelineCreated ? "OK " : "NO "}  legacy-attn ${p.legacyAttnPipelineCreated ? "OK " : "NO "}` +
        `  p50 ${p.msPerOpP50 === null || p.msPerOpP50 === undefined ? "n/d" : p.msPerOpP50.toFixed(4) + " ms"}` +
        (p.error ? `  err ${String(p.error).slice(0, 140)}` : ""),
      );
    }
    const ok = sw.points.filter((p) => p.msPerOpP50 !== null && p.msPerOpP50 !== undefined);
    if (ok.length > 1) {
      const lo = Math.min(...ok.map((p) => p.msPerOpP50));
      const hi = Math.max(...ok.map((p) => p.msPerOpP50));
      console.log(`[tt]   spread fra i tetti concessi: ${(100 * (hi / lo - 1)).toFixed(1)} %`);
    }
  }
  const first = sweep[0]?.points?.[0];
  if (first?.legacyAttnError) console.log(`[tt] legacy attn a 16.384 B: ${String(first.legacyAttnError).slice(0, 200)}`);
}
