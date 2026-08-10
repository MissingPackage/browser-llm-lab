// Probe EMPIRICO subgroup-matrix (q1 fase 6, spike da contratto): il nostro
// Chrome espone la feature? Prova con flag progressivamente più permissivi
// (stable → unsafe-apis → experimental-subgroup-matrix) e registra adapter
// features + esito requestDevice. Output: JSON su stdout (il chiamante lo
// committa in results/engine/).
// Uso: node scripts/webgpu-subgroup-matrix-probe.mjs
import { chromium } from "playwright";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const CANDIDATE_FEATURES = [
  "chromium-experimental-subgroup-matrix",
  "subgroup-matrix",
  "chromium-experimental-subgroups",
  "subgroups",
  "shader-f16",
];

async function probe(label, args) {
  // launchPersistentContext + headed: lo stesso setup dei runner funzionanti
  // (landmine SwiftShader: launch() semplice non espone navigator.gpu)
  const profile = mkdtempSync(join(tmpdir(), "blab-sgm-probe-"));
  const browser = await chromium.launchPersistentContext(profile, { headless: false, channel: "chrome", args });
  const page = browser.pages()[0] ?? (await browser.newPage());
  // WebGPU non è esposto su about:blank: serve un origin http (i runner
  // funzionanti navigano sempre su localhost)
  const BASE_URL = process.env.BASE_URL ?? "http://localhost:5199";
  await page.goto(`${BASE_URL}/index.html`, { waitUntil: "load" });
  const out = await page.evaluate(async (cands) => {
    if (!navigator.gpu) return { error: "navigator.gpu assente" };
    const first = await navigator.gpu.requestAdapter();
    if (!first) return { error: "requestAdapter null" };
    const features = [...first.features].sort();
    const info = first.info ? { vendor: first.info.vendor, architecture: first.info.architecture } : null;
    // configs subgroup-matrix (Dawn experimental: adapter.info.subgroupMatrixConfigs)
    let configs = null;
    const rawCfg = first.info && first.info.subgroupMatrixConfigs;
    if (rawCfg) {
      configs = [...rawCfg].map((c) => ({ component: c.componentType, result: c.resultComponentType, M: c.M, N: c.N, K: c.K }));
    }
    const per = {};
    for (const f of cands) {
      // adapter FRESCO per ogni requestDevice (Chrome consuma l'adapter)
      const a = await navigator.gpu.requestAdapter();
      per[f] = { onAdapter: a.features.has(f), deviceOk: null };
      if (per[f].onAdapter) {
        try {
          const d = await a.requestDevice({ requiredFeatures: [f] });
          per[f].deviceOk = d.features.has(f);
          d.destroy?.();
        } catch (e) {
          per[f].deviceOk = `ERROR: ${String(e).slice(0, 120)}`;
        }
      }
    }
    // tentativo di COMPILAZIONE di uno shader subgroup-matrix minimale (la
    // praticabilita' si prova, non si assume) — enable directive sperimentale
    let compile = null;
    if (per["chromium-experimental-subgroup-matrix"] && per["chromium-experimental-subgroup-matrix"].onAdapter) {
      const a2 = await navigator.gpu.requestAdapter();
      const dev = await a2.requestDevice({ requiredFeatures: ["chromium-experimental-subgroup-matrix"] });
      const cfg = configs && configs[0];
      const ct = cfg ? cfg.component : "f16";
      const rt = cfg ? cfg.result : "f16";
      const M = cfg ? cfg.M : 8, N = cfg ? cfg.N : 8, K = cfg ? cfg.K : 8;
      const code = `
enable chromium_experimental_subgroup_matrix;
@group(0) @binding(0) var<storage, read_write> a: array<${ct}>;
@group(0) @binding(1) var<storage, read_write> b: array<${ct}>;
@group(0) @binding(2) var<storage, read_write> c: array<${rt}>;
@compute @workgroup_size(32)
fn main() {
  let l: subgroup_matrix_left<${ct}, ${K}, ${M}> = subgroupMatrixLoad<subgroup_matrix_left<${ct}, ${K}, ${M}>>(&a, 0u, false, ${K}u);
  let r: subgroup_matrix_right<${ct}, ${N}, ${K}> = subgroupMatrixLoad<subgroup_matrix_right<${ct}, ${N}, ${K}>>(&b, 0u, false, ${N}u);
  var acc: subgroup_matrix_result<${rt}, ${N}, ${M}>;
  acc = subgroupMatrixMultiplyAccumulate(l, r, acc);
  subgroupMatrixStore(&c, 0u, acc, false, ${N}u);
}`;
      const mod = dev.createShaderModule({ code });
      const ci = await mod.getCompilationInfo();
      const errs = ci.messages.filter((m) => m.type === "error").map((m) => m.message.slice(0, 200));
      compile = { config: cfg ?? "nessuna esposta, tentato f16 8x8x8", ok: errs.length === 0, errors: errs.slice(0, 3) };
      dev.destroy?.();
    }
    return { info, features, configs, per, compile };
  }, CANDIDATE_FEATURES);
  await browser.close();
  return { label, args, ...out };
}

const results = [];
results.push(await probe("stable-default", ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPUService", "--ignore-gpu-blocklist"]));
results.push(await probe("unsafe-apis", ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPUService", "--ignore-gpu-blocklist", "--enable-dawn-features=allow_unsafe_apis"]));
results.push(await probe("developer-features", ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPUService", "--ignore-gpu-blocklist", "--enable-dawn-features=allow_unsafe_apis", "--enable-webgpu-developer-features"]));

console.log(JSON.stringify({
  schemaVersion: 1,
  kind: "webgpu-subgroup-matrix-probe",
  date: new Date().toISOString().slice(0, 10),
  chromeChannel: "chrome (stable, playwright)",
  probes: results,
}, null, 1));
