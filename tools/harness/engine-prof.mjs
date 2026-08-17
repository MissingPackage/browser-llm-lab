// Verifica meccanica dei contatori L1-L3 sul MOTORE (goal engine-fase-a, fase 5):
// patch dei prototype WebGPU nel worker della pagina engine (stesso metodo di
// dispatch-profile), snapshot a inizio decode e a fine run.
// Attese: createBindGroup Δ=0 (L1), submit/forward = 1 (L2), dispatch/forward = 123 (L3).
// Uso: node tools/harness/engine-prof.mjs
import { chromium } from "playwright";
import { mkdirSync, writeFileSync } from "node:fs";

const PROFILE = process.env.E2E_PROFILE ?? "/tmp/blab-e2e-profile";
const BASE_URL = process.env.BASE_URL ?? "http://localhost:5173";
const args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPUService", "--ignore-gpu-blocklist"];

const browser = await chromium.launchPersistentContext(PROFILE, { headless: false, channel: "chrome", args });
const page = browser.pages()[0] ?? (await browser.newPage());
page.on("pageerror", (e) => console.log("[eprof][pageerror]", e.message.slice(0, 200)));
await page.goto(`${BASE_URL}/engine.html?bench=1`, { waitUntil: "load" });

let worker = page.workers()[0];
for (let i = 0; !worker && i < 50; i++) {
  await new Promise((r) => setTimeout(r, 100));
  worker = page.workers()[0];
}
if (!worker) { console.error("[eprof] nessun worker"); process.exit(1); }
await worker.evaluate(() => {
  const c = { dispatch: 0, bindGroup: 0, submit: 0, writeBuffer: 0, mapAsync: 0 };
  self.__prof = c;
  const wrap = (proto, name, key) => {
    const orig = proto[name];
    proto[name] = function (...a) { c[key]++; return orig.apply(this, a); };
  };
  wrap(GPUComputePassEncoder.prototype, "dispatchWorkgroups", "dispatch");
  wrap(GPUDevice.prototype, "createBindGroup", "bindGroup");
  wrap(GPUQueue.prototype, "submit", "submit");
  wrap(GPUQueue.prototype, "writeBuffer", "writeBuffer");
  const om = GPUBuffer.prototype.mapAsync;
  GPUBuffer.prototype.mapAsync = function (...a) { c.mapAsync++; return om.apply(this, a); };
});
console.log("[eprof] patch installata");

// FINESTRA DECODE PURA (fase B1: il prefill è chunked, non più una sequenza di
// forward — la finestra fase-A "dal warmup a fine run" mescolerebbe i due profili):
// snapshot fra i progress i=32 e i=224 dell'ULTIMA replica OFF ⇒ 192 forward di
// decode esatti fra i due snapshot (progress ogni 32 token; skew di polling ≤50 ms).
const waitStatus = async (needle) => page.waitForFunction(
  (n) => (document.querySelector("#status")?.textContent ?? "").includes(n),
  needle, { timeout: 900000, polling: 50 },
);
await waitStatus("replica OFF 3/3: 32/");
const atStart = await worker.evaluate(() => ({ ...self.__prof }));
await waitStatus("replica OFF 3/3: 224/");
const atEnd = await worker.evaluate(() => ({ ...self.__prof }));
await page.waitForFunction(
  () => {
    const s = document.querySelector("#status")?.textContent ?? "";
    return s === "done" || s.startsWith("ERROR");
  },
  null, { timeout: 900000, polling: 1000 },
);
const report = await page.evaluate(() => window.__report ?? null);
await browser.close();
if (!report) { console.error("[eprof] niente report"); process.exit(2); }

const totalForwards = 192; // finestra deterministica (i=32 → i=224)
const d = {
  dispatch: atEnd.dispatch - atStart.dispatch,
  bindGroup: atEnd.bindGroup - atStart.bindGroup,
  submit: atEnd.submit - atStart.submit,
  writeBuffer: atEnd.writeBuffer - atStart.writeBuffer,
  mapAsync: atEnd.mapAsync - atStart.mapAsync,
};
const forwardsInWindow = totalForwards;
// fase B2: col decode multi-step (report.k>1) i gate cambiano forma —
// submit/forward atteso = 1/k (una submit+mapAsync per batch), dispatch/forward
// atteso = dispatchesPerToken + 1 (l'embedGather per step non è nel piano
// statico contato da dispatchesPerToken). Bound di spec §Soglie: ≤160.
const k = report.k ?? 1;
const dispPerForward = d.dispatch / forwardsInWindow;
const expectedDisp = report.dispatchesPerToken + (k > 1 ? 1 : 0);
const out = {
  schemaVersion: 2, kind: "engine-prof", ts: new Date().toISOString(),
  deviceLabel: "4090-linux", totalForwardsExpected: totalForwards,
  decodePath: report.decodePath ?? "per-token", k,
  deltasDuranteRun: d,
  perForward: {
    dispatch: dispPerForward,
    submit: d.submit / forwardsInWindow,
    bindGroup: d.bindGroup / forwardsInWindow,
    writeBufferPerForward: d.writeBuffer / forwardsInWindow,
    mapAsync: d.mapAsync / forwardsInWindow,
  },
  gates: {
    L1_bindGroupZero: d.bindGroup === 0,
    L2_submitPerForward: Math.abs(d.submit / forwardsInWindow - 1 / k) < 0.1,
    L3_dispatchPerToken: dispPerForward,
    L3_matchesPlan: Math.abs(dispPerForward - expectedDisp) < 0.5,
    specBoundLe160: dispPerForward <= 160,
  },
};
console.log(JSON.stringify(out, null, 2));
mkdirSync("results/engine", { recursive: true });
writeFileSync(`results/engine/engine-prof-4090-${out.ts.replace(/[:.]/g, "-")}.json`, JSON.stringify(out, null, 2));
const pass = out.gates.L1_bindGroupZero && out.gates.L2_submitPerForward && out.gates.L3_matchesPlan && out.gates.specBoundLe160;
process.exit(pass ? 0 : 1);
