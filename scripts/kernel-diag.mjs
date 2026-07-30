// Test kernel GEMM chunk in isolamento vs dequant CPU (fase 3 B1; scaffolding di
// fase, rimozione a fase 6 con gli altri knob diag). Uso: node scripts/kernel-diag.mjs
// (dev server attivo su BASE_URL). Guardia contro il bug Tint decl-in-loop (vedi wgsl.ts).
const BASE_URL = "http://localhost:5199";
const args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPUService", "--ignore-gpu-blocklist"];
const browser = await chromium.launchPersistentContext(PROFILE, { headless: false, channel: "chrome", args });
const page = browser.pages()[0] ?? (await browser.newPage());
page.on("pageerror", (e) => console.log("[sdiag][pageerror]", e.message.slice(0, 300)));
await page.goto(`${BASE_URL}/engine.html?kerneldiag=1`, { waitUntil: "load" });
await page.waitForFunction(() => {
  const s = document.querySelector("#status")?.textContent ?? "";
  return s === "done" || s.startsWith("ERROR");
}, null, { timeout: 600000, polling: 1000 });
const status = await page.evaluate(() => document.querySelector("#status").textContent);
const report = await page.evaluate(() => window.__report ?? null);
await browser.close();
if (status !== "done" || !report) { console.error("[sdiag] FALLITO:", status); process.exit(2); }
console.log("[sdiag]", JSON.stringify(report, null, 1));
