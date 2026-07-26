// Driver di sequenza per verificare il ruling docket #5b (warm-up).
// Riproduce l'ordine del run 4090-linux-2026-07-26T19-54-55-278Z.json:
//   transformersjs, webllm, transformersjs, webllm
// e logga i clock GPU prima/dopo ogni cella.
import { chromium } from "playwright";
import { execSync } from "node:child_process";
import { copyFileSync } from "node:fs";

const PROFILE = process.env.E2E_PROFILE ?? "/tmp/blab-seq-profile";
const HEADED = process.env.HEADED === "1";
const CHANNEL = process.env.CHANNEL; // "chrome" per il branded (necessario per la GPU reale su Fedora/NVIDIA)
const args = process.env.CHROME_ARGS
  ? process.env.CHROME_ARGS.split(" ")
  : ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPUService", "--ignore-gpu-blocklist"];

const MODELS = {
  transformersjs: { id: "onnx-community/Qwen2.5-0.5B-Instruct", quant: "q4" },
  webllm: { id: "Qwen2.5-0.5B-Instruct-q4f32_1-MLC", quant: "q4f32_1" },
};
const SEQUENCE = (process.env.SEQ ?? "transformersjs,webllm,transformersjs,webllm").split(",");

const clocks = () =>
  execSync(
    "nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,memory.used,clocks_throttle_reasons.active --format=csv,noheader",
  )
    .toString()
    .trim();

const browser = await chromium.launchPersistentContext(PROFILE, {
  headless: !HEADED,
  args,
  ...(CHANNEL ? { channel: CHANNEL } : {}),
});
const page = browser.pages()[0] ?? (await browser.newPage());
page.on("pageerror", (e) => console.log(`[pageerror] ${e.message.slice(0, 300)}`));

await page.goto("http://localhost:5174", { waitUntil: "load" });
await page.waitForFunction(() => document.querySelector("#probe-box")?.textContent?.includes("webgpu"), null, { timeout: 30000 });
const probe = JSON.parse(await page.evaluate(() => document.querySelector("#probe-box").textContent));
const vendor = (probe.adapterInfo?.vendor ?? "").toLowerCase();
const desc = `${probe.adapterInfo?.description ?? ""} ${probe.adapterInfo?.architecture ?? ""}`;
if (!probe.webgpu || vendor !== "nvidia" || /swiftshader|llvmpipe|software/i.test(desc)) {
  console.log("[seq] STOP: adapter non reale —", JSON.stringify(probe.adapterInfo));
  await browser.close();
  process.exit(2);
}
console.log("[seq] GPU:", probe.adapterInfo.description || probe.adapterInfo.architecture);

for (const [i, stack] of SEQUENCE.entries()) {
  const m = MODELS[stack];
  await page.evaluate(
    ([stack, id, quant]) => {
      const stackSel = document.querySelector("#stack");
      stackSel.value = stack;
      stackSel.dispatchEvent(new Event("change"));
      const sel = document.querySelector("#model");
      const opt = Array.from(sel.options).find((o) => o.value === id);
      if (!opt) throw new Error(`option mancante: ${id}`);
      if (opt.dataset.quant !== quant) throw new Error(`quant mismatch: ${opt.dataset.quant} != ${quant}`);
      sel.value = id;
    },
    [stack, m.id, m.quant],
  );
  console.log(`[seq] cella ${i + 1}/${SEQUENCE.length} — ${stack} — clock pre: ${clocks()}`);
  // Azzera lo status: resta "done" dalla cella precedente, e senza reset la waitForFunction
  // qui sotto ritornerebbe immediatamente, facendo cliccare #run mentre il bench è in corso.
  await page.evaluate(() => { document.querySelector("#status").textContent = ""; });
  await page.click("#run");
  await page.waitForFunction(
    () => {
      const s = document.querySelector("#status")?.textContent ?? "";
      return s === "done" || s.startsWith("ERROR");
    },
    null,
    { timeout: 900000, polling: 2000 },
  );
  const status = await page.evaluate(() => document.querySelector("#status").textContent);
  if (status.startsWith("ERROR")) {
    console.log(`[seq] ERRORE cella ${i + 1}: ${status}`);
    await browser.close();
    process.exit(3);
  }
  console.log(`[seq] cella ${i + 1} done — clock post: ${clocks()}`);
}

const [download] = await Promise.all([page.waitForEvent("download", { timeout: 30000 }), page.click("#export")]);
const out = `/tmp/${download.suggestedFilename()}`;
copyFileSync(await download.path(), out);
console.log(`[seq] EXPORT: ${out}`);
await browser.close();
