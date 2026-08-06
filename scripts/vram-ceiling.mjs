// Probe del TETTO VRAM ALLOCABILE reale da Chrome/Dawn (goal C3a, fase 4c
// slice A', emendamento 5). Nasce perche' il bilancio del design 4c usava
// "usabile = 16376 MiB - 763 di desktop" — un numero ARITMETICO mai
// verificato contro i 429 MiB di memory.reserved del driver. La regola
// pre-dichiarata dell'emendamento 5 decide sul numero che QUESTO probe
// misura: tetto >= parco+non-expert+KV(525)+riserva => residenza totale
// Q4_0 puro; gap => docket, mai degradazione.
//
// Metodo: alloca storage buffer da --chunk-mib finche' Dawn non da' OOM
// (error scope 'out-of-memory'), poi raffina a --fine-mib; per ogni buffer
// misura la banda di lettura (copy verso uno scratch pre-allocato) per
// distinguere VRAM residente (~200-400 GB/s) da spill host/PCIe (~10-25):
// il verdetto usa SOLO i byte residenti. La curva nvidia-smi (used/reserved)
// e' campionata dal lato node a ogni chunk.
//
// Uso:
//   node scripts/vram-ceiling.mjs --label full|minimal|headless [--headless]
//     [--chunk-mib 256] [--fine-mib 32] [--out results/engine/...json]
// Exit: 0 report scritto, 2 errore infrastruttura (niente adapter/device).
import { chromium } from "playwright";
import { execFileSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const arg = (name, dflt) => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : dflt;
};
const has = (name) => process.argv.includes(`--${name}`);

const LABEL = arg("label", "full");
const CHUNK = Number(arg("chunk-mib", "256")) * 2 ** 20;
const FINE = Number(arg("fine-mib", "32")) * 2 ** 20;
const HEADLESS = has("headless");
const ROOT = new URL("..", import.meta.url).pathname;
const BASE_URL = process.env.BASE_URL ?? "http://localhost:5199";
const PROFILE = process.env.VRAM_PROBE_PROFILE ?? join(homedir(), ".cache/blab-vram-probe-profile");
const STAMP = new Date().toISOString().slice(0, 10);
const OUT = arg("out", join(ROOT, `results/engine/vram-ceiling-${LABEL}-${STAMP}.json`));

// Fabbisogno della residenza totale a ctx 525 (design 4c §2.1, ricalcolato):
// il verdetto della regola pre-dichiarata si prende contro QUESTO numero.
const PARK_B = 15_678_308_352;      // 2944 expert, layout motore
const NON_EXPERT_B = 1_354_078_720; // attention+denso+shexp+router+head+norm
const KV_PER_TOKEN_B = 108_288;
const CTX_GATE = 525;
const RESERVE_B = 64 * 2 ** 20;     // VRAM_RESERVE dichiarata dal design
const REQUIRED_B = PARK_B + NON_EXPERT_B + CTX_GATE * KV_PER_TOKEN_B + RESERVE_B;
const SLAB_Q4_0 = 5_308_416;

// Banda sotto cui un buffer si considera SPILL (host/PCIe): PCIe4 vale
// ~12-25 GB/s, la VRAM di questo device ~300+. 60 separa con margine.
const RESIDENT_GBPS_MIN = 60;

const smi = (fields) =>
  execFileSync("nvidia-smi", [`--query-gpu=${fields}`, "--format=csv,noheader,nounits"], { encoding: "utf8" }).trim();
const smiProcs = () => {
  try {
    return execFileSync("nvidia-smi", ["--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], { encoding: "utf8" }).trim();
  } catch {
    return "";
  }
};
const vramUsedMiB = () => {
  const [used, reserved] = smi("memory.used,memory.reserved").split(",").map((s) => Number(s.trim()));
  return { used, reserved };
};
const hostSnapshot = () => ({
  gpu: smi("memory.total,memory.used,memory.reserved,temperature.gpu,clocks.sm,power.draw"),
  procs: execFileSync("nvidia-smi", [], { encoding: "utf8" }).split("\n").filter((l) => /MiB \|$/.test(l)).map((l) => l.trim()),
  computeApps: smiProcs(),
});

const before = hostSnapshot();
console.log(`[vram] regime=${LABEL} headless=${HEADLESS} chunk=${CHUNK / 2 ** 20}MiB fine=${FINE / 2 ** 20}MiB`);
console.log(`[vram] host prima: ${before.gpu}`);

mkdirSync(PROFILE, { recursive: true });
const args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPUService", "--ignore-gpu-blocklist", "--no-first-run", "--no-default-browser-check"];
const browser = await chromium.launchPersistentContext(PROFILE, {
  headless: HEADLESS,
  channel: "chrome",
  args: HEADLESS ? [...args, "--headless=new", "--use-angle=vulkan", "--enable-gpu", "--disable-vulkan-surface"] : args,
});
const page = browser.pages()[0] ?? (await browser.newPage());
page.on("pageerror", (e) => console.log("[vram][pageerror]", e.message.slice(0, 300)));
await page.goto(`${BASE_URL}/vramprobe.html`, { waitUntil: "load" });

// Setup: device coi limiti di buffer negoziati al massimo dell'adapter +
// scratch da 256 MiB PRE-allocato (a tetto raggiunto non si potrebbe piu'
// creare: va preso prima di riempire).
const setup = await page.evaluate(async () => {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) return { error: "niente adapter WebGPU" };
  const info = adapter.info ?? {};
  // Guardia: su un adapter software (SwiftShader/llvmpipe) il probe
  // "allocherebbe" RAM host spacciandola per VRAM — successo silenzioso,
  // numero senza senso. Rifiuto esplicito (postura R6).
  const desc = `${info.vendor ?? ""} ${info.architecture ?? ""} ${info.description ?? ""}`.toLowerCase();
  if (/swiftshader|llvmpipe|software/.test(desc)) return { error: `adapter software: ${desc}` };
  const want = {
    maxBufferSize: Number(adapter.limits.maxBufferSize),
    maxStorageBufferBindingSize: Number(adapter.limits.maxStorageBufferBindingSize),
  };
  let device;
  try {
    device = await adapter.requestDevice({ requiredLimits: want });
  } catch (e) {
    return { error: `requestDevice: ${e}` };
  }
  const g = (globalThis.__vramProbe = { device, buffers: [], lost: null, uncaptured: [] });
  device.lost.then((info) => { g.lost = { reason: info.reason, message: info.message }; });
  device.addEventListener("uncapturederror", (ev) => g.uncaptured.push(String(ev.error?.message ?? ev.error)));
  device.pushErrorScope("out-of-memory");
  g.scratch = device.createBuffer({ size: 256 * 2 ** 20, usage: GPUBufferUsage.COPY_DST });
  const err = await device.popErrorScope();
  if (err) return { error: `scratch OOM: ${err.message}` };
  return {
    limits: want,
    scratchBytes: 256 * 2 ** 20,
    adapterInfo: { vendor: info.vendor, architecture: info.architecture, description: info.description },
  };
});
if (setup.error) {
  console.error(`[vram] ${setup.error}`);
  await browser.close();
  process.exit(2);
}

// Un chunk: alloca sotto error scope; l'OOM e' l'esito atteso al tetto,
// non un errore del probe.
const allocChunk = (bytes) =>
  page.evaluate(async (size) => {
    const g = globalThis.__vramProbe;
    if (g.lost) return { ok: false, lost: g.lost };
    g.device.pushErrorScope("out-of-memory");
    const buf = g.device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const err = await g.device.popErrorScope();
    if (err) { try { buf.destroy(); } catch {} return { ok: false, oom: err.message }; }
    g.buffers.push(buf);
    return { ok: true };
  }, bytes);

const timeline = [];
let allocated = setup.scratchBytes;
let oomMessage = null, lost = null;

const fill = async (chunkBytes, phase) => {
  for (;;) {
    const r = await allocChunk(chunkBytes);
    if (r.lost) { lost = r.lost; return false; }
    if (!r.ok) { oomMessage = r.oom ?? null; return true; }
    allocated += chunkBytes;
    const v = vramUsedMiB();
    timeline.push({ phase, allocatedMiB: Math.round(allocated / 2 ** 20), smiUsedMiB: v.used, smiReservedMiB: v.reserved });
    if (timeline.length % 8 === 0) console.log(`[vram] ${phase}: allocati ${(allocated / 2 ** 30).toFixed(2)} GiB — smi used ${v.used} MiB`);
  }
};

let cont = await fill(CHUNK, "coarse");
if (cont && !lost) cont = await fill(FINE, "fine");

// Dopo il primo OOM duro: quanto pool HOST-BACKED (sysmem, GPU-indirizzabile)
// il driver concede davvero? La coda del regime full e' arrivata a ~200 MiB
// a 13 GB/s prima dell'OOM — se il pool fosse ~700 MiB, la coda fredda del
// parco ci starebbe SENZA degradazione e senza CPU (niente sync). Si insiste
// con chunk fine e brevi pause: l'esito (concesso/negato) e' il dato.
let overflowB = 0;
if (!lost && oomMessage) {
  for (let round = 0; round < 24 && !lost; round++) {
    await new Promise((r) => setTimeout(r, 250));
    const r = await allocChunk(FINE);
    if (r.lost) { lost = r.lost; break; }
    if (!r.ok) break;
    allocated += FINE;
    overflowB += FINE;
    const v = vramUsedMiB();
    timeline.push({ phase: "overflow", allocatedMiB: Math.round(allocated / 2 ** 20), smiUsedMiB: v.used, smiReservedMiB: v.reserved });
  }
  if (overflowB) console.log(`[vram] overflow post-OOM concesso: ${(overflowB / 2 ** 20).toFixed(0)} MiB`);
}

// Stabilita' del tier host-backed: 5 s di attesa (finestra per un'eventuale
// migrazione del driver), poi la banda si misura due volte per buffer. Un
// tier che il driver rimescola in silenzio non e' un tier utilizzabile.
if (!lost) await new Promise((r) => setTimeout(r, 5000));

// Banda di lettura per buffer: copy 256 MiB buffer->scratch, wall su
// onSubmittedWorkDone. Discrimina residente vs spill; il primo giro puo'
// pagare la residency migration, quindi due giri e si tiene il migliore.
const bw = lost ? [] : await page.evaluate(async () => {
  const g = globalThis.__vramProbe;
  const out = [];
  for (let i = 0; i < g.buffers.length; i++) {
    const buf = g.buffers[i];
    const n = Math.min(buf.size, 256 * 2 ** 20);
    let best = 0;
    for (let rep = 0; rep < 2; rep++) {
      const enc = g.device.createCommandEncoder();
      enc.copyBufferToBuffer(buf, 0, g.scratch, 0, n);
      const t0 = performance.now();
      g.device.queue.submit([enc.finish()]);
      await g.device.queue.onSubmittedWorkDone();
      const dt = (performance.now() - t0) / 1000;
      best = Math.max(best, n / dt / 1e9);
    }
    out.push({ i, sizeMiB: Math.round(buf.size / 2 ** 20), gbps: Number(best.toFixed(1)) });
  }
  return out;
});
const lostAfter = await page.evaluate(() => globalThis.__vramProbe.lost);
const uncaptured = await page.evaluate(() => globalThis.__vramProbe.uncaptured);
await browser.close();
const after = hostSnapshot();

const residentB = bw.filter((b) => b.gbps >= RESIDENT_GBPS_MIN).reduce((s, b) => s + b.sizeMiB * 2 ** 20, 0) + setup.scratchBytes;
const spilledB = bw.filter((b) => b.gbps < RESIDENT_GBPS_MIN).reduce((s, b) => s + b.sizeMiB * 2 ** 20, 0);
const gapB = REQUIRED_B - residentB;

const report = {
  schema: "vram-ceiling-v1",
  date: new Date().toISOString(),
  label: LABEL,
  headlessChrome: HEADLESS,
  chunkBytes: CHUNK,
  fineBytes: FINE,
  adapterInfo: setup.adapterInfo,
  deviceLimits: setup.limits,
  hostBefore: before,
  hostAfter: after,
  allocatedBytes: allocated,
  residentBytes: residentB,
  spilledBytes: spilledB,
  overflowPostOomBytes: overflowB,
  residentGbpsMin: RESIDENT_GBPS_MIN,
  oomMessage,
  deviceLost: lost ?? lostAfter ?? null,
  uncapturedErrors: uncaptured,
  bandwidthPerBuffer: bw,
  timeline,
  required: {
    parkBytes: PARK_B,
    nonExpertBytes: NON_EXPERT_B,
    kvBytes: CTX_GATE * KV_PER_TOKEN_B,
    reserveBytes: RESERVE_B,
    totalBytes: REQUIRED_B,
    ctx: CTX_GATE,
  },
  verdict: {
    covers: gapB <= 0,
    gapBytes: Math.max(0, gapB),
    gapMiB: Math.max(0, Math.round(gapB / 2 ** 20)),
    gapSlabsQ4_0: Math.max(0, Math.ceil(gapB / SLAB_Q4_0)),
    rule: "emendamento 5: copre => residenza totale Q4_0 puro; gap => docket, mai degradazione",
  },
};
mkdirSync(join(ROOT, "results/engine"), { recursive: true });
writeFileSync(OUT, JSON.stringify(report, null, 2));
console.log(`[vram] regime=${LABEL}: allocati ${(allocated / 2 ** 30).toFixed(3)} GiB, residenti ${(residentB / 2 ** 30).toFixed(3)} GiB, spill ${(spilledB / 2 ** 30).toFixed(3)} GiB`);
console.log(`[vram] fabbisogno ${(REQUIRED_B / 2 ** 30).toFixed(3)} GiB => ${report.verdict.covers ? "COPRE" : `GAP ${report.verdict.gapMiB} MiB (${report.verdict.gapSlabsQ4_0} slab q4_0)`}`);
console.log(`[vram] report: ${OUT}`);
