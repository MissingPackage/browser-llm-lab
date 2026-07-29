// Banda OPFS in lettura/scrittura — fase 2 del goal engine-fase-a (direction §9.2).
//
// Worker inline con FileSystemSyncAccessHandle (OPFS sync è worker-only). Blocchi
// scelti sulle taglie che contano per il motore: 64 KB (metadata), 1 MB, 5.3 MB
// (expert GLM-4.7-Flash q4), 19 MB (slab colibri). Più read random a taglia expert.
//
// Limite dichiarato: il file di test viene scritto subito prima delle letture, quindi
// le read sono con page cache OS calda — è il regime "expert riletto durante il decode"
// (upper bound), NON il cold-load del primo avvio. Il cold puro richiederebbe
// drop_caches tra write e read: fuori portata dal browser.
//
// ATTENZIONE profilo: mai /tmp (tmpfs su Fedora ⇒ misureresti la RAM). Default:
// ~/.cache/blab-opfs-profile (disco vero).
//
// Uso (dev server attivo): node .harness/tools/opfs-bench.mjs
import { chromium } from "playwright";
import { mkdirSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";

const OUT_DIR = "results/opfs-bench";
const PROFILE = process.env.OPFS_PROFILE ?? `${homedir()}/.cache/blab-opfs-profile`;
const BASE_URL = process.env.BASE_URL ?? "http://localhost:5173";
const DEVICE_LABEL = process.env.DEVICE_LABEL ?? "4090-linux";
const FILE_MB = Number(process.env.FILE_MB ?? 512);

const workerCode = String.raw`
const MB = 1024 * 1024;
self.onmessage = async (e) => {
  const { fileMB } = e.data;
  const root = await navigator.storage.getDirectory();
  const fh = await root.getFileHandle("bench.bin", { create: true });
  const h = await fh.createSyncAccessHandle();
  const chunk = new Uint8Array(8 * MB);
  for (let i = 0; i < chunk.length; i += 4096) chunk[i] = (i >> 12) & 0xff; // anti-compressione
  h.truncate(0);
  let t0 = performance.now();
  for (let off = 0; off < fileMB * MB; off += chunk.length) h.write(chunk, { at: off });
  h.flush();
  const writeMs = performance.now() - t0;
  const size = h.getSize();
  const results = [];
  for (const blkBytes of [64 * 1024, 1 * MB, Math.round(5.3 * MB), 19 * MB]) {
    const buf = new Uint8Array(blkBytes);
    t0 = performance.now();
    let bytes = 0;
    for (let off = 0; off + blkBytes <= size; off += blkBytes) bytes += h.read(buf, { at: off });
    const ms = performance.now() - t0;
    results.push({ mode: "seq", blockBytes: blkBytes, bytes, ms, gBps: bytes / ms / 1e6 });
  }
  {
    // random a taglia expert; offset deterministici (LCG), riproducibili
    const blkBytes = Math.round(5.3 * MB);
    const buf = new Uint8Array(blkBytes);
    const n = 48;
    let seed = 12345;
    const rnd = () => (seed = (seed * 1103515245 + 12345) >>> 0) / 2 ** 32;
    t0 = performance.now();
    let bytes = 0;
    for (let i = 0; i < n; i++) {
      const off = 4096 * Math.floor((rnd() * (size - blkBytes)) / 4096);
      bytes += h.read(buf, { at: off });
    }
    const ms = performance.now() - t0;
    results.push({ mode: "rand", blockBytes: blkBytes, n, bytes, ms, gBps: bytes / ms / 1e6 });
  }
  h.close();
  await root.removeEntry("bench.bin");
  self.postMessage({ writeMs, writeGBps: size / writeMs / 1e6, size, results });
};
`;

const browser = await chromium.launchPersistentContext(PROFILE, { headless: false, channel: "chrome" });
const page = browser.pages()[0] ?? (await browser.newPage());
await page.goto(BASE_URL, { waitUntil: "load" });

const out = await page.evaluate(
  ({ code, fileMB }) =>
    new Promise((resolve, reject) => {
      const w = new Worker(URL.createObjectURL(new Blob([code], { type: "text/javascript" })));
      w.onmessage = (e) => resolve(e.data);
      w.onerror = (e) => reject(new Error(e.message));
      w.postMessage({ fileMB });
      setTimeout(() => reject(new Error("timeout 300s")), 300000);
    }),
  { code: workerCode, fileMB: FILE_MB },
);

console.log(`[opfs] write: ${out.writeGBps.toFixed(2)} GB/s (${FILE_MB} MB)`);
for (const r of out.results) {
  console.log(`[opfs] read ${r.mode} ${(r.blockBytes / 1048576).toFixed(2)} MB: ${r.gBps.toFixed(2)} GB/s`);
}

mkdirSync(OUT_DIR, { recursive: true });
const record = {
  schemaVersion: 1,
  kind: "opfs-bench",
  deviceLabel: DEVICE_LABEL,
  ts: new Date().toISOString(),
  fileMB: FILE_MB,
  cacheState: "warm-page-cache", // vedi header: upper bound dichiarato
  profileDir: PROFILE,
  ...out,
};
const file = `${OUT_DIR}/opfs-bench-${DEVICE_LABEL}-${record.ts.replace(/[:.]/g, "-")}.json`;
writeFileSync(file, JSON.stringify(record, null, 2));
console.log("[opfs] scritto", file);
await browser.close();
