// WP banda fredda browser — goal engine-fase-c3c, fase 1 (PHASES riga 1).
//
// Misura IN CHROME la lettura OPFS (FileSystemSyncAccessHandle, worker-only) a
// page cache OS **FREDDA**, su blocchi expert-size (5 325 512 B = expert GLM-4.7
// -Flash q4, stesso numero di tools/cold-read-bench.py) e sequenziali. Chiude il
// buco dichiarato dal tool warm di fase A (tools/harness/opfs-bench.mjs): "il
// cold puro richiederebbe drop_caches tra write e read: fuori portata dal
// browser" — qui l'eviction è fatta DA FUORI (orchestratore Node) sui file di
// backing OPFS del profilo Chrome, con posix_fadvise(DONTNEED)×2 (python3),
// la stessa tecnica del tool OS di C1 ⇒ confronto browser-vs-OS appaiato.
//
// La freddezza NON è asserita: è dimostrata nel JSON dal protocollo —
//   (a) residenza in page cache dei file di backing PRIMA e DOPO ogni drop,
//       misurata con `fincore` (byte residenti);
//   (b) controllo warm sugli stessi offset (il delta cold/warm è la firma).
//
// Protocollo (ogni lettura "cold" è preceduta da un drop verificato):
//   1. write file 6 GiB di dati crypto.getRandomValues (incomprimibili:
//      ~/.cache è btrfs compress=zstd:1 — dati comprimibili gonfierebbero la
//      banda fredda) + flush
//   2. drop → seq COLD blocchi 1 MiB su 4 GiB   (confronto: OS 3.22 GB/s)
//   3.        seq WARM re-read 1 GiB @offset 3 GiB (sanity: stack di misura)
//   4. drop → seq COLD blocchi expert-size su 2 GiB, latenza per blocco
//             (regime streaming-load / instant-on)
//   5. drop → rand COLD 192 × expert-size, offset page-aligned deterministici
//             (LCG seed 12345)                  (confronto: OS 1.63 GB/s, 3.74 ms p50)
//   6.        rand WARM stessi offset            (confronto: OS 10.92 GB/s, 0.46 ms p50)
//
// ATTENZIONE profilo: mai /tmp (tmpfs ⇒ misureresti la RAM). Default:
// ~/.cache/blab-opfs-cold-profile (disco vero). Il tool serve da sé una pagina
// su 127.0.0.1 (secure context): NON richiede il dev server vite.
//
// Uso: node tools/opfs-cold/opfs-cold-bench.mjs
//   env: OPFS_PROFILE, DEVICE_LABEL, FILE_GIB, N_RAND, KEEP_FILE=1 (riusa il
//   file fra run: skip write su size-match, come l'import del motore)
import { chromium } from "playwright";
import { mkdirSync, writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { createServer } from "node:http";
import { homedir, loadavg } from "node:os";

const OUT_DIR = "results/opfs-bench";
const PROFILE = process.env.OPFS_PROFILE ?? `${homedir()}/.cache/blab-opfs-cold-profile`;
const DEVICE_LABEL = process.env.DEVICE_LABEL ?? "4090-linux";
const FILE_GIB = Number(process.env.FILE_GIB ?? 6);
const EXPERT_BYTES = 5325512; // byte/expert GLM-4.7-Flash q4 (residency-sim, = tool OS C1)
const N_RAND = Number(process.env.N_RAND ?? 192); // = tool OS C1
const GIB = 1024 ** 3;

// ---------- worker (gira nella pagina; tiene il SyncAccessHandle aperto) ----------
const workerCode = String.raw`
const MB = 1024 * 1024;
let h = null;
const handlers = {
  async setup({ fileGiB, keep }) {
    const root = await navigator.storage.getDirectory();
    const fh = await root.getFileHandle("cold-bench.bin", { create: true });
    h = await fh.createSyncAccessHandle();
    const size = fileGiB * 1024 * MB;
    if (keep && h.getSize() === size) return { size, skippedWrite: true };
    // tile 8 MiB di dati random veri (incomprimibile per-extent sotto zstd)
    const tile = new Uint8Array(8 * MB);
    for (let o = 0; o < tile.length; o += 65536) crypto.getRandomValues(tile.subarray(o, o + 65536));
    h.truncate(0);
    const t0 = performance.now();
    for (let off = 0; off < size; off += tile.length) h.write(tile, { at: off });
    h.flush();
    return { size, skippedWrite: false, writeMs: performance.now() - t0 };
  },
  seq({ bytes, blockBytes, offset, withLat }) {
    const buf = new Uint8Array(blockBytes);
    const lat = withLat ? [] : null;
    const t0 = performance.now();
    let got = 0, off = offset;
    while (got < bytes) {
      const t1 = withLat ? performance.now() : 0;
      const n = h.read(buf, { at: off });
      if (withLat) lat.push(performance.now() - t1);
      if (n === 0) break;
      got += n; off += n;
    }
    return { bytes: got, ms: performance.now() - t0, lat };
  },
  randExpert({ n, expertBytes, seed }) {
    const size = h.getSize();
    const buf = new Uint8Array(expertBytes);
    let s = seed >>> 0;
    const rnd = () => (s = (s * 1103515245 + 12345) >>> 0) / 2 ** 32; // LCG deterministico
    const offs = [];
    for (let i = 0; i < n; i++) offs.push(4096 * Math.floor((rnd() * (size - expertBytes)) / 4096));
    const lat = [];
    const t0 = performance.now();
    let got = 0;
    for (const off of offs) {
      const t1 = performance.now();
      got += h.read(buf, { at: off });
      lat.push(performance.now() - t1);
    }
    return { bytes: got, ms: performance.now() - t0, lat };
  },
  close() { h.flush(); h.close(); h = null; return {}; },
};
self.onmessage = async (e) => {
  const { id, cmd, params } = e.data;
  try { self.postMessage({ id, ok: true, res: await handlers[cmd](params) }); }
  catch (err) { self.postMessage({ id, ok: false, err: String(err && err.stack || err) }); }
};
`;

// ---------- eviction + prova di residenza (lato OS, fuori dal browser) ----------
const PY_EVICT = `
import os, sys, time
for p in sys.argv[1:]:
    fd = os.open(p, os.O_RDONLY)
    os.fsync(fd)  # le pagine dirty non si droppano: prima si scrivono
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    time.sleep(1)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
`;

function findBackingFiles() {
  // il backing store OPFS di Chrome: cerchiamo per taglia, non per path hardcoded
  const out = execFileSync("find", [PROFILE, "-type", "f", "-size", "+512M"], { encoding: "utf8" });
  return out.split("\n").filter(Boolean);
}
function residentBytes(files) {
  let tot = 0;
  for (const f of files) {
    const out = execFileSync("fincore", ["--bytes", "--noheadings", "--output", "RES", f], { encoding: "utf8" });
    tot += Number(out.trim());
  }
  return tot;
}
const evictionLog = [];
function evict(files, phase) {
  const before = residentBytes(files);
  execFileSync("python3", ["-c", PY_EVICT, ...files]);
  const after = residentBytes(files);
  evictionLog.push({ phase, residentBytesBefore: before, residentBytesAfter: after });
  console.log(`[evict] ${phase}: resident ${(before / 1e9).toFixed(2)} GB -> ${(after / 1e6).toFixed(1)} MB`);
  if (after > 0.01 * before && after > 64 * 1024 * 1024) {
    throw new Error(`eviction NON efficace (${after} B residenti): freddezza non dimostrabile`);
  }
  return { before, after };
}

// ---------- pagina self-served (secure context senza dev server) ----------
const server = createServer((_, res) => {
  res.writeHead(200, { "content-type": "text/html" });
  res.end("<!doctype html><title>opfs-cold</title>");
});
// porta FISSA: l'origin determina il bucket OPFS — con porta effimera ogni run
// creerebbe un nuovo bucket da FILE_GIB nel profilo e KEEP_FILE non funzionerebbe
const PORT = Number(process.env.OPFS_PORT ?? 5327);
await new Promise((r, j) => { server.on("error", j); server.listen(PORT, "127.0.0.1", r); });
const BASE_URL = `http://127.0.0.1:${PORT}/`;

const browser = await chromium.launchPersistentContext(PROFILE, { headless: false, channel: "chrome" });
try {
  const page = browser.pages()[0] ?? (await browser.newPage());
  await page.goto(BASE_URL, { waitUntil: "load" });
  await page.evaluate((code) => {
    const w = new Worker(URL.createObjectURL(new Blob([code], { type: "text/javascript" })));
    const pending = new Map();
    let nextId = 1;
    w.onmessage = (e) => {
      const { id, ok, res, err } = e.data;
      const p = pending.get(id); pending.delete(id);
      ok ? p.resolve(res) : p.reject(new Error(err));
    };
    w.onerror = (e) => { for (const p of pending.values()) p.reject(new Error(e.message)); pending.clear(); };
    window.__opfs = (cmd, params) => new Promise((resolve, reject) => {
      const id = nextId++; pending.set(id, { resolve, reject }); w.postMessage({ id, cmd, params });
    });
  }, workerCode);
  const cmd = (c, params = {}) => page.evaluate(([c, params]) => window.__opfs(c, params), [c, params]);

  // 1. write (o riuso su size-match con KEEP_FILE=1)
  const setup = await cmd("setup", { fileGiB: FILE_GIB, keep: process.env.KEEP_FILE === "1" });
  console.log(`[opfs-cold] file ${FILE_GIB} GiB ${setup.skippedWrite ? "riusato" : `scritto in ${(setup.writeMs / 1000).toFixed(1)} s (${(setup.size / setup.writeMs / 1e6).toFixed(2)} GB/s)`}`);
  const backing = findBackingFiles();
  if (backing.length === 0) throw new Error(`nessun file di backing >512M trovato in ${PROFILE}: impossibile provare la freddezza`);
  console.log(`[opfs-cold] backing: ${backing.join(", ")}`);

  const stats = (lat) => {
    const s = [...lat].sort((a, b) => a - b);
    return { p50: s[Math.floor(s.length * 0.5)], p95: s[Math.floor(s.length * 0.95)], n: s.length };
  };
  const gbps = (r) => r.bytes / r.ms / 1e6;

  // 2-3. sequenziale 1 MiB cold + warm re-read
  evict(backing, "pre-seq-1MiB");
  const seqCold = await cmd("seq", { bytes: 4 * GIB, blockBytes: 1024 * 1024, offset: 0, withLat: false });
  console.log(`[opfs-cold] seq COLD 1MiB: ${gbps(seqCold).toFixed(2)} GB/s (${(seqCold.bytes / GIB).toFixed(1)} GiB in ${(seqCold.ms / 1000).toFixed(1)} s)`);
  const seqWarm = await cmd("seq", { bytes: 1 * GIB, blockBytes: 1024 * 1024, offset: 3 * GIB, withLat: false });
  console.log(`[opfs-cold] seq WARM re-read: ${gbps(seqWarm).toFixed(2)} GB/s`);

  // 4. sequenziale expert-size cold con latenze (regime streaming-load)
  evict(backing, "pre-seq-expert");
  const seqExp = await cmd("seq", { bytes: 2 * GIB, blockBytes: EXPERT_BYTES, offset: 0, withLat: true });
  const seqExpStats = stats(seqExp.lat);
  console.log(`[opfs-cold] seq COLD expert-size: ${gbps(seqExp).toFixed(2)} GB/s | p50 ${seqExpStats.p50.toFixed(2)} p95 ${seqExpStats.p95.toFixed(2)} ms/expert (n=${seqExpStats.n})`);

  // 5-6. random expert-size cold + warm (stessi offset: stesso seed)
  evict(backing, "pre-rand-expert");
  const randCold = await cmd("randExpert", { n: N_RAND, expertBytes: EXPERT_BYTES, seed: 12345 });
  const randColdStats = stats(randCold.lat);
  console.log(`[opfs-cold] rand COLD expert: ${gbps(randCold).toFixed(2)} GB/s | p50 ${randColdStats.p50.toFixed(2)} p95 ${randColdStats.p95.toFixed(2)} ms/expert (n=${N_RAND})`);
  const randWarm = await cmd("randExpert", { n: N_RAND, expertBytes: EXPERT_BYTES, seed: 12345 });
  const randWarmStats = stats(randWarm.lat);
  console.log(`[opfs-cold] rand WARM expert (stessi offset): ${gbps(randWarm).toFixed(2)} GB/s | p50 ${randWarmStats.p50.toFixed(2)} ms/expert`);

  // il file resta nell'OPFS del profilo dedicato: con KEEP_FILE=1 il prossimo
  // run salta il write (size-match); il profilo si butta intero per pulire
  await cmd("close");

  // ---------- record ----------
  const os = { // results/opfs-bench/cold-read-os-4090-linux-2026-07-31.json (C1 item 4)
    ref: "results/opfs-bench/cold-read-os-4090-linux-2026-07-31.json",
    seqCold1MiBGBps: 3.22,
    randExpertColdGBps: 1.63,
    randExpertColdLatencyMsP50: 3.74,
    randExpertColdLatencyMsP95: 4.34,
    randExpertWarmGBps: 10.92,
    randExpertWarmLatencyMsP50: 0.46,
  };
  const browserNums = {
    seqCold1MiBGBps: +gbps(seqCold).toFixed(2),
    seqWarmGBps: +gbps(seqWarm).toFixed(2),
    seqColdExpert: { gBps: +gbps(seqExp).toFixed(2), latencyMsP50: +seqExpStats.p50.toFixed(2), latencyMsP95: +seqExpStats.p95.toFixed(2), n: seqExpStats.n, bytesRead: seqExp.bytes },
    randExpertCold: { gBps: +gbps(randCold).toFixed(2), latencyMsP50: +randColdStats.p50.toFixed(2), latencyMsP95: +randColdStats.p95.toFixed(2), n: N_RAND },
    randExpertWarm: { gBps: +gbps(randWarm).toFixed(2), latencyMsP50: +randWarmStats.p50.toFixed(2), latencyMsP95: +randWarmStats.p95.toFixed(2), n: N_RAND },
    expertBytes: EXPERT_BYTES,
  };
  const record = {
    schemaVersion: 1,
    kind: "opfs-cold-bench",
    deviceLabel: DEVICE_LABEL,
    ts: new Date().toISOString(),
    tool: "tools/opfs-cold/opfs-cold-bench.mjs",
    protocol: {
      fileGiB: FILE_GIB,
      data: "crypto.getRandomValues per-tile (incomprimibile: profilo su btrfs compress=zstd:1)",
      profileDir: PROFILE,
      backingFiles: backing,
      eviction: "fsync + posix_fadvise(POSIX_FADV_DONTNEED) x2 con sleep 1 s (python3, da fuori il browser) sui file di backing OPFS; identica al tool OS di C1 (tools/cold-read-bench.py) => confronto appaiato",
      evictionEvidence: evictionLog,
      coldnessProof: "ogni misura COLD e' preceduta da un drop con residenza fincore registrata prima/dopo (evictionEvidence); controprova: delta cold/warm sugli STESSI offset in randExpert{Cold,Warm}",
      hostLoadavg: loadavg().map((x) => +x.toFixed(2)),
    },
    results: browserNums,
    comparisonVsOsC1: {
      ref: os.ref,
      os,
      browser: {
        seqCold1MiBGBps: browserNums.seqCold1MiBGBps,
        randExpertColdGBps: browserNums.randExpertCold.gBps,
        randExpertColdLatencyMsP50: browserNums.randExpertCold.latencyMsP50,
        randExpertColdLatencyMsP95: browserNums.randExpertCold.latencyMsP95,
        randExpertWarmGBps: browserNums.randExpertWarm.gBps,
        randExpertWarmLatencyMsP50: browserNums.randExpertWarm.latencyMsP50,
      },
      ratioBrowserOverOs: {
        seqCold1MiB: +(browserNums.seqCold1MiBGBps / os.seqCold1MiBGBps).toFixed(3),
        randExpertCold: +(browserNums.randExpertCold.gBps / os.randExpertColdGBps).toFixed(3),
        randExpertColdLatencyP50: +(browserNums.randExpertCold.latencyMsP50 / os.randExpertColdLatencyMsP50).toFixed(3),
      },
    },
    caveats: [
      "fadvise DONTNEED e' best-effort: validato da fincore (evictionEvidence) + delta cold/warm",
      "layout extent del backing store Chrome puo' differire dal file scritto dal tool OS (secondo ordine)",
      "SyncAccessHandle.read passa dallo storage service di Chrome: il numero include l'overhead browser reale - e' IL numero che serve al modello di banda (fase 6), non un bound",
    ],
  };
  mkdirSync(OUT_DIR, { recursive: true });
  const file = `${OUT_DIR}/opfs-cold-${DEVICE_LABEL}-${record.ts.replace(/[:.]/g, "-")}.json`;
  writeFileSync(file, JSON.stringify(record, null, 2));
  console.log("[opfs-cold] scritto", file);
} finally {
  await browser.close();
  server.close();
}
