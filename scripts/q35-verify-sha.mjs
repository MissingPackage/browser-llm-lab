// Verifica SHA dei GGUF pinnati q1 (fase 2, done-when "SHA verificate, exit 0").
// Manifest: scripts/q35-manifest.json (SHA = oid LFS dei pointer HF, spec §1).
// Gate secco: byte E sha256 devono coincidere per TUTTI i file, altrimenti exit 1.
// Uso: node scripts/q35-verify-sha.mjs [--models-dir ~/.cache/blab-models]
import { createHash } from "node:crypto";
import { createReadStream, statSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const argDir = process.argv.indexOf("--models-dir");
const modelsDir = argDir > -1 ? process.argv[argDir + 1] : join(homedir(), ".cache", "blab-models");
const manifest = JSON.parse(readFileSync(join(dirname(fileURLToPath(import.meta.url)), "q35-manifest.json"), "utf8"));

async function sha256File(path) {
  const h = createHash("sha256");
  await new Promise((res, rej) => {
    createReadStream(path, { highWaterMark: 1 << 22 })
      .on("data", (c) => h.update(c))
      .on("end", res)
      .on("error", rej);
  });
  return h.digest("hex");
}

let fail = 0;
for (const f of manifest.files) {
  const path = join(modelsDir, f.path);
  let size;
  try {
    size = statSync(path).size;
  } catch {
    console.error(`FAIL ${f.model}: file assente (${path})`);
    fail++;
    continue;
  }
  if (size !== f.bytes) {
    console.error(`FAIL ${f.model}: byte ${size} != ${f.bytes} attesi`);
    fail++;
    continue;
  }
  const t0 = performance.now();
  const sha = await sha256File(path);
  const s = ((performance.now() - t0) / 1000).toFixed(1);
  if (sha !== f.sha256) {
    console.error(`FAIL ${f.model}: sha256 ${sha} != ${f.sha256} attesa`);
    fail++;
  } else {
    console.log(`PASS ${f.model}: ${f.bytes} B, sha256 ok (${s}s)`);
  }
}
process.exit(fail === 0 ? 0 : 1);
