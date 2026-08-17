// OVERLAP DEL ROUTER FRA POSIZIONI ADIACENTI — spike (2) di engine-velocita-decode.
//
// LA DOMANDA. Lo spec-dec verifica M token in una passata. Sul segmento expert
// di un MoE le M righe si sparpagliano su expert diversi, quindi ciascun expert
// ne vede POCHE anche con M grande. Con top-8 su 256, se il routing di posizioni
// adiacenti fosse INDIPENDENTE l'atteso sarebbe 8·8/256 = 0,25 expert condivisi:
// a M=2 si avrebbero ~15,75 expert distinti per 16 selezioni, cioe' ~1,02 righe
// per expert e AMMORTAMENTO ZERO. La curva cost(M) dello spike (1) non si
// applicherebbe affatto al segmento che pesa di piu'.
//
// IL PRIOR CHE NON VALE, e perche' questo script esiste. Avevo citato il recall
// 82,67% di `q35-looka-run.mjs` come evidenza di correlazione: E' IL NUMERO
// SBAGLIATO. Quello script predice il router del layer l dall'hidden PRE-ATTENTION
// DELLA STESSA POSIZIONE — misura quanto l'attention sposta il routing DENTRO un
// token, e non dice nulla sull'overlap FRA token. Il prior giusto e'
// `baseline_prev` di `tools/oracle-moe/trace.cpp`, che pero' e' stato misurato
// solo su GLM (64 expert, top-4), mai sul 35B.
//
// LA METRICA PRIMARIA NON E' L'OVERLAP MEDIO: e' D(M), il numero di expert
// DISTINTI nell'unione di una finestra di M posizioni, misurato direttamente su
// finestre scorrevoli. L'overlap a coppie ov(d) e' il diagnostico del decadimento
// ma NON ricostruisce D(M) (l'inclusione-esclusione vorrebbe i termini di ordine
// superiore). Coi set loggati l'unione e' gratis.
//
// IL CONFONDIMENTO DEL CONTENUTO, e come si separa. Un testo ripetitivo avra'
// overlap piu' alto: si misurerebbe il corpus invece del modello. Il controllo e'
// la BASELINE A LUNGA DISTANZA nello stesso prompt (coppie a d >= 64): quella e'
// la componente stazionaria/topica, e l'ECCESSO ov(d) - ov(lungo) a d piccolo e'
// la correlazione locale del modello.
//
// SUBSET DICHIARATO di prompt INTERI (regola full-corpus del progetto:
// esplorazione = subset di prompt interi, MAI un cap sulle posizioni). Default p7
// (08-prosa-en, 269 tok) come kill-check; `--prompts 7,4` per la misura.
//
// Uso: npx vite-node scripts/q35-router-overlap-run.mjs [--prompts 7] [--out PATH]
import { fstatSync, openSync, readFileSync, readSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const arg = (n, d) => { const i = process.argv.indexOf(`--${n}`); return i >= 0 ? process.argv[i + 1] : d; };
const PROMPTS = arg("prompts", "7").split(",").map(Number);
const OUT = arg("out", `results/engine/q35-router-overlap-35b-${new Date().toISOString().slice(0, 10)}.json`);

const { Q35CpuRefModel, q35MoeFfnRefF64 } = await import("../src/engine/q35cpurefmodel.ts");
const MODEL = join(homedir(), ".cache/blab-models/q35/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf");
const fd = openSync(MODEL, "r");
const src = {
  size: fstatSync(fd).size,
  slice(off, len) {
    const out = new Uint8Array(len);
    let g = 0;
    while (g < len) { const n = readSync(fd, out, g, len - g, off + g); if (n <= 0) throw new Error("short read"); g += n; }
    return out;
  },
};
const m = new Q35CpuRefModel(src);
const S = m.shape;
const golden = JSON.parse(readFileSync("results/engine/golden/q35/golden-q35-35b-full-2026-08-10.json", "utf8"));
const rms = (x, w, eps) => {
  let ss = 0;
  for (let i = 0; i < x.length; i++) ss += x[i] * x[i];
  const sc = 1 / Math.sqrt(ss / x.length + eps);
  const o = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) o[i] = x[i] * sc * w[i];
  return o;
};

/** sel[prompt][layer][pos] = Uint8Array(8) — i SET GREZZI, non le metriche.
 *  Le metriche si ricalcolano, i set no: e' il motivo per cui si loggano questi. */
const sel = {};
const t0 = performance.now();
for (const pi of PROMPTS) {
  const tokens = golden.prompts[pi].promptTokens;
  const perLayer = Array.from({ length: S.nLayer }, () => []);
  let hidden = tokens.map((tk) => m.embedRow(tk));
  for (let l = 0; l < S.nLayer; l++) {
    const b = `blk.${l}.`;
    const post = m.dequant(`${b}post_attention_norm.weight`);
    const router = m.dequant(`${b}ffn_gate_inp.weight`);
    const attnOut = m.attnLayerRef(l, hidden);
    const cache = new Map();
    const moeW = {
      router,
      sharedGate: m.dequant(`${b}ffn_gate_inp_shexp.weight`),
      shGate: m.dequant(`${b}ffn_gate_shexp.weight`),
      shUp: m.dequant(`${b}ffn_up_shexp.weight`),
      shDown: m.dequant(`${b}ffn_down_shexp.weight`),
      expert: (e) => {
        let h = cache.get(e);
        if (!h) {
          h = { gate: m.dequantExpert(`${b}ffn_gate_exps.weight`, e), up: m.dequantExpert(`${b}ffn_up_exps.weight`, e), down: m.dequantExpert(`${b}ffn_down_exps.weight`, e) };
          cache.set(e, h);
        }
        return h;
      },
    };
    const next = [];
    for (let t = 0; t < tokens.length; t++) {
      const after = new Float64Array(S.dModel);
      for (let i = 0; i < S.dModel; i++) after[i] = hidden[t][i] + attnOut[t][i];
      const xn = rms(after, post, S.rmsEps);
      const { out, selected } = q35MoeFfnRefF64(xn, moeW, S.nExpert, S.nExpertUsed, S.dFfnExpert);
      perLayer[l].push(Array.from(selected));
      const h2 = new Float64Array(S.dModel);
      for (let i = 0; i < S.dModel; i++) h2[i] = after[i] + out[i];
      next.push(h2);
    }
    hidden = next;
    const el = (performance.now() - t0) / 1000;
    console.log(`[ovl] p${pi} layer ${l + 1}/${S.nLayer} · ${tokens.length} pos · ${el.toFixed(0)}s`);
  }
  sel[pi] = perLayer;
}

// ---------------------------------------------------------------- metriche
const K = S.nExpertUsed, E = S.nExpert;
const inter = (a, b) => { const s = new Set(a); let n = 0; for (const x of b) if (s.has(x)) n++; return n; };
const report = { schemaVersion: 1, kind: "q35-router-overlap", date: new Date().toISOString(), model: "Qwen3.6-35B-A3B-UD-Q4_K_S",
  nExpert: E, topK: K, indipendente: (K * K) / E, prompts: PROMPTS, note: "PREFILL: posizioni adiacenti del prompt. Il decode e' un regime diverso e NON e' misurato qui.", perPrompt: {} };

for (const pi of PROMPTS) {
  const L = sel[pi], n = L[0].length;
  const ov = {}, D = {};
  for (const d of [1, 2, 3, 4, 64]) {
    let s = 0, c = 0;
    for (let l = 0; l < L.length; l++) for (let t = 0; t + d < n; t++) { s += inter(L[l][t], L[l][t + d]); c++; }
    if (c) ov[d] = s / c;
  }
  for (const M of [2, 3, 4, 5, 8, 16]) {
    let s = 0, c = 0;
    for (let l = 0; l < L.length; l++) for (let t = 0; t + M <= n; t++) { const u = new Set(); for (let j = 0; j < M; j++) for (const e of L[l][t + j]) u.add(e); s += u.size; c++; }
    if (c) D[M] = s / c;
  }
  const perLayerOv1 = L.map((rows) => { let s = 0, c = 0; for (let t = 0; t + 1 < n; t++) { s += inter(rows[t], rows[t + 1]); c++; } return c ? s / c : null; });
  report.perPrompt[pi] = { file: golden.prompts[pi].file.split("/").pop(), positions: n, ov, D,
    righePerExpert: Object.fromEntries(Object.entries(D).map(([M, d]) => [M, (K * Number(M)) / d])), perLayerOv1 };
}
writeFileSync(OUT, JSON.stringify(report, null, 2));
console.log(`\n[ovl] scritto ${OUT}  (${((performance.now() - t0) / 60000).toFixed(1)} min)`);
for (const pi of PROMPTS) {
  const r = report.perPrompt[pi];
  console.log(`[ovl] p${pi} ${r.file} · ${r.positions} pos`);
  console.log(`[ovl]   ov(d):  ${[1, 2, 3, 4, 64].map((d) => `d${d}=${(r.ov[d] ?? NaN).toFixed(2)}`).join("  ")}   (indipendente: ${report.indipendente.toFixed(2)})`);
  console.log(`[ovl]   D(M):   ${[2, 4, 8, 16].map((M) => `M${M}=${(r.D[M] ?? NaN).toFixed(1)}`).join("  ")}`);
  console.log(`[ovl]   righe/expert: ${[2, 4, 8, 16].map((M) => `M${M}=${r.righePerExpert[M].toFixed(2)}`).join("  ")}`);
}
