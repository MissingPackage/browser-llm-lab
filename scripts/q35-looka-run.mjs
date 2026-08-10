// Recall dell'ORACOLO LOOKAHEAD sul router 256-wide del 35B (q1 fase 8,
// metodo C1): a fine layer l-1 (hidden x PRIMA dell'attn di l) si replica la
// selezione del router l con rms(x, postnorm_l) → top-K; recall vs la
// selezione VERA (che usa xn_l = rms(x + attnOut_l, postnorm_l), cioè
// ATTRAVERSA l'attention). È l'analogo esatto del LOOKA GLM (tap hidden L →
// router L+1, recall 91.92% @K=8): misura quanto l'attention del layer
// sposta il routing. SUBSET DICHIARATO di prompt INTERI (regola full-corpus:
// esplorazione = subset di prompt interi, mai --cap): p4 (05-math) + p7
// (08-prosa-en) del golden full. NON è un gate: lo scostamento si SPIEGA.
// Uso: npx vite-node scripts/q35-looka-run.mjs   (~35-40 min CPU)
import { fstatSync, openSync, readFileSync, readSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const { Q35CpuRefModel, q35MoeFfnRefF64 } = await import("../src/engine/q35cpurefmodel.ts");

const MODEL = join(homedir(), ".cache/blab-models/q35/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf");
const fd = openSync(MODEL, "r");
const src = {
  size: fstatSync(fd).size,
  slice(off, len) {
    const out = new Uint8Array(len);
    let g = 0;
    while (g < len) {
      const n = readSync(fd, out, g, len - g, off + g);
      if (n <= 0) throw new Error("short read");
      g += n;
    }
    return out;
  },
};
const m = new Q35CpuRefModel(src);
const S = m.shape;
const golden = JSON.parse(readFileSync("results/engine/golden/q35/golden-q35-35b-full-2026-08-10.json", "utf8"));
const SUBSET = [4, 7]; // prompt INTERI dichiarati
const KS = [4, 8];

const rms = (x, w, eps) => {
  let ss = 0;
  for (let i = 0; i < x.length; i++) ss += x[i] * x[i];
  const sc = 1 / Math.sqrt(ss / x.length + eps);
  const o = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) o[i] = x[i] * sc * w[i];
  return o;
};
const topKof = (probs, k) =>
  Array.from({ length: probs.length }, (_, e) => e).sort((a, b) => probs[b] - probs[a] || a - b).slice(0, k);
const routerProbs = (router, x, nE) => {
  const logits = new Float64Array(nE);
  for (let e = 0; e < nE; e++) {
    let acc = 0;
    const base = e * x.length;
    for (let i = 0; i < x.length; i++) acc += router[base + i] * x[i];
    logits[e] = acc;
  }
  let mx = -Infinity;
  for (let e = 0; e < nE; e++) if (logits[e] > mx) mx = logits[e];
  const p = new Float64Array(nE);
  let sm = 0;
  for (let e = 0; e < nE; e++) { p[e] = Math.exp(logits[e] - mx); sm += p[e]; }
  for (let e = 0; e < nE; e++) p[e] /= sm;
  return p;
};

const perLayer = Array.from({ length: S.nLayer }, () => ({ hit: { 4: 0, 8: 0 }, tot: 0 }));
let positions = 0;
const t0 = performance.now();
for (const pi of SUBSET) {
  const tokens = golden.prompts[pi].promptTokens;
  let hidden = tokens.map((tk) => m.embedRow(tk));
  for (let l = 0; l < S.nLayer; l++) {
    const b = `blk.${l}.`;
    const post = m.dequant(`${b}post_attention_norm.weight`);
    const router = m.dequant(`${b}ffn_gate_inp.weight`);
    // predizione LOOKAHEAD: dall'hidden PRIMA dell'attn di l
    const preds = hidden.map((x) => {
      const p = routerProbs(router, rms(x, post, S.rmsEps), S.nExpert);
      return { p4: topKof(p, 4), p8: topKof(p, 8) };
    });
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
          h = {
            gate: m.dequantExpert(`${b}ffn_gate_exps.weight`, e),
            up: m.dequantExpert(`${b}ffn_up_exps.weight`, e),
            down: m.dequantExpert(`${b}ffn_down_exps.weight`, e),
          };
          cache.set(e, h);
        }
        return h;
      },
    };
    for (let t = 0; t < tokens.length; t++) {
      const after = new Float64Array(S.dModel);
      for (let i = 0; i < S.dModel; i++) after[i] = hidden[t][i] + attnOut[t][i];
      const xn = rms(after, post, S.rmsEps);
      const { out, selected } = q35MoeFfnRefF64(xn, moeW, S.nExpert, S.nExpertUsed, S.dFfnExpert);
      const trueSet = new Set(selected);
      for (const K of KS) {
        const hitK = preds[t][`p${K}`].filter((e) => trueSet.has(e)).length;
        perLayer[l].hit[K] += hitK;
      }
      perLayer[l].tot += S.nExpertUsed;
      for (let i = 0; i < S.dModel; i++) after[i] += out[i];
      hidden[t] = after;
    }
    console.log(`p${pi} layer ${l}/${S.nLayer} (${((performance.now() - t0) / 60000).toFixed(1)} min)`);
  }
  positions += tokens.length;
}

const agg = { 4: 0, 8: 0 };
let tot = 0;
for (const pl of perLayer) {
  agg[4] += pl.hit[4];
  agg[8] += pl.hit[8];
  tot += pl.tot;
}
const out = {
  schemaVersion: 1,
  kind: "q35-looka-35b",
  date: "2026-08-10",
  model: "Qwen3.6-35B-A3B-UD-Q4_K_S.gguf",
  method:
    "metodo C1 su cpuref-f64: predizione = router_l su rms(hidden pre-attn_l, postnorm_l), verita' = selezione con xn_l post-attn; recall@K = |topK(pred) ∩ top8(vero)|/8. NON un gate: scostamento da spiegare.",
  subset: { prompts: SUBSET.map((i) => golden.prompts[i].file), positions, declared: "subset di prompt INTERI (regola full-corpus)" },
  glmReference: { recallAt8: 0.9192, note: "GLM C1: router sigmoid+bias 64-wide top-4, hidden L->router L+1" },
  recall: {
    at4: agg[4] / tot,
    at8: agg[8] / tot,
  },
  perLayer: perLayer.map((pl, l) => ({ layer: l, at4: +(pl.hit[4] / pl.tot).toFixed(4), at8: +(pl.hit[8] / pl.tot).toFixed(4) })),
  minutes: +((performance.now() - t0) / 60000).toFixed(1),
};
writeFileSync("results/engine/q35-looka-35b-2026-08-10.json", JSON.stringify(out, null, 1));
console.log(`OK recall@4 ${(out.recall.at4 * 100).toFixed(2)}% @8 ${(out.recall.at8 * 100).toFixed(2)}% su ${positions} posizioni (${out.minutes} min)`);
