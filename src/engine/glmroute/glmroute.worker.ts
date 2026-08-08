// Conformance routing GLM (goal C2 fase 5 slice 3b, spec §7): replay
// teacher-forced dei prompt della traccia C1 nel motore (createGlmModel a 47
// layer, pesi reali da OPFS) e set-match dei top-4 per (posizione, layer)
// contro la traccia oracolo. GATE: match ≥99% sulle posizioni DECODE — sotto
// soglia ci si ferma e si debugga il router (niente fase 6).
//
// Il replay usa i token della traccia (non i propri argmax): la conformance
// dei logits è il gate di fase 6; qui si misura il routing a parità di input.
import { GLM47_FLASH as G, GLM47_FLASH_SHA256 } from "../shape";
import { dequantQ4_0Row } from "../quant";
import { createGlmModel } from "../glmmodel";
import { GlmOpfsSource } from "../glmsource";
import { slabBufferCap, grantedLimits } from "../gpulimits";
import { createEngineDevice } from "../gpudevice";
import { arenaNeeds } from "../residency";

interface TraceRow { p: number; i: number; tok: number; ph: "p" | "d"; e: number[] }
interface TraceFile {
  header: { ggufSha256: string; llamaCppCommit: string; corpusHash: string; nMoe: number };
  rows: TraceRow[];
}
interface Cfg { cap?: number; prompts?: number[]; budgetGiB: number; prefetch?: "inforward" }

const post = (m: unknown) => (self as unknown as Worker).postMessage(m);
const progress = (msg: string) => post({ type: "progress", msg });

interface LayerCounter { total: number; match: number }

// Fedelta' del router GPU (C3a fase 4 slice B). Il modello gira in modo
// `shadow`: il router+resolve su GPU scrive Sel in una regione ombra mentre a
// decidere resta il routerSelect f64 su CPU. Qui si confronta quello che il
// router GPU AVREBBE scelto con quello che la CPU ha scelto, sulle 31 274
// posizioni del corpus vero invece che sulle 64 estrazioni sintetiche di it.9.
// Il gate e' l'insieme (l'ordine dentro i 4 non cambia la matematica del layer:
// id e peso viaggiano appaiati in Sel); i pesi si confrontano solo quando
// l'insieme coincide — su insiemi diversi sarebbe un confronto fra quantita'
// diverse, non un errore numerico.
const GPU_AGREEMENT_GATE = 99.99;
// f32 (GPU) vs f64 (CPU) sullo stesso ingresso: it.9 ha misurato maxRel 1.6e-7
// su 64 estrazioni. La tolleranza dichiarata e' quella del gate ktest, 1e-5.
const GPU_WEIGHT_REL_TOL = 1e-5;

async function main(cfg: Cfg): Promise<void> {
  const t0 = performance.now();
  const source = await GlmOpfsSource.open("/models/GLM-4.7-Flash-Q4_0.gguf", progress);
  const trace = (await (await fetch("/models/glm-route-trace.json")).json()) as TraceFile;
  if (trace.header.ggufSha256 !== GLM47_FLASH_SHA256) throw new Error("trace: SHA GGUF diverso dal canonico");

  let rows = trace.rows;
  if (cfg.prompts) rows = rows.filter((r) => cfg.prompts!.includes(r.p));
  const byPrompt = new Map<number, TraceRow[]>();
  for (const r of rows) {
    let a = byPrompt.get(r.p);
    if (!a) { a = []; byPrompt.set(r.p, a); }
    a.push(r);
  }
  for (const a of byPrompt.values()) a.sort((x, y) => x.i - y.i);
  if (cfg.cap) for (const [p, a] of byPrompt) byPrompt.set(p, a.slice(0, cfg.cap));
  const totalRows = [...byPrompt.values()].reduce((s, a) => s + a.length, 0);
  const ctxMax = Math.max(...[...byPrompt.values()].map((a) => a.length));

  // Device creato DOPO ctxMax: mlaAttnDecode tiene scores[ctxMax] in workgroup
  // memory (4*ctxMax+256 B) e il corpus arriva a 6688 pos — il requisito si
  // DERIVA dal contesto invece di chiedere il massimo (C3a it.6).
  const { adapter, device } = await createEngineDevice({
    label: "glmroute",
    needs: (adapter) => ({
      ctxMax, head: { vocab: G.vocab, dModel: G.dModel },
      slabClassBytes: Math.floor(cfg.budgetGiB * (1 << 30)),
      // arena expert (C3a fase 4 strato 1): binding d'arena e finestra
      ...arenaNeeds({
        budgetBytes: Math.floor(cfg.budgetGiB * (1 << 30)),
        maxBufferBytes: adapter.limits.maxBufferSize,
        maxBindingBytes: adapter.limits.maxStorageBufferBindingSize,
      }),
    }),
  });
  const { maxBindingBytes: maxBind, maxBufferBytes: maxBuf } = slabBufferCap(device);
  progress(`adapter ${adapter.info?.vendor ?? "?"} — maxBuffer ${(maxBuf / 2 ** 30).toFixed(1)} GiB`);

  // embedding: l'intero token_embd in RAM (178 MB), dequant Q4_0 per riga
  progress("carico token_embd…");
  const embd = source.nonExpert("token_embd.weight");
  const xRow = new Float32Array(G.dModel);

  progress(`costruisco il modello (47 layer, ctxMax ${ctxMax})…`);
  const model = createGlmModel(device, source, {
    ctxMax,
    // il path che decide NON cambia: shadow aggiunge un dispatch per layer MoE e
    // una regione di Sel, e lascia CPU-side ogni esito (setMatch verso l'oracolo
    // deve restare identico all'artefatto del 31-07)
    select: "shadow",
    // C3c fase 4: prefetch in-forward + telemetria (serve per i contatori di
    // recall; nella run di firma — senza flag — resta tutto spento)
    prefetch: cfg.prefetch,
    telemetry: cfg.prefetch !== undefined,
    cache: {
      budgetBytes: Math.floor(cfg.budgetGiB * (1 << 30)),
      maxBindingBytes: maxBind, maxBufferBytes: maxBuf, timing: true,
    },
  });
  const slots = model.cacheStats().slots;
  progress(`cache expert: ${slots.q4_0}+${slots.q4_1} slot (${(((slots.q4_0 * 5308416 + slots.q4_1 * 5505024) / 2 ** 30)).toFixed(1)} GiB)`);

  const perLayerDecode: LayerCounter[] = Array.from({ length: G.nLayer }, () => ({ total: 0, match: 0 }));
  const phase = { p: { total: 0, match: 0 }, d: { total: 0, match: 0 } };
  const samples: Array<{ p: number; i: number; layer: number; got: number[]; want: number[] }> = [];
  // accumulatori del confronto GPU-vs-CPU (uno per (posizione, layer MoE))
  const gpuAgr = {
    total: 0, setMatch: 0, orderMatch: 0, weightOutOfTol: 0, weightMaxRel: 0,
    slotResolved: 0, slotMiss: 0, missing: 0,
    // Sel di PRODUZIONE riletta dalla VRAM: e' quello che i kernel expert hanno
    // letto davvero. Confrontarla con la decisione di routerSelect e' l'unica
    // verifica diretta dell'ordine writeBuffer→dispatch sul corpus vero (R5).
    vramChecked: 0, vramMismatch: 0,
  };
  const gpuSamples: Array<{ p: number; i: number; layer: number; gpu: number[]; cpu: number[] }> = [];
  let doneRows = 0;
  const tReplay0 = performance.now();

  for (const [p, a] of byPrompt) {
    for (const row of a) {
      dequantQ4_0Row(embd, 0, G.dModel, row.tok, xRow);
      const { routing } = await model.forward(xRow, row.i);
      if (routing.length !== G.nLayer - G.denseLead) throw new Error(`routing ${routing.length} layer`);
      for (const r of routing) {
        const moeIdx = r.layer - G.denseLead;
        const want = row.e.slice(moeIdx * 4, moeIdx * 4 + 4);
        const got = Array.from(r.experts);
        const ok = got.length === 4 && want.every((e) => got.includes(e));
        phase[row.ph].total++;
        if (ok) phase[row.ph].match++;
        if (row.ph === "d") {
          perLayerDecode[r.layer].total++;
          if (ok) perLayerDecode[r.layer].match++;
        }
        if (!ok && samples.length < 30) samples.push({ p, i: row.i, layer: r.layer, got, want });

        // --- fedelta' del router GPU (shadow): GPU vs CPU, non vs oracolo ---
        gpuAgr.total++;
        if (!r.gpu) { gpuAgr.missing++; continue; }
        const gpuIds = Array.from(r.gpu.experts);
        const sameSet = new Set(gpuIds).size === 4 && gpuIds.every((e) => got.includes(e));
        if (sameSet) {
          gpuAgr.setMatch++;
          if (gpuIds.every((e, k) => e === got[k])) gpuAgr.orderMatch++;
          for (let k = 0; k < 4; k++) {
            const want2 = r.weights[got.indexOf(gpuIds[k])];
            const rel = Math.abs(r.gpu.weights[k] - want2) / Math.max(Math.abs(want2), 1e-9);
            gpuAgr.weightMaxRel = Math.max(gpuAgr.weightMaxRel, rel);
            if (rel > GPU_WEIGHT_REL_TOL) gpuAgr.weightOutOfTol++;
          }
        } else if (gpuSamples.length < 30) {
          gpuSamples.push({ p, i: row.i, layer: r.layer, gpu: gpuIds, cpu: got });
        }
        // Slot risolti: NON un gate in shadow. Il router GPU gira prima degli
        // `ensure` del suo layer, quindi un expert che il token sta per caricare
        // risulta MISS per costruzione — e' la residenza parziale vista da GPU,
        // ed e' la misura di quanto lo slice C dipenda dalla residenza totale.
        for (const f of r.gpu.flags) (f & 1 ? gpuAgr.slotMiss++ : gpuAgr.slotResolved++);
        // R5 lato produzione: id e peso in VRAM devono essere quelli che
        // routerSelect ha deciso (il peso in f32, come Sel lo tiene).
        if (r.vram) {
          for (let k = 0; k < r.experts.length; k++) {
            gpuAgr.vramChecked++;
            if (r.vram.experts[k] !== r.experts[k]
              || r.vram.weights[k] !== Math.fround(r.weights[k])
              || r.vram.flags[k] !== 0) gpuAgr.vramMismatch++;
          }
        }
      }
      doneRows++;
      if (doneRows % 25 === 0) {
        const st = model.cacheStats();
        const dm = phase.d.total ? (100 * phase.d.match / phase.d.total).toFixed(3) : "—";
        const pm = phase.p.total ? (100 * phase.p.match / phase.p.total).toFixed(3) : "—";
        const rate = doneRows / ((performance.now() - tReplay0) / 1000);
        post({
          type: "tick",
          msg: `${doneRows}/${totalRows} pos (${rate.toFixed(1)}/s, prompt ${p}) — match p ${pm}% d ${dm}% — ` +
            `gpuRouter ${(100 * gpuAgr.setMatch / Math.max(1, gpuAgr.total)).toFixed(4)}% — ` +
            `hit ${(100 * st.hits / Math.max(1, st.hits + st.misses)).toFixed(1)}% (${st.misses} miss, ${(st.bytesRead / 2 ** 30).toFixed(1)} GiB letti)`,
        });
      }
    }
  }

  const st = model.cacheStats();
  // C3c fase 4: contatori del prefetch PRIMA del destroy (telemetria accesa
  // solo con cfg.prefetch — nella run di firma questo e' null)
  const telem = cfg.prefetch ? await model.telemetry() : null;
  model.destroy();
  source.close();
  const report = {
    kind: "glm-routing-conformance",
    // v2 (C3a fase 4 slice B): aggiunto `gpuRouterAgreement`. Tutto il resto del
    // report ha lo stesso significato di v1 — e' il confronto con l'artefatto
    // del 31-07 a dirlo, e deve restare identico.
    schemaVersion: 2,
    date: new Date().toISOString(),
    ggufSha256: GLM47_FLASH_SHA256,
    oracle: { llamaCppCommit: trace.header.llamaCppCommit, corpusHash: trace.header.corpusHash },
    config: { cap: cfg.cap ?? null, prompts: cfg.prompts ?? null, budgetGiB: cfg.budgetGiB, ctxMax, prefetch: cfg.prefetch ?? null },
    positions: totalRows,
    setMatch: {
      prefill: { ...phase.p, pct: phase.p.total ? 100 * phase.p.match / phase.p.total : null },
      decode: { ...phase.d, pct: phase.d.total ? 100 * phase.d.match / phase.d.total : null },
    },
    gate: { threshold: 99, pass: phase.d.total > 0 && 100 * phase.d.match / phase.d.total >= 99 },
    // Gate dello slice B: il router GPU concorda con la CPU su almeno il 99.99%
    // delle (posizione, layer). `missing` > 0 significa che il modo shadow non
    // ha prodotto la regione ombra — il confronto non esiste, non "e' passato".
    gpuRouterAgreement: {
      mode: "shadow",
      ...gpuAgr,
      pct: gpuAgr.total ? 100 * gpuAgr.setMatch / gpuAgr.total : null,
      orderPct: gpuAgr.total ? 100 * gpuAgr.orderMatch / gpuAgr.total : null,
      weightRelTol: GPU_WEIGHT_REL_TOL,
      gate: {
        threshold: GPU_AGREEMENT_GATE, weightRelTol: GPU_WEIGHT_REL_TOL,
        pass: gpuAgr.total > 0 && gpuAgr.missing === 0
          && 100 * gpuAgr.setMatch / gpuAgr.total >= GPU_AGREEMENT_GATE
          && gpuAgr.weightOutOfTol === 0
          // la Sel che decide dev'essere ESATTA: qui non c'e' tolleranza da
          // concedere, e' la scrittura della CPU riletta com'e' arrivata
          && gpuAgr.vramChecked === gpuAgr.total * G.nExpertUsed && gpuAgr.vramMismatch === 0,
      },
      mismatchSamples: gpuSamples,
    },
    perLayerDecode: perLayerDecode
      .map((c, l) => ({ layer: l, ...c, pct: c.total ? 100 * c.match / c.total : null }))
      .filter((c) => c.total > 0),
    mismatchSamples: samples,
    residency: {
      ...st,
      hitRate: st.hits + st.misses > 0 ? st.hits / (st.hits + st.misses) : null,
      slots,
      importMs: source.importMs,
    },
    // C3c fase 4: recall in-engine del prefetch (spec §3) — confronto con
    // l'oracolo C1, scostamento SPIEGATO nel journal, non gateato
    prefetch: telem?.prefetch ? {
      ...telem.prefetch,
      recallAt4Pct: telem.prefetch.recallPreds ? 100 * telem.prefetch.recallHits4 / (telem.prefetch.recallPreds * G.nExpertUsed) : null,
      recallAt8Pct: telem.prefetch.recallPreds ? 100 * telem.prefetch.recallHits8 / (telem.prefetch.recallPreds * G.nExpertUsed) : null,
      oracleRef: {
        recallAt8Pct: 92.0, recallAt4Pct: 77.5,
        source: "C1 (results/engine/moe-oracle/): lookahead di un layer sull'oracolo llama.cpp",
        note: "tap in-engine = router L+1 su ffn_norm_L(x) f32; l'oracolo usa il suo hidden — scostamento atteso, spiegato nel journal",
      },
    } : null,
    dispatchesPerTokenPlanned: model.dispatchesPerTokenPlanned, // DERIVATO dal piano, non contato (C3a it.3)
    deviceLimits: grantedLimits(device), // limiti CONCESSI: senza, un confronto fra run non e' falsificabile
    wallMs: performance.now() - t0,
    replayMs: performance.now() - tReplay0,
  };
  post({ type: "done", report });
}

self.onmessage = (ev: MessageEvent) => {
  void main(ev.data as Cfg).catch((e) => post({ type: "error", message: e instanceof Error ? `${e.message}\n${e.stack}` : String(e) }));
};
