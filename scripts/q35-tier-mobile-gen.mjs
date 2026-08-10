// Tier MOBILE 4B (q1 fase 8, spec §6): il riferimento NON è una emulazione di
// paging (il denso non pagina) — è (a) la MISURA full-resident del 4B col
// VINCOLO di cap 3 GiB DICHIARATO e verificato dal footprint, + (b) la
// PROIEZIONE PARAMETRICA del TTFT mobile con la banda storage come PARAMETRO
// LIBERO (spec §6: i numeri di banda mobile NON si estrapolano — arrivano
// solo dal device, PI-gated). Deriva tutto dal bench committato di it.10.
import { readFileSync, writeFileSync } from "node:fs";

const bench = JSON.parse(readFileSync("results/engine/q35-bench-4b-fullresident-2026-08-10.json", "utf8"));

// footprint VRAM 4B (contabile, dal file: pesi repacked ≈ raw + scratch):
const GGUF_BYTES = 2583221408;
const HEAD_Q6K = 521472000 * (212 / 210 > 1 ? 1 : 1); // tied embd su GPU (kquant, 36w/44w esatti per Q6_K: 53w → 212/210)
const KV_CTX512 = 8 * 512 * 1024 * 4; // 8 layer full × ctx × kvDim(1024) × f32
const SCRATCH = 64 * 1024 * 1024; // logits 1MB + scratch vari, stima larga
const footprint = GGUF_BYTES + KV_CTX512 + SCRATCH; // (head è DENTRO il gguf: tied)
const CAP = 3 * 2 ** 30;

const out = {
  schemaVersion: 1,
  kind: "q35-tier-mobile-4b",
  date: "2026-08-10",
  model: bench.model,
  modelSha256: bench.modelSha256,
  declared:
    "tier mobile = classe 8 GB RAM unificata (S22, spec §6). Il 4B è FULL-RESIDENT sotto il cap: nessun paging da emulare (il paging è del MoE). I numeri di compute sono la MISURA su 4090 (frame correttezza-prima, NON proiettabili al silicio mobile); il TTFT mobile è una PROIEZIONE PARAMETRICA con banda storage e fattore compute LIBERI — i numeri veri arrivano solo dal device (PI-gated, docket).",
  capBytes: CAP,
  footprintBytes: { weights: GGUF_BYTES, kvCtx512: KV_CTX512, scratchEst: SCRATCH, total: footprint },
  capRespected: footprint <= CAP,
  measured4090: {
    source: "results/engine/q35-bench-4b-fullresident-2026-08-10.json",
    decodeTokS: bench.decode.tokS,
    prefillSeqTokS: bench.prefill.tokS,
    ttftMs: bench.ttftMs,
    hostState: bench.hostState ?? "v. file sorgente",
  },
  projectionMobile: {
    formula: "TTFT(bandaStorageGBs, computeFactor) = pesiGB/bandaStorageGBs + prefillTokens/(prefillSeqTokS_4090/computeFactor)",
    note: "computeFactor = rapporto compute 4090/mobile (LIBERO, tipico 5-15x per GPU mobile); bandaStorageGBs = banda lettura storage del device (LIBERO). Esempi ILLUSTRATIVI, non claim:",
    examples: [1, 2].map((banda) =>
      [5, 10].map((cf) => ({
        bandaStorageGBs: banda,
        computeFactor: cf,
        prefillTokens: 387,
        ttftS: +(GGUF_BYTES / 1e9 / banda + 387 / (bench.prefill.tokS / cf)).toFixed(1),
      })),
    ).flat(),
  },
};
writeFileSync("results/engine/q35-tier-mobile-4b-2026-08-10.json", JSON.stringify(out, null, 1));
console.log("OK: cap 3 GiB rispettato =", out.capRespected, "| footprint", (footprint / 2 ** 30).toFixed(2), "GiB");
