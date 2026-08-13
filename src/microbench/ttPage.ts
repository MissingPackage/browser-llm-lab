// Pagina delle sonde della riga 1 di engine-ttft. Stesso pattern di kdPage.ts:
// parte da sola al load e pubblica il run file su `window.__report`. Il piano
// della spazzata dei limiti (sonda d) finisce su `window.__sweepPlan`: la esegue
// il DRIVER, perche' richiede device distinti con `requiredLimits` espliciti e
// il punto unico di creazione device in src/ negozia sempre.
import { runTtftProbeBench } from "./ttRunner";

const q = new URLSearchParams(location.search);
const statusEl = document.getElementById("status")!;
const logEl = document.getElementById("log")!;

const log = (m: string): void => {
  statusEl.textContent = m;
  logEl.textContent = `${m}\n${logEl.textContent}`.split("\n").slice(0, 60).join("\n");
};

void (async () => {
  try {
    const { runFile, sweepPlan } = await runTtftProbeBench({
      deviceLabel: q.get("label") ?? "4090-linux",
      hostState: q.get("host") ?? "quiescent",
      onProgress: log,
    });
    const w = window as unknown as { __report: unknown; __sweepPlan: unknown };
    w.__report = runFile;
    w.__sweepPlan = sweepPlan;
    document.getElementById("results")!.textContent = JSON.stringify(
      runFile.cells.map((c) => ({
        k: c.kernel, v: c.variant, shape: c.shape, M: c.M,
        p50ms: +c.msPerOp.p50.toFixed(4),
        tokS: c.tokensPerSecond ? +c.tokensPerSecond.toFixed(1) : null,
        wBytesPerTok: c.weightBytesPerToken,
        tflops: c.tflops ? +c.tflops.toFixed(2) : null,
        wgStorage: c.workgroupStorageBytes,
      })), null, 1);
    statusEl.textContent = "done";
  } catch (e) {
    statusEl.textContent = `ERROR: ${e instanceof Error ? e.message : String(e)}`;
    console.error(e);
  }
})();
