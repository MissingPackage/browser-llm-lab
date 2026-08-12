// Pagina del micro-bench di fase 0 (engine-kernel-decode). Parte da sola al
// load e pubblica il run file su `window.__report`: stesso pattern di ktest.html
// e engine.html?attnbench=1, che i runner playwright dell'harness gia' guidano.
import { runKernelDecodeBench } from "./kdRunner";

const q = new URLSearchParams(location.search);
const statusEl = document.getElementById("status")!;
const logEl = document.getElementById("log")!;

const log = (m: string): void => {
  statusEl.textContent = m;
  logEl.textContent = `${m}\n${logEl.textContent}`.split("\n").slice(0, 40).join("\n");
};

void (async () => {
  try {
    const runFile = await runKernelDecodeBench({
      deviceLabel: q.get("label") ?? "4090-linux",
      hostState: q.get("host") ?? "quiescent",
      onProgress: log,
    });
    (window as unknown as { __report: unknown }).__report = runFile;
    document.getElementById("results")!.textContent = JSON.stringify(
      runFile.cells.map((c) => ({
        k: c.kernel, v: c.variant, shape: c.shape,
        p50ms: +c.msPerOp.p50.toFixed(4), gbps: +c.effectiveGBps.toFixed(1),
        gws: c.weightsPerSecond ? +(c.weightsPerSecond / 1e9).toFixed(1) : null,
      })), null, 1);
    statusEl.textContent = "done";
  } catch (e) {
    statusEl.textContent = `ERROR: ${e instanceof Error ? e.message : String(e)}`;
    console.error(e);
  }
})();
