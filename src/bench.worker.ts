import { BenchServer } from "./benchServer";
import { WebLLMAdapter } from "./adapters/webllm";
import { probeWebGPU } from "./probe";

const server = new BenchServer({
  adapterFactory: () => new WebLLMAdapter(),
  probe: () => probeWebGPU(navigator.gpu, navigator as { userAgent: string; deviceMemory?: number }),
  post: (m) => self.postMessage(m),
});

self.onmessage = (e: MessageEvent) => void server.handle(e.data);
