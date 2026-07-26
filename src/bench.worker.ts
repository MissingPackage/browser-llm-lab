import { BenchServer } from "./benchServer";
import { WebLLMAdapter } from "./adapters/webllm";
import { TransformersJsAdapter } from "./adapters/transformersjs";
import { WllamaAdapter } from "./adapters/wllama";
import { probeWebGPU } from "./probe";

const server = new BenchServer({
  adapters: {
    webllm: () => new WebLLMAdapter(),
    transformersjs: () => new TransformersJsAdapter(),
    wllama: () => new WllamaAdapter(),
  },
  probe: () => probeWebGPU(navigator.gpu, navigator as { userAgent: string; deviceMemory?: number }),
  post: (m) => self.postMessage(m),
});

self.onmessage = (e: MessageEvent) => void server.handle(e.data);
