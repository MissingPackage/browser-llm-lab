import type { LoadReport } from "../schema";
import type { GenTimeline } from "../metrics";

export interface AdapterCapabilities {
  logprobs: boolean; // 1a: false; usato dal modulo qualità in 1b
  streaming: boolean;
  seed: boolean;
}

export interface GenerateRequest {
  prompt: string;
  maxTokens: number;
}

export interface InferenceAdapter {
  readonly id: "webllm"; // union estesa in 1b
  capabilities(): AdapterCapabilities;
  load(modelId: string, onProgress: (text: string, progress: number) => void): Promise<LoadReport>;
  generate(req: GenerateRequest): Promise<GenTimeline>;
  dispose(): Promise<void>;
}
