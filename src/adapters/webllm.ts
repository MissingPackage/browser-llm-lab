import {
  CreateMLCEngine,
  hasModelInCache,
  type MLCEngineInterface,
} from "@mlc-ai/web-llm";
import type { InferenceAdapter, AdapterCapabilities, GenerateRequest } from "./types";
import type { LoadReport } from "../schema";
import type { GenTimeline } from "../metrics";

type EngineFactory = (
  modelId: string,
  onProgress: (text: string, progress: number) => void,
) => Promise<MLCEngineInterface>;

const defaultFactory: EngineFactory = (modelId, onProgress) =>
  CreateMLCEngine(modelId, {
    initProgressCallback: (p) => onProgress(p.text, p.progress),
  });

export class WebLLMAdapter implements InferenceAdapter {
  readonly id = "webllm" as const;
  private engine: MLCEngineInterface | null = null;
  private engineFactory: EngineFactory;
  private hasCache: (modelId: string) => Promise<boolean>;
  private now: () => number;

  constructor(deps?: {
    engineFactory?: EngineFactory;
    hasCache?: (modelId: string) => Promise<boolean>;
    now?: () => number;
  }) {
    this.engineFactory = deps?.engineFactory ?? defaultFactory;
    this.hasCache = deps?.hasCache ?? ((id) => hasModelInCache(id).catch(() => false));
    this.now = deps?.now ?? (() => performance.now());
  }

  capabilities(): AdapterCapabilities {
    return { logprobs: false, streaming: true, seed: false }; // logprobs: rivalutare in 1b
  }

  async load(modelId: string, onProgress: (text: string, progress: number) => void): Promise<LoadReport> {
    const cached = await this.hasCache(modelId);
    const t0 = this.now();
    this.engine = await this.engineFactory(modelId, onProgress);
    return { loadMs: this.now() - t0, cacheState: cached ? "warm" : "cold" };
  }

  async generate(req: GenerateRequest): Promise<GenTimeline> {
    if (!this.engine) throw new Error("not loaded");
    const tRequestStart = this.now();
    const chunkTimestamps: number[] = [];
    let promptTokens: number | null = null;
    let completionTokens: number | null = null;

    const stream = await this.engine.chat.completions.create({
      messages: [{ role: "user", content: req.prompt }],
      temperature: 0,
      max_tokens: req.maxTokens,
      stream: true,
      stream_options: { include_usage: true },
    });

    for await (const chunk of stream) {
      if (chunk.choices[0]?.delta?.content) chunkTimestamps.push(this.now());
      if (chunk.usage) {
        promptTokens = chunk.usage.prompt_tokens ?? null;
        completionTokens = chunk.usage.completion_tokens ?? null;
      }
    }
    return { tRequestStart, chunkTimestamps, promptTokens, completionTokens };
  }

  async dispose(): Promise<void> {
    await this.engine?.unload();
    this.engine = null;
  }
}
