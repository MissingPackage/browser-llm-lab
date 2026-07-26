import { pipeline, TextStreamer, ModelRegistry } from "@huggingface/transformers";
import type { InferenceAdapter, AdapterCapabilities, GenerateRequest } from "./types";
import type { LoadReport } from "../schema";
import type { GenTimeline } from "../metrics";

const DTYPE = "q4";
const DEVICE = "webgpu";
const TASK = "text-generation";

// Contratto minimo che usiamo davvero dell'oggetto restituito da pipeline():
// chiamabile, `.tokenizer` (passato a TextStreamer), `.dispose()`. Il tipo reale
// di pipeline() dipende dal task a runtime e non è utilmente esprimibile qui.
type RawPipeline = {
  tokenizer: ConstructorParameters<typeof TextStreamer>[0];
  dispose(): Promise<void>;
  (input: Array<{ role: string; content: string }>, options: Record<string, unknown>): Promise<unknown>;
};

interface TextGenerationEngine {
  generate(
    messages: Array<{ role: string; content: string }>,
    maxTokens: number,
    onToken: () => void,
  ): Promise<void>;
  dispose(): Promise<void>;
}

type EngineFactory = (
  modelId: string,
  onProgress: (text: string, progress: number) => void,
) => Promise<TextGenerationEngine>;

const defaultEngineFactory: EngineFactory = async (modelId, onProgress) => {
  const pipe = (await pipeline(TASK, modelId, {
    dtype: DTYPE,
    device: DEVICE,
    progress_callback: (info: { status: string; file?: string; progress?: number }) => {
      const pct = typeof info.progress === "number" ? info.progress / 100 : 0;
      onProgress(info.file ? `${info.status}: ${info.file}` : info.status, pct);
    },
  })) as unknown as RawPipeline;
  return {
    generate: async (messages, maxTokens, onToken) => {
      const streamer = new TextStreamer(pipe.tokenizer, {
        skip_prompt: true,
        skip_special_tokens: true,
        callback_function: () => onToken(),
      });
      await pipe(messages, { max_new_tokens: maxTokens, do_sample: false, streamer });
    },
    dispose: () => pipe.dispose(),
  };
};

const defaultIsCached = (modelId: string) =>
  ModelRegistry.is_pipeline_cached(TASK, modelId, { dtype: DTYPE }).catch(() => false);

export class TransformersJsAdapter implements InferenceAdapter {
  readonly id = "transformersjs" as const;
  private engine: TextGenerationEngine | null = null;
  private engineFactory: EngineFactory;
  private isCached: (modelId: string) => Promise<boolean>;
  private now: () => number;

  constructor(deps?: {
    engineFactory?: EngineFactory;
    isCached?: (modelId: string) => Promise<boolean>;
    now?: () => number;
  }) {
    this.engineFactory = deps?.engineFactory ?? defaultEngineFactory;
    this.isCached = deps?.isCached ?? defaultIsCached;
    this.now = deps?.now ?? (() => performance.now());
  }

  capabilities(): AdapterCapabilities {
    return { logprobs: false, streaming: true, seed: false }; // logprobs: rivalutare nel modulo qualità (Fase 3)
  }

  async load(modelId: string, onProgress: (text: string, progress: number) => void): Promise<LoadReport> {
    const cached = await this.isCached(modelId);
    const t0 = this.now();
    this.engine = await this.engineFactory(modelId, onProgress);
    return { loadMs: this.now() - t0, cacheState: cached ? "warm" : "cold" };
  }

  async generate(req: GenerateRequest): Promise<GenTimeline> {
    if (!this.engine) throw new Error("not loaded");
    const tRequestStart = this.now();
    const chunkTimestamps: number[] = [];
    await this.engine.generate([{ role: "user", content: req.prompt }], req.maxTokens, () => {
      chunkTimestamps.push(this.now());
    });
    return { tRequestStart, chunkTimestamps, promptTokens: null, completionTokens: null };
  }

  async dispose(): Promise<void> {
    await this.engine?.dispose();
    this.engine = null;
  }
}
