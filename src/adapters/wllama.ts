// Deep import in `esm/` invece del bare specifier `@wllama/wllama` che il README mostra:
// il pacchetto 3.5.1 dichiara `main: "index.js"` ma non pubblica nessun `index.js` alla
// root (solo `index.ts`) e non ha campo `exports`. Con il bare specifier TypeScript
// ripiega sul sorgente `.ts`, che viola `erasableSyntaxOnly` di questo progetto. Il
// README usa già la stessa forma per `esm/wasm-from-cdn.js`.
import { Wllama } from "@wllama/wllama/esm/index.js";
import type {
  ChatCompletionChunk,
  ChatCompletionMessage,
  ChatCompletionUsage,
} from "@wllama/wllama/esm/types/oai-compat.js";
import wllamaWasmUrl from "@wllama/wllama/esm/wasm/wllama.wasm?url";
import type { InferenceAdapter, AdapterCapabilities, GenerateRequest } from "./types";
import type { LoadReport } from "../schema";
import type { GenTimeline } from "../metrics";

// Contesto: il prompt di bench è ~512 token e ne generiamo fino a 256, quindi 2048 dà
// margine senza allocare KV cache inutile (wllama gira su WASM, la memoria è il vincolo).
const N_CTX = 2048;

/**
 * `modelId` per questo stack è `<owner>/<repo>/<percorso-file.gguf>` — il file GGUF è
 * nominato per esteso, non derivato da un `quant`.
 *
 * Motivo: `loadModelFromHF({ quant })` fa **fallback silenzioso** (Q4_K_M → Q8_0 →
 * non quantizzato) se il file richiesto non c'è. Una cella etichettata `quant: "Q4_K_M"`
 * potrebbe così contenere una misura Q8_0 senza che nulla lo dica — esattamente la classe
 * di mislabel che `src/stacks.ts` esiste per impedire. Con il file esplicito il quant è
 * incorporato nel modelId (come per i modelId MLC di WebLLM) e non esiste fallback.
 */
export function hfUrlFromModelId(modelId: string): string {
  const parts = modelId.split("/").filter((p) => p.length > 0);
  if (parts.length < 3) {
    throw new Error(
      `wllama modelId must be "<owner>/<repo>/<file.gguf>", got "${modelId}" — ` +
        "the GGUF file must be named explicitly (no quant fallback, see hfUrlFromModelId)",
    );
  }
  const [owner, repo, ...filePath] = parts;
  const file = filePath.join("/");
  if (!file.endsWith(".gguf")) {
    throw new Error(`wllama modelId must point at a .gguf file, got "${file}"`);
  }
  return `https://huggingface.co/${owner}/${repo}/resolve/main/${file}`;
}

/**
 * Un chunk di stream conta come token generato solo se porta contenuto.
 *
 * wllama emette due tipi di chunk senza contenuto: quello di apertura
 * (`delta.role="assistant"`, `content=null`) e quello di chiusura (`content=undefined`,
 * `finish_reason` valorizzato). Contarli produce timestamp in più del dovuto (17 per 16
 * token — rilevato dal contratto di conformance, non dai test unitari) e gonfia le
 * metriche di decode che questo adapter esiste per misurare.
 *
 * Nota: il chunk di chiusura viene consegnato **all'inizio della chiamata successiva**
 * (wllama lo emette dopo aver già segnalato `has_more=false`). Questo filtro lo scarta,
 * quindi non contamina i timestamp; resta però il conteggio a scalare fra generate
 * consecutive — vedi docket #8.
 */
export function chunkIsToken(chunk: ChatCompletionChunk): boolean {
  const content = chunk.choices[0]?.delta?.content;
  return typeof content === "string" && content.length > 0;
}

export interface WllamaGenerateResult {
  promptTokens: number | null;
  completionTokens: number | null;
}

interface WllamaEngine {
  generate(
    messages: ChatCompletionMessage[],
    maxTokens: number,
    onToken: () => void,
  ): Promise<WllamaGenerateResult>;
  dispose(): Promise<void>;
}

type EngineFactory = (
  modelUrl: string,
  onProgress: (text: string, progress: number) => void,
) => Promise<WllamaEngine>;

function defaultEngineFactory(): EngineFactory {
  return async (modelUrl, onProgress) => {
    const wllama = new Wllama({ default: wllamaWasmUrl });
    await wllama.loadModelFromUrl(modelUrl, {
      n_ctx: N_CTX,
      progressCallback: ({ loaded, total }) =>
        onProgress("downloading model", total > 0 ? loaded / total : 0),
    });
    return {
      generate: async (messages, maxTokens, onToken) => {
        // L'usage arriva sull'ultimo chunk: è il conteggio token autorevole di llama.cpp,
        // non una stima ricavata dal numero di callback. Contenitore invece di una `let`
        // perché TypeScript, vedendo l'unica assegnazione dentro la callback, restringerebbe
        // la variabile a `never` dopo l'await.
        const captured: { usage: ChatCompletionUsage | null } = { usage: null };
        await wllama.createChatCompletion({
          messages,
          max_tokens: maxTokens,
          temperature: 0, // greedy: il bench misura la velocità, non il campionamento
          // Il prompt cache di llama.cpp riuserebbe la KV cache del prompt fra chiamate
          // successive: le repliche 2 e 3 di una cella salterebbero il prefill dei ~512
          // token del prompt di bench, con un TTFT artificialmente basso — mentre WebLLM e
          // Transformers.js il prefill lo rifanno ogni volta. Il confronto cross-stack
          // misurerebbe il caching di uno stack, non la sua velocità.
          // (Verificato attivo: con `true` il log llama.cpp mostra `n_past was set to …`,
          // con `false` no. NON è invece la causa del check di determinismo che fallisce —
          // quella è lo sforamento del chunk di chiusura, vedi docket #8.)
          cache_prompt: false,
          stream: true,
          onData: (chunk: ChatCompletionChunk) => {
            if (chunkIsToken(chunk)) onToken();
            if (chunk.usage) captured.usage = chunk.usage;
          },
        });
        return {
          promptTokens: captured.usage?.prompt_tokens ?? null,
          completionTokens: captured.usage?.completion_tokens ?? null,
        };
      },
      dispose: () => wllama.exit(),
    };
  };
}

function defaultIsCached(modelUrl: string): Promise<boolean> {
  // Istanza usata solo per interrogare la cache: `new Wllama` non carica nulla finché non
  // gli si chiede un modello, quindi non c'è WASM da liberare qui.
  const probe = new Wllama({ default: wllamaWasmUrl });
  return probe.cacheManager
    .getNameFromURL(modelUrl)
    .then((name) => probe.cacheManager.getSize(name))
    .then((size) => size > 0)
    .catch(() => false);
}

export class WllamaAdapter implements InferenceAdapter {
  readonly id = "wllama" as const;
  private engine: WllamaEngine | null = null;
  private engineFactory: EngineFactory;
  private isCached: (modelUrl: string) => Promise<boolean>;
  private now: () => number;

  constructor(deps?: {
    engineFactory?: EngineFactory;
    isCached?: (modelUrl: string) => Promise<boolean>;
    now?: () => number;
  }) {
    this.engineFactory = deps?.engineFactory ?? defaultEngineFactory();
    this.isCached = deps?.isCached ?? defaultIsCached;
    this.now = deps?.now ?? (() => performance.now());
  }

  capabilities(): AdapterCapabilities {
    // seed: wllama lo espone (LoadModelParams.seed / RawCompletionParams.seed), ma questo
    // adapter non lo usa — genera a temperature 0. Dichiarare `true` significherebbe
    // affermare una capacità che il percorso di codice reale non esercita.
    return { logprobs: false, streaming: true, seed: false };
  }

  async load(modelId: string, onProgress: (text: string, progress: number) => void): Promise<LoadReport> {
    const modelUrl = hfUrlFromModelId(modelId);
    const cached = await this.isCached(modelUrl);
    const t0 = this.now();
    this.engine = await this.engineFactory(modelUrl, onProgress);
    return { loadMs: this.now() - t0, cacheState: cached ? "warm" : "cold" };
  }

  async generate(req: GenerateRequest): Promise<GenTimeline> {
    if (!this.engine) throw new Error("not loaded");
    const tRequestStart = this.now();
    const chunkTimestamps: number[] = [];
    const { promptTokens, completionTokens } = await this.engine.generate(
      [{ role: "user", content: req.prompt }],
      req.maxTokens,
      () => {
        chunkTimestamps.push(this.now());
      },
    );
    return { tRequestStart, chunkTimestamps, promptTokens, completionTokens };
  }

  async dispose(): Promise<void> {
    await this.engine?.dispose();
    this.engine = null;
  }
}
