import { GPU } from "./webgpu-llm/gpu.js";
import { Loader } from "./webgpu-llm/loader.js";
import { Model } from "./webgpu-llm/model.js";
import { Tokenizer } from "./webgpu-llm/tokenizer.js";
import { RT } from "./webgpu-llm/config.js";
import { GemmaLoader } from "./webgpu-llm/gemma-loader.js";
import { GemmaModel } from "./webgpu-llm/gemma-model.js";
import { GemmaTokenizer } from "./webgpu-llm/gemma-tokenizer.js";
import { GEMMA_E2B_CFG, GEMMA_E4B_CFG } from "./webgpu-llm/gemma-config.js";
import type { GemmaWeightsMap } from "./webgpu-llm/gemma-loader.js";
import type { WeightsMap } from "./webgpu-llm/loader.js";
import {
  DEFAULT_MODEL_ID,
  GEMMA_E2B_MODEL_ID,
  GEMMA_E4B_MODEL_ID,
  QWEN_MODEL_ID,
  assertModelRegistered,
  availableModels,
  getModelProfile,
  isModelId,
  isGemmaModelId,
  type ModelId,
  type ModelProfile,
} from "./webgpu-llm/model-registry";

export interface ProgressEvent {
  stage: string;
  message: string;
  progress: number | null;
}

export type OnProgress = (event: ProgressEvent) => void;

export interface LoadOptions {
  /** Select a registered model. Omit after load to keep using the active one. */
  model?: ModelId;
  maxContext?: number;
  batchSize?: number;
  /** Prompt rows processed per GPU submission; rounded to a 32-row tile. */
  prefillChunk?: number;
  mtp?: boolean;
  reload?: boolean;
}

export interface DeviceInfo {
  vendor: string;
  architecture: string;
  vramBytes: number;
  subgroups?: boolean;
  subgroupMinSize?: number;
  subgroupMaxSize?: number;
}

export interface EngineStatus {
  model: string;
  activeModel: ModelId;
  defaultModel: ModelId;
  availableModels: Array<Pick<ModelProfile, "id" | "label" | "modalities" | "maxContext">>;
  modalities: readonly string[];
  webgpu: boolean;
  adapter: boolean;
  cached: boolean | null;
  loaded: boolean;
  generating: boolean;
  hasMtp: boolean;
  batchSize: number;
  prefillChunk: number;
  contextUsedTokens: number;
  contextMaxTokens: number;
  device: DeviceInfo | null;
}

export interface ListedModel extends Pick<ModelProfile, "id" | "label" | "modalities" | "maxContext"> {
  object: "model";
  cached: boolean | null;
}

const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

function resolveLoadOptions(options: LoadOptions, model: ModelId) {
  const profile = getModelProfile(model);
  const gemma = isGemmaModelId(model);
  return {
    // Gemma's FP8 cache is material for every decoder layer, unlike Qwen's
    // sparse full-attention cache.  Keep its default practical while still
    // allowing callers to opt into its full checkpoint context window.
    maxContext: Math.round(
      clamp(options.maxContext ?? (gemma ? 8192 : RT.maxCtx), 1024, profile.maxContext)
    ),
    // Long decode command buffers monopolize the same GPU queue Chrome uses
    // for compositing. With the flash-decoding attention path a Gemma forward
    // is short enough that four tokens amortize the per-batch readback while
    // keeping each burst well under the old two-forward wall-clock.
    batchSize: Math.round(clamp(options.batchSize ?? (gemma ? 4 : 8), 1, 8)),
    prefillChunk: Math.round(
      clamp(
        options.prefillChunk ?? (model === GEMMA_E4B_MODEL_ID
          ? 64
          : model === GEMMA_E2B_MODEL_ID
            ? 128
            : RT.chunk),
        32,
        256
      ) / 32
    ) * 32,
    mtp: gemma ? false : options.mtp ?? false,
  };
}

type RuntimeLoader = Loader | GemmaLoader;
type RuntimeModel = Model | GemmaModel;
type RuntimeTokenizer = Tokenizer | GemmaTokenizer;

type BufferLike = { size: number };
type WeightLike = {
  q?: BufferLike;
  s?: BufferLike;
  size?: number;
  shards?: Array<{ q: BufferLike; s: BufferLike }>;
};

function weightsVramBytes(weights: Record<string, unknown>): number {
  let total = 0;
  for (const entry of Object.values(weights) as WeightLike[]) {
    if (!entry) continue;
    if (entry.q && entry.s) total += entry.q.size + entry.s.size;
    else if (entry.shards) {
      for (const shard of entry.shards) total += shard.q.size + shard.s.size;
    } else if (entry.size) total += entry.size;
  }
  return total;
}

export class Engine {
  private gpu: GPU | null = null;
  private loader: RuntimeLoader | null = null;
  private modelInstance: RuntimeModel | null = null;
  private tokInstance: RuntimeTokenizer | null = null;
  private loadPromise: Promise<void> | null = null;
  private deviceInfo: DeviceInfo | null = null;
  private mtpEnabled = false;
  private activeModel = DEFAULT_MODEL_ID;

  generating = false;
  /**
   * GPU-resident conversation prefix from the last clean completion. When the
   * next request strictly extends `sigs` under the same `model` + `promptKey`
   * (thinking flag + tool declarations rendered into the prompt prefix), the
   * runtime skips model.reset() and prefills only the suffix. `pending` names
   * canonical text the cache is missing because the final stop token was
   * sampled but never fed: the turn-close marker, the tool-call close tag, or
   * nothing (a lookahead already fed the close).
   */
  committed: {
    model: ModelId;
    promptKey: string;
    sigs: string[];
    toolCallCount: number;
    pending: "turn-close" | "tool-close" | "none";
  } | null = null;

  get ready() {
    return !!this.modelInstance && !!this.tokInstance;
  }

  get activeModelId(): ModelId {
    return this.activeModel;
  }

  async listModels(): Promise<ListedModel[]> {
    return Promise.all(
      availableModels().map(async ({ id, label, modalities, maxContext }) => ({
        id,
        object: "model" as const,
        label,
        modalities,
        maxContext,
        cached: await this.probeCache(id),
      }))
    );
  }

  get model(): RuntimeModel {
    if (!this.modelInstance) throw new Error("Model is not loaded");
    return this.modelInstance;
  }

  get tok(): RuntimeTokenizer {
    if (!this.tokInstance) throw new Error("Model is not loaded");
    return this.tokInstance;
  }

  get qwenModel(): Model {
    if (this.activeModel !== QWEN_MODEL_ID || !(this.modelInstance instanceof Model)) {
      throw new Error("Qwen is not the loaded model");
    }
    return this.modelInstance;
  }

  get qwenTok(): Tokenizer {
    if (this.activeModel !== QWEN_MODEL_ID || !(this.tokInstance instanceof Tokenizer)) {
      throw new Error("Qwen is not the loaded model");
    }
    return this.tokInstance;
  }

  get gemmaModel(): GemmaModel {
    if (!isGemmaModelId(this.activeModel) || !(this.modelInstance instanceof GemmaModel)) {
      throw new Error("Gemma is not the loaded model");
    }
    return this.modelInstance;
  }

  get gemmaTok(): GemmaTokenizer {
    if (!isGemmaModelId(this.activeModel) || !(this.tokInstance instanceof GemmaTokenizer)) {
      throw new Error("Gemma is not the loaded model");
    }
    return this.tokInstance;
  }

  private requestedModel(value: unknown): ModelId {
    const candidate = value ?? this.activeModel;
    if (!isModelId(candidate)) {
      throw new Error(`Unsupported model: ${String(candidate)}`);
    }
    return candidate;
  }

  checkGpu(): void {
    if (typeof navigator === "undefined" || !navigator.gpu) {
      throw new Error(
        'WebGPU is not available in this browser. Use Chrome/Edge 121+ (on Linux you may need chrome://flags → "Vulkan" + "Unsafe WebGPU").'
      );
    }
  }

  async probeCache(model = this.activeModel): Promise<boolean | null> {
    try {
      const probe = isGemmaModelId(model)
        ? new GemmaLoader(null, () => {}, model === GEMMA_E2B_MODEL_ID ? GEMMA_E2B_CFG : GEMMA_E4B_CFG)
        : new Loader(null, () => {});
      return !!(await probe.cacheValid());
    } catch {
      return null;
    }
  }

  async status(): Promise<EngineStatus> {
    const profile = getModelProfile(this.activeModel);
    const webgpu = typeof navigator !== "undefined" && !!navigator.gpu;
    const adapter = this.ready
      ? true
      : webgpu && !!(await navigator.gpu.requestAdapter().catch(() => null));
    return {
      model: this.activeModel,
      activeModel: this.activeModel,
      defaultModel: DEFAULT_MODEL_ID,
      // Status keeps its existing lightweight registry summary; callers that
      // need per-model cache state use models.list().
      availableModels: availableModels().map(({ id, label, modalities, maxContext }) => ({
        id,
        label,
        modalities,
        maxContext,
      })),
      modalities: profile.modalities,
      webgpu,
      adapter,
      cached: await this.probeCache(),
      loaded: this.ready,
      generating: this.generating,
      hasMtp: !!this.modelInstance?.hasMtp,
      batchSize: this.modelInstance?.BATCH ?? 0,
      prefillChunk: this.modelInstance?.chunk ?? 0,
      contextUsedTokens: Math.max(0, this.modelInstance?.pos ?? 0),
      contextMaxTokens: this.modelInstance?.maxCtx ?? 0,
      device: this.deviceInfo,
    };
  }

  async ensureLoaded(options: LoadOptions, onProgress: OnProgress): Promise<void> {
    const model = this.requestedModel(options.model);
    assertModelRegistered(model);
    if (this.ready && this.activeModel === model && !options.reload) return;
    if (this.loadPromise) {
      // A second caller may legitimately request a different registered
      // model while the first model is still downloading.  Wait for the
      // current transition, then re-evaluate rather than silently returning
      // the wrong runtime.
      await this.loadPromise;
      return this.ensureLoaded(options, onProgress);
    }
    this.loadPromise = this.loadModel({ ...options, model }, onProgress).finally(() => {
      this.loadPromise = null;
    });
    return this.loadPromise;
  }

  private async loadModel(options: LoadOptions, onProgress: OnProgress): Promise<void> {
    const target = this.requestedModel(options.model);
    const profile = assertModelRegistered(target);
    this.checkGpu();
    if (this.generating) {
      throw new Error("Cannot load the model while a response is generating.");
    }

    const resolved = resolveLoadOptions(options, target);
    this.mtpEnabled = resolved.mtp;

    this.modelInstance = null;
    this.tokInstance = null;
    this.loader = null;
    this.gpu?.destroy();
    this.gpu = null;
    this.deviceInfo = null;
    this.committed = null;

    onProgress({ stage: "tokenizer", message: `Loading ${profile.label} tokenizer…`, progress: null });
    const gemmaConfig = target === GEMMA_E2B_MODEL_ID ? GEMMA_E2B_CFG : GEMMA_E4B_CFG;
    const tok = isGemmaModelId(target)
      ? await GemmaTokenizer.load((message: string) =>
          onProgress({ stage: "tokenizer", message, progress: null })
        , gemmaConfig)
      : await Tokenizer.load((message: string) =>
          onProgress({ stage: "tokenizer", message, progress: null })
        );

    onProgress({ stage: "gpu", message: "Initializing WebGPU…", progress: null });
    let gpuError: string | null = null;
    const gpu = new GPU();
    let loader: RuntimeLoader;
    let model: RuntimeModel;
    let vram: number;
    try {
      await gpu.init((message: string) => {
        gpuError = message;
      });
      if (gpuError) throw new Error(gpuError);

      loader = isGemmaModelId(target)
        ? new GemmaLoader(gpu, (message: string, phase?: string, frac?: number | null) =>
            onProgress({ stage: phase ?? "weights", message, progress: frac ?? null })
          , gemmaConfig)
        : new Loader(gpu, (message: string, phase?: string, frac?: number | null) =>
            onProgress({ stage: phase ?? "weights", message, progress: frac ?? null })
          );
      const weights = await loader.load();

      onProgress({ stage: "pipelines", message: `Building ${profile.label} pipelines…`, progress: 1 });
      model = isGemmaModelId(target)
        ? new GemmaModel(gpu, weights as GemmaWeightsMap, {
            maxCtx: resolved.maxContext,
            chunk: resolved.prefillChunk,
            config: gemmaConfig,
          })
        : new Model(gpu, weights as WeightsMap, {
            maxCtx: resolved.maxContext,
            chunk: resolved.prefillChunk,
          });
      model.BATCH = resolved.batchSize;
      model.spec = !!model.hasMtp && resolved.mtp;
      await model.reset();
      vram = weightsVramBytes(weights as Record<string, unknown>);
    } catch (error) {
      gpu.destroy();
      throw error;
    }

    this.gpu = gpu;
    this.loader = loader;
    this.tokInstance = tok;
    this.modelInstance = model;
    this.activeModel = target;
    this.deviceInfo = {
      vendor: gpu.info?.vendor ?? "unknown",
      architecture: gpu.info?.architecture ?? "",
      vramBytes: vram,
      subgroups: !!gpu.subgroups,
      subgroupMinSize: typeof gpu.info?.subgroupMinSize === "number"
        ? gpu.info.subgroupMinSize
        : undefined,
      subgroupMaxSize: typeof gpu.info?.subgroupMaxSize === "number"
        ? gpu.info.subgroupMaxSize
        : undefined,
    };
    onProgress({ stage: "ready", message: `${profile.label} ready.`, progress: 1 });
  }

  updateRuntime(options: { batchSize?: number; mtp?: boolean }): void {
    if (options.mtp !== undefined) this.mtpEnabled = !!options.mtp;
    if (!this.modelInstance) return;
    if (options.batchSize !== undefined) {
      this.modelInstance.BATCH = Math.round(clamp(options.batchSize, 1, 8));
    }
    this.modelInstance.spec = !!this.modelInstance.hasMtp && this.mtpEnabled;
  }

  unload(): void {
    if (this.generating) {
      throw new Error("Cannot unload the model while a response is generating.");
    }
    if (this.loadPromise) {
      throw new Error("Cannot unload the model while it is loading.");
    }

    this.gpu?.destroy();
    this.gpu = null;
    this.loader = null;
    this.modelInstance = null;
    this.tokInstance = null;
    this.deviceInfo = null;
    this.mtpEnabled = false;
    this.committed = null;
  }

  async wipeCache(options: { model?: ModelId } = {}): Promise<void> {
    const model = this.requestedModel(options.model);
    assertModelRegistered(model);
    const loader = this.loader && model === this.activeModel
      ? this.loader
      : isGemmaModelId(model)
        ? new GemmaLoader(null, () => {}, model === GEMMA_E2B_MODEL_ID ? GEMMA_E2B_CFG : GEMMA_E4B_CFG)
        : new Loader(null, () => {});
    await loader.clearCache();
  }
}

export const engine = new Engine();
