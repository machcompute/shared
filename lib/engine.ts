import { GPU } from "./webgpu-llm/gpu.js";
import { Loader } from "./webgpu-llm/loader.js";
import { Model } from "./webgpu-llm/model.js";
import { Tokenizer } from "./webgpu-llm/tokenizer.js";
import { CFG, RT } from "./webgpu-llm/config.js";

export interface ProgressEvent {
  stage: string;
  message: string;
  progress: number | null;
}

export type OnProgress = (event: ProgressEvent) => void;

export interface LoadOptions {
  maxContext?: number;
  batchSize?: number;
  mtp?: boolean;
  reload?: boolean;
}

export interface DeviceInfo {
  vendor: string;
  architecture: string;
  vramBytes: number;
}

export interface EngineStatus {
  model: string;
  webgpu: boolean;
  adapter: boolean;
  cached: boolean | null;
  loaded: boolean;
  generating: boolean;
  hasMtp: boolean;
  contextUsedTokens: number;
  contextMaxTokens: number;
  device: DeviceInfo | null;
}

const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

function resolveLoadOptions(options: LoadOptions) {
  return {
    maxContext: Math.round(clamp(options.maxContext ?? RT.maxCtx, 1024, RT.maxCtx)),
    batchSize: Math.round(clamp(options.batchSize ?? 8, 1, 8)),
    mtp: options.mtp ?? false,
  };
}

class Engine {
  private gpu: GPU | null = null;
  private loader: Loader | null = null;
  private modelInstance: Model | null = null;
  private tokInstance: Tokenizer | null = null;
  private loadPromise: Promise<void> | null = null;
  private deviceInfo: DeviceInfo | null = null;
  private mtpEnabled = false;

  generating = false;
  committedSigs: string[] | null = null;

  get ready() {
    return !!this.modelInstance && !!this.tokInstance;
  }

  get model(): Model {
    if (!this.modelInstance) throw new Error("Model is not loaded");
    return this.modelInstance;
  }

  get tok(): Tokenizer {
    if (!this.tokInstance) throw new Error("Model is not loaded");
    return this.tokInstance;
  }

  checkGpu(): void {
    if (typeof navigator === "undefined" || !navigator.gpu) {
      throw new Error(
        'WebGPU is not available in this browser. Use Chrome/Edge 121+ (on Linux you may need chrome://flags → "Vulkan" + "Unsafe WebGPU").'
      );
    }
  }

  async probeCache(): Promise<boolean | null> {
    try {
      const probe = new Loader(null, () => {});
      return !!(await probe.cacheValid());
    } catch {
      return null;
    }
  }

  async status(): Promise<EngineStatus> {
    const webgpu = typeof navigator !== "undefined" && !!navigator.gpu;
    const adapter = this.ready
      ? true
      : webgpu && !!(await navigator.gpu.requestAdapter().catch(() => null));
    return {
      model: CFG.repo,
      webgpu,
      adapter,
      cached: await this.probeCache(),
      loaded: this.ready,
      generating: this.generating,
      hasMtp: !!this.modelInstance?.hasMtp,
      contextUsedTokens: Math.max(0, this.modelInstance?.pos ?? 0),
      contextMaxTokens: this.modelInstance?.maxCtx ?? 0,
      device: this.deviceInfo,
    };
  }

  async ensureLoaded(options: LoadOptions, onProgress: OnProgress): Promise<void> {
    if (this.ready && !options.reload) return;
    if (!this.loadPromise) {
      this.loadPromise = this.loadModel(options, onProgress).finally(() => {
        this.loadPromise = null;
      });
    }
    return this.loadPromise;
  }

  private async loadModel(options: LoadOptions, onProgress: OnProgress): Promise<void> {
    this.checkGpu();
    if (this.generating) {
      throw new Error("Cannot load the model while a response is generating.");
    }

    const resolved = resolveLoadOptions(options);
    this.mtpEnabled = resolved.mtp;

    this.modelInstance = null;
    this.tokInstance = null;
    this.loader = null;
    this.gpu?.destroy();
    this.gpu = null;
    this.deviceInfo = null;
    this.committedSigs = null;

    onProgress({ stage: "tokenizer", message: "Loading tokenizer…", progress: null });
    const tok = await Tokenizer.load((message: string) =>
      onProgress({ stage: "tokenizer", message, progress: null })
    );

    onProgress({ stage: "gpu", message: "Initializing WebGPU…", progress: null });
    let gpuError: string | null = null;
    const gpu = await new GPU().init((message: string) => {
      gpuError = message;
    });
    if (gpuError) throw new Error(gpuError);

    this.loader = new Loader(gpu, (message: string, phase?: string, frac?: number | null) =>
      onProgress({ stage: phase ?? "weights", message, progress: frac ?? null })
    );
    const weights = await this.loader.load();

    onProgress({ stage: "pipelines", message: "Building pipelines…", progress: 1 });
    const model = new Model(gpu, weights, { maxCtx: resolved.maxContext });
    model.BATCH = resolved.batchSize;
    model.spec = !!model.hasMtp && resolved.mtp;
    await model.reset();

    type BufLike = { size: number };
    type WeightEntry = { q?: BufLike; s?: BufLike; size?: number };
    let vram = 0;
    for (const w of Object.values(weights) as WeightEntry[]) {
      if (w?.q && w.s) vram += w.q.size + w.s.size;
      else if (w?.size) vram += w.size;
    }
    const embShards = (weights as Record<string, unknown>).embShards;
    if (Array.isArray(embShards)) {
      for (const sh of embShards as Array<{ q: BufLike; s: BufLike }>) {
        vram += sh.q.size + sh.s.size;
      }
    }

    this.gpu = gpu;
    this.tokInstance = tok;
    this.modelInstance = model;
    this.deviceInfo = {
      vendor: gpu.info?.vendor ?? "unknown",
      architecture: gpu.info?.architecture ?? "",
      vramBytes: vram,
    };
    onProgress({ stage: "ready", message: "Model ready.", progress: 1 });
  }

  async wipeCache(): Promise<void> {
    const loader = this.loader ?? new Loader(null, () => {});
    await loader.clearCache();
  }
}

export const engine = new Engine();
