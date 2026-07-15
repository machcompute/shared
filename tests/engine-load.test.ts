import { beforeEach, describe, expect, it, vi } from "vitest";

const controls = vi.hoisted(() => ({
  gpuInit: vi.fn(),
  gpuDestroy: vi.fn(),
  loaderLoad: vi.fn(),
  tokenizerLoad: vi.fn(),
  modelReset: vi.fn(),
  modelConstruct: vi.fn(),
}));

vi.mock("../lib/webgpu-llm/gpu.js", () => ({
  GPU: class GPU {
    info = { vendor: "test", architecture: "fake" };
    async init() {
      await controls.gpuInit();
      return this;
    }
    destroy() {
      controls.gpuDestroy();
    }
  },
}));

vi.mock("../lib/webgpu-llm/tokenizer.js", () => ({
  Tokenizer: class Tokenizer {
    static load(...args: unknown[]) {
      return controls.tokenizerLoad(...args);
    }
  },
}));

vi.mock("../lib/webgpu-llm/loader.js", () => ({
  Loader: class Loader {
    load() {
      return controls.loaderLoad();
    }
    async cacheValid() {
      return false;
    }
    async clearCache() {}
  },
}));

vi.mock("../lib/webgpu-llm/model.js", () => ({
  Model: class Model {
    BATCH = 1;
    hasMtp = false;
    spec = false;
    pos = 0;
    maxCtx = 1024;
    chunk = 256;
    constructor(_gpu: unknown, _weights: unknown, options: { chunk?: number }) {
      this.chunk = options.chunk ?? 256;
      controls.modelConstruct(options);
    }
    reset() {
      return controls.modelReset();
    }
  },
}));

import { Engine } from "../lib/engine";
import { QWEN_MODEL_ID } from "../lib/webgpu-llm/model-registry";

describe("Engine model-load cleanup", () => {
  beforeEach(() => {
    vi.stubGlobal("navigator", { gpu: { requestAdapter: vi.fn().mockResolvedValue({}) } });
    controls.gpuInit.mockReset().mockResolvedValue(undefined);
    controls.gpuDestroy.mockReset();
    controls.loaderLoad.mockReset().mockResolvedValue({});
    controls.tokenizerLoad.mockReset().mockResolvedValue({});
    controls.modelReset.mockReset().mockResolvedValue(undefined);
    controls.modelConstruct.mockReset();
  });

  it("destroys a partially initialized GPU when device initialization fails", async () => {
    controls.gpuInit.mockRejectedValueOnce(new Error("init failed"));
    const engine = new Engine();
    await expect(engine.ensureLoaded({ model: QWEN_MODEL_ID }, () => {})).rejects.toThrow("init failed");
    expect(controls.gpuDestroy).toHaveBeenCalledOnce();
    expect(engine.ready).toBe(false);
  });

  it("destroys downloaded allocations when weight loading fails", async () => {
    controls.loaderLoad.mockRejectedValueOnce(new Error("weights failed"));
    const engine = new Engine();
    await expect(engine.ensureLoaded({ model: QWEN_MODEL_ID }, () => {})).rejects.toThrow("weights failed");
    expect(controls.gpuDestroy).toHaveBeenCalledOnce();
    expect(engine.ready).toBe(false);
  });

  it("destroys model allocations when pipeline reset fails", async () => {
    controls.modelReset.mockRejectedValueOnce(new Error("reset failed"));
    const engine = new Engine();
    await expect(engine.ensureLoaded({ model: QWEN_MODEL_ID }, () => {})).rejects.toThrow("reset failed");
    expect(controls.gpuDestroy).toHaveBeenCalledOnce();
    expect(engine.ready).toBe(false);
  });

  it("publishes resources only after a fully successful load", async () => {
    const engine = new Engine();
    await engine.ensureLoaded({ model: QWEN_MODEL_ID }, () => {});
    expect(engine.ready).toBe(true);
    expect(controls.gpuDestroy).not.toHaveBeenCalled();
  });

  it("normalizes prefill chunks to the 32-row kernel tile", async () => {
    const engine = new Engine();
    await engine.ensureLoaded({ model: QWEN_MODEL_ID, prefillChunk: 70 }, () => {});

    expect(controls.modelConstruct).toHaveBeenCalledWith(expect.objectContaining({ chunk: 64 }));
    await expect(engine.status()).resolves.toMatchObject({ prefillChunk: 64 });
  });

  it("unloads a ready model and releases its GPU", async () => {
    const engine = new Engine();
    await engine.ensureLoaded({ model: QWEN_MODEL_ID }, () => {});

    engine.unload();

    expect(engine.ready).toBe(false);
    expect(controls.gpuDestroy).toHaveBeenCalledOnce();
    await expect(engine.status()).resolves.toMatchObject({
      loaded: false,
      device: null,
      hasMtp: false,
      contextUsedTokens: 0,
      contextMaxTokens: 0,
    });
  });

  it("does not unload while generation is active", async () => {
    const engine = new Engine();
    await engine.ensureLoaded({ model: QWEN_MODEL_ID }, () => {});
    engine.generating = true;

    expect(() => engine.unload()).toThrow("while a response is generating");
    expect(engine.ready).toBe(true);
    expect(controls.gpuDestroy).not.toHaveBeenCalled();
  });
});
