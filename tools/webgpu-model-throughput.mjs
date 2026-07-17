#!/usr/bin/env node

// Native-Dawn model throughput benchmark. This runs the production WebGPU
// loaders and model implementations in Node without launching a browser.

import { access, copyFile, mkdir, open, readFile, rename, rm, stat } from "node:fs/promises";
import { homedir } from "node:os";
import path from "node:path";
import { Worker as NodeWorker } from "node:worker_threads";
import { fileURLToPath, pathToFileURL } from "node:url";
import { create, globals } from "webgpu";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const QUANT_WORKER = path.join(ROOT, "public/webgpu-llm/quant-worker.js");
const DEFAULT_CACHE_DIR = path.join(homedir(), ".cache", "mach-compute", "webgpu");

const MODEL_SPECS = {
  qwen: {
    id: "Qwen/Qwen3.5-4B",
    label: "Qwen 3.5 4B",
    maxContext: 65_536,
    batchSize: 8,
    prefillChunk: 256,
    promptToken: 0,
    kind: "qwen",
  },
  "gemma-e4b": {
    id: "google/gemma-4-E4B",
    label: "Gemma 4 E4B",
    maxContext: 131_072,
    batchSize: 8,
    prefillChunk: 64,
    promptToken: 2,
    kind: "gemma",
    configExport: "GEMMA_E4B_CFG",
  },
  "gemma-e2b": {
    id: "google/gemma-4-E2B",
    label: "Gemma 4 E2B",
    maxContext: 131_072,
    batchSize: 8,
    prefillChunk: 128,
    promptToken: 2,
    kind: "gemma",
    configExport: "GEMMA_E2B_CFG",
  },
};

const MODEL_ALIASES = new Map([
  ...Object.entries(MODEL_SPECS).map(([name, spec]) => [name, name]),
  ...Object.entries(MODEL_SPECS).map(([name, spec]) => [spec.id.toLowerCase(), name]),
  ["qwen3.5-4b", "qwen"],
  ["gemma-4-e4b", "gemma-e4b"],
  ["gemma-4-e2b", "gemma-e2b"],
]);

function usage() {
  return `Usage: npm run test:webgpu -- [options]

Runs real model prefill and batched decode through native Dawn/WebGPU.

Options:
  --model <name>          qwen (default), gemma-e2b, or gemma-e4b
  --prompt-tokens <n>     Synthetic prompt length (default: 1024)
  --decode-tokens <n>     Timed generated tokens per run (default: 128)
  --warmup-tokens <n>     Untimed decode warmup tokens (default: 8)
  --runs <n>              Timed runs (default: 3)
  --context <n>           Context capacity (default: enough for this run, min 1024)
  --batch-size <n>        Decode batch size (default: 8)
  --prefill-chunk <n>     Prompt chunk, a multiple of 32 from 32 to 256
  --cache-dir <path>      Persistent quantized-weight cache
                          (default: ${DEFAULT_CACHE_DIR})
  --backend <name>        Dawn backend, e.g. vulkan, metal, or d3d12
  --adapter <name>        Dawn adapter name filter
  --probe                 Initialize the production GPU wrapper and exit
  --help                  Show this help

Environment equivalents: WEBGPU_MODEL, WEBGPU_CACHE_DIR, WEBGPU_BACKEND,
WEBGPU_ADAPTER, WEBGPU_PROMPT_TOKENS, WEBGPU_DECODE_TOKENS,
WEBGPU_WARMUP_TOKENS, WEBGPU_RUNS, WEBGPU_CONTEXT, WEBGPU_BATCH_SIZE,
WEBGPU_PREFILL_CHUNK.`;
}

function parseArgs(argv) {
  const raw = {};
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--help" || arg === "-h") raw.help = true;
    else if (arg === "--probe") raw.probe = true;
    else if (arg.startsWith("--")) {
      const equal = arg.indexOf("=");
      const key = arg.slice(2, equal < 0 ? undefined : equal);
      const value = equal < 0 ? argv[++i] : arg.slice(equal + 1);
      if (value === undefined || value.startsWith("--")) throw new Error(`Missing value for --${key}`);
      raw[key] = value;
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  return raw;
}

function envOr(raw, key, envName, fallback) {
  return raw[key] ?? process.env[envName] ?? fallback;
}

function integer(value, label, min, max) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < min || parsed > max) {
    throw new Error(`${label} must be an integer from ${min} to ${max}`);
  }
  return parsed;
}

function resolveOptions(raw) {
  const requestedModel = String(envOr(raw, "model", "WEBGPU_MODEL", "qwen")).toLowerCase();
  const modelName = MODEL_ALIASES.get(requestedModel);
  if (!modelName) throw new Error(`Unknown model: ${requestedModel}`);
  const model = MODEL_SPECS[modelName];
  const promptTokens = integer(envOr(raw, "prompt-tokens", "WEBGPU_PROMPT_TOKENS", 1024), "prompt tokens", 1, model.maxContext - 2);
  const decodeTokens = integer(envOr(raw, "decode-tokens", "WEBGPU_DECODE_TOKENS", 128), "decode tokens", 1, 4096);
  const warmupTokens = integer(envOr(raw, "warmup-tokens", "WEBGPU_WARMUP_TOKENS", 8), "warmup tokens", 0, 256);
  const runs = integer(envOr(raw, "runs", "WEBGPU_RUNS", 3), "runs", 1, 10);
  const minContext = Math.max(1024, promptTokens + decodeTokens + 1, Math.min(promptTokens, model.prefillChunk) + warmupTokens + 1);
  const context = integer(envOr(raw, "context", "WEBGPU_CONTEXT", minContext), "context", minContext, model.maxContext);
  const batchSize = integer(envOr(raw, "batch-size", "WEBGPU_BATCH_SIZE", model.batchSize), "batch size", 1, 8);
  const prefillChunk = integer(envOr(raw, "prefill-chunk", "WEBGPU_PREFILL_CHUNK", model.prefillChunk), "prefill chunk", 32, 256);
  if (prefillChunk % 32 !== 0) throw new Error("prefill chunk must be a multiple of 32");

  return {
    modelName,
    model,
    promptTokens,
    decodeTokens,
    warmupTokens,
    runs,
    context,
    batchSize,
    prefillChunk,
    cacheDir: path.resolve(String(envOr(raw, "cache-dir", "WEBGPU_CACHE_DIR", DEFAULT_CACHE_DIR))),
    backend: envOr(raw, "backend", "WEBGPU_BACKEND", null),
    adapter: envOr(raw, "adapter", "WEBGPU_ADAPTER", null),
    probe: !!raw.probe,
  };
}

function safeChild(parent, name) {
  if (typeof name !== "string" || !name || name === "." || name === ".." || name.includes("/") || name.includes("\\")) {
    throw new TypeError(`Invalid filesystem entry name: ${String(name)}`);
  }
  return path.join(parent, name);
}

async function exists(target) {
  try {
    await access(target);
    return true;
  } catch {
    return false;
  }
}

function notFound(target) {
  return new DOMException(`No such file or directory: ${target}`, "NotFoundError");
}

class NodeFileSlice {
  constructor(filename, start, end) {
    this.filename = filename;
    this.start = start;
    this.end = end;
    this.size = Math.max(0, end - start);
  }

  async arrayBuffer() {
    const output = new Uint8Array(this.size);
    const handle = await open(this.filename, "r");
    try {
      let offset = 0;
      while (offset < output.byteLength) {
        const { bytesRead } = await handle.read(output, offset, output.byteLength - offset, this.start + offset);
        if (!bytesRead) break;
        offset += bytesRead;
      }
      if (offset !== output.byteLength) throw new Error(`Short read from ${this.filename}: ${offset} / ${output.byteLength}`);
      return output.buffer;
    } finally {
      await handle.close();
    }
  }

  async text() {
    return new TextDecoder().decode(await this.arrayBuffer());
  }
}

class NodeFile {
  constructor(filename, size) {
    this.filename = filename;
    this.size = size;
    this.name = path.basename(filename);
    this.type = "";
  }

  slice(start = 0, end = this.size) {
    const from = Math.min(this.size, Math.max(0, Number(start) || 0));
    const to = Math.min(this.size, Math.max(from, end === undefined ? this.size : Number(end) || 0));
    return new NodeFileSlice(this.filename, from, to);
  }

  async arrayBuffer() {
    return this.slice().arrayBuffer();
  }

  async text() {
    return readFile(this.filename, "utf8");
  }
}

function bytesOf(data) {
  if (typeof data === "string") return new TextEncoder().encode(data);
  if (data instanceof ArrayBuffer) return new Uint8Array(data);
  if (ArrayBuffer.isView(data)) return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  throw new TypeError(`Unsupported writable data: ${Object.prototype.toString.call(data)}`);
}

let writableId = 0;

class NodeWritableFileStream {
  constructor(filename, temporary, handle) {
    this.filename = filename;
    this.temporary = temporary;
    this.handle = handle;
    this.position = 0;
    this.closed = false;
  }

  static async create(filename, keepExistingData) {
    await mkdir(path.dirname(filename), { recursive: true });
    const temporary = `${filename}.tmp-${process.pid}-${++writableId}`;
    if (keepExistingData && await exists(filename)) await copyFile(filename, temporary);
    const handle = await open(temporary, keepExistingData && await exists(temporary) ? "r+" : "w+");
    return new NodeWritableFileStream(filename, temporary, handle);
  }

  async write(value) {
    if (this.closed) throw new TypeError("Writable stream is closed");
    let data = value;
    let position = this.position;
    if (value && typeof value === "object" && value.type === "write") {
      data = value.data;
      position = Number(value.position);
    }
    if (!Number.isSafeInteger(position) || position < 0) throw new TypeError(`Invalid write position: ${position}`);
    const bytes = bytesOf(data);
    let offset = 0;
    while (offset < bytes.byteLength) {
      const { bytesWritten } = await this.handle.write(bytes, offset, bytes.byteLength - offset, position + offset);
      if (!bytesWritten) throw new Error(`Short write to ${this.temporary}`);
      offset += bytesWritten;
    }
    this.position = position + bytes.byteLength;
  }

  async seek(position) {
    if (!Number.isSafeInteger(position) || position < 0) throw new TypeError(`Invalid seek position: ${position}`);
    this.position = position;
  }

  async close() {
    if (this.closed) return;
    this.closed = true;
    await this.handle.close();
    await rename(this.temporary, this.filename);
  }

  async abort() {
    if (!this.closed) {
      this.closed = true;
      await this.handle.close().catch(() => {});
    }
    await rm(this.temporary, { force: true }).catch(() => {});
  }
}

class NodeFileHandle {
  constructor(filename) {
    this.filename = filename;
    this.kind = "file";
    this.name = path.basename(filename);
  }

  async getFile() {
    let info;
    try {
      info = await stat(this.filename);
    } catch {
      throw notFound(this.filename);
    }
    if (!info.isFile()) throw notFound(this.filename);
    return new NodeFile(this.filename, info.size);
  }

  async createWritable(options = {}) {
    return NodeWritableFileStream.create(this.filename, !!options.keepExistingData);
  }
}

class NodeDirectoryHandle {
  constructor(dirname) {
    this.dirname = dirname;
    this.kind = "directory";
    this.name = path.basename(dirname);
  }

  async getDirectoryHandle(name, options = {}) {
    const target = safeChild(this.dirname, name);
    if (options.create) await mkdir(target, { recursive: true });
    else {
      try {
        if (!(await stat(target)).isDirectory()) throw notFound(target);
      } catch {
        throw notFound(target);
      }
    }
    return new NodeDirectoryHandle(target);
  }

  async getFileHandle(name, options = {}) {
    const target = safeChild(this.dirname, name);
    if (options.create) {
      await mkdir(path.dirname(target), { recursive: true });
      const handle = await open(target, "a");
      await handle.close();
    } else {
      try {
        if (!(await stat(target)).isFile()) throw notFound(target);
      } catch {
        throw notFound(target);
      }
    }
    return new NodeFileHandle(target);
  }

  async removeEntry(name, options = {}) {
    const target = safeChild(this.dirname, name);
    if (!await exists(target)) throw notFound(target);
    await rm(target, { recursive: !!options.recursive, force: false });
  }
}

const WORKER_BOOTSTRAP = String.raw`
const { parentPort, workerData } = require("node:worker_threads");
const { readFileSync } = require("node:fs");
const vm = require("node:vm");
globalThis.postMessage = (data, transfer) => parentPort.postMessage(data, transfer);
vm.runInThisContext(readFileSync(workerData.script, "utf8"), { filename: workerData.script });
parentPort.on("message", (data) => globalThis.onmessage({ data }));
`;

class BrowserWorker {
  constructor(specifier) {
    const filename = specifier === "/webgpu-llm/quant-worker.js"
      ? QUANT_WORKER
      : specifier instanceof URL && specifier.protocol === "file:"
        ? fileURLToPath(specifier)
        : null;
    if (!filename) throw new Error(`Unsupported worker URL in native benchmark: ${String(specifier)}`);
    this.worker = new NodeWorker(WORKER_BOOTSTRAP, { eval: true, workerData: { script: filename } });
    this.worker.on("message", (data) => this.onmessage?.({ data }));
    this.worker.on("error", (error) => this.onerror?.({ message: error.message, error }));
  }

  postMessage(data, transfer) {
    this.worker.postMessage(data, transfer);
  }

  terminate() {
    return this.worker.terminate();
  }
}

function installNodePlatform(cacheDir) {
  const navigatorObject = globalThis.navigator ?? {};
  if (!globalThis.navigator) Object.defineProperty(globalThis, "navigator", { value: navigatorObject, configurable: true });
  const storageDescriptor = Object.getOwnPropertyDescriptor(navigatorObject, "storage");
  const workerDescriptor = Object.getOwnPropertyDescriptor(globalThis, "Worker");
  const root = new NodeDirectoryHandle(cacheDir);
  Object.defineProperty(navigatorObject, "storage", {
    configurable: true,
    value: {
      persist: async () => true,
      getDirectory: async () => {
        await mkdir(cacheDir, { recursive: true });
        return root;
      },
    },
  });
  Object.defineProperty(globalThis, "Worker", { configurable: true, writable: true, value: BrowserWorker });

  return () => {
    if (storageDescriptor) Object.defineProperty(navigatorObject, "storage", storageDescriptor);
    else delete navigatorObject.storage;
    if (workerDescriptor) Object.defineProperty(globalThis, "Worker", workerDescriptor);
    else delete globalThis.Worker;
  };
}

function dawnOptions(options) {
  const out = [];
  if (options.backend) out.push(`backend=${options.backend}`);
  if (options.adapter) out.push(`adapter=${options.adapter}`);
  return out;
}

function installNativeWebGPU(options) {
  Object.assign(globalThis, globals);
  const navigatorObject = globalThis.navigator ?? {};
  if (!globalThis.navigator) Object.defineProperty(globalThis, "navigator", { value: navigatorObject, configurable: true });
  const gpuDescriptor = Object.getOwnPropertyDescriptor(navigatorObject, "gpu");
  let nativeGpu = create(dawnOptions(options));
  Object.defineProperty(navigatorObject, "gpu", { configurable: true, value: nativeGpu });
  return () => {
    if (gpuDescriptor) Object.defineProperty(navigatorObject, "gpu", gpuDescriptor);
    else delete navigatorObject.gpu;
    nativeGpu = null;
  };
}

async function probeNodePlatform() {
  const root = await navigator.storage.getDirectory();
  const dirname = `.native-webgpu-probe-${process.pid}`;
  try {
    const directory = await root.getDirectoryHandle(dirname, { create: true });
    const handle = await directory.getFileHandle("probe.bin", { create: true });
    const writable = await handle.createWritable();
    await writable.write(new Uint8Array([3, 1, 4, 1, 5]));
    await writable.close();
    const file = await handle.getFile();
    const bytes = new Uint8Array(await file.slice(1, 4).arrayBuffer());
    if (bytes.length !== 3 || bytes[0] !== 1 || bytes[1] !== 4 || bytes[2] !== 1) {
      throw new Error("Filesystem cache bridge returned incorrect bytes");
    }
  } finally {
    await root.removeEntry(dirname, { recursive: true }).catch(() => {});
  }

  const worker = new Worker("/webgpu-llm/quant-worker.js");
  try {
    const result = await new Promise((resolve, reject) => {
      worker.onmessage = ({ data }) => resolve(data);
      worker.onerror = ({ error, message }) => reject(error ?? new Error(message));
      worker.postMessage({ id: 1, u16: new Uint16Array(32).buffer, rows: 1, K: 32 });
    });
    if (result.id !== 1 || result.qdata.byteLength !== 16 || result.scales.byteLength !== 4) {
      throw new Error("Quantization worker bridge returned an invalid layout");
    }
  } finally {
    await worker.terminate();
  }
  return { filesystemCache: true, quantizationWorker: true };
}

function selectedLimits(limits) {
  return {
    maxBufferSize: limits.maxBufferSize,
    maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
    maxComputeWorkgroupStorageSize: limits.maxComputeWorkgroupStorageSize,
    maxStorageBuffersPerShaderStage: limits.maxStorageBuffersPerShaderStage,
    maxComputeInvocationsPerWorkgroup: limits.maxComputeInvocationsPerWorkgroup,
  };
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function greedyCandidate(candidates) {
  if (!candidates?.ids?.length || !candidates?.vals?.length) throw new Error("Model returned no next-token candidates");
  let best = 0;
  for (let i = 1; i < candidates.ids.length; i++) {
    if (candidates.vals[i] > candidates.vals[best]) best = i;
  }
  return Number(candidates.ids[best]);
}

const sampling = {
  temperature: 0,
  topP: 1,
  topK: 1,
  presencePenalty: 0,
  stopIds: [],
  seed: 1,
};

async function runDecode(model, firstToken, tokens, batchSize) {
  let next = firstToken;
  let produced = 0;
  let submissions = 0;
  while (produced < tokens) {
    const count = Math.min(batchSize, tokens - produced);
    const result = await model.decodeBatch(next, count, sampling);
    if (!result.ids.length) throw new Error("Decode batch produced no tokens");
    produced += result.ids.length;
    submissions++;
    next = result.ids[result.ids.length - 1];
  }
  return { produced, submissions, next };
}

async function benchmarkRun(model, options, prompt) {
  await model.reset();
  const prefillStart = performance.now();
  const candidates = await model.prefill(prompt);
  const prefillEnd = performance.now();
  const firstToken = greedyCandidate(candidates);
  const decodeStart = performance.now();
  const decode = await runDecode(model, firstToken, options.decodeTokens, options.batchSize);
  const end = performance.now();
  const prefillSeconds = (prefillEnd - prefillStart) / 1000;
  const decodeSeconds = (end - decodeStart) / 1000;
  return {
    promptTokens: prompt.length,
    prefillSeconds,
    promptTokensPerSecond: prompt.length / prefillSeconds,
    decodeTokens: decode.produced,
    decodeSubmissions: decode.submissions,
    decodeSeconds,
    tokensPerSecond: decode.produced / decodeSeconds,
    millisecondsPerToken: decodeSeconds * 1000 / decode.produced,
    totalSeconds: (end - prefillStart) / 1000,
  };
}

function makeProgressReporter() {
  let lastAt = 0;
  let lastStage = null;
  let lastBucket = -1;
  return (message, stage = "load", fraction = null) => {
    const now = performance.now();
    const bucket = Number.isFinite(fraction) ? Math.floor(fraction * 20) : -1;
    if (stage !== lastStage || bucket !== lastBucket || now - lastAt > 1000) {
      process.stderr.write(`[${stage}] ${message}\n`);
      lastAt = now;
      lastStage = stage;
      lastBucket = bucket;
    }
  };
}

async function loadRuntime(gpu, options, report) {
  if (options.model.kind === "qwen") {
    const [{ Loader }, { Model }] = await Promise.all([
      import(pathToFileURL(path.join(ROOT, "lib/webgpu-llm/loader.js"))),
      import(pathToFileURL(path.join(ROOT, "lib/webgpu-llm/model.js"))),
    ]);
    const weights = await new Loader(gpu, report).load();
    const model = new Model(gpu, weights, { maxCtx: options.context, chunk: options.prefillChunk });
    model.BATCH = options.batchSize;
    model.spec = false;
    await model.reset();
    return model;
  }

  const [{ GemmaLoader }, { GemmaModel }, configs] = await Promise.all([
    import(pathToFileURL(path.join(ROOT, "lib/webgpu-llm/gemma-loader.js"))),
    import(pathToFileURL(path.join(ROOT, "lib/webgpu-llm/gemma-model.js"))),
    import(pathToFileURL(path.join(ROOT, "lib/webgpu-llm/gemma-config.js"))),
  ]);
  const config = configs[options.model.configExport];
  const weights = await new GemmaLoader(gpu, report, config).load();
  const model = new GemmaModel(gpu, weights, {
    maxCtx: options.context,
    chunk: options.prefillChunk,
    config,
  });
  model.BATCH = options.batchSize;
  await model.reset();
  return model;
}

async function main() {
  const raw = parseArgs(process.argv.slice(2));
  if (raw.help) {
    console.log(usage());
    return;
  }
  const options = resolveOptions(raw);
  const restorePlatform = installNodePlatform(options.cacheDir);
  const restoreWebGPU = installNativeWebGPU(options);
  let gpu = null;
  try {
    const { GPU } = await import(pathToFileURL(path.join(ROOT, "lib/webgpu-llm/gpu.js")));
    gpu = await new GPU().init((message, kind) => process.stderr.write(`[gpu:${kind ?? "status"}] ${message}\n`));
    const adapter = {
      vendor: gpu.info?.vendor ?? "unknown",
      architecture: gpu.info?.architecture ?? "",
      device: gpu.info?.device ?? "",
      description: gpu.info?.description ?? "",
      subgroups: gpu.subgroups,
      subgroupMinSize: gpu.info?.subgroupMinSize ?? null,
      subgroupMaxSize: gpu.info?.subgroupMaxSize ?? null,
      packed4x8IntegerDotProduct: gpu.dp4a,
      backend: options.backend ?? "auto",
    };
    if (options.probe) {
      const platform = await probeNodePlatform();
      console.log(JSON.stringify({ ok: true, adapter, limits: selectedLimits(gpu.limits), platform }, null, 2));
      return;
    }

    process.stderr.write(`Loading ${options.model.label}; cache: ${options.cacheDir}\n`);
    const loadStart = performance.now();
    const model = await loadRuntime(gpu, options, makeProgressReporter());
    const loadSeconds = (performance.now() - loadStart) / 1000;
    const prompt = new Array(options.promptTokens).fill(options.model.promptToken);

    if (options.warmupTokens) {
      process.stderr.write(`Warming up ${options.warmupTokens} decode tokens...\n`);
      await model.reset();
      const warmPrompt = prompt.slice(0, Math.min(prompt.length, options.prefillChunk));
      const candidates = await model.prefill(warmPrompt);
      await runDecode(model, greedyCandidate(candidates), options.warmupTokens, options.batchSize);
    }

    const runs = [];
    for (let i = 0; i < options.runs; i++) {
      process.stderr.write(`Benchmark run ${i + 1}/${options.runs}...\n`);
      const result = await benchmarkRun(model, options, prompt);
      runs.push(result);
      process.stderr.write(
        `  prefill ${result.promptTokensPerSecond.toFixed(1)} tok/s; decode ${result.tokensPerSecond.toFixed(2)} tok/s\n`,
      );
    }

    const output = {
      benchmark: "native-dawn-model-throughput",
      model: { id: options.model.id, label: options.model.label },
      adapter,
      parameters: {
        promptTokens: options.promptTokens,
        decodeTokens: options.decodeTokens,
        warmupTokens: options.warmupTokens,
        runs: options.runs,
        context: options.context,
        batchSize: options.batchSize,
        prefillChunk: options.prefillChunk,
      },
      cacheDir: options.cacheDir,
      loadSeconds,
      summary: {
        promptTokensPerSecond: median(runs.map((run) => run.promptTokensPerSecond)),
        tokensPerSecond: median(runs.map((run) => run.tokensPerSecond)),
        millisecondsPerToken: median(runs.map((run) => run.millisecondsPerToken)),
      },
      runs,
    };
    console.log(JSON.stringify(output, null, 2));
  } finally {
    gpu?.destroy();
    gpu = null;
    restoreWebGPU();
    restorePlatform();
  }
}

main().catch((error) => {
  process.stderr.write(`${error?.stack ?? error}\n`);
  process.exitCode = 1;
});
