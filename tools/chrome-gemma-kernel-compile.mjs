#!/usr/bin/env node
// Compile every Gemma 4 E4B WGSL generator in a real Chrome WebGPU device.
//
// Usage:
//   node tools/chrome-gemma-kernel-compile.mjs [page-url] [cdp-port]
//
// Defaults to http://localhost:3002 and Chrome's remote-debugging port 9223.
// The URL is only used as an execution origin: the kernel source is read from
// this checkout and imported into Chrome as a Blob module, so this catches the
// WGSL emitted by the working tree rather than a stale dev-server bundle.

import { readFile } from "node:fs/promises";

const target = process.argv[2] || process.env.GEMMA_KERNEL_TEST_URL || "http://localhost:3002";
const port = Number(process.argv[3] || process.env.GEMMA_CHROME_CDP_PORT || 9223);
const sourceUrl = new URL("../lib/webgpu-llm/gemma-kernels.js", import.meta.url);
const targetOrigin = new URL(target).origin;

if (!Number.isInteger(port) || port < 1 || port > 65535) {
  throw new Error(`Invalid Chrome CDP port: ${port}`);
}

const kernelSource = await readFile(sourceUrl, "utf8");
const kernelSourceBase64 = Buffer.from(kernelSource).toString("base64");

// These are real E4B shapes.  Keep every exported generator represented here:
// the browser-side check reports a missing case when a new generator is added.
// A few generators are intentionally represented more than once where E4B has
// materially different compiled paths (local versus global text attention).
const cases = [
  { name: "gemvQ4", modality: "text", args: { N: 2560, K: 2560 } },
  { name: "gemvQ4", label: "e2b-subgroups", modality: "text", args: { N: 1536, K: 1536, SUBGROUPS: 1 } },
  { name: "gemvQ4", label: "lovelace-direct-subgroup", modality: "text", args: { N: 2560, K: 2560, SUBGROUPS: 1, ROW_LANES: 32, DIRECT_SUBGROUP: 1 } },
  { name: "gemvQ4", label: "e2b-gated", modality: "text", args: { N: 1536, K: 9216, SUBGROUPS: 1, GATED: 1 } },
  { name: "gemmQ4", modality: "text", args: { N: 2560, K: 2560 } },
  { name: "gemmQ4", label: "e2b", modality: "text", args: { N: 12288, K: 1536 } },
  { name: "gatherQ4", modality: "text", args: { START: 0, NUM: 256, K: 2560, SCALE: Math.sqrt(2560) } },
  { name: "rmsnorm", modality: "text", args: { K: 2560 } },
  { name: "rmsnormAdd", modality: "text", args: { K: 2560 } },
  { name: "rmsnormAddScale", label: "e2b", modality: "text", args: { K: 1536 } },
  { name: "layernorm", modality: "audio", args: { K: 1024 } },
  { name: "clearF32", modality: "shared", args: null },
  { name: "add", modality: "shared", args: null },
  { name: "addBias", modality: "audio", args: { K: 1536 } },
  { name: "scale", modality: "shared", args: { SCALE: Math.SQRT1_2 } },
  { name: "geluMul", modality: "text", args: { K: 10240 } },
  { name: "geluMulPair", modality: "text", args: { K: 256 } },
  { name: "pleGateMul", modality: "text", args: { LAYERS: 42, DIM: 256, LAYER: 0 } },
  { name: "softcap", modality: "text", args: { CAP: 30 } },
  { name: "pleCombine", modality: "text", args: { LAYERS: 42, DIM: 256 } },
  { name: "pleCombineMasked", modality: "text", args: { LAYERS: 42, DIM: 256 } },
  { name: "scatterRows", modality: "shared", args: { H: 2560 } },
  {
    name: "textKvPrep",
    label: "sliding",
    modality: "text",
    args: { HEADS: 8, KV_HEADS: 2, HEAD_DIM: 256, ROTARY_PAIRS: 128, MAXCTX: 512 },
  },
  {
    name: "textKvPrep",
    label: "e2b-sliding",
    modality: "text",
    args: { HEADS: 8, KV_HEADS: 1, HEAD_DIM: 256, ROTARY_PAIRS: 128, MAXCTX: 512 },
  },
  {
    name: "textKvPrep",
    label: "global",
    modality: "text",
    args: { HEADS: 8, KV_HEADS: 2, HEAD_DIM: 512, ROTARY_PAIRS: 64, MAXCTX: 512 },
  },
  {
    name: "textQPrep",
    label: "shared-sliding",
    modality: "text",
    args: { HEADS: 8, KV_HEADS: 2, HEAD_DIM: 256, ROTARY_PAIRS: 128, MAXCTX: 512 },
  },
  {
    name: "textQPrep",
    label: "shared-global",
    modality: "text",
    args: { HEADS: 8, KV_HEADS: 2, HEAD_DIM: 512, ROTARY_PAIRS: 64, MAXCTX: 512 },
  },
  {
    name: "textCausalAttention",
    label: "sliding",
    modality: "text",
    args: { HEADS: 8, KV_HEADS: 2, HEAD_DIM: 256, MAXCTX: 512, WINDOW: 512, ATTENTION_SCALE: 1 },
  },
  {
    name: "textCausalAttention",
    label: "e2b-sliding",
    modality: "text",
    args: { HEADS: 8, KV_HEADS: 1, HEAD_DIM: 256, MAXCTX: 512, WINDOW: 512, ATTENTION_SCALE: 1 },
  },
  {
    name: "textCausalAttention",
    label: "global",
    modality: "text",
    args: { HEADS: 8, KV_HEADS: 2, HEAD_DIM: 512, MAXCTX: 512, WINDOW: 0, ATTENTION_SCALE: 1 },
  },
  {
    name: "textFlashAttention",
    label: "sliding",
    modality: "text",
    args: { HEADS: 8, KV_HEADS: 2, HEAD_DIM: 256, MAXCTX: 8192, WINDOW: 512, ATTENTION_SCALE: 1 },
  },
  {
    // E2B's single KV head puts all eight Q heads in one group, compiling the
    // two-batch (vec4 + vec4) head path.
    name: "textFlashAttention",
    label: "e2b-sliding",
    modality: "text",
    args: { HEADS: 8, KV_HEADS: 1, HEAD_DIM: 256, MAXCTX: 8192, WINDOW: 512, ATTENTION_SCALE: 1 },
  },
  {
    name: "textFlashAttention",
    label: "global",
    modality: "text",
    args: { HEADS: 8, KV_HEADS: 2, HEAD_DIM: 512, MAXCTX: 8192, WINDOW: 0, ATTENTION_SCALE: 1 },
  },
  {
    name: "textFlashCombine",
    label: "sliding",
    modality: "text",
    args: { HEADS: 8, HEAD_DIM: 256, MAXCTX: 8192, WINDOW: 512 },
  },
  {
    name: "textFlashCombine",
    label: "global",
    modality: "text",
    args: { HEADS: 8, HEAD_DIM: 512, MAXCTX: 8192, WINDOW: 0 },
  },
  { name: "headRmsnorm", modality: "vision", args: { HEADS: 12, HEAD_DIM: 64 } },
  { name: "vision2DRope", modality: "vision", args: { HEADS: 12, HEAD_DIM: 64, THETA: 100 } },
  { name: "addVisionPositions", modality: "vision", args: { H: 768, TABLE_SIZE: 10240 } },
  { name: "addVisionPositionsQ4", modality: "vision", args: { H: 768, TABLE_SIZE: 10240 } },
  { name: "clampByScalars", modality: "shared", args: null },
  { name: "clampByBounds", label: "input", modality: "shared", args: { OFFSET: 0 } },
  { name: "clampByBounds", label: "output", modality: "shared", args: { OFFSET: 2 } },
  { name: "scaleByScalar", modality: "shared", args: null },
  {
    name: "denseAttention",
    modality: "vision",
    args: { HEADS: 12, KV_HEADS: 12, HEAD_DIM: 64, CAUSAL: 0, WINDOW: 0, ATTENTION_SCALE: 1 },
  },
  { name: "visionPool", modality: "vision", args: { H: 768, POOL: 3, SCALE: Math.sqrt(768) } },
  { name: "relu", modality: "audio", args: null },
  { name: "silu", modality: "audio", args: null },
  { name: "conv2d3x3", modality: "audio", args: { STRIDE: 2, PAD: 1 } },
  { name: "depthwiseConv1d", modality: "audio", args: { C: 1024, K: 5 } },
  { name: "glu", modality: "audio", args: { C: 1024 } },
  { name: "audioQkScale", modality: "audio", args: { HEADS: 8, HEAD_DIM: 128 } },
  { name: "audioRelativePositions", modality: "audio", args: { HIDDEN: 1024, REL: 13 } },
  {
    name: "audioChunkAttention",
    modality: "audio",
    // Gemma config's `attention_context_left=13` includes the current
    // position; the implementation's max past horizon is therefore 12.
    args: { HEADS: 8, HEAD_DIM: 128, CHUNK: 12, LEFT: 12, RIGHT: 0, SOFTCAP: 50 },
  },
];

function formatCdpError(error) {
  return error?.message || JSON.stringify(error);
}

class CdpClient {
  #socket;
  #nextId = 0;
  #pending = new Map();
  #events = new Map();

  constructor(socket) {
    this.#socket = socket;
    socket.addEventListener("message", ({ data }) => {
      const message = JSON.parse(data);
      if (message.id) {
        const pending = this.#pending.get(message.id);
        if (!pending) return;
        this.#pending.delete(message.id);
        clearTimeout(pending.timeout);
        if (message.error) pending.reject(new Error(`${pending.method}: ${formatCdpError(message.error)}`));
        else pending.resolve(message.result);
        return;
      }
      const listeners = this.#events.get(message.method);
      if (!listeners) return;
      for (const listener of [...listeners]) listener(message.params);
    });
    socket.addEventListener("close", () => {
      for (const pending of this.#pending.values()) {
        clearTimeout(pending.timeout);
        pending.reject(new Error("Chrome closed the CDP connection."));
      }
      this.#pending.clear();
    });
  }

  send(method, params = {}, timeoutMs = 30_000) {
    return new Promise((resolve, reject) => {
      const id = ++this.#nextId;
      const timeout = setTimeout(() => {
        this.#pending.delete(id);
        reject(new Error(`${method} timed out after ${timeoutMs}ms.`));
      }, timeoutMs);
      this.#pending.set(id, { method, resolve, reject, timeout });
      this.#socket.send(JSON.stringify({ id, method, params }));
    });
  }

  once(method, timeoutMs = 30_000) {
    return new Promise((resolve, reject) => {
      const listeners = this.#events.get(method) || new Set();
      this.#events.set(method, listeners);
      const listener = (params) => {
        clearTimeout(timeout);
        listeners.delete(listener);
        if (!listeners.size) this.#events.delete(method);
        resolve(params);
      };
      const timeout = setTimeout(() => {
        listeners.delete(listener);
        if (!listeners.size) this.#events.delete(method);
        reject(new Error(`Timed out waiting for ${method}.`));
      }, timeoutMs);
      listeners.add(listener);
    });
  }

  close() {
    this.#socket.close();
  }
}

async function connectCdp(webSocketDebuggerUrl) {
  const socket = new WebSocket(webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    socket.addEventListener("open", resolve, { once: true });
    socket.addEventListener("error", () => reject(new Error("Could not open the Chrome CDP WebSocket.")), { once: true });
  });
  return new CdpClient(socket);
}

async function createIsolatedPage() {
  let response;
  try {
    response = await fetch(`http://127.0.0.1:${port}/json/new?${encodeURIComponent("about:blank")}`, {
      method: "PUT",
    });
  } catch (error) {
    throw new Error(
      `Could not reach Chrome CDP on port ${port}. Launch the isolated Chrome with remote debugging and WebGPU flags. ${formatCdpError(error)}`,
    );
  }
  if (!response.ok) {
    throw new Error(`Chrome CDP refused to create an isolated test page (${response.status} ${response.statusText}).`);
  }
  const page = await response.json();
  if (!page?.id || !page.webSocketDebuggerUrl) {
    throw new Error("Chrome CDP created a page without an inspectable WebSocket target.");
  }
  return page;
}

async function closeIsolatedPage(page) {
  if (!page?.id) return;
  try {
    await fetch(`http://127.0.0.1:${port}/json/close/${encodeURIComponent(page.id)}`);
  } catch {
    // The check is complete; a failed best-effort tab cleanup is not a shader
    // result and should not hide the compilation report.
  }
}

function browserExpression(sourceBase64, testCases) {
  return `
    (async () => {
      const sourceBytes = Uint8Array.from(atob(${JSON.stringify(sourceBase64)}), (character) => character.codePointAt(0));
      const source = new TextDecoder().decode(sourceBytes);
      const blobUrl = URL.createObjectURL(new Blob([source], { type: "text/javascript" }));
      let kernels;
      try {
        kernels = await import(blobUrl);
      } finally {
        URL.revokeObjectURL(blobUrl);
      }

      const exported = Object.entries(kernels)
        .filter(([, value]) => typeof value === "function")
        .map(([name]) => name)
        .sort();
      const cases = ${JSON.stringify(testCases)};
      const covered = new Set(cases.map((testCase) => testCase.name));
      const missingCases = exported.filter((name) => !covered.has(name));
      const staleCases = [...covered].filter((name) => !exported.includes(name));

      const webgpu = !!navigator.gpu;
      const adapter = webgpu ? await navigator.gpu.requestAdapter() : null;
      if (!adapter) {
        return { webgpu, adapter: false, exported, missingCases, staleCases, cases: [] };
      }
      const requiredFeatures = adapter.features.has("subgroups") ? ["subgroups"] : [];
      const device = await adapter.requestDevice({ requiredFeatures });
      const results = [];
      for (const testCase of cases) {
        const generator = kernels[testCase.name];
        const label = testCase.label ? testCase.name + ":" + testCase.label : testCase.name;
        if (typeof generator !== "function") {
          results.push({ label, name: testCase.name, modality: testCase.modality, error: "No matching exported generator" });
          continue;
        }
        try {
          const code = testCase.args === null ? generator() : generator(testCase.args);
          if (typeof code !== "string") throw new Error("Generator did not return WGSL source.");
          device.pushErrorScope("validation");
          const module = device.createShaderModule({ code, label });
          const compilation = await module.getCompilationInfo();
          const validationError = await device.popErrorScope();
          const messages = compilation.messages.map((message) => ({
            type: message.type,
            message: message.message,
            lineNum: message.lineNum,
            linePos: message.linePos,
            offset: message.offset,
            length: message.length,
          }));
          results.push({
            label,
            name: testCase.name,
            modality: testCase.modality,
            args: testCase.args,
            shaderErrors: messages.filter((message) => message.type === "error"),
            warnings: messages.filter((message) => message.type === "warning"),
            validationError: validationError ? validationError.message : null,
          });
        } catch (error) {
          results.push({
            label,
            name: testCase.name,
            modality: testCase.modality,
            args: testCase.args,
            generatorError: error instanceof Error ? error.message : String(error),
          });
        }
      }
      device.destroy();
      return {
        webgpu,
        adapter: true,
        adapterInfo: adapter.info ? { vendor: adapter.info.vendor, architecture: adapter.info.architecture } : null,
        exported,
        missingCases,
        staleCases,
        cases: results,
      };
    })()
  `;
}

const page = await createIsolatedPage();
const cdp = await connectCdp(page.webSocketDebuggerUrl);
try {
  await cdp.send("Page.enable");
  const load = cdp.once("Page.loadEventFired");
  const navigation = await cdp.send("Page.navigate", { url: target });
  if (navigation.errorText) throw new Error(`Could not navigate to ${target}: ${navigation.errorText}`);
  await load;
  const location = await cdp.send("Runtime.evaluate", { expression: "location.href", returnByValue: true });
  const actualUrl = location.result?.value;
  if (!actualUrl || new URL(actualUrl).origin !== targetOrigin) {
    throw new Error(`Chrome did not reach ${target}; current URL is ${actualUrl || "unknown"}.`);
  }

  const evaluation = await cdp.send("Runtime.evaluate", {
    expression: browserExpression(kernelSourceBase64, cases),
    awaitPromise: true,
    returnByValue: true,
    userGesture: true,
  }, 120_000);
  if (evaluation.exceptionDetails) {
    throw new Error(`Chrome evaluation failed: ${evaluation.exceptionDetails.text || "unknown exception"}`);
  }
  const report = evaluation.result?.value;
  if (!report) throw new Error("Chrome returned no Gemma kernel compilation report.");

  console.log(JSON.stringify(report, null, 2));
  const hasShaderErrors = report.cases.some((testCase) =>
    testCase.shaderErrors?.length || testCase.validationError || testCase.generatorError || testCase.error,
  );
  if (!report.webgpu || !report.adapter || report.missingCases.length || report.staleCases.length || hasShaderErrors) {
    process.exitCode = 1;
  }
} finally {
  cdp.close();
  await closeIsolatedPage(page);
}
