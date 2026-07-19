#!/usr/bin/env node
// Load the Gemma E4B runtime through the live postMessage API
// and print bounded progress updates. This exercises the actual browser
// tokenizer, GGUF range validation, verified OPFS cache, native GPU uploads,
// and model graph construction.
// Usage: node tools/chrome-gemma-load.mjs [page-url] [cdp-port] [max-context]

const target = process.argv[2] || "http://localhost:3002";
const port = Number(process.argv[3] || 9223);
const maxContext = Number(process.argv[4] || 1024);
const model = "google/gemma-4-E4B";

if (!Number.isInteger(maxContext) || maxContext < 1024) {
  throw new Error("max-context must be an integer of at least 1024.");
}

const pages = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
const page = pages.find((entry) => entry.type === "page");
if (!page?.webSocketDebuggerUrl) throw new Error(`No Chrome page found on CDP port ${port}.`);

const socket = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {
  socket.addEventListener("open", resolve, { once: true });
  socket.addEventListener("error", reject, { once: true });
});
let nextId = 0;
const pending = new Map();
socket.addEventListener("message", ({ data }) => {
  const message = JSON.parse(data);
  const request = pending.get(message.id);
  if (!request) return;
  pending.delete(message.id);
  clearTimeout(request.timer);
  if (message.error) request.reject(new Error(message.error.message));
  else request.resolve(message.result);
});
const send = (method, params = {}, timeoutMs = 30_000) => new Promise((resolve, reject) => {
  const id = ++nextId;
  const timer = setTimeout(() => {
    pending.delete(id);
    reject(new Error(`${method} timed out after ${timeoutMs}ms.`));
  }, timeoutMs);
  pending.set(id, { resolve, reject, timer });
  socket.send(JSON.stringify({ id, method, params }));
});

const evaluate = async (expression, timeoutMs) => {
  const response = await send("Runtime.evaluate", {
    expression,
    awaitPromise: true,
    returnByValue: true,
  }, timeoutMs);
  if (response.exceptionDetails) {
    throw new Error(response.exceptionDetails.text || "Chrome evaluation failed.");
  }
  return response.result?.value;
};

try {
  await send("Page.navigate", { url: target });
  await new Promise((resolve) => setTimeout(resolve, 750));
  await evaluate(`
    (() => {
      const id = "gemma-load-" + Math.random().toString(36).slice(2);
      window.__gemmaLoadState = { events: [], done: false, result: null, error: null };
      const state = window.__gemmaLoadState;
      const onMessage = (event) => {
        const data = event.data;
        if (!data || data.ns !== "mach-llm" || data.id !== id) return;
        if (data.type === "chunk" && data.data?.event === "progress") {
          state.events.push(data.data.progress);
          if (state.events.length > 64) state.events.splice(0, state.events.length - 64);
          return;
        }
        if (data.type === "result") {
          state.done = true;
          state.result = data.data;
          window.removeEventListener("message", onMessage);
          return;
        }
        if (data.type === "error") {
          state.done = true;
          state.error = data.data;
          window.removeEventListener("message", onMessage);
        }
      };
      window.addEventListener("message", onMessage);
      window.postMessage({
        ns: "mach-llm",
        id,
        method: "load",
        params: { model: ${JSON.stringify(model)}, maxContext: ${maxContext}, batchSize: 1, mtp: false },
      }, location.origin);
    })()
  `);

  let last = "";
  const deadline = Date.now() + 3 * 60 * 60 * 1000;
  while (Date.now() < deadline) {
    const state = await evaluate("window.__gemmaLoadState", 30_000);
    const event = state?.events?.at(-1);
    const line = event
      ? `${event.stage}: ${event.message}${event.progress === null || event.progress === undefined ? "" : ` (${Math.round(event.progress * 100)}%)`}`
      : "waiting for Gemma loader…";
    if (line !== last) {
      console.log(line);
      last = line;
    }
    if (state?.done) {
      console.log(JSON.stringify(state, null, 2));
      if (state.error) process.exitCode = 1;
      break;
    }
    await new Promise((resolve) => setTimeout(resolve, 2_000));
  }
  if (Date.now() >= deadline) {
    throw new Error("Gemma load exceeded the 3-hour verification timeout.");
  }
} finally {
  for (const request of pending.values()) clearTimeout(request.timer);
  socket.close();
}
