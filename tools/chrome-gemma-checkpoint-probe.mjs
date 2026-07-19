#!/usr/bin/env node
// Verify the Gemma E4B checkpoint endpoint from the same Chrome origin used
// by the WebGPU runtime, without reading the multi-gigabyte response body.
// Usage: node tools/chrome-gemma-checkpoint-probe.mjs [page-url] [cdp-port]

const target = process.argv[2] || "http://localhost:3002";
const port = Number(process.argv[3] || 9223);
const revision = "06f24bb269339b2a19a5167199b81e89ef813c10";
const checkpoint = `https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/${revision}/gemma-4-E4B-it-Q4_0.gguf`;

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
  const resolve = pending.get(message.id);
  if (!resolve) return;
  pending.delete(message.id);
  resolve(message);
});
const send = (method, params = {}) => new Promise((resolve) => {
  const id = ++nextId;
  pending.set(id, resolve);
  socket.send(JSON.stringify({ id, method, params }));
});

try {
  await send("Page.navigate", { url: target });
  await new Promise((resolve) => setTimeout(resolve, 500));
  const response = await send("Runtime.evaluate", {
    awaitPromise: true,
    returnByValue: true,
    expression: `
      (async () => {
        const url = ${JSON.stringify(checkpoint)};
        try {
          const response = await fetch(url, {
            headers: { Range: "bytes=0-7" },
            credentials: "omit",
          });
          const report = {
            url: response.url,
            type: response.type,
            status: response.status,
            ok: response.ok,
            contentRange: response.headers.get("content-range"),
            contentLength: response.headers.get("content-length"),
            acceptsRanges: response.headers.get("accept-ranges"),
          };
          await response.body?.cancel();
          return report;
        } catch (error) {
          return { error: error instanceof Error ? error.message : String(error) };
        }
      })()
    `,
  });
  if (response.error) throw new Error(response.error.message);
  if (response.result?.exceptionDetails) {
    throw new Error(response.result.exceptionDetails.text || "Chrome evaluation failed.");
  }
  const value = response.result?.result?.value;
  if (!value) throw new Error("Chrome returned no checkpoint probe result.");
  console.log(JSON.stringify(value, null, 2));
  if (value.status !== 206 || !value.contentRange) process.exitCode = 1;
} finally {
  socket.close();
}
