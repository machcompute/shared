#!/usr/bin/env node
// Query a live postMessage method in the isolated Chrome/WebGPU test page.
// Usage: node tools/chrome-protocol-status.mjs [page-url] [cdp-port] [method]

const target = process.argv[2] || "http://localhost:3002";
const port = Number(process.argv[3] || 9223);
const protocolMethod = process.argv[4] || "status";
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
  if (resolve) {
    pending.delete(message.id);
    resolve(message);
  }
});
const send = (method, params = {}) => new Promise((resolve) => {
  const id = ++nextId;
  pending.set(id, resolve);
  socket.send(JSON.stringify({ id, method, params }));
});

try {
  await send("Page.navigate", { url: target });
  // Navigation is asynchronous. Starting evaluation in the old document can
  // resolve as `undefined` when Chrome tears it down, so let the target page
  // mount its client-side protocol host first.
  await new Promise((resolve) => setTimeout(resolve, 900));
  const response = await send("Runtime.evaluate", {
    awaitPromise: true,
    returnByValue: true,
    expression: `
      (async () => {
        await new Promise((resolve) => setTimeout(resolve, 300));
        const id = "protocol-status-" + Math.random().toString(36).slice(2);
        return await new Promise((resolve, reject) => {
          const timeout = setTimeout(() => {
            window.removeEventListener("message", onMessage);
            reject(new Error("Timed out waiting for mach-llm response."));
          }, 5000);
          const onMessage = (event) => {
            const data = event.data;
            if (!data || data.ns !== "mach-llm" || data.id !== id || (data.type !== "result" && data.type !== "error")) return;
            clearTimeout(timeout);
            window.removeEventListener("message", onMessage);
            if (data.type === "error") reject(new Error(data.data?.message || "Protocol error"));
            else resolve(data.data);
          };
          window.addEventListener("message", onMessage);
          window.postMessage(
            { ns: "mach-llm", id, method: ${JSON.stringify(protocolMethod)} },
            location.origin
          );
        });
      })()
    `,
  });
  if (response.error) throw new Error(response.error.message);
  if (response.result?.exceptionDetails) {
    throw new Error(response.result.exceptionDetails.text || "Chrome evaluation failed.");
  }
  const value = response.result?.result?.value;
  if (value === undefined) {
    throw new Error(`Chrome returned no protocol value: ${JSON.stringify(response.result)}`);
  }
  console.log(JSON.stringify(value, null, 2));
} finally {
  socket.close();
}
