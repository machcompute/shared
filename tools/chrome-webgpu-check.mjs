// Minimal CDP check for the isolated Chrome used by local WebGPU verification.
// Usage: node tools/chrome-webgpu-check.mjs [page-url] [cdp-port]
const target = process.argv[2] || "http://localhost:3002";
const port = Number(process.argv[3] || 9223);
const pages = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
const page = pages.find((entry) => entry.type === "page");
if (!page?.webSocketDebuggerUrl) throw new Error(`No CDP page found on port ${port}`);

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

await send("Page.navigate", { url: target });
const response = await send("Runtime.evaluate", {
  awaitPromise: true,
  returnByValue: true,
  expression: `
    (async () => {
      const webgpu = !!navigator.gpu;
      const adapter = webgpu ? await navigator.gpu.requestAdapter() : null;
      if (!adapter) return { webgpu, adapter: false };
      const device = await adapter.requestDevice();
      const module = device.createShaderModule({ code: \`
        @compute @workgroup_size(1) fn main() {}
      \` });
      const compilation = await module.getCompilationInfo();
      device.destroy();
      return {
        webgpu,
        adapter: true,
        adapterInfo: adapter.info ? { vendor: adapter.info.vendor, architecture: adapter.info.architecture } : null,
        shaderErrors: compilation.messages.filter((message) => message.type === "error").map((message) => message.message),
      };
    })()
  `,
});
socket.close();
if (response.error) throw new Error(response.error.message);
console.log(JSON.stringify(response.result?.result?.value, null, 2));
