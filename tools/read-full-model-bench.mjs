const pages = await (await fetch('http://127.0.0.1:9222/json/list')).json();
const page = pages.find((p) => p.url.includes('/tools/full-model-bench.html'));
if (!page) throw new Error('Full-model benchmark tab not found');
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {
  ws.addEventListener('open', resolve, { once: true });
  ws.addEventListener('error', reject, { once: true });
});
let nextId = 0;
const pending = new Map();
ws.addEventListener('message', (event) => {
  const message = JSON.parse(event.data);
  const resolve = pending.get(message.id);
  if (resolve) { pending.delete(message.id); resolve(message); }
});
const send = (method, params = {}) => new Promise((resolve) => {
  const id = ++nextId;
  pending.set(id, resolve);
  ws.send(JSON.stringify({ id, method, params }));
});
const response = await send('Runtime.evaluate', {
  expression: 'window.__fullModelBench ?? { pending: true, display: document.querySelector("#out")?.textContent }',
  returnByValue: true,
});
ws.close();
console.log(JSON.stringify(response.result.result.value ?? { pending: true }, null, 2));
