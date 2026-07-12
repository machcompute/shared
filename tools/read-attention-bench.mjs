const pages = await (await fetch('http://127.0.0.1:9222/json/list')).json();
const page = pages.find((p) => p.url.includes('/tools/attention-bench.html'));
if (!page) throw new Error('Benchmark tab not found');

const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {
  ws.addEventListener('open', resolve, { once: true });
  ws.addEventListener('error', reject, { once: true });
});

let id = 0;
const pending = new Map();
ws.addEventListener('message', (event) => {
  const message = JSON.parse(event.data);
  if (!message.id) return;
  const callback = pending.get(message.id);
  if (callback) { pending.delete(message.id); callback(message); }
});
const send = (method, params = {}) => new Promise((resolve) => {
  const callId = ++id;
  pending.set(callId, resolve);
  ws.send(JSON.stringify({ id: callId, method, params }));
});

const response = await send('Runtime.evaluate', {
  expression: `new Promise((resolve) => {
    const deadline = performance.now() + 120000;
    const poll = () => {
      if (window.__attentionBench) resolve(window.__attentionBench);
      else if (performance.now() > deadline) resolve({ error: 'benchmark timeout' });
      else setTimeout(poll, 100);
    };
    poll();
  })`,
  awaitPromise: true,
  returnByValue: true,
});
ws.close();
console.log(JSON.stringify(response.result.result.value, null, 2));
