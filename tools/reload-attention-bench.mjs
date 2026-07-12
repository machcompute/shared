const pages = await (await fetch('http://127.0.0.1:9222/json/list')).json();
const page = pages.find((p) => p.url.includes('/tools/attention-bench.html'));
if (!page) throw new Error('Benchmark tab not found');
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {
  ws.addEventListener('open', resolve, { once: true });
  ws.addEventListener('error', reject, { once: true });
});
ws.send(JSON.stringify({ id: 1, method: 'Page.reload', params: { ignoreCache: true } }));
await new Promise((resolve) => setTimeout(resolve, 250));
ws.close();
