const target = process.argv[2] || 'http://localhost:4173/tools/full-model-bench.html';
const pages = await (await fetch('http://127.0.0.1:9222/json/list')).json();
const page = pages.find((p) => p.url.includes('/tools/full-model-bench.html'));
if (!page) throw new Error('Full-model benchmark tab not found');
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {
  ws.addEventListener('open', resolve, { once: true });
  ws.addEventListener('error', reject, { once: true });
});
ws.send(JSON.stringify({ id: 1, method: 'Page.navigate', params: { url: target } }));
await new Promise((resolve) => setTimeout(resolve, 250));
ws.close();
