import { MachLLM } from 'http://localhost:3001/client.js';

const out = document.querySelector('#out');
const query = new URLSearchParams(location.search);
const repeats = Number(query.get('repeats') || 16000);
const shouldRun = query.get('run') === '1';
const log = [];
const show = (value) => { out.textContent = JSON.stringify(value, null, 2); };

async function main() {
  const llm = await MachLLM.connect({
    engineUrl: 'http://localhost:3001',
    timeoutMs: 30000,
    onProgress: (event) => {
      log.push({ at: performance.now(), ...event });
      show({ stage: event, recent: log.slice(-5) });
    },
  });
  const initialStatus = await llm.status();
  if (!shouldRun) return { initialStatus };

  await llm.load({ maxContext: 65536, batchSize: 8, mtp: false });
  const prompt = 'test '.repeat(repeats) +
    '\nContinue by outputting only the word test separated by spaces. Do not stop early.';
  const requestStart = performance.now();
  const stream = await llm.chat.completions.create({
    messages: [{ role: 'user', content: prompt }],
    stream: true,
    max_tokens: 256,
    temperature: 0,
    top_p: 1,
    top_k: 1,
    presence_penalty: 0,
    thinking: false,
  });
  let firstTokenAt = null;
  for await (const chunk of stream) {
    const delta = chunk.choices?.[0]?.delta;
    if (firstTokenAt == null && (delta?.content || delta?.reasoning_content)) {
      firstTokenAt = performance.now();
    }
  }
  const end = performance.now();
  const completion = stream.completion;
  const tokens = completion.usage.completion_tokens;
  const decodeSeconds = firstTokenAt == null ? null : (end - firstTokenAt) / 1000;
  return {
    initialStatus,
    finalStatus: await llm.status(),
    repeats,
    usage: completion.usage,
    finishReason: completion.choices[0].finish_reason,
    prefillAndDecodeSeconds: (end - requestStart) / 1000,
    decodeSeconds,
    measuredDecodeTokens: Math.max(0, tokens - 1),
    tps: decodeSeconds ? Math.max(0, tokens - 1) / decodeSeconds : null,
  };
}

main().then(result => {
  window.__fullModelBench = result;
  show(result);
}).catch(error => {
  window.__fullModelBench = { error: error.stack || String(error), recent: log.slice(-10) };
  show(window.__fullModelBench);
});
