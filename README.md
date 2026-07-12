# shared.machcomputing.com

Headless, embeddable WebGPU LLM engine. Qwen3.5-4B is the default Q4 text model; `google/gemma-4-E2B` and `google/gemma-4-E4B` add Q4 text/vision/audio weights, FP8 E4M3FN KV caches, and image/audio/video chat content. It has no UI: the page at `/` is an "API" that other `machcomputing.com` sites embed as a hidden iframe and drive over `postMessage` through the client SDK at `/client.js`.

The engine origin owns the weights cache (OPFS, ~3 GB after quantization). Because browsers partition embedded-iframe storage by top-level *site* (eTLD+1), every `*.machcomputing.com` page embedding this engine shares one cache: the model downloads and quantizes once per device, then loads from disk everywhere.

## Integration

```html
<script type="module">
  import { MachLLM } from "https://shared.machcomputing.com/client.js";

  const llm = await MachLLM.connect({
    onProgress: (p) => console.log(p.stage, p.message, p.progress),
  });

  const completion = await llm.chat.completions.create({
    model: "google/gemma-4-E4B", // selects the multimodal runtime
    messages: [
      { role: "system", content: "You are a helpful assistant." },
      { role: "user", content: "Hello!" },
    ],
  });
  console.log(completion.choices[0].message.content);
</script>
```

Streaming:

```js
const stream = await llm.chat.completions.create({ messages, stream: true });
for await (const chunk of stream) {
  const delta = chunk.choices[0].delta;
  if (delta.reasoning_content) renderThinking(delta.reasoning_content);
  if (delta.content) renderAnswer(delta.content);
}
console.log(stream.completion.usage);
```

That is the whole integration. `connect()` injects a hidden `<iframe allow="webgpu">` pointing at the engine; the first `create()` call auto-loads the model (progress is reported through `onProgress` / `llm.on("progress", fn)`).

## Client API

`MachLLM.connect(options?)` → `Promise<MachLLM>`

- `engineUrl` — engine origin, defaults to `https://shared.machcomputing.com`. Point it at `http://localhost:3001` during local development.
- `onProgress` — shorthand for `llm.on("progress", fn)`. Events: `{ stage, message, progress }` with stages `tokenizer`, `gpu`, `download`, `quantize`, `cache`, `pipelines`, `prefill`, `ready`.
- `timeoutMs` — handshake timeout, default 20000.

`llm.models.list()` → `{ object: "list", data: Model[] }` — lists every registered runtime before loading it. Each `Model` has `{ id, object: "model", label, modalities, maxContext }`.

`llm.chat.completions.create(params, options?)`

- `params.messages` — `{ role: "system" | "user" | "assistant" | "tool", content: string | ContentPart[] }[]`. At most one system message, first; the last message must be a user or tool message. Assistant messages may carry OpenAI-style `tool_calls`; tool messages may carry `tool_call_id`.
- `params.model` — optional registered model ID. The request loads and uses that specific model; omit it to keep the currently loaded model (initially `Qwen/Qwen3.5-4B`). Use `google/gemma-4-E2B` or `google/gemma-4-E4B` for image, audio, and video content.
- Gemma user messages may use OpenAI-style ordered content parts: `{ type: "text", text }`, `{ type: "image_url", image_url: { url, detail? } }`, `{ type: "audio_url", audio_url: { url } }`, and `{ type: "video_url", video_url: { url, frames? } }` (the direct `image`/`audio`/`video` + `url` spellings work too). Use image `detail: "low"` for 70 soft tokens; the default/high path uses 280. `frames` caps video sampling between 1 and 32 frames. Image, audio, and video are encoded into Gemma soft tokens in place, so mixed text/media order is preserved. Qwen accepts text content only.
- `params.tools` — OpenAI-style function definitions (`{ type: "function", function: { name, description, parameters } }[]`). When the model calls a tool, the stream emits `delta.tool_calls` (note: streamed `function.arguments` fragments are the model's raw on-the-wire grammar, for display only) and the completion ends with `finish_reason: "tool_calls"` and a parsed `message.tool_calls` whose `function.arguments` is a JSON string. Execute the tool yourself, then send a follow-up request appending the assistant message (with its `tool_calls`) and a `{ role: "tool", tool_call_id, content }` message.

  Both runtimes support tools. Qwen uses its trained `<tool_call>`/`<function>`/`<parameter>` XML grammar and resumes from live GPU state without re-prefilling. Gemma serializes tools through its instruction-tuned (`-it`) chat template — tool declarations, model-emitted calls, and results ride the `<|tool>`/`<|tool_call>`/`<|tool_response>` control tokens — and re-prefills each turn. Gemma tool calls compose with image, audio, and video content in the same request; its `function.arguments` are JSON. When tools are supplied, native tool-call names and close tokens are GPU-constrained; a call can close only after its arguments satisfy the supported schema subset (`object`, `properties`, `required`, `array/items`, scalar types, `enum`, and nesting). Unsupported JSON Schema keywords reject the request.
- `params.parallel_tool_calls` (default `false`) — when `true`, the engine keeps decoding after a tool call to collect consecutive calls, so `message.tool_calls` may hold several entries (streamed deltas are distinguished by `index`). Execute them all, then append one `role: "tool"` message per call, in call order. When `false`, generation stops at the first call.
- `params.stream` — `false` (default): resolves to an OpenAI-style chat completion object. `true`: resolves to an async-iterable stream of chunks with an OpenAI-style `choices[0].delta`; after iteration ends, `stream.completion` holds the final completion object.
- `params.temperature` (0.6), `params.top_p` (0.95), `params.top_k` (20), `params.presence_penalty` (1.5), `params.max_tokens`, `params.thinking` (default `true`; reasoning is streamed as `delta.reasoning_content` and returned as `message.reasoning_content`, never mixed into `content`).
- `options.signal` — an `AbortSignal`; aborting ends generation and yields a final result with `finish_reason: "abort"` and the partial text. Streams also expose `stream.controller.abort()`.
- `finish_reason` — `"stop"` (EOS), `"length"` (hit `max_tokens` or the context window), `"abort"`.
- `usage.prompt_tokens` counts the tokens actually prefilled: on a continuation turn (see below) that is just the new suffix, not the whole conversation.

`llm.status()` → `{ model, activeModel, defaultModel, availableModels, modalities, webgpu, adapter, cached, loaded, generating, hasMtp, contextUsedTokens, contextMaxTokens, device }` — `webgpu` says the API exists, `adapter` says a usable GPU adapter was actually found; gate any "Load model" UI on both.

`llm.load(options?)` — optional explicit preload; `{ model?, maxContext?, batchSize?, mtp?, reload? }`. Loading Gemma uses `batchSize: 1` and a practical default 8k context; a larger context can be requested explicitly up to 131,072 tokens, subject to GPU memory.

`llm.updateSettings(options?)` — applies `{ batchSize?, mtp? }` to a loaded model without reloading (`maxContext` requires a reload).

`llm.wipeCache({ model? })` — deletes the active model's OPFS weights cache, or the named registered model's cache.

`llm.close()` — removes the iframe and rejects pending requests.

One completion runs at a time per engine; a second concurrent `create()` rejects with `code: "busy"`.

## How prompts are built

Messages are rendered through the model's own Jinja `chat_template` (fetched with the tokenizer from HuggingFace) with `add_generation_prompt` and `enable_thinking`. If a request extends the previous one by exactly one user message (same system prompt, same history, plus the assistant reply the engine just produced), the engine skips the full re-prefill and feeds only the new turn into the live GPU state.

## Security

The engine only answers `postMessage` requests from allowed domains and serves a matching `frame-ancestors` CSP. Every other origin is blocked from embedding and ignored if it somehow posts a request; foreign embedders would get an empty, separate storage partition anyway.

The allowlist comes from `NEXT_PUBLIC_ALLOWED_DOMAINS` (required, no default — the build fails without it; see `.env.example`) — a comma-separated list of bare hostnames, `*.` wildcards, or `localhost`/`127.0.0.1` (which accept http and any port; everything else must be https). It is read at **build time**: both the CSP header and the origin check (`lib/allowed-origins.ts`) are baked in, so changing it requires a rebuild. `.env.development` sets `localhost` for local dev.

## Local development

```sh
npm run dev   # engine on http://localhost:3001 (port is pinned; the OPFS dev cache lives under this origin)
```

For an isolated Gemma/WebGPU test server without disturbing a port-3001 session:

```sh
NEXT_DIST_DIR=.next-gemma-test ./node_modules/.bin/next dev -p 3002
```

Launch Chrome with Vulkan and Unsafe WebGPU enabled when your Linux setup needs those flags. The Gemma checkpoint is large (~16 GB source, roughly 5 GB Q4 browser cache before the FP8 KV cache), so test only on a GPU with sufficient headroom.

- Consumer apps set their engine URL to `http://localhost:3001` in development (e.g. `NEXT_PUBLIC_LLM_ENGINE_URL`) and omit it in production.
- All `localhost` ports share one top-level site, so every local app reuses the same dev cache; it is separate from the production cache.
- WebGPU needs Chrome/Edge 121+; on Linux you may need `chrome://flags` → Vulkan + Unsafe WebGPU.

## Protocol (for non-JS clients)

Requests to the iframe: `{ ns: "mach-llm", id, method, params }` with methods `ping`, `status`, `models.list`, `load`, `chat.completions.create`, `abort` (`{ targetId }`), `settings.update`, `cache.wipe`. Replies: `{ ns, id, type: "chunk" | "result" | "error", data }`; `models.list` replies with `{ object: "list", data: Model[] }`; chunks are `{ event: "progress", progress }` or `{ event: "chunk", chunk }`. On boot the engine broadcasts `{ ns, type: "ready", version, model }` to its parent.
