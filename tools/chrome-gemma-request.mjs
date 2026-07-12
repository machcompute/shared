#!/usr/bin/env node
// Exercise a loaded Gemma E4B instance through the public completion API.
// Usage: node tools/chrome-gemma-request.mjs [text|audio|image|video] [cdp-port] [max-tokens]

const kind = process.argv[2] || "text";
const port = Number(process.argv[3] || 9223);
const maxTokens = Number(process.argv[4] || 1);
const model = "google/gemma-4-E4B";

if (!Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > 16) {
  throw new Error("max-tokens must be an integer between 1 and 16.");
}

function silenceWavDataUrl(samples = 1920, sampleRate = 16_000) {
  const wav = Buffer.alloc(44 + samples * 2);
  wav.write("RIFF", 0);
  wav.writeUInt32LE(wav.length - 8, 4);
  wav.write("WAVEfmt ", 8);
  wav.writeUInt32LE(16, 16);
  wav.writeUInt16LE(1, 20);
  wav.writeUInt16LE(1, 22);
  wav.writeUInt32LE(sampleRate, 24);
  wav.writeUInt32LE(sampleRate * 2, 28);
  wav.writeUInt16LE(2, 32);
  wav.writeUInt16LE(16, 34);
  wav.write("data", 36);
  wav.writeUInt32LE(samples * 2, 40);
  return `data:audio/wav;base64,${wav.toString("base64")}`;
}

const request = kind === "audio"
  ? {
      model,
      messages: [{
        role: "user",
        content: [
          { type: "text", text: "Reply with one word." },
          { type: "audio_url", audio_url: { url: silenceWavDataUrl() } },
        ],
      }],
      max_tokens: maxTokens,
      temperature: 0,
      thinking: false,
    }
  : kind === "image"
    ? {
        model,
        messages: [{
          role: "user",
          content: [
            { type: "text", text: "Reply with one word." },
            {
              type: "image_url",
              image_url: {
                // Valid 1×1 PNG; the low-detail setting validates the vision
                // graph with 70 configured soft tokens rather than 280.
                url: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
                detail: "low",
              },
            },
          ],
        }],
        max_tokens: maxTokens,
        temperature: 0,
        thinking: false,
      }
  : kind === "video"
    ? {
        model,
        messages: [{
          role: "user",
          content: [
            { type: "text", text: "Reply with one word." },
            { type: "video_url", video_url: { url: "__generated_webm__", frames: 1 } },
          ],
        }],
        max_tokens: maxTokens,
        temperature: 0,
        thinking: false,
      }
  : {
      model,
      messages: [{ role: "user", content: "Reply with one word." }],
      max_tokens: maxTokens,
      temperature: 0,
      thinking: false,
    };

if (kind !== "text" && kind !== "audio" && kind !== "image" && kind !== "video") {
  throw new Error("request kind must be text, audio, image, or video.");
}

const pages = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
const page = pages.find((entry) => entry.type === "page" && entry.url.startsWith("http://localhost:3002"));
if (!page?.webSocketDebuggerUrl) throw new Error("No loaded Gemma page found at localhost:3002.");

const socket = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {
  socket.addEventListener("open", resolve, { once: true });
  socket.addEventListener("error", reject, { once: true });
});
let nextId = 0;
const pending = new Map();
socket.addEventListener("message", ({ data }) => {
  const message = JSON.parse(data);
  const pendingRequest = pending.get(message.id);
  if (!pendingRequest) return;
  pending.delete(message.id);
  pendingRequest.resolve(message);
});
const send = (method, params = {}) => new Promise((resolve) => {
  const id = ++nextId;
  pending.set(id, { resolve });
  socket.send(JSON.stringify({ id, method, params }));
});

try {
  const response = await send("Runtime.evaluate", {
    awaitPromise: true,
    returnByValue: true,
    expression: `
      (async () => {
        const id = "gemma-request-" + Math.random().toString(36).slice(2);
        const progress = [];
        const request = ${JSON.stringify(request)};
        if (request.messages[0].content?.[1]?.video_url?.url === "__generated_webm__") {
          if (!(window.MediaRecorder && HTMLCanvasElement.prototype.captureStream)) {
            throw new Error("This Chrome session cannot generate the tiny WebM test fixture.");
          }
          const canvas = document.createElement("canvas");
          canvas.width = canvas.height = 48;
          const context = canvas.getContext("2d");
          context.fillStyle = "#2563eb";
          context.fillRect(0, 0, canvas.width, canvas.height);
          const stream = canvas.captureStream(8);
          const chunks = [];
          const recorder = new MediaRecorder(stream, { mimeType: "video/webm;codecs=vp8" });
          const stopped = new Promise((resolve, reject) => {
            recorder.addEventListener("dataavailable", (event) => chunks.push(event.data));
            recorder.addEventListener("stop", resolve, { once: true });
            recorder.addEventListener("error", () => reject(new Error("MediaRecorder failed.")), { once: true });
          });
          recorder.start();
          await new Promise((resolve) => setTimeout(resolve, 500));
          recorder.stop();
          await stopped;
          stream.getTracks().forEach((track) => track.stop());
          const dataUrl = await new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.addEventListener("load", () => resolve(reader.result), { once: true });
            reader.addEventListener("error", () => reject(reader.error || new Error("Could not read WebM.")), { once: true });
            reader.readAsDataURL(new Blob(chunks, { type: "video/webm" }));
          });
          request.messages[0].content[1].video_url.url = dataUrl;
        }
        return await new Promise((resolve, reject) => {
          const timeout = setTimeout(() => {
            window.removeEventListener("message", onMessage);
            reject(new Error("Timed out waiting for Gemma completion."));
          }, 180000);
          const onMessage = (event) => {
            const data = event.data;
            if (!data || data.ns !== "mach-llm" || data.id !== id) return;
            if (data.type === "chunk" && data.data?.event === "progress") {
              progress.push(data.data.progress);
              return;
            }
            if (data.type === "result" || data.type === "error") {
              clearTimeout(timeout);
              window.removeEventListener("message", onMessage);
              if (data.type === "error") reject(new Error(data.data?.message || "Gemma request failed."));
              else resolve({ result: data.data, progress });
            }
          };
          window.addEventListener("message", onMessage);
          window.postMessage({
            ns: "mach-llm",
            id,
            method: "chat.completions.create",
            params: request,
          }, location.origin);
        });
      })()
    `,
  });
  if (response.error) throw new Error(response.error.message);
  if (response.result?.exceptionDetails) {
    throw new Error(response.result.exceptionDetails.text || "Gemma browser request failed.");
  }
  const value = response.result?.result?.value;
  if (!value) throw new Error("Chrome returned no Gemma completion.");
  console.log(JSON.stringify(value, null, 2));
} finally {
  socket.close();
}
