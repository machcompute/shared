import { engine, type LoadOptions, type ProgressEvent } from "./engine";
import {
  runCompletion,
  validateMessages,
  type CompletionDelta,
  type CompletionRequest,
} from "./completions";
import { CFG } from "./webgpu-llm/config.js";
import { originAllowed } from "./allowed-origins";

export const NS = "mach-llm";
export const PROTOCOL_VERSION = 1;

interface RequestMessage {
  ns: string;
  id: string;
  method: string;
  params?: unknown;
}

function isRequest(data: unknown): data is RequestMessage {
  const d = data as RequestMessage | null;
  return (
    !!d &&
    d.ns === NS &&
    typeof d.id === "string" &&
    typeof d.method === "string"
  );
}

export function attachProtocol(win: Window): () => void {
  const inflight = new Map<string, AbortController>();
  let busy = false;

  const onMessage = async (event: MessageEvent) => {
    if (!isRequest(event.data) || !originAllowed(event.origin)) return;
    const source = event.source as Window | null;
    if (!source) return;

    const { id, method, params } = event.data;
    const reply = (type: "chunk" | "result" | "error", data: unknown) =>
      source.postMessage({ ns: NS, id, type, data }, event.origin);
    const progressChunk = (progress: ProgressEvent) =>
      reply("chunk", { event: "progress", progress });

    try {
      switch (method) {
        case "ping": {
          reply("result", { version: PROTOCOL_VERSION, model: CFG.repo });
          break;
        }
        case "status": {
          reply("result", await engine.status());
          break;
        }
        case "load": {
          await engine.ensureLoaded((params ?? {}) as LoadOptions, progressChunk);
          reply("result", await engine.status());
          break;
        }
        case "settings.update": {
          engine.updateRuntime((params ?? {}) as { batchSize?: number; mtp?: boolean });
          reply("result", await engine.status());
          break;
        }
        case "cache.wipe": {
          if (busy) throw Object.assign(new Error("The engine is busy."), { code: "busy" });
          await engine.wipeCache();
          reply("result", { wiped: true });
          break;
        }
        case "abort": {
          const targetId = (params as { targetId?: string } | undefined)?.targetId;
          const controller = targetId ? inflight.get(targetId) : undefined;
          controller?.abort();
          reply("result", { aborted: !!controller });
          break;
        }
        case "chat.completions.create": {
          if (busy) throw Object.assign(new Error("A completion is already running."), { code: "busy" });
          const request = (params ?? {}) as CompletionRequest;
          validateMessages(request.messages);
          busy = true;
          const controller = new AbortController();
          inflight.set(id, controller);
          try {
            await engine.ensureLoaded({}, progressChunk);
            const completionId = `chatcmpl-${Math.random().toString(36).slice(2)}${Date.now().toString(36)}`;
            const created = Math.floor(Date.now() / 1000);
            const chunkOf = (
              delta: CompletionDelta & { role?: "assistant" },
              finishReason: string | null
            ) => ({
              id: completionId,
              object: "chat.completion.chunk",
              created,
              model: CFG.repo,
              choices: [{ index: 0, delta, finish_reason: finishReason }],
            });

            engine.generating = true;
            let first = true;
            let result;
            try {
              result = await runCompletion(
                request,
                controller.signal,
                (delta) => {
                  reply("chunk", {
                    event: "chunk",
                    chunk: chunkOf(first ? { role: "assistant", ...delta } : delta, null),
                  });
                  first = false;
                },
                progressChunk
              );
            } finally {
              engine.generating = false;
            }

            reply("chunk", { event: "chunk", chunk: chunkOf({}, result.finish_reason) });
            reply("result", {
              id: completionId,
              object: "chat.completion",
              created,
              model: CFG.repo,
              choices: [
                {
                  index: 0,
                  message: {
                    role: "assistant",
                    content: result.content,
                    reasoning_content: result.reasoning_content,
                    ...(result.tool_calls ? { tool_calls: result.tool_calls } : {}),
                  },
                  finish_reason: result.finish_reason,
                },
              ],
              usage: result.usage,
              context: result.context,
            });
          } finally {
            busy = false;
            inflight.delete(id);
          }
          break;
        }
        default:
          throw Object.assign(new Error(`Unknown method: ${method}`), {
            code: "unknown_method",
          });
      }
    } catch (error) {
      const e = error as Error & { code?: string };
      reply("error", { message: e?.message ?? String(error), code: e?.code });
    }
  };

  win.addEventListener("message", onMessage);
  if (win.parent && win.parent !== win) {
    win.parent.postMessage(
      { ns: NS, type: "ready", version: PROTOCOL_VERSION, model: CFG.repo },
      "*"
    );
  }
  return () => win.removeEventListener("message", onMessage);
}
