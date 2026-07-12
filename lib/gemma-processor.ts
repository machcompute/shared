import type { ChatMessage, ToolCall } from "./completions";
import type { FunctionToolDef } from "./webgpu-llm/chat-template";
import { formatToolCall, formatToolDeclaration, formatToolResponse } from "./webgpu-llm/gemma-tool-format";
import type { GemmaModel } from "./webgpu-llm/gemma-model.js";
import type { GemmaTokenizer } from "./webgpu-llm/gemma-tokenizer.js";
import {
  preprocessGemmaMedia,
  type GemmaMediaInputs,
  type GemmaPreparedContentPart,
} from "./webgpu-llm/gemma-media";
import { multimodalPlaceholder, type GemmaMediaType } from "./webgpu-llm/gemma-template";

const HIDDEN = 2560;

export interface PreparedGemmaPrompt {
  tokenIds: number[];
  /** Absolute decoder positions whose regular embedding is replaced by media. */
  overrides: Map<number, Float32Array>;
  mediaTokenCount: number;
}

function throwIfAborted(signal: AbortSignal) {
  if (signal.aborted) throw new DOMException("The completion was aborted.", "AbortError");
}

function toolDeclaration(tool: FunctionToolDef): string {
  return `<|tool>${formatToolDeclaration(tool)}<tool|>`;
}

function toolCallText(call: ToolCall): string {
  let args: Record<string, unknown> = {};
  try {
    const parsed = JSON.parse(call.function.arguments || "{}");
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) args = parsed;
  } catch {
    args = {};
  }
  return formatToolCall(call.function.name, args);
}

function toolResponseText(name: string, content: string, sanitize: (text: string) => string): string {
  return formatToolResponse(name, sanitize(content));
}

function rowsOf(values: Float32Array, label: string): number {
  if (!values.length || values.length % HIDDEN) {
    throw new Error(`Gemma ${label} encoder returned an invalid soft-embedding matrix.`);
  }
  return values.length / HIDDEN;
}

async function mediaEmbeddings(
  prepared: GemmaMediaInputs,
  model: GemmaModel,
  signal: AbortSignal,
  onStage: (message: string) => void,
): Promise<Map<GemmaPreparedContentPart, Float32Array>> {
  const embeddings = new Map<GemmaPreparedContentPart, Float32Array>();
  for (const part of prepared.parts) {
    if (part.type === "text") continue;
    throwIfAborted(signal);
    if (part.type === "image") {
      onStage("Encoding image…");
      embeddings.set(part, await model.encodeImage(prepared.images[part.imageIndex]));
    } else if (part.type === "video") {
      onStage("Encoding video frames…");
      embeddings.set(part, await model.encodeVideo(prepared.videos[part.videoIndex]));
    } else {
      onStage("Encoding audio…");
      embeddings.set(part, await model.encodeAudio(prepared.audios[part.audioIndex]));
    }
  }
  return embeddings;
}

/**
 * Preprocess mixed OpenAI-style content, encode each modality, and construct
 * the exact Gemma turn protocol plus embedding overrides.  Boundary tokens
 * remain in the language sequence; only repeated media placeholder IDs are
 * replaced with their 2560-wide tower projections.
 */
export async function prepareGemmaPrompt(
  messages: ChatMessage[],
  tokenizer: GemmaTokenizer,
  model: GemmaModel,
  options: {
    thinking: boolean;
    signal: AbortSignal;
    tools?: FunctionToolDef[];
    onStage?: (message: string) => void;
  },
): Promise<PreparedGemmaPrompt> {
  const { thinking, signal } = options;
  const onStage = options.onStage ?? (() => {});
  const tools = options.tools ?? [];
  const sanitize = (text: string) => tokenizer.sanitize(text);
  const tokens: number[] = [];
  const overrides = new Map<number, Float32Array>();
  let mediaTokenCount = 0;

  const appendText = (text: string) => tokens.push(...tokenizer.encode(text));
  const appendMedia = (type: GemmaMediaType, values: Float32Array) => {
    const rows = rowsOf(values, type);
    const placeholderId = type === "image" ? tokenizer.image : type === "audio" ? tokenizer.audio : tokenizer.video;
    const ids = tokenizer.encode(multimodalPlaceholder(type, rows));
    let row = 0;
    for (const id of ids) {
      if (id === placeholderId) {
        overrides.set(tokens.length, values.subarray(row * HIDDEN, (row + 1) * HIDDEN));
        tokens.push(tokenizer.pad);
        row++;
      } else {
        tokens.push(id);
      }
    }
    if (row !== rows) {
      throw new Error(`Gemma ${type} prompt expansion produced ${row} placeholders for ${rows} embeddings.`);
    }
    mediaTokenCount += rows;
  };

  const emitContent = async (message: ChatMessage): Promise<{ hasContent: boolean; mediaType: GemmaMediaType | null }> => {
    let hasContent = false;
    let mediaType: GemmaMediaType | null = null;
    if (typeof message.content === "string") {
      const text = sanitize(message.content).trim();
      if (text) {
        appendText(text);
        hasContent = true;
      }
    } else {
      if (message.role !== "user") {
        throw new Error("Gemma only accepts multimodal content on user messages.");
      }
      onStage("Preparing image, audio, and video content…");
      const prepared = await preprocessGemmaMedia(message.content, { signal });
      const encoded = await mediaEmbeddings(prepared, model, signal, onStage);
      for (const part of prepared.parts) {
        if (part.type === "text") {
          const text = sanitize(part.text).trim();
          if (text) {
            appendText(text);
            hasContent = true;
          }
          continue;
        }
        const values = encoded.get(part);
        if (!values) throw new Error(`Gemma ${part.type} preprocessing did not produce embeddings.`);
        appendMedia(part.type, values);
        hasContent = true;
        mediaType = part.type;
      }
    }
    return { hasContent, mediaType };
  };

  appendText("<bos>");
  const first = messages[0];
  const firstIsSystem = first?.role === "system";
  const toolDeclText = tools.map((tool) => toolDeclaration(tool)).join("");
  if (thinking || toolDeclText || firstIsSystem) {
    appendText("<|turn>system\n");
    if (thinking) appendText("<|think|>\n");
    if (firstIsSystem && typeof first.content === "string") appendText(sanitize(first.content).trim());
    if (toolDeclText) appendText(toolDeclText);
    appendText("<turn|>\n");
  }

  const rest = firstIsSystem ? messages.slice(1) : messages;
  let endType: "tool_call" | "tool_response" | GemmaMediaType | null = null;

  for (let i = 0; i < rest.length; i++) {
    const message = rest[i];
    if (message.role === "tool") continue;
    throwIfAborted(signal);

    const role = message.role === "assistant" ? "model" : message.role;
    let prevNonTool: ChatMessage | null = null;
    for (let j = i - 1; j >= 0; j--) {
      if (rest[j].role !== "tool") {
        prevNonTool = rest[j];
        break;
      }
    }
    const continueModelTurn = role === "model" && prevNonTool?.role === "assistant";
    if (!continueModelTurn) appendText(`<|turn>${role}\n`);

    let prevType: "tool_call" | "tool_response" | GemmaMediaType | null = null;
    let toolResponsesOut = false;
    if (message.role === "assistant" && message.tool_calls?.length) {
      for (const call of message.tool_calls) appendText(toolCallText(call));
      prevType = "tool_call";
      for (let k = i + 1; k < rest.length; k++) {
        const follow = rest[k];
        if (follow.role !== "tool") break;
        const name = message.tool_calls.find((tc) => tc.id === follow.tool_call_id)?.function.name ?? "unknown";
        appendText(toolResponseText(name, follow.content, sanitize));
        toolResponsesOut = true;
        prevType = "tool_response";
      }
    }

    const { hasContent, mediaType } = await emitContent(message);
    if (mediaType) prevType = mediaType;

    if (prevType === "tool_call" && !toolResponsesOut) {
      appendText("<|tool_response>");
    } else if (!(toolResponsesOut && !hasContent)) {
      appendText("<turn|>\n");
    }
    endType = prevType;
  }

  if (endType !== "tool_response" && endType !== "tool_call") {
    appendText("<|turn>model\n");
  }
  return { tokenIds: tokens, overrides, mediaTokenCount };
}
