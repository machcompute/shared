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

function rowsOf(values: Float32Array, width: number, label: string): number {
  if (!Number.isInteger(width) || width < 1 || !values.length || values.length % width) {
    throw new Error(
      `Gemma ${label} encoder returned an invalid soft-embedding matrix ` +
      `(${values.length} values for width ${width}).`
    );
  }
  return values.length / width;
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

interface GemmaPromptBuilder {
  appendText(text: string): void;
  emitContent(message: ChatMessage): Promise<{ hasContent: boolean; mediaType: GemmaMediaType | null }>;
  finish(): PreparedGemmaPrompt;
}

/**
 * Shared token/override assembly for full prompts and suffix continuations.
 * `basePos` offsets media override keys: `#submitEmbeddings` resolves them as
 * absolute decoder positions, so a suffix prefilled at `model.pos > 0` must
 * key its overrides from that position, not from the suffix start.
 */
function promptBuilder(
  tokenizer: GemmaTokenizer,
  model: GemmaModel,
  signal: AbortSignal,
  onStage: (message: string) => void,
  basePos = 0,
): GemmaPromptBuilder {
  const sanitize = (text: string) => tokenizer.sanitize(text);
  const tokens: number[] = [];
  const overrides = new Map<number, Float32Array>();
  let mediaTokenCount = 0;
  const embeddingWidth = model.embeddingWidth;

  const appendText = (text: string) => tokens.push(...tokenizer.encode(text));
  const appendMedia = (type: GemmaMediaType, values: Float32Array) => {
    const rows = rowsOf(values, embeddingWidth, type);
    const placeholderId = type === "image" ? tokenizer.image : type === "audio" ? tokenizer.audio : tokenizer.video;
    const ids = tokenizer.encode(multimodalPlaceholder(type, rows));
    let row = 0;
    for (const id of ids) {
      if (id === placeholderId) {
        overrides.set(
          basePos + tokens.length,
          values.subarray(row * embeddingWidth, (row + 1) * embeddingWidth)
        );
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

  return {
    appendText,
    emitContent,
    finish: () => ({ tokenIds: tokens, overrides, mediaTokenCount }),
  };
}

/**
 * Preprocess mixed OpenAI-style content, encode each modality, and construct
 * the exact Gemma turn protocol plus embedding overrides. Boundary tokens
 * remain in the language sequence; repeated media placeholder IDs are
 * replaced with projections matching the loaded text decoder's width.
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
  const { appendText, emitContent, finish } = promptBuilder(tokenizer, model, signal, onStage);

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
  let lastUserIdx = -1;
  for (let i = 0; i < rest.length; i++) if (rest[i].role === "user") lastUserIdx = i;

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

    // The checkpoint's chat template restores reasoning as a thought channel
    // on in-flight tool exchanges (assistant tool calls after the last user
    // turn). Without it, thinking-mode tool turns render out-of-distribution
    // — a model turn that opens directly with <|tool_call> — and the model
    // continues poorly after the tool response.
    if (
      message.role === "assistant" &&
      message.tool_calls?.length &&
      message.reasoning_content &&
      i > lastUserIdx
    ) {
      appendText(`<|channel>thought\n${sanitize(message.reasoning_content)}\n<channel|>`);
    }

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
  return finish();
}

/**
 * Continuation suffix for one new user turn on top of committed GPU state.
 *
 * The cache ends at the last generated content token — the sampled turn-end
 * stop was never fed — so the suffix opens with the canonical `<turn|>\n`.
 * The appendText segmentation deliberately mirrors `prepareGemmaPrompt` so
 * both paths tokenize turn boundaries identically. Only this message's media
 * is encoded; committed turns' media rows are already resident in the cache.
 */
export async function prepareGemmaUserTurnSuffix(
  message: ChatMessage,
  tokenizer: GemmaTokenizer,
  model: GemmaModel,
  options: {
    signal: AbortSignal;
    basePos: number;
    onStage?: (message: string) => void;
    /** False when the model turn is already open-ended in the cache — a user
     * turn directly following tool responses, which the renderer (and the
     * checkpoint's template) leave unclosed. Default true. */
    closeTurn?: boolean;
  },
): Promise<PreparedGemmaPrompt> {
  if (message.role !== "user") {
    throw new Error("Gemma user-turn continuation requires a user message.");
  }
  const { appendText, emitContent, finish } = promptBuilder(
    tokenizer, model, options.signal, options.onStage ?? (() => {}), options.basePos,
  );
  if (options.closeTurn ?? true) appendText("<turn|>\n");
  appendText("<|turn>user\n");
  await emitContent(message);
  appendText("<turn|>\n");
  appendText("<|turn>model\n");
  return finish();
}

/**
 * Continuation suffix feeding tool responses back into an open model turn.
 *
 * Matches the full renderer: responses follow the tool call directly and the
 * model turn stays open (no turn markers). `pending: "tool-close"` prepends
 * the `<tool_call|>` close tag when it was sampled as a stop token and never
 * fed; a parallel-lookahead session already has it in the cache ("none").
 */
export function prepareGemmaToolResponsesSuffix(
  responses: { name: string; content: string }[],
  tokenizer: GemmaTokenizer,
  options: { pending: "tool-close" | "none" },
): PreparedGemmaPrompt {
  const sanitize = (text: string) => tokenizer.sanitize(text);
  const tokens: number[] = [];
  if (options.pending === "tool-close") tokens.push(...tokenizer.encode("<tool_call|>"));
  for (const response of responses) {
    tokens.push(...tokenizer.encode(toolResponseText(response.name, response.content, sanitize)));
  }
  return { tokenIds: tokens, overrides: new Map(), mediaTokenCount: 0 };
}
