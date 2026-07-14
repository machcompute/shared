import { engine, type ProgressEvent } from "./engine";
import {
  renderFullPrompt,
  buildUserTurnSuffix,
  buildToolResponsesSuffix,
  type RenderMessage,
  type FunctionToolDef,
} from "./webgpu-llm/chat-template";
import { ToolCallStreamParser } from "./webgpu-llm/tool-call-parser";
import { sample } from "./webgpu-llm/tokenizer.js";
import { isGemmaModelId, type ModelId } from "./webgpu-llm/model-registry";
import { runGemmaCompletion } from "./gemma-completions";
import { canCommitCompletionPrefix, qwenDecodeBatchSize } from "./completion-policy";
import { ToolConstraint, validateTools } from "./webgpu-llm/tool-constraint";
import { parseToolCallBody } from "./webgpu-llm/tool-call-parser";

export interface ToolCall {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;
  };
}

export type ChatContent = string | unknown[];

export type ChatMessage =
  | { role: "system" | "user"; content: ChatContent }
  | { role: "assistant"; content: string; tool_calls?: ToolCall[]; reasoning_content?: string }
  | { role: "tool"; content: string; tool_call_id?: string };

type QwenChatMessage = ChatMessage & { content: string };

export interface CompletionRequest {
  model?: ModelId;
  messages: ChatMessage[];
  tools?: FunctionToolDef[];
  parallel_tool_calls?: boolean;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  presence_penalty?: number;
  max_tokens?: number | null;
  thinking?: boolean;
}

export interface ToolCallDelta {
  index: number;
  id?: string;
  type?: "function";
  function: {
    name?: string;
    arguments?: string;
  };
}

export interface CompletionDelta {
  content?: string;
  reasoning_content?: string;
  tool_calls?: ToolCallDelta[];
}

export interface CompletionResult {
  content: string;
  reasoning_content: string;
  tool_calls: ToolCall[] | null;
  finish_reason: "stop" | "length" | "abort" | "tool_calls";
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  context: {
    used_tokens: number;
    max_tokens: number;
  };
}

const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

const randSeed = () => (Math.random() * 0x100000000) >>> 0;

/** cyrb53-style string hash; committed records must stay small even when a
 * message embeds multi-megabyte data-URL media parts. */
function hashString(text: string): string {
  let h1 = 0xdeadbeef ^ text.length;
  let h2 = 0x41c6ce57 ^ text.length;
  for (let i = 0; i < text.length; i++) {
    const ch = text.charCodeAt(i);
    h1 = Math.imul(h1 ^ ch, 2654435761);
    h2 = Math.imul(h2 ^ ch, 1597334677);
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  return (h2 >>> 0).toString(36) + "-" + (h1 >>> 0).toString(36);
}

function deepSortJson(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(deepSortJson);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.keys(value as Record<string, unknown>)
        .sort()
        .map((key) => [key, deepSortJson((value as Record<string, unknown>)[key])])
    );
  }
  return value;
}

/** Tool arguments as the renderer sees them: parsed and key-sorted, so a
 * client re-serializing the same call does not invalidate the prefix. */
function argsSig(argsJson: string): string {
  try {
    const parsed = JSON.parse(argsJson);
    if (parsed && typeof parsed === "object") return JSON.stringify(deepSortJson(parsed));
  } catch {
    // Unparseable arguments participate verbatim.
  }
  return argsJson;
}

export function messageSig(message: ChatMessage): string {
  // Normalization mirrors what the renderers actually emit — content is
  // trimmed and tool arguments render with sorted keys — so only changes
  // that would alter the rendered prompt invalidate the committed prefix.
  const body =
    message.role === "assistant" && message.tool_calls?.length
      ? JSON.stringify([
          "assistant",
          (message.content ?? "").trim(),
          message.tool_calls.map((tc) => [tc.function.name, argsSig(tc.function.arguments)]),
        ])
      : JSON.stringify([
          message.role,
          typeof message.content === "string" ? message.content.trim() : message.content,
        ]);
  // Long bodies (multimodal content arrays) hash down; any content change —
  // including a swapped image — still invalidates the committed prefix.
  return body.length > 256 ? `#${body.length}:${hashString(body)}` : body;
}

/** Everything besides the messages that renders into the prompt prefix; a
 * change means the cached prefix is stale even when all sigs match. */
export function promptPrefixKey(thinking: boolean, tools: FunctionToolDef[] | undefined): string {
  return hashString(JSON.stringify({ thinking, tools: tools ?? null }));
}

export function validateMessages(input: unknown): ChatMessage[] {
  if (!Array.isArray(input) || input.length === 0) {
    throw new Error("messages must be a non-empty array");
  }
  const messages = input.map((raw, i): ChatMessage => {
    const m = (raw ?? {}) as {
      role?: unknown;
      content?: unknown;
      tool_calls?: unknown;
      tool_call_id?: unknown;
    };
    const { role, content } = m;
    if (role !== "system" && role !== "user" && role !== "assistant" && role !== "tool") {
      throw new Error(`messages[${i}].role must be "system", "user", "assistant" or "tool"`);
    }
    if (typeof content !== "string" && !Array.isArray(content)) {
      throw new Error(`messages[${i}].content must be a string or a content-parts array`);
    }
    if (role === "system" && i !== 0) {
      throw new Error("a system message is only allowed as the first message");
    }
    if (role !== "assistant") {
      if (role !== "user" && typeof content !== "string") {
        throw new Error(`messages[${i}].content must be a string for ${role} messages`);
      }
      if (m.tool_calls !== undefined) {
        throw new Error(`messages[${i}].tool_calls is only allowed on assistant messages`);
      }
      return role === "tool"
        ? { role, content: content as string, tool_call_id: typeof m.tool_call_id === "string" ? m.tool_call_id : undefined }
        : { role, content };
    }
    if (typeof content !== "string") {
      throw new Error(`messages[${i}].content must be a string for assistant messages`);
    }
    // Clients that echo reasoning back (DeepSeek-style reasoning_content) let
    // the Gemma renderer restore the thought channel on in-flight tool turns.
    const reasoning_content =
      typeof (m as { reasoning_content?: unknown }).reasoning_content === "string"
        ? ((m as { reasoning_content: string }).reasoning_content)
        : undefined;
    if (m.tool_calls === undefined) return { role, content, reasoning_content };
    if (!Array.isArray(m.tool_calls)) {
      throw new Error(`messages[${i}].tool_calls must be an array`);
    }
    const tool_calls = m.tool_calls.map((rawCall, j): ToolCall => {
      const call = (rawCall ?? {}) as { id?: unknown; function?: { name?: unknown; arguments?: unknown } };
      if (typeof call.function?.name !== "string" || typeof call.function?.arguments !== "string") {
        throw new Error(
          `messages[${i}].tool_calls[${j}] must have function.name and function.arguments strings`
        );
      }
      return {
        id: typeof call.id === "string" ? call.id : `call_${i}_${j}`,
        type: "function",
        function: { name: call.function.name, arguments: call.function.arguments },
      };
    });
    return { role, content, tool_calls, reasoning_content };
  });
  const last = messages[messages.length - 1];
  if (last.role !== "user" && last.role !== "tool") {
    throw new Error('the last message must have role "user" or "tool"');
  }
  return messages;
}

function assertQwenTextMessages(messages: ChatMessage[]): asserts messages is QwenChatMessage[] {
  for (let i = 0; i < messages.length; i++) {
    if (typeof messages[i].content !== "string") {
      throw new Error(
        "Image, audio, and video content requires a Gemma 4 E2B or E4B model."
      );
    }
  }
}

function parseArgs(argsJson: string): Record<string, unknown> {
  try {
    const parsed = JSON.parse(argsJson);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function toRenderMessages(messages: QwenChatMessage[], sanitize: (s: string) => string): RenderMessage[] {
  return messages.map((m): RenderMessage => {
    if (m.role === "assistant" && m.tool_calls?.length) {
      return {
        role: "assistant",
        content: sanitize(m.content),
        tool_calls: m.tool_calls.map((tc) => ({
          name: tc.function.name,
          arguments: parseArgs(tc.function.arguments),
        })),
      };
    }
    return { role: m.role, content: sanitize(m.content) };
  });
}

export async function runCompletion(
  request: CompletionRequest,
  signal: AbortSignal,
  emitDelta: (delta: CompletionDelta) => void,
  emitProgress: (event: ProgressEvent) => void
): Promise<CompletionResult> {
  if (isGemmaModelId(engine.activeModelId)) {
    return runGemmaCompletion(request, signal, emitDelta, emitProgress);
  }
  const messages = validateMessages(request.messages);
  validateTools(request.tools, { grammar: "qwen" });
  assertQwenTextMessages(messages);
  const model = engine.qwenModel;
  const tok = engine.qwenTok;
  const thinking = request.thinking ?? true;

  const samplingParams = {
    temperature: clamp(request.temperature ?? 0.6, 0, 2),
    topP: clamp(request.top_p ?? 0.95, 0.05, 1),
    topK: Math.round(clamp(request.top_k ?? 20, 1, 128)),
    presencePenalty: clamp(request.presence_penalty ?? 1.5, 0, 2),
  };
  model.resetPenaltyWindow();
  const sanitize = (s: string) => tok.sanitize(s);

  const sigs = messages.map(messageSig);
  const promptKey = promptPrefixKey(thinking, request.tools);
  const committed = engine.committed;
  engine.committed = null;
  const extra =
    committed &&
    committed.model === engine.activeModelId &&
    committed.promptKey === promptKey &&
    messages.length > committed.sigs.length &&
    committed.sigs.every((sig, i) => sig === sigs[i])
      ? messages.slice(committed.sigs.length)
      : null;
  const closePrefix = committed?.pending === "tool-close" ? "</tool_call>" : "";

  let promptText: string;
  if (extra && extra.length === 1 && extra[0].role === "user") {
    promptText = closePrefix + buildUserTurnSuffix(sanitize(extra[0].content), thinking);
  } else if (
    extra &&
    committed!.toolCallCount > 0 &&
    extra.length === committed!.toolCallCount &&
    extra.every((m) => m.role === "tool")
  ) {
    promptText =
      closePrefix +
      buildToolResponsesSuffix(extra.map((m) => sanitize(m.content)), thinking);
  } else {
    await model.reset();
    promptText = renderFullPrompt(
      tok.chatTemplate,
      toRenderMessages(messages, sanitize),
      request.tools,
      { thinking }
    );
  }

  const ids = tok.encode(promptText);
  if (model.pos + ids.length >= model.maxCtx - 1) {
    throw new Error(
      `Prompt is too long: ${model.pos + ids.length} tokens does not fit the ${model.maxCtx}-token context window.`
    );
  }

  let cands = await model.prefill(ids, (done: number, total: number) =>
    emitProgress({
      stage: "prefill",
      message: `Processing prompt… ${done} / ${total} tokens`,
      progress: total ? done / total : null,
    })
  );

  const toolCallCloseId = tok.special.get("</tool_call>");
  if (toolCallCloseId === undefined) {
    throw new Error("Tokenizer is missing the </tool_call> special token");
  }
  const constraint = request.tools?.length
    ? new ToolConstraint(request.tools, tok, {
        closeToken: "</tool_call>",
        grammar: "qwen",
        forbiddenTokenIds: [tok.eos],
      })
    : null;
  const maxNew = Math.max(1, Math.min(request.max_tokens ?? model.maxCtx, model.maxCtx));
  const stopIds = [tok.eos, toolCallCloseId];
  const decodeTok = tok.makeDecoder();
  const parallelCalls = !!request.parallel_tool_calls;
  const toolCalls: ToolCall[] = [];
  let toolParser = new ToolCallStreamParser();
  let toolIndex = 0;
  let toolCallId = `call_${randSeed().toString(36)}_${toolIndex}`;

  let reasoningText = "";
  let contentText = "";
  let inThink = thinking;
  let pending = "";
  let toolNameSent = false;
  let toolRawSent = 0;

  const emitReasoning = (chunk: string) => {
    if (!chunk) return;
    reasoningText += chunk;
    emitDelta({ reasoning_content: chunk });
  };
  const emitContent = (chunk: string) => {
    if (!chunk) return;
    contentText += chunk;
    emitDelta({ content: chunk });
  };
  const emitToolDelta = () => {
    if (!toolParser.isOpen && !toolParser.isComplete) return;
    const name = toolParser.toolName ?? "";
    const argsDelta = toolParser.rawBuffer.slice(toolRawSent);
    const nameDelta = name && !toolNameSent ? name : undefined;
    if (!nameDelta && !argsDelta) return;
    emitDelta({
      tool_calls: [
        {
          index: toolIndex,
          id: toolCallId,
          type: "function",
          function: { ...(nameDelta ? { name: nameDelta } : {}), arguments: argsDelta },
        },
      ],
    });
    if (nameDelta) toolNameSent = true;
    toolRawSent = toolParser.rawBuffer.length;
  };
  // Whitespace between consecutive <tool_call> blocks is template glue, not
  // prose — folding it into content would poison the committed signature.
  const emitVisible = (chunk: string) => {
    if (toolCalls.length && chunk.trim() === "") return;
    emitContent(chunk);
  };
  const processAnswer = (piece: string) => {
    emitVisible(toolParser.feed(piece));
    emitToolDelta();
  };
  const emit = (piece: string) => {
    if (!piece) return;
    if (inThink) {
      pending += piece;
      const idx = pending.indexOf("</think>");
      if (idx >= 0) {
        emitReasoning(pending.slice(0, idx));
        const after = pending.slice(idx + 8).replace(/^\n+/, "");
        pending = "";
        inThink = false;
        if (after) processAnswer(after);
      } else if (pending.length > 12) {
        emitReasoning(pending.slice(0, -12));
        pending = pending.slice(-12);
      }
    } else {
      processAnswer(piece);
    }
  };

  let totalN = 0;
  let closePending = false;
  let next = sample(cands, { ...samplingParams, recentIds: model.recentSet() });
  let hitEos = next === tok.eos;
  if (!hitEos && next !== toolCallCloseId) model.notePenaltyToken(next);

  for (;;) {
    while (totalN < maxNew && !signal.aborted && !hitEos && !toolParser.isComplete) {
      emit(decodeTok(next));
      totalN++;
      if (toolParser.isComplete) break;

      const constrained = !!constraint && toolParser.isOpen;
      // Qwen's recurrent state cannot rewind an unconstrained batch if a
      // tool opener appears in its middle. Requests with tools therefore
      // decode one token at a time, including while the parser is idle.
      const k = Math.min(
        qwenDecodeBatchSize(!!constraint, model.BATCH),
        maxNew - totalN,
        model.maxCtx - model.pos - 1,
      );
      if (k < 1) break;

      if (!constraint && model.spec && model.hasMtp && k >= 3) {
        const maxSpecRounds = (model as unknown as { R?: number }).R ?? 1;
        const r = await model.specChain(
          next,
          Math.max(1, Math.min(maxSpecRounds, Math.floor(k / 3))),
          { ...samplingParams, stopIds, seed: randSeed() }
        );

        let roundInput: number | null = next;
        for (let j = 0; j < r.rounds.length; j++) {
          const round = r.rounds[j];
          if (j > 0 && roundInput != null) {
            emit(decodeTok(roundInput));
            totalN++;
            if (toolParser.isComplete) break;
          }
          if (round.a >= 2 && round.d0 != null) {
            emit(decodeTok(round.d0));
            totalN++;
            if (toolParser.isComplete) break;
          }
          if (round.a >= 3 && round.d1 != null) {
            emit(decodeTok(round.d1));
            totalN++;
            if (toolParser.isComplete) break;
          }
          if (round.next == null) {
            roundInput = null;
            break;
          }
          if (r.stopped && round.next === r.stopId) {
            if (round.next === tok.eos) {
              hitEos = true;
              roundInput = null;
            } else if (toolParser.isOpen) {
              emit(decodeTok(round.next));
              totalN++;
              roundInput = null;
            } else {
              model.notePenaltyToken(round.next);
              roundInput = round.next;
            }
            break;
          }
          roundInput = round.next;
        }
        if (toolParser.isComplete || hitEos) break;
        if (roundInput == null) break;
        next = roundInput;
        continue;
      }

      const allowedTokenIds = constrained
        ? constraint!.allowed(toolParser.rawBuffer, parseToolCallBody)
        : undefined;
      if (allowedTokenIds && !allowedTokenIds.length) throw new Error("Tool call cannot satisfy the declared schema.");
      const r = await model.decodeBatch(next, k, {
        ...samplingParams,
        stopIds,
        seed: randSeed(),
        allowedTokenIds,
      });
      for (let j = 0; j < r.ids.length; j++) {
        const id = r.ids[j];
        if (id === tok.eos) {
          hitEos = true;
          break;
        }
        if (j === r.ids.length - 1 && (!r.stopped || !toolParser.isOpen)) {
          if (r.stopped) model.notePenaltyToken(id);
          next = id;
          break;
        }
        emit(decodeTok(id));
        totalN++;
        if (toolParser.isComplete) break;
      }
    }

    const parsedCall = toolParser.isComplete ? toolParser.result : null;
    if (!parsedCall) break;
    toolCalls.push({
      id: toolCallId,
      type: "function",
      function: { name: parsedCall.name, arguments: JSON.stringify(parsedCall.arguments) },
    });
    closePending = true;
    if (
      !parallelCalls ||
      signal.aborted ||
      totalN >= maxNew ||
      model.maxCtx - model.pos - 1 < 2
    ) {
      break;
    }

    // Look ahead for another <tool_call>: the close tag was frozen out as a
    // stop token, so feed it into the state before decoding further.
    model.notePenaltyToken(toolCallCloseId);
    cands = await model.prefill([toolCallCloseId]);
    closePending = false;
    toolParser = new ToolCallStreamParser();
    toolIndex++;
    toolCallId = `call_${randSeed().toString(36)}_${toolIndex}`;
    toolNameSent = false;
    toolRawSent = 0;
    next = sample(cands, { ...samplingParams, recentIds: model.recentSet() });
    hitEos = next === tok.eos;
    if (hitEos) break;
    if (next !== toolCallCloseId) model.notePenaltyToken(next);
  }

  if (pending) emitReasoning(pending);
  emitVisible(toolParser.flush());

  const finalToolCalls = toolCalls.length ? toolCalls : null;
  const aborted = signal.aborted;
  if (canCommitCompletionPrefix({
    aborted,
    stopped: hitEos,
    hasToolCalls: finalToolCalls !== null,
  })) {
    engine.committed = {
      model: engine.activeModelId,
      promptKey,
      sigs: [
        ...sigs,
        messageSig({
          role: "assistant",
          content: contentText,
          tool_calls: finalToolCalls ?? undefined,
        }),
      ],
      toolCallCount: toolCalls.length,
      // Qwen's suffix builders always open with <|im_end|>, so a normal end
      // maps to "turn-close"; only an unfed </tool_call> needs the prefix.
      pending: closePending ? "tool-close" : "turn-close",
    };
  }

  return {
    content: contentText,
    reasoning_content: reasoningText,
    tool_calls: finalToolCalls,
    finish_reason: finalToolCalls ? "tool_calls" : hitEos ? "stop" : aborted ? "abort" : "length",
    usage: {
      prompt_tokens: ids.length,
      completion_tokens: totalN,
      total_tokens: ids.length + totalN,
    },
    context: {
      used_tokens: Math.max(0, model.pos),
      max_tokens: model.maxCtx,
    },
  };
}
