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

export interface ToolCall {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;
  };
}

export type ChatMessage =
  | { role: "system" | "user"; content: string }
  | { role: "assistant"; content: string; tool_calls?: ToolCall[] }
  | { role: "tool"; content: string; tool_call_id?: string };

export interface CompletionRequest {
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

function messageSig(message: ChatMessage): string {
  if (message.role === "assistant" && message.tool_calls?.length) {
    return JSON.stringify([
      "assistant",
      message.content,
      message.tool_calls.map((tc) => [tc.function.name, tc.function.arguments]),
    ]);
  }
  return JSON.stringify([message.role, message.content]);
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
    if (typeof content !== "string") {
      throw new Error(`messages[${i}].content must be a string`);
    }
    if (role === "system" && i !== 0) {
      throw new Error("a system message is only allowed as the first message");
    }
    if (role !== "assistant") {
      if (m.tool_calls !== undefined) {
        throw new Error(`messages[${i}].tool_calls is only allowed on assistant messages`);
      }
      return role === "tool"
        ? { role, content, tool_call_id: typeof m.tool_call_id === "string" ? m.tool_call_id : undefined }
        : { role, content };
    }
    if (m.tool_calls === undefined) return { role, content };
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
    return { role, content, tool_calls };
  });
  const last = messages[messages.length - 1];
  if (last.role !== "user" && last.role !== "tool") {
    throw new Error('the last message must have role "user" or "tool"');
  }
  return messages;
}

function parseArgs(argsJson: string): Record<string, unknown> {
  try {
    const parsed = JSON.parse(argsJson);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function toRenderMessages(messages: ChatMessage[], sanitize: (s: string) => string): RenderMessage[] {
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
  const messages = validateMessages(request.messages);
  const model = engine.model;
  const tok = engine.tok;
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
  const committed = engine.committed;
  engine.committed = null;
  const extra =
    committed &&
    messages.length > committed.sigs.length &&
    committed.sigs.every((sig, i) => sig === sigs[i])
      ? messages.slice(committed.sigs.length)
      : null;
  const closePrefix = committed?.closePending ? "</tool_call>" : "";

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

      const k = Math.min(model.BATCH, maxNew - totalN, model.maxCtx - model.pos - 1);
      if (k < 1) break;

      if (model.spec && model.hasMtp && k >= 3) {
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

      const r = await model.decodeBatch(next, k, {
        ...samplingParams,
        stopIds,
        seed: randSeed(),
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
  if (!aborted) {
    engine.committed = {
      sigs: [
        ...sigs,
        messageSig({
          role: "assistant",
          content: contentText,
          tool_calls: finalToolCalls ?? undefined,
        }),
      ],
      toolCallCount: toolCalls.length,
      closePending,
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
