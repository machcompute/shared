import { engine, type ProgressEvent } from "./engine";
import {
  validateMessages,
  type CompletionDelta,
  type CompletionRequest,
  type CompletionResult,
  type ToolCall,
} from "./completions";
import { prepareGemmaPrompt } from "./gemma-processor";
import { GemmaToolCallParser, parseGemmaToolCallBody } from "./webgpu-llm/gemma-tool-parser";
import { ToolConstraint } from "./webgpu-llm/tool-constraint";
import { sample } from "./webgpu-llm/tokenizer.js";

const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

const makeToolCallId = (index: number) =>
  `call_${Math.random().toString(36).slice(2, 10)}_${index}`;

type GemmaChannelState = {
  reasoning: boolean;
  /** Channel names often decode separately from `<|channel>`. */
  pendingLabel: string | null;
  skipLabelNewline: boolean;
};

const channelLabels = ["thought", "analysis", "final", "commentary"];

function stripChannelLabel(piece: string, state: GemmaChannelState): string {
  if (state.skipLabelNewline) {
    piece = piece.replace(/^\r?\n/, "");
    state.skipLabelNewline = false;
  }
  if (state.pendingLabel === null) return piece;

  const candidate = state.pendingLabel + piece;
  if (channelLabels.includes(candidate)) {
    state.pendingLabel = null;
    state.skipLabelNewline = true;
    return "";
  }
  const label = channelLabels.find((value) => candidate.startsWith(value));
  const isPrefix = channelLabels.some((value) => value.startsWith(candidate));
  if (isPrefix) {
    // A BPE token may hold only part of `thought`; wait until its label is
    // complete so none of it leaks into `reasoning_content`.
    state.pendingLabel = candidate;
    return "";
  }
  if (label) {
    state.pendingLabel = null;
    return candidate.slice(label.length).replace(/^\r?\n/, "");
  }

  // This is an unlabelled channel. It is real model output, not control text.
  state.pendingLabel = null;
  return candidate;
}

function cleanGemmaControl(piece: string, state: GemmaChannelState): string {
  if (!piece) return "";

  const closesChannel = piece.includes("<channel|>");
  if (piece.includes("<|channel>")) {
    state.reasoning = true;
    state.pendingLabel = "";
  }

  let cleaned = piece
    .replace(/<\|channel\>/g, "")
    .replace(/<channel\|>\n?/g, "")
    .replace(/<\|think\|>\n?/g, "");
  cleaned = stripChannelLabel(cleaned, state);

  if (closesChannel) state.reasoning = false;
  return cleaned;
}

export async function runGemmaCompletion(
  request: CompletionRequest,
  signal: AbortSignal,
  emitDelta: (delta: CompletionDelta) => void,
  emitProgress: (event: ProgressEvent) => void,
): Promise<CompletionResult> {
  const messages = validateMessages(request.messages);
  const model = engine.gemmaModel;
  const tok = engine.gemmaTok;
  const thinking = request.thinking ?? true;

  const wantsTools = !!request.tools?.length;
  const toolCallClose = tok.toolCallClose;
  const canTool = wantsTools && toolCallClose !== undefined && tok.toolCallOpen !== undefined;
  if (wantsTools && !canTool) {
    throw new Error("The loaded Gemma checkpoint has no tool-call tokens; function calling requires the -it checkpoint.");
  }
  const constraint = canTool
    ? new ToolConstraint(request.tools!, tok, {
        closeToken: "<tool_call|>",
        grammar: "gemma",
        forbiddenTokenIds: [tok.eos, tok.turnEnd],
      })
    : null;

  model.resetPenaltyWindow();
  await model.reset();

  const prompt = await prepareGemmaPrompt(messages, tok, model, {
    thinking,
    signal,
    tools: request.tools,
    onStage: (message) => emitProgress({ stage: "media", message, progress: null }),
  });
  if (model.pos + prompt.tokenIds.length >= model.maxCtx - 1) {
    throw new Error(
      `Prompt is too long: ${prompt.tokenIds.length} tokens does not fit the ${model.maxCtx}-token Gemma context window.`
    );
  }
  const sampling = {
    temperature: clamp(request.temperature ?? 0.6, 0, 2),
    topP: clamp(request.top_p ?? 0.95, 0.05, 1),
    topK: Math.round(clamp(request.top_k ?? 20, 1, 128)),
    presencePenalty: clamp(request.presence_penalty ?? 1.5, 0, 2),
  };
  const cands = await model.prefill(
    prompt.tokenIds,
    (done, total) =>
      emitProgress({
        stage: "prefill",
        message: `Processing Gemma prompt… ${done} / ${total} tokens`,
        progress: total ? done / total : null,
      }),
    prompt.overrides,
  );

  const stopIds = [tok.eos, tok.turnEnd, ...(canTool ? [toolCallClose as number] : [])];
  const maxNew = Math.max(1, Math.min(request.max_tokens ?? model.maxCtx, model.maxCtx - model.pos - 1));
  const decode = tok.makeDecoder();
  const channel: GemmaChannelState = { reasoning: false, pendingLabel: null, skipLabelNewline: false };
  const parallelCalls = !!request.parallel_tool_calls;

  let content = "";
  let reasoning = "";
  let total = 0;

  const toolCalls: ToolCall[] = [];
  let toolParser = new GemmaToolCallParser();
  let toolIndex = 0;
  let toolCallId = makeToolCallId(toolIndex);
  let toolNameSent = false;
  let toolRawSent = 0;

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

  const emitContent = (chunk: string) => {
    if (!chunk) return;
    if (toolCalls.length && chunk.trim() === "") return;
    content += chunk;
    emitDelta({ content: chunk });
  };

  const processVisible = (text: string) => {
    if (!canTool) {
      emitContent(text);
      return;
    }
    emitContent(toolParser.feed(text));
    emitToolDelta();
  };

  const consume = (id: number) => {
    const visible = cleanGemmaControl(decode(id), channel);
    if (!visible) return;
    if (channel.reasoning) {
      reasoning += visible;
      emitDelta({ reasoning_content: visible });
    } else {
      processVisible(visible);
    }
  };

  let next = sample(cands as unknown as { ids: number[]; vals: number[] }, {
    ...sampling,
    recentIds: model.recentSet(),
  });
  let stopped = stopIds.includes(next);
  if (!stopped) model.notePenaltyToken(next);

  for (;;) {
    while (!signal.aborted && total < maxNew && !stopped && !toolParser.isComplete) {
      consume(next);
      total++;
      if (toolParser.isComplete) break;
      if (total >= maxNew || model.pos >= model.maxCtx - 1) break;
      const allowedTokenIds = constraint && toolParser.isOpen
        ? constraint.allowed(toolParser.rawBuffer, parseGemmaToolCallBody)
        : undefined;
      if (allowedTokenIds && !allowedTokenIds.length) throw new Error("Tool call cannot satisfy the declared schema.");
      const result = await model.decodeBatch(next, 1, { ...sampling, stopIds, eosId: tok.eos, allowedTokenIds });
      next = result.ids[0];
      stopped = result.stopped || stopIds.includes(next);
    }

    if (canTool && !toolParser.isComplete && toolParser.isOpen && next === toolCallClose) {
      consume(next);
    }

    const parsed = toolParser.isComplete ? toolParser.result : null;
    if (!parsed) break;
    toolCalls.push({
      id: toolCallId,
      type: "function",
      function: { name: parsed.name, arguments: JSON.stringify(parsed.arguments) },
    });
    if (!parallelCalls || signal.aborted || total >= maxNew || model.maxCtx - model.pos - 1 < 2) break;

    const cont = await model.decodeBatch(next, 1, { ...sampling, stopIds, eosId: tok.eos });
    next = cont.ids[0];
    stopped = cont.stopped || stopIds.includes(next);
    toolParser = new GemmaToolCallParser();
    toolIndex++;
    toolCallId = makeToolCallId(toolIndex);
    toolNameSent = false;
    toolRawSent = 0;
  }

  emitContent(toolParser.flush());

  engine.committed = null;
  const finalToolCalls = toolCalls.length ? toolCalls : null;
  return {
    content,
    reasoning_content: reasoning,
    tool_calls: finalToolCalls,
    finish_reason: finalToolCalls
      ? "tool_calls"
      : signal.aborted
        ? "abort"
        : stopped
          ? "stop"
          : "length",
    usage: {
      prompt_tokens: prompt.tokenIds.length,
      completion_tokens: total,
      total_tokens: prompt.tokenIds.length + total,
    },
    context: { used_tokens: model.pos, max_tokens: model.maxCtx },
  };
}
