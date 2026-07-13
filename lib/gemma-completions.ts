import { engine, type ProgressEvent } from "./engine";
import {
  messageSig,
  promptPrefixKey,
  validateMessages,
  type CompletionDelta,
  type CompletionRequest,
  type CompletionResult,
  type ChatMessage,
  type ToolCall,
} from "./completions";
import {
  prepareGemmaPrompt,
  prepareGemmaToolResponsesSuffix,
  prepareGemmaUserTurnSuffix,
  type PreparedGemmaPrompt,
} from "./gemma-processor";
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
  // The tool-constraint mask survives on the model between requests; only
  // reset() cleared it before suffix continuations existed.
  model.clearAllowedTokenIds();

  const onStage = (message: string) => emitProgress({ stage: "media", message, progress: null });
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

  // Suffix continuations skip model.reset() and prefill on top of the
  // GPU-resident prefix. The cache keeps the model's own generated stream —
  // including reasoning the full render would drop — so a continuation and a
  // fresh render condition slightly differently; that trade matches Qwen.
  let prompt: PreparedGemmaPrompt | null = null;
  const priorToolMessages = extra?.filter(
    (m): m is ChatMessage & { role: "tool" } => m.role === "tool",
  );
  if (extra && committed && extra.length === 1 && extra[0].role === "user" && committed.toolCallCount === 0) {
    prompt = await prepareGemmaUserTurnSuffix(extra[0], tok, model, {
      signal,
      basePos: model.pos,
      onStage,
    });
  } else if (
    extra &&
    committed &&
    committed.toolCallCount > 0 &&
    priorToolMessages &&
    priorToolMessages.length === committed.toolCallCount &&
    extra.slice(0, priorToolMessages.length).every((m) => m.role === "tool") &&
    // The delta is the round's tool responses, optionally followed by one
    // injected user turn (e.g. a screenshot of the tool's result).
    (extra.length === committed.toolCallCount ||
      (extra.length === committed.toolCallCount + 1 && extra[extra.length - 1].role === "user"))
  ) {
    const prior = messages[committed.sigs.length - 1];
    const priorCalls = prior.role === "assistant" ? prior.tool_calls ?? [] : [];
    const toolSuffix = prepareGemmaToolResponsesSuffix(
      priorToolMessages.map((m) => ({
        name: priorCalls.find((tc) => tc.id === m.tool_call_id)?.function.name ?? "unknown",
        content: m.content,
      })),
      tok,
      { pending: committed.pending === "tool-close" ? "tool-close" : "none" },
    );
    const trailingUser = extra.length > committed.toolCallCount ? extra[extra.length - 1] : null;
    if (trailingUser) {
      // A user turn directly after tool responses does not close the model
      // turn first — the renderer leaves it open after responses.
      const userSuffix = await prepareGemmaUserTurnSuffix(trailingUser, tok, model, {
        signal,
        basePos: model.pos + toolSuffix.tokenIds.length,
        onStage,
        closeTurn: false,
      });
      prompt = {
        tokenIds: [...toolSuffix.tokenIds, ...userSuffix.tokenIds],
        overrides: userSuffix.overrides,
        mediaTokenCount: userSuffix.mediaTokenCount,
      };
    } else {
      prompt = toolSuffix;
    }
  }
  if (prompt && model.pos + prompt.tokenIds.length >= model.maxCtx - 1) {
    // The continuation overflows; a full render is more compact (it drops
    // prior generated reasoning) and may still fit.
    prompt = null;
  }
  if (!prompt) {
    await model.reset();
    prompt = await prepareGemmaPrompt(messages, tok, model, {
      thinking,
      signal,
      tools: request.tools,
      onStage,
    });
  }
  if (model.pos + prompt.tokenIds.length >= model.maxCtx - 1) {
    throw new Error(
      `Prompt is too long: ${prompt.tokenIds.length} tokens does not fit the ${model.maxCtx}-token Gemma context window.`
    );
  }
  // Gemma 4's model card standardizes sampling at temperature 1.0 / top-p
  // 0.95 / top-k 64 with repetition penalties off. A presence penalty is
  // actively harmful under constrained tool-call decoding — digits, commas,
  // and quote markers all live inside the penalty window. Top-k stays at the
  // GPU top-k gather width (20 candidates).
  const sampling = {
    temperature: clamp(request.temperature ?? 1.0, 0, 2),
    topP: clamp(request.top_p ?? 0.95, 0.05, 1),
    topK: Math.round(clamp(request.top_k ?? 20, 1, 128)),
    presencePenalty: clamp(request.presence_penalty ?? 0, 0, 2),
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
  // Whether the latest tool call's close tag reached the KV cache (only the
  // parallel-call lookahead feeds it; as a stop token it is sampled unfed).
  let lastCloseFed = false;
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

  // True when `next` was already consumed by the mid-batch early exit below;
  // consuming it again would feed the tool-call opener into the parser's body
  // buffer and derail the schema constraint.
  let nextConsumed = false;
  for (;;) {
    while (!signal.aborted && total < maxNew && !stopped && !toolParser.isComplete) {
      if (!nextConsumed) {
        consume(next);
        total++;
      }
      nextConsumed = false;
      if (toolParser.isComplete) break;
      if (total >= maxNew || model.pos >= model.maxCtx - 1) break;
      const allowedTokenIds = constraint && toolParser.isOpen
        ? constraint.allowed(toolParser.rawBuffer, parseGemmaToolCallBody)
        : undefined;
      if (allowedTokenIds && !allowedTokenIds.length) throw new Error("Tool call cannot satisfy the declared schema.");
      const k = Math.min(
        toolParser.isOpen ? 1 : model.BATCH,
        maxNew - total,
        model.maxCtx - model.pos,
      );
      const result = await model.decodeBatch(next, k, { ...sampling, stopIds, eosId: tok.eos, allowedTokenIds });
      for (let i = 0; i < result.ids.length; i++) {
        const id = result.ids[i];
        const last = i === result.ids.length - 1;
        if (last) {
          next = id;
          stopped = result.stopped || stopIds.includes(id);
          break;
        }
        consume(id);
        total++;
        if (toolParser.isOpen || toolParser.isComplete || total >= maxNew) {
          const unused = result.ids.length - i - 1;
          model.rewindDecode(unused);
          next = id;
          nextConsumed = true;
          break;
        }
      }
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
    lastCloseFed = false;
    if (!parallelCalls || signal.aborted || total >= maxNew || model.maxCtx - model.pos - 1 < 2) break;

    const cont = await model.decodeBatch(next, 1, { ...sampling, stopIds, eosId: tok.eos });
    // That call fed the pending close into the cache as its input token.
    lastCloseFed = true;
    next = cont.ids[0];
    // `next` is a fresh token even if the loop above exited with a consumed
    // pivot (a tool call completing mid-batch leaves nextConsumed set).
    nextConsumed = false;
    // A sampled <|tool_response> opener means the model is done calling and
    // awaits results. Treat it as a stop: consuming it would let the model
    // hallucinate a tool response into the cache, and the real response
    // suffix supplies its own opener (this token was sampled, never fed).
    stopped = cont.stopped || stopIds.includes(next)
      || next === tok.specialTokenId("<|tool_response>");
    toolParser = new GemmaToolCallParser();
    toolIndex++;
    toolCallId = makeToolCallId(toolIndex);
    toolNameSent = false;
    toolRawSent = 0;
  }

  emitContent(toolParser.flush());

  const finalToolCalls = toolCalls.length ? toolCalls : null;
  // Commit only clean stops: a length-capped turn leaves its final sampled
  // token consumed into `content` but never fed, so the cache would diverge
  // from the assistant message the client echoes back.
  if (stopped && !signal.aborted) {
    engine.committed = {
      model: engine.activeModelId,
      promptKey,
      sigs: [
        ...sigs,
        messageSig({ role: "assistant", content, tool_calls: finalToolCalls ?? undefined }),
      ],
      toolCallCount: toolCalls.length,
      pending: toolCalls.length ? (lastCloseFed ? "none" : "tool-close") : "turn-close",
    };
  }
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
