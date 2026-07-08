import { engine, type ProgressEvent } from "./engine";
import {
  renderFullPrompt,
  buildUserTurnSuffix,
  type RenderMessage,
} from "./webgpu-llm/chat-template";
import { sample } from "./webgpu-llm/tokenizer.js";

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface CompletionRequest {
  messages: ChatMessage[];
  temperature?: number;
  top_p?: number;
  top_k?: number;
  presence_penalty?: number;
  max_tokens?: number | null;
  thinking?: boolean;
}

export interface CompletionDelta {
  content?: string;
  reasoning_content?: string;
}

export interface CompletionResult {
  content: string;
  reasoning_content: string;
  finish_reason: "stop" | "length" | "abort";
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

const randSeed = () => (Math.random() * 0x100000000) >>> 0;

const messageSig = (message: ChatMessage) =>
  JSON.stringify([message.role, message.content]);

export function validateMessages(input: unknown): ChatMessage[] {
  if (!Array.isArray(input) || input.length === 0) {
    throw new Error("messages must be a non-empty array");
  }
  const messages = input.map((raw, i): ChatMessage => {
    const { role, content } = (raw ?? {}) as { role?: unknown; content?: unknown };
    if (role !== "system" && role !== "user" && role !== "assistant") {
      throw new Error(`messages[${i}].role must be "system", "user" or "assistant"`);
    }
    if (typeof content !== "string") {
      throw new Error(`messages[${i}].content must be a string`);
    }
    if (role === "system" && i !== 0) {
      throw new Error("a system message is only allowed as the first message");
    }
    return { role, content };
  });
  if (messages[messages.length - 1].role !== "user") {
    throw new Error('the last message must have role "user"');
  }
  return messages;
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
  const lastMessage = messages[messages.length - 1];
  const committed = engine.committedSigs;
  const isContinuation =
    !!committed &&
    messages.length === committed.length + 1 &&
    committed.every((sig, i) => sig === sigs[i]);
  engine.committedSigs = null;

  let promptText: string;
  if (isContinuation) {
    promptText = buildUserTurnSuffix(sanitize(lastMessage.content), thinking);
  } else {
    await model.reset();
    const history: RenderMessage[] = messages.map((m) => ({
      role: m.role,
      content: sanitize(m.content),
    }));
    promptText = renderFullPrompt(tok.chatTemplate, history, undefined, { thinking });
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

  const maxNew = Math.max(1, Math.min(request.max_tokens ?? model.maxCtx, model.maxCtx));
  const stopIds = [tok.eos];
  const decodeTok = tok.makeDecoder();

  let reasoningText = "";
  let contentText = "";
  let inThink = thinking;
  let pending = "";

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
        emitContent(after);
      } else if (pending.length > 12) {
        emitReasoning(pending.slice(0, -12));
        pending = pending.slice(-12);
      }
    } else {
      emitContent(piece);
    }
  };

  let totalN = 0;
  let next = sample(cands, { ...samplingParams, recentIds: model.recentSet() });
  let hitEos = next === tok.eos;
  if (!hitEos) model.notePenaltyToken(next);

  while (totalN < maxNew && !signal.aborted && !hitEos) {
    emit(decodeTok(next));
    totalN++;

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
        }
        if (round.a >= 2 && round.d0 != null) {
          emit(decodeTok(round.d0));
          totalN++;
        }
        if (round.a >= 3 && round.d1 != null) {
          emit(decodeTok(round.d1));
          totalN++;
        }
        if (round.next == null) {
          roundInput = null;
          break;
        }
        if (r.stopped && round.next === r.stopId) {
          hitEos = round.next === tok.eos;
          roundInput = null;
          break;
        }
        roundInput = round.next;
      }
      if (hitEos || roundInput == null) break;
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
      if (j === r.ids.length - 1 && !r.stopped) {
        next = id;
        break;
      }
      emit(decodeTok(id));
      totalN++;
    }
  }

  if (pending) emitReasoning(pending);

  const aborted = signal.aborted;
  if (!aborted) {
    engine.committedSigs = [...sigs, messageSig({ role: "assistant", content: contentText })];
  }

  return {
    content: contentText,
    reasoning_content: reasoningText,
    finish_reason: hitEos ? "stop" : aborted ? "abort" : "length",
    usage: {
      prompt_tokens: ids.length,
      completion_tokens: totalN,
      total_tokens: ids.length + totalN,
    },
  };
}
