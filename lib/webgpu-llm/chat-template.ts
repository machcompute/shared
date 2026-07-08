import { Template } from "@huggingface/jinja";

export type ChatRole = "system" | "user" | "assistant" | "tool";

export interface RenderMessage {
  role: ChatRole;
  content: string;
  tool_calls?: { name: string; arguments: Record<string, unknown> }[];
}

export interface FunctionToolDef {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
}

// Full prompt rebuild — only used for the first turn of a conversation or
// whenever committed GPU state diverges from the thread's message history.
// Uses the model's own live-fetched chat_template (not a hand reconstruction)
// since Qwen's tool-call format is a custom <tool_call>/<function>/<parameter>
// grammar, not OpenAI-style JSON.
export function renderFullPrompt(
  chatTemplate: string,
  messages: RenderMessage[],
  tools: FunctionToolDef[] | undefined,
  { thinking }: { thinking: boolean }
): string {
  const template = new Template(chatTemplate);
  return template.render({
    messages,
    tools: tools && tools.length ? tools : undefined,
    add_generation_prompt: true,
    enable_thinking: thinking,
  });
}

// Incremental suffixes — verified against the real template's own output for
// these exact transitions (see plan). Rendering the full prompt on every turn
// isn't a stable-prefix operation (the <think> block on historical assistant
// turns depends on which turn is "last"), so continuations feed only the new
// suffix directly into the live GPU-resident KV/DeltaNet state.
export function buildUserTurnSuffix(userText: string, thinking: boolean): string {
  return (
    `<|im_end|>\n<|im_start|>user\n${userText.trim()}<|im_end|>\n<|im_start|>assistant\n` +
    (thinking ? "<think>\n" : "<think>\n\n</think>\n\n")
  );
}

// Deliberately does NOT include the leading `</tool_call>`: whether the close
// tag is already in GPU state depends on how the turn ended (fed via lookahead
// in parallel mode, frozen out as a stop token in single-call mode), so the
// caller prepends it only when it is still pending.
export function buildToolResponsesSuffix(resultTexts: string[], thinking: boolean): string {
  return (
    `<|im_end|>\n<|im_start|>user\n` +
    resultTexts.map((t) => `<tool_response>\n${t}\n</tool_response>`).join("\n") +
    `<|im_end|>\n<|im_start|>assistant\n` +
    (thinking ? "<think>\n" : "<think>\n\n</think>\n\n")
  );
}
