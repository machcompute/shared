/** Prompt construction for Gemma 4's documented `<|turn>` protocol.
 *
 * We intentionally do not use Qwen's Jinja/XML tool template here: Gemma 4
 * has different reserved tokens and multimodal placeholder expansion rules.
 */
export type GemmaMediaType = "image" | "audio" | "video";

/**
 * URL-only normalized content shape.  This deliberately matches
 * gemma-media.ts so template users and the actual media preprocessor accept
 * the same OpenAI-style content representation.
 */
export type GemmaContentPart =
  | { type: "text"; text: string }
  | { type: "image"; url: string; detail?: "low" | "high" }
  | { type: "audio"; url: string }
  | { type: "video"; url: string; frames?: number };

export interface GemmaRenderMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string | GemmaContentPart[];
}

export type GemmaPromptPart =
  | { type: "text"; text: string }
  | { type: GemmaMediaType; url: string };

function partsForContent(
  content: string | GemmaContentPart[],
  sanitize: (text: string) => string
): GemmaPromptPart[] {
  if (typeof content === "string") return [{ type: "text", text: sanitize(content) }];
  return content.map((part) => {
    if (part.type === "text") return { type: "text", text: sanitize(part.text) };
    if (part.type === "image") return { type: "image", url: part.url };
    if (part.type === "audio") return { type: "audio", url: part.url };
    return { type: "video", url: part.url };
  });
}

/**
 * Build a sequence rather than a single string because image/audio/video
 * placeholder counts are known only after their browser preprocessing step.
 */
export function renderGemmaPrompt(
  messages: GemmaRenderMessage[],
  sanitize: (text: string) => string,
  { thinking }: { thinking: boolean }
): GemmaPromptPart[] {
  const out: GemmaPromptPart[] = [{ type: "text", text: "<bos>" }];
  let sawSystem = false;

  for (const message of messages) {
    // Gemma documents system/user/model roles. Tool turns remain textual
    // context until Gemma-specific function-call serialization is enabled.
    const role = message.role === "assistant" ? "model" : message.role === "tool" ? "user" : message.role;
    out.push({ type: "text", text: `<|turn>${role}\n` });
    if (role === "system" && thinking && !sawSystem) {
      out.push({ type: "text", text: "<|think|>" });
      sawSystem = true;
    }
    out.push(...partsForContent(message.content, sanitize));
    out.push({ type: "text", text: "<turn|>\n" });
  }

  if (thinking && !sawSystem) {
    // Thinking is a conversation-level control located in a system turn.
    out.splice(1, 0, { type: "text", text: "<|turn>system\n<|think|><turn|>\n" });
  }
  out.push({ type: "text", text: "<|turn>model\n" });
  return out;
}

export function multimodalPlaceholder(type: GemmaMediaType, count: number): string {
  if (!Number.isInteger(count) || count < 1) {
    throw new Error(`Gemma ${type} preprocessing produced no soft tokens.`);
  }
  if (type === "image") return `<|image>${"<|image|>".repeat(count)}<image|>`;
  if (type === "audio") return `<|audio>${"<|audio|>".repeat(count)}<audio|>`;
  // Videos use the vision tower; Gemma's processor wraps video placeholders
  // in image boundaries while retaining distinct video placeholder IDs.
  return `<|image>${"<|video|>".repeat(count)}<image|>`;
}
