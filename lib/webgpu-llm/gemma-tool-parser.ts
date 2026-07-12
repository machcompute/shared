import { parseToolArguments } from "./gemma-tool-format";

export interface ParsedGemmaToolCall {
  name: string;
  arguments: Record<string, unknown>;
}

const TOOL_CALL_OPEN = "<|tool_call>";
const TOOL_CALL_CLOSE = "<tool_call|>";
const NAME_RE = /^\s*(?:call:)?\s*([A-Za-z_][\w.-]*)\s*/;

type ParserState = "idle" | "in-call" | "complete";

export class GemmaToolCallParser {
  private state: ParserState = "idle";
  private preBuffer = "";
  private buffer = "";
  private _toolName: string | null = null;
  private _result: ParsedGemmaToolCall | null = null;

  get isOpen(): boolean {
    return this.state === "in-call";
  }

  get isComplete(): boolean {
    return this.state === "complete";
  }

  get toolName(): string | null {
    return this._toolName;
  }

  get rawBuffer(): string {
    return this.buffer;
  }

  get result(): ParsedGemmaToolCall | null {
    return this._result;
  }

  feed(piece: string): string {
    if (this.state === "complete") return piece;

    if (this.state === "idle") {
      this.preBuffer += piece;
      const idx = this.preBuffer.indexOf(TOOL_CALL_OPEN);
      if (idx < 0) {
        const keep = Math.min(this.preBuffer.length, TOOL_CALL_OPEN.length - 1);
        const visible = this.preBuffer.slice(0, this.preBuffer.length - keep);
        this.preBuffer = this.preBuffer.slice(this.preBuffer.length - keep);
        return visible;
      }
      const visible = this.preBuffer.slice(0, idx);
      this.buffer = this.preBuffer.slice(idx + TOOL_CALL_OPEN.length);
      this.preBuffer = "";
      this.state = "in-call";
      this.tryParseName();
      this.tryClose();
      return visible;
    }

    this.buffer += piece;
    this.tryParseName();
    this.tryClose();
    return "";
  }

  flush(): string {
    if (this.state !== "idle") return "";
    const rest = this.preBuffer;
    this.preBuffer = "";
    return rest;
  }

  private tryParseName() {
    if (this._toolName) return;
    const m = NAME_RE.exec(this.buffer);
    if (m) this._toolName = m[1];
  }

  private tryClose() {
    const closeIdx = this.buffer.indexOf(TOOL_CALL_CLOSE);
    if (closeIdx < 0) return;
    this._result = parseGemmaToolCallBody(this.buffer.slice(0, closeIdx));
    this.state = "complete";
  }
}

export function parseGemmaToolCallBody(inner: string): ParsedGemmaToolCall {
  const m = NAME_RE.exec(inner);
  const name = m ? m[1] : "";
  const rest = (m ? inner.slice(m[0].length) : inner).trim();
  const start = rest.indexOf("{");
  return { name, arguments: start < 0 ? {} : parseToolArguments(rest.slice(start)) };
}
