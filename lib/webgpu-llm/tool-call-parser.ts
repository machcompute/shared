// Incremental parser for Qwen's tool-call grammar, which is XML-ish rather
// than JSON:
//   <tool_call>
//   <function=NAME>
//   <parameter=PARAM>
//   value
//   </parameter>
//   </function>
//   </tool_call>
// All of this app's existing tools take only string parameters, so parameter
// values are used as raw strings — no JSON parsing of scalar args for v1.

export interface ParsedToolCall {
  name: string;
  arguments: Record<string, string>;
}

const TOOL_CALL_OPEN = "<tool_call>";
const TOOL_CALL_CLOSE = "</tool_call>";
const FUNCTION_OPEN_RE = /<function=([^>]+)>/;
const PARAMETER_RE = /<parameter=([^>]+)>\n?([\s\S]*?)\n?<\/parameter>/g;

type ParserState = "idle" | "in-call" | "complete";

export class ToolCallStreamParser {
  private state: ParserState = "idle";
  private preBuffer = "";
  private buffer = "";
  private _toolName: string | null = null;
  private _result: ParsedToolCall | null = null;

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

  get result(): ParsedToolCall | null {
    return this._result;
  }

  // Feed a piece of newly decoded answer text (not <think> content). Returns
  // the portion of the piece that should still render as visible chat prose
  // (empty once a <tool_call> block has been entered).
  feed(piece: string): string {
    if (this.state === "complete") return piece;

    if (this.state === "idle") {
      this.preBuffer += piece;
      const idx = this.preBuffer.indexOf(TOOL_CALL_OPEN);
      if (idx < 0) {
        // hold back a tail in case the open tag itself spans chunks
        const keep = Math.min(this.preBuffer.length, TOOL_CALL_OPEN.length - 1);
        const visible = this.preBuffer.slice(0, this.preBuffer.length - keep);
        this.preBuffer = this.preBuffer.slice(this.preBuffer.length - keep);
        return visible;
      }
      const visible = this.preBuffer.slice(0, idx);
      this.buffer = this.preBuffer.slice(idx + TOOL_CALL_OPEN.length);
      this.preBuffer = "";
      this.state = "in-call";
      this.tryParseFunctionName();
      this.tryClose();
      return visible;
    }

    this.buffer += piece;
    this.tryParseFunctionName();
    this.tryClose();
    return "";
  }

  // End-of-stream: release the tail withheld while watching for a split
  // <tool_call> open tag. Only meaningful in the idle state — once a call has
  // opened, buffered content is tool-call body, not prose.
  flush(): string {
    if (this.state !== "idle") return "";
    const rest = this.preBuffer;
    this.preBuffer = "";
    return rest;
  }

  private tryParseFunctionName() {
    if (this._toolName) return;
    const m = FUNCTION_OPEN_RE.exec(this.buffer);
    if (m) this._toolName = m[1].trim();
  }

  private tryClose() {
    const closeIdx = this.buffer.indexOf(TOOL_CALL_CLOSE);
    if (closeIdx < 0) return;
    this._result = parseToolCallBody(this.buffer.slice(0, closeIdx));
    this.state = "complete";
  }
}

function parseToolCallBody(inner: string): ParsedToolCall {
  const fnMatch = FUNCTION_OPEN_RE.exec(inner);
  const name = fnMatch ? fnMatch[1].trim() : "";
  const args: Record<string, string> = {};
  for (const m of inner.matchAll(PARAMETER_RE)) {
    args[m[1].trim()] = m[2];
  }
  return { name, arguments: args };
}
