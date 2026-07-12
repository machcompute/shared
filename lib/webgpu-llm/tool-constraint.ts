/**
 * Shared tool-call constraint contract.  It deliberately operates on token
 * IDs, leaving each model's prompt grammar/parser as a small adapter.
 */
export type JsonSchema = Record<string, unknown>;

export interface ConstraintTokenizer {
  encode(text: string): number[];
  vocabSize(): number;
  specialTokenId(text: string): number | undefined;
}

export interface ConstraintTool {
  function: { name: string; parameters?: JsonSchema };
}

const SUPPORTED = new Set(["type", "properties", "required", "items", "enum", "nullable", "description"]);

function checkSchema(schema: unknown, path = "parameters"): void {
  if (!schema || typeof schema !== "object" || Array.isArray(schema)) return;
  const record = schema as JsonSchema;
  for (const key of Object.keys(record)) {
    if (!SUPPORTED.has(key)) throw new Error(`Unsupported tool JSON Schema keyword at ${path}: ${key}`);
  }
  if (record.properties && typeof record.properties === "object" && !Array.isArray(record.properties)) {
    for (const [key, child] of Object.entries(record.properties as JsonSchema)) checkSchema(child, `${path}.properties.${key}`);
  }
  if (record.items) checkSchema(record.items, `${path}.items`);
}

function typeMatches(value: unknown, schema: JsonSchema): boolean {
  if (schema.enum && Array.isArray(schema.enum) && !schema.enum.some((item) => JSON.stringify(item) === JSON.stringify(value))) return false;
  const type = schema.type ?? (schema.properties ? "object" : undefined);
  if (type === "string") return typeof value === "string";
  if (type === "number") return typeof value === "number" && Number.isFinite(value);
  if (type === "integer") return typeof value === "number" && Number.isInteger(value);
  if (type === "boolean") return typeof value === "boolean";
  if (type === "null") return value === null;
  if (type === "array") return Array.isArray(value) && (!schema.items || value.every((item) => typeMatches(item, schema.items as JsonSchema)));
  if (type === "object") {
    if (!value || typeof value !== "object" || Array.isArray(value)) return false;
    const obj = value as JsonSchema;
    const props = (schema.properties ?? {}) as JsonSchema;
    if (Object.keys(obj).some((key) => !(key in props))) return false;
    if (Array.isArray(schema.required) && schema.required.some((key) => !(key in obj))) return false;
    return Object.entries(obj).every(([key, item]) => typeMatches(item, props[key] as JsonSchema));
  }
  // Omitted `type` is permitted by OpenAI-style schemas.
  return true;
}

/**
 * The constrained segment starts only after the native tool opener. Names are
 * token-constrained immediately; argument completion is schema-validated
 * before the native close token is ever admitted.
 */
export class ToolConstraint {
  private readonly tools = new Map<string, JsonSchema>();
  private readonly all: Uint32Array;
  private readonly closeId: number | undefined;
  private readonly namePrefix: string;
  private readonly forbiddenIds: Set<number>;

  constructor(
    tools: ConstraintTool[],
    tokenizer: ConstraintTokenizer,
    options: { closeToken: string; namePrefix: string; forbiddenTokenIds?: readonly number[] }
  ) {
    for (const tool of tools) {
      if (!tool.function?.name) throw new Error("Tool constraint requires function names.");
      checkSchema(tool.function.parameters);
      this.tools.set(tool.function.name, tool.function.parameters ?? { type: "object" });
    }
    this.all = Uint32Array.from({ length: tokenizer.vocabSize() }, (_, id) => id);
    this.closeId = tokenizer.specialTokenId(options.closeToken);
    this.namePrefix = options.namePrefix;
    this.forbiddenIds = new Set(options.forbiddenTokenIds ?? []);
    this.tokenizer = tokenizer;
  }

  private readonly tokenizer: ConstraintTokenizer;

  allowed(raw: string, parseArguments: (raw: string) => { name: string; arguments: Record<string, unknown> }): Uint32Array {
    const emitted = raw.trimStart();
    // Include the native function-tag terminator. Stopping at the name left
    // the model free to emit malformed XML immediately after a valid name.
    const targets = [...this.tools.keys()].map((toolName) => `${this.namePrefix}${toolName}>`);
    if (targets.some((target) => target.startsWith(emitted)) && !targets.includes(emitted)) {
      const ids = new Set<number>();
      for (const target of targets) {
        if (!target.startsWith(emitted)) continue;
        const suffix = target.slice(emitted.length);
        const token = this.tokenizer.encode(suffix)[0];
        if (token !== undefined) ids.add(token);
      }
      return Uint32Array.from([...ids].sort((a, b) => a - b));
    }

    // Keep every vocabulary token reachable for value generation. Once the
    // native body is complete and schema-valid, however, force its close
    // token: continuing as prose leaves the streaming parser in-call and
    // makes a perfectly valid invocation look truncated to clients.
    const parsed = parseArguments(raw);
    const schema = this.tools.get(parsed.name);
    if (this.closeId !== undefined && schema && this.isStructurallyComplete(raw) && typeMatches(parsed.arguments, schema)) {
      return Uint32Array.of(this.closeId);
    }

    // Before the body is valid, exclude the native close control token and
    // terminal tokens. Otherwise a malformed call can end as ordinary text
    // and never reach the consumer's tool dispatcher.
    const allowed = Array.from(this.all).filter(
      (id) => id !== this.closeId && !this.forbiddenIds.has(id)
    );
    return Uint32Array.from(allowed.sort((a, b) => a - b));
  }

  private isStructurallyComplete(raw: string): boolean {
    if (this.namePrefix === "<function=") {
      const parametersOpen = (raw.match(/<parameter=/g) ?? []).length;
      const parametersClose = (raw.match(/<\/parameter>/g) ?? []).length;
      return /<function=[^>]+>[\s\S]*<\/function>\s*$/.test(raw) && parametersOpen === parametersClose;
    }
    let depth = 0;
    let quotes = 0;
    for (let i = 0; i < raw.length; i++) {
      if (raw.startsWith('<|"|>', i)) { quotes++; i += 4; continue; }
      if (quotes % 2) continue;
      if (raw[i] === '{') depth++;
      if (raw[i] === '}') depth--;
      if (depth < 0) return false;
    }
    return depth === 0 && quotes % 2 === 0 && raw.includes('{');
  }
}
