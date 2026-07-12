/**
 * Shared tool-call constraint contract.  It deliberately operates on token
 * IDs, leaving each model's prompt grammar/parser as a small adapter.
 */
import { parseToolArguments } from "./gemma-tool-format";

export type JsonSchema = Record<string, unknown>;

export interface ConstraintTokenizer {
  encode(text: string): number[];
  vocabSize(): number;
  specialTokenId(text: string): number | undefined;
}

export interface ConstraintTool {
  function: { name: string; parameters?: JsonSchema };
}

export type ToolGrammar = "qwen" | "gemma";

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
  private readonly grammar: ToolGrammar;
  private readonly forbiddenIds: Set<number>;

  constructor(
    tools: ConstraintTool[],
    tokenizer: ConstraintTokenizer,
    options: { closeToken: string; grammar: ToolGrammar; forbiddenTokenIds?: readonly number[] }
  ) {
    for (const tool of tools) {
      if (!tool.function?.name) throw new Error("Tool constraint requires function names.");
      checkSchema(tool.function.parameters);
      this.tools.set(tool.function.name, tool.function.parameters ?? { type: "object" });
    }
    this.all = Uint32Array.from({ length: tokenizer.vocabSize() }, (_, id) => id);
    this.closeId = tokenizer.specialTokenId(options.closeToken);
    this.grammar = options.grammar;
    this.forbiddenIds = new Set(options.forbiddenTokenIds ?? []);
    this.tokenizer = tokenizer;
  }

  private readonly tokenizer: ConstraintTokenizer;

  private unrestricted(): Uint32Array {
    return Uint32Array.from(
      Array.from(this.all).filter(
        (id) => id !== this.closeId && !this.forbiddenIds.has(id)
      )
    );
  }

  private requirePrefix(emitted: string, targets: string[]): Uint32Array | null {
    const matches = targets.filter((target) => target.startsWith(emitted));
    if (!matches.length) return new Uint32Array();
    if (matches.includes(emitted)) return null;
    const ids = new Set<number>();
    for (const target of matches) {
      const token = this.tokenizer.encode(target.slice(emitted.length))[0];
      if (token !== undefined && !this.forbiddenIds.has(token)) ids.add(token);
    }
    return Uint32Array.from([...ids].sort((a, b) => a - b));
  }

  private qwenAllowed(raw: string): Uint32Array {
    const emitted = raw.trimStart();
    const functionTargets = [...this.tools.keys()].map((name) => `<function=${name}>`);
    const fnMatch = /^<function=([^>]+)>/.exec(emitted);
    if (!fnMatch) return this.requirePrefix(emitted, functionTargets) ?? this.unrestricted();

    const name = fnMatch[1].trim();
    const schema = this.tools.get(name);
    if (!schema) return new Uint32Array();
    const properties = (schema.properties ?? {}) as JsonSchema;
    const required = new Set(Array.isArray(schema.required) ? schema.required as string[] : []);
    const seen = new Set<string>();
    let rest = emitted.slice(fnMatch[0].length);

    for (;;) {
      rest = rest.trimStart();
      if (!rest) {
        const next = Object.keys(properties)
          .filter((key) => !seen.has(key))
          .map((key) => `<parameter=${key}>`);
        if ([...required].every((key) => seen.has(key))) next.push("</function>");
        return this.requirePrefix("", next) ?? this.unrestricted();
      }

      if (rest.startsWith("</function>")) {
        if (rest.slice("</function>".length).trim()) return new Uint32Array();
        if (![...required].every((key) => seen.has(key))) return new Uint32Array();
        const args = parseQwenArguments(emitted);
        if (!typeMatches(args, schema)) return new Uint32Array();
        return this.closeId === undefined ? new Uint32Array() : Uint32Array.of(this.closeId);
      }

      if (!rest.startsWith("<parameter=")) {
        const targets = Object.keys(properties)
          .filter((key) => !seen.has(key))
          .map((key) => `<parameter=${key}>`);
        if ([...required].every((key) => seen.has(key))) targets.push("</function>");
        return this.requirePrefix(rest, targets) ?? this.unrestricted();
      }

      const tagEnd = rest.indexOf(">");
      if (tagEnd < 0) {
        const targets = Object.keys(properties)
          .filter((key) => !seen.has(key))
          .map((key) => `<parameter=${key}>`);
        return this.requirePrefix(rest, targets) ?? this.unrestricted();
      }
      const parameterName = rest.slice("<parameter=".length, tagEnd).trim();
      if (!(parameterName in properties) || seen.has(parameterName)) return new Uint32Array();

      const valueStart = tagEnd + 1;
      const closeAt = rest.indexOf("</parameter>", valueStart);
      if (closeAt < 0) return this.unrestricted();
      const valueText = rest.slice(valueStart, closeAt).trim();
      const value = parseScalar(valueText);
      if (!typeMatches(value, properties[parameterName] as JsonSchema)) return new Uint32Array();
      seen.add(parameterName);
      rest = rest.slice(closeAt + "</parameter>".length);
    }
  }

  private gemmaAllowed(raw: string): Uint32Array {
    const emitted = raw.trimStart();
    const targets = [...this.tools.keys()].map((name) => `call:${name}{`);
    const match = /^(?:call:)?\s*([A-Za-z_][\w.-]*)\s*\{/.exec(emitted);
    if (!match) return this.requirePrefix(emitted, targets) ?? this.unrestricted();

    const schema = this.tools.get(match[1]);
    if (!schema) return new Uint32Array();
    const properties = (schema.properties ?? {}) as JsonSchema;
    const required = new Set(Array.isArray(schema.required) ? schema.required as string[] : []);
    const objectStart = emitted.indexOf("{", match.index + match[0].length - 1);
    const keyState = gemmaTopLevelKeyState(emitted.slice(objectStart), new Set(Object.keys(properties)));
    if (keyState.invalid) return new Uint32Array();
    if (keyState.prefix !== null) {
      const targets = [...keyState.remaining].map((key) => `${key}:`);
      if ([...required].every((key) => keyState.seen.has(key))) targets.push("}");
      return this.requirePrefix(keyState.prefix, targets) ?? this.unrestricted();
    }

    const parsed = parseGemmaArguments(emitted);
    if (this.isGemmaStructurallyComplete(emitted) && typeMatches(parsed, schema)) {
      return this.closeId === undefined ? new Uint32Array() : Uint32Array.of(this.closeId);
    }
    return this.unrestricted();
  }

  allowed(raw: string, parseArguments: (raw: string) => { name: string; arguments: Record<string, unknown> }): Uint32Array {
    // parseArguments remains part of the public adapter contract for callers
    // that need the finalized representation; prefix validation itself is
    // grammar-specific so Qwen and Gemma cannot accidentally share syntax.
    void parseArguments;
    return this.grammar === "qwen" ? this.qwenAllowed(raw) : this.gemmaAllowed(raw);
  }

  private isGemmaStructurallyComplete(raw: string): boolean {
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

function parseScalar(text: string): unknown {
  try { return JSON.parse(text); } catch { return text; }
}

function parseQwenArguments(raw: string): Record<string, unknown> {
  const args: Record<string, unknown> = {};
  for (const match of raw.matchAll(/<parameter=([^>]+)>\n?([\s\S]*?)\n?<\/parameter>/g)) {
    args[match[1].trim()] = parseScalar(match[2].trim());
  }
  return args;
}

function parseGemmaArguments(raw: string): Record<string, unknown> {
  const start = raw.indexOf("{");
  return start < 0 ? {} : parseToolArguments(raw.slice(start));
}

interface GemmaKeyState {
  invalid: boolean;
  prefix: string | null;
  seen: Set<string>;
  remaining: Set<string>;
}

/** Tracks top-level Gemma argument keys while leaving nested value syntax to
 * the shared schema validator. Keys are emitted unquoted by Gemma's native
 * tool grammar, so rejecting an unknown prefix prevents it before dispatch. */
function gemmaTopLevelKeyState(body: string, declared: Set<string>): GemmaKeyState {
  const seen = new Set<string>();
  const remaining = () => new Set([...declared].filter((key) => !seen.has(key)));
  let i = 1;
  let expectingKey = true;
  let braces = 1;
  let brackets = 0;
  let quoted = false;

  while (i < body.length) {
    if (body.startsWith('<|"|>', i)) {
      quoted = !quoted;
      i += 5;
      continue;
    }
    if (quoted) { i++; continue; }

    if (expectingKey && braces === 1 && brackets === 0) {
      while (i < body.length && /\s/.test(body[i])) i++;
      if (i >= body.length) return { invalid: false, prefix: "", seen, remaining: remaining() };
      if (body[i] === "}") return { invalid: false, prefix: null, seen, remaining: remaining() };
      const start = i;
      while (i < body.length && ![":", ",", "}"].includes(body[i])) i++;
      const prefix = body.slice(start, i).trim();
      if (i >= body.length) {
        if (![...remaining()].some((key) => key.startsWith(prefix))) {
          return { invalid: true, prefix: null, seen, remaining: remaining() };
        }
        return { invalid: false, prefix, seen, remaining: remaining() };
      }
      if (body[i] !== ":" || !declared.has(prefix) || seen.has(prefix)) {
        return { invalid: true, prefix: null, seen, remaining: remaining() };
      }
      seen.add(prefix);
      expectingKey = false;
      i++;
      continue;
    }

    const c = body[i];
    if (c === "{") braces++;
    else if (c === "}") {
      braces--;
      if (braces === 0) return { invalid: false, prefix: null, seen, remaining: remaining() };
    } else if (c === "[") brackets++;
    else if (c === "]") brackets--;
    else if (c === "," && braces === 1 && brackets === 0) expectingKey = true;
    if (braces < 0 || brackets < 0) return { invalid: true, prefix: null, seen, remaining: remaining() };
    i++;
  }

  return expectingKey
    ? { invalid: false, prefix: "", seen, remaining: remaining() }
    : { invalid: false, prefix: null, seen, remaining: remaining() };
}
