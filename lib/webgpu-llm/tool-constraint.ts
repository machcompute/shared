/**
 * Shared tool-call constraint contract.  It deliberately operates on token
 * IDs, leaving each model's prompt grammar/parser as a small adapter.
 */
import { parseToolArguments, parseToolValue } from "./gemma-tool-format";

export type JsonSchema = Record<string, unknown>;

const GEMMA_QUOTE = '<|"|>';
const TOOL_NAME_RE = /^[A-Za-z_][\w.-]*$/;
const PROPERTY_NAME_RE = /^[A-Za-z_][\w.-]*$/;
const RESERVED_GEMMA_TEXT = [
  GEMMA_QUOTE,
  "<|tool>",
  "<tool|>",
  "<|tool_call>",
  "<tool_call|>",
  "<|tool_response>",
  "<tool_response|>",
];

export interface ConstraintTokenizer {
  encode(text: string): number[];
  vocabSize(): number;
  specialTokenId(text: string): number | undefined;
  /** Plain text of a regular (non-special) token; used to classify numeric
   * and bounded-string tokens. Without it, numbers are steered one digit at
   * a time and maxLength strings conservatively reject unknown tokens. */
  tokenText?(id: number): string | undefined;
}

export interface ConstraintTool {
  type: "function";
  function: { name: string; description?: string; parameters?: JsonSchema };
}

export type ToolGrammar = "qwen" | "gemma";

export type TokenMask =
  | { kind: "all" }
  | { kind: "allow"; tokenIds: Uint32Array }
  | { kind: "deny"; tokenIds: Uint32Array };

const SUPPORTED = new Set([
  "$schema", "type", "properties", "required", "items", "enum", "nullable", "description",
  "minLength", "maxLength", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
  "minItems", "maxItems", "anyOf", "oneOf", "allOf", "prefixItems",
  "additionalProperties", "propertyNames",
]);

function assertSafeGemmaText(value: unknown, path: string): void {
  if (typeof value !== "string") return;
  const marker = RESERVED_GEMMA_TEXT.find((candidate) => value.includes(candidate));
  if (marker) throw new Error(`${path} contains reserved Gemma control text: ${marker}`);
}

function checkSchema(schema: unknown, path = "parameters", gemma = true): void {
  if (!schema || typeof schema !== "object" || Array.isArray(schema)) return;
  const record = schema as JsonSchema;
  for (const key of Object.keys(record)) {
    if (!SUPPORTED.has(key)) throw new Error(`Unsupported tool JSON Schema keyword at ${path}: ${key}`);
  }
  if (gemma && Array.isArray(record.enum)) {
    record.enum.forEach((member, index) => assertSafeGemmaText(member, `${path}.enum[${index}]`));
  }
  if (record.properties !== undefined) {
    if (!record.properties || typeof record.properties !== "object" || Array.isArray(record.properties)) {
      throw new Error(`${path}.properties must be an object`);
    }
    for (const [key, child] of Object.entries(record.properties as JsonSchema)) {
      if (gemma && !PROPERTY_NAME_RE.test(key)) {
        throw new Error(`${path}.properties has a key that Gemma cannot serialize unquoted: ${key}`);
      }
      checkSchema(child, `${path}.properties.${key}`, gemma);
    }
  }
  if (record.required !== undefined) {
    if (!Array.isArray(record.required) || record.required.some((key) => typeof key !== "string")) {
      throw new Error(`${path}.required must be an array of strings`);
    }
    for (const key of record.required as string[]) {
      if (gemma && !PROPERTY_NAME_RE.test(key)) {
        throw new Error(`${path}.required contains a key that Gemma cannot serialize unquoted: ${key}`);
      }
    }
  }
  if (record.items) checkSchema(record.items, `${path}.items`, gemma);
  if (Array.isArray(record.prefixItems)) {
    record.prefixItems.forEach((child, index) => checkSchema(child, `${path}.prefixItems[${index}]`, gemma));
  }
  for (const keyword of ["anyOf", "oneOf", "allOf"] as const) {
    if (record[keyword] !== undefined && !Array.isArray(record[keyword])) {
      throw new Error(`${path}.${keyword} must be an array`);
    }
    (record[keyword] as unknown[] | undefined)?.forEach((child, index) =>
      checkSchema(child, `${path}.${keyword}[${index}]`, gemma)
    );
  }
  if (record.additionalProperties !== undefined) {
    checkSchema(record.additionalProperties, `${path}.additionalProperties`, gemma);
  }
  if (record.propertyNames !== undefined) checkSchema(record.propertyNames, `${path}.propertyNames`, gemma);
}

/** Validate the runtime tool shape before prompt rendering or GPU prefill. */
export function validateTools(
  input: unknown,
  options: { grammar?: ToolGrammar } = {},
): asserts input is ConstraintTool[] | undefined {
  if (input === undefined) return;
  if (!Array.isArray(input)) throw new Error("tools must be an array");
  const seen = new Set<string>();
  input.forEach((raw, index) => {
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
      throw new Error(`tools[${index}] must be an object`);
    }
    const tool = raw as { type?: unknown; function?: unknown };
    if (tool.type !== "function") throw new Error(`tools[${index}].type must be "function"`);
    if (!tool.function || typeof tool.function !== "object" || Array.isArray(tool.function)) {
      throw new Error(`tools[${index}].function must be an object`);
    }
    const fn = tool.function as { name?: unknown; description?: unknown; parameters?: unknown };
    if (typeof fn.name !== "string" || !TOOL_NAME_RE.test(fn.name)) {
      throw new Error(
        `tools[${index}].function.name must match ${TOOL_NAME_RE.source}`
      );
    }
    if (seen.has(fn.name)) throw new Error(`Duplicate tool function name: ${fn.name}`);
    seen.add(fn.name);
    if (fn.description !== undefined && typeof fn.description !== "string") {
      throw new Error(`tools[${index}].function.description must be a string`);
    }
    const gemma = (options.grammar ?? "gemma") === "gemma";
    if (fn.parameters === undefined) return;
    if (!fn.parameters || typeof fn.parameters !== "object" || Array.isArray(fn.parameters)) {
      throw new Error(`tools[${index}].function.parameters must be an object schema`);
    }
    const parameters = fn.parameters as JsonSchema;
    const rootTypes = Array.isArray(parameters.type)
      ? parameters.type
      : parameters.type === undefined
        ? []
        : [parameters.type];
    if (rootTypes.length && !rootTypes.includes("object")) {
      throw new Error(`tools[${index}].function.parameters must describe an object`);
    }
    checkSchema(parameters, `tools[${index}].function.parameters`, gemma);
  });
}

function numberKeyword(schema: JsonSchema, keyword: string): number | undefined {
  const value = schema[keyword];
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function typeMatches(value: unknown, inputSchema: unknown): boolean {
  if (inputSchema === true || inputSchema === undefined) return true;
  if (inputSchema === false || !inputSchema || typeof inputSchema !== "object" || Array.isArray(inputSchema)) return false;
  const schema = inputSchema as JsonSchema;

  if (value === null && schema.nullable === true) return true;
  if (Array.isArray(schema.allOf) && !schema.allOf.every((child) => typeMatches(value, child))) return false;
  if (Array.isArray(schema.anyOf) && !schema.anyOf.some((child) => typeMatches(value, child))) return false;
  if (Array.isArray(schema.oneOf) && schema.oneOf.filter((child) => typeMatches(value, child)).length !== 1) return false;
  if (schema.enum && Array.isArray(schema.enum) && !schema.enum.some((item) => JSON.stringify(item) === JSON.stringify(value))) return false;

  const declaredTypes = Array.isArray(schema.type) ? schema.type : schema.type ? [schema.type] : [];
  if (declaredTypes.length > 1) {
    return declaredTypes.some((type) => typeMatches(value, { ...schema, type }));
  }
  const inferredType = schema.properties || schema.required || schema.additionalProperties !== undefined || schema.propertyNames
    ? "object"
    : schema.items !== undefined || schema.prefixItems || schema.minItems !== undefined || schema.maxItems !== undefined
      ? "array"
      : (schema.minLength !== undefined || schema.maxLength !== undefined) && typeof value === "string"
        ? "string"
        : (schema.minimum !== undefined || schema.maximum !== undefined || schema.exclusiveMinimum !== undefined || schema.exclusiveMaximum !== undefined)
            && typeof value === "number"
          ? "number"
          : undefined;
  const type = declaredTypes[0] ?? inferredType;

  if (type === "string") {
    if (typeof value !== "string") return false;
    const length = [...value].length;
    const minLength = numberKeyword(schema, "minLength");
    const maxLength = numberKeyword(schema, "maxLength");
    return (minLength === undefined || length >= minLength) && (maxLength === undefined || length <= maxLength);
  }
  if (type === "number" || type === "integer") {
    if (typeof value !== "number" || !Number.isFinite(value) || (type === "integer" && !Number.isInteger(value))) return false;
    const minimum = numberKeyword(schema, "minimum");
    const maximum = numberKeyword(schema, "maximum");
    const exclusiveMinimum = numberKeyword(schema, "exclusiveMinimum");
    const exclusiveMaximum = numberKeyword(schema, "exclusiveMaximum");
    return (minimum === undefined || value >= minimum)
      && (maximum === undefined || value <= maximum)
      && (exclusiveMinimum === undefined || value > exclusiveMinimum)
      && (exclusiveMaximum === undefined || value < exclusiveMaximum);
  }
  if (type === "boolean") return typeof value === "boolean";
  if (type === "null") return value === null;
  if (type === "array") {
    if (!Array.isArray(value)) return false;
    const minItems = numberKeyword(schema, "minItems");
    const maxItems = numberKeyword(schema, "maxItems");
    if ((minItems !== undefined && value.length < minItems) || (maxItems !== undefined && value.length > maxItems)) return false;
    const prefixItems = Array.isArray(schema.prefixItems) ? schema.prefixItems : [];
    for (let index = 0; index < Math.min(value.length, prefixItems.length); index++) {
      if (!typeMatches(value[index], prefixItems[index])) return false;
    }
    if (schema.items !== undefined) {
      const start = prefixItems.length ? prefixItems.length : 0;
      for (let index = start; index < value.length; index++) {
        if (!typeMatches(value[index], schema.items)) return false;
      }
    }
    return true;
  }
  if (type === "object") {
    if (!value || typeof value !== "object" || Array.isArray(value)) return false;
    const obj = value as JsonSchema;
    const props = (schema.properties ?? {}) as JsonSchema;
    if (Array.isArray(schema.required) && schema.required.some((key) => typeof key === "string" && !(key in obj))) return false;
    if (schema.propertyNames !== undefined && Object.keys(obj).some((key) => !typeMatches(key, schema.propertyNames))) return false;
    for (const [key, item] of Object.entries(obj)) {
      if (key in props) {
        if (!typeMatches(item, props[key])) return false;
        continue;
      }
      const additional = schema.additionalProperties;
      // JSON Schema permits additional properties by default. Zod's closed
      // objects transmit `additionalProperties: false`, while z.record()
      // transmits a schema (often `{}`) for every undeclared value.
      if (additional === false) return false;
      if (additional !== undefined && additional !== true && !typeMatches(item, additional)) return false;
    }
    return true;
  }
  // Omitted `type` is permitted by JSON Schema; composition and enum checks
  // above still apply.
  return true;
}

/**
 * The constrained segment starts only after the native tool opener. Names are
 * token-constrained immediately; argument completion is schema-validated
 * before the native close token is ever admitted.
 */
export class ToolConstraint {
  private readonly tools = new Map<string, JsonSchema>();
  private all?: Uint32Array;
  private readonly closeId: number | undefined;
  private readonly grammar: ToolGrammar;
  private readonly forbiddenIds: Set<number>;

  constructor(
    tools: ConstraintTool[],
    tokenizer: ConstraintTokenizer,
    options: { closeToken: string; grammar: ToolGrammar; forbiddenTokenIds?: readonly number[] }
  ) {
    validateTools(tools, { grammar: options.grammar });
    for (const tool of tools) {
      this.tools.set(tool.function.name, tool.function.parameters ?? { type: "object" });
    }
    this.closeId = tokenizer.specialTokenId(options.closeToken);
    this.grammar = options.grammar;
    this.forbiddenIds = new Set(options.forbiddenTokenIds ?? []);
    this.tokenizer = tokenizer;
  }

  private readonly tokenizer: ConstraintTokenizer;
  private unrestrictedCache?: Uint32Array;
  private gemmaIds?: GemmaTokenIds;
  private gemmaUnrestrictedCache?: TokenMask;
  private readonly gemmaUnrestrictedWithoutCache = new Map<number, TokenMask>();

  private allTokens(): Uint32Array {
    if (!this.all) {
      this.all = Uint32Array.from({ length: this.tokenizer.vocabSize() }, (_, id) => id);
    }
    return this.all;
  }

  private unrestricted(): Uint32Array {
    if (!this.unrestrictedCache) {
      this.unrestrictedCache = Uint32Array.from(
        Array.from(this.allTokens()).filter(
          (id) => id !== this.closeId && !this.forbiddenIds.has(id)
        )
      );
    }
    return this.unrestrictedCache;
  }

  private gemmaUnrestricted(excluded?: number): TokenMask {
    if (excluded !== undefined) {
      const cached = this.gemmaUnrestrictedWithoutCache.get(excluded);
      if (cached) return cached;
    } else if (this.gemmaUnrestrictedCache) {
      return this.gemmaUnrestrictedCache;
    }
    const denied = new Set(this.forbiddenIds);
    if (this.closeId !== undefined) denied.add(this.closeId);
    if (excluded !== undefined) denied.add(excluded);
    const tokenIds = sortedIds(denied);
    const mask: TokenMask = tokenIds.length ? { kind: "deny", tokenIds } : { kind: "all" };
    if (excluded === undefined) this.gemmaUnrestrictedCache = mask;
    else this.gemmaUnrestrictedWithoutCache.set(excluded, mask);
    return mask;
  }

  private allow(tokenIds: Uint32Array): TokenMask {
    return { kind: "allow", tokenIds };
  }

  private gemmaPrefix(emitted: string, targets: string[]): TokenMask {
    const tokenIds = this.requirePrefix(emitted, targets);
    return tokenIds === null ? this.gemmaUnrestricted() : this.allow(tokenIds);
  }

  /** First token of each literal continuation, the same steering primitive
   * requirePrefix uses: a BPE first token is always a prefix of its target. */
  private firstTokens(texts: string[]): number[] {
    const ids = new Set<number>();
    for (const text of texts) {
      if (!text) continue;
      const token = this.tokenizer.encode(text)[0];
      if (token !== undefined && !this.forbiddenIds.has(token)) ids.add(token);
    }
    return [...ids];
  }

  private gemmaTokenIds(): GemmaTokenIds {
    if (this.gemmaIds) return this.gemmaIds;
    const single = (text: string): number | undefined => this.tokenizer.encode(text)[0];
    // Numeric tokens split by shape so the walker can keep a numeral on a
    // schema-valid trajectory: pure digit runs extend any number, dotted
    // tokens start/continue a decimal fraction exactly once. Exponent and
    // sign-bearing multi-char tokens are excluded outright — tool arguments
    // never need them and each one opens a can-never-recover trap.
    let vocabulary = gemmaVocabularyCache.get(this.tokenizer);
    if (!vocabulary) {
      const stringIds: number[] = [];
      const stringLengths: number[] = [];
      const digits: [number, string][] = [];
      const dotted: [number, string][] = [];
      const classify = (id: number, text: string) => {
        stringIds.push(id);
        stringLengths.push([...text].length);
        if (/^[0-9]+$/.test(text)) digits.push([id, text]);
        else if (/^[0-9]*\.[0-9]*$/.test(text)) dotted.push([id, text]);
      };
      const textOf = this.tokenizer.tokenText?.bind(this.tokenizer);
      if (textOf) {
        const size = this.tokenizer.vocabSize();
        for (let id = 0; id < size; id++) {
          const text = textOf(id);
          if (text) classify(id, text);
        }
      } else {
        // Char-by-char fallback keeps numbers writable without tokenText.
        for (const ch of ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.']) {
          const id = single(ch);
          if (id !== undefined) classify(id, ch);
        }
      }
      vocabulary = {
        stringIds: Uint32Array.from(stringIds),
        stringLengths: Uint32Array.from(stringLengths),
        digits,
        dotted,
      };
      gemmaVocabularyCache.set(this.tokenizer, vocabulary);
    }
    const usable = (entry: [number, string]) =>
      entry[0] !== this.closeId && !this.forbiddenIds.has(entry[0]);
    const digitEntries = vocabulary.digits.filter(usable).sort(([a], [b]) => a - b);
    const dottedEntries = vocabulary.dotted.filter(usable).sort(([a], [b]) => a - b);
    this.gemmaIds = {
      quote: this.tokenizer.specialTokenId(GEMMA_QUOTE) ?? single(GEMMA_QUOTE),
      openBrace: single('{'),
      closeBrace: single('}'),
      openBracket: single('['),
      closeBracket: single(']'),
      comma: single(','),
      minus: single('-'),
      digits: Uint32Array.from(digitEntries.map(([id]) => id)),
      digitTexts: digitEntries.map(([, text]) => text),
      dotted: Uint32Array.from(dottedEntries.map(([id]) => id)),
      dottedTexts: dottedEntries.map(([, text]) => text),
      stringIds: vocabulary.stringIds,
      stringLengths: vocabulary.stringLengths,
    };
    return this.gemmaIds;
  }

  /**
   * Numeral continuations that can still reach a schema-valid value.
   *
   * Every candidate token is admitted only when `text + token` remains a
   * viable prefix of SOME value satisfying the bounds (reachability, not a
   * one-step look): integers never admit a decimal point, numeric enums are
   * steered like word literals, and past double precision only a valid stop
   * remains.
   */
  private gemmaNumberTokens(text: string, schema: unknown, frame: GemmaFrame | undefined): Uint32Array {
    const T = this.gemmaTokenIds();
    const s = (schema && typeof schema === "object" && !Array.isArray(schema) ? schema : {}) as JsonSchema;
    const info = gemmaSchemaKinds(schema);
    const integerOnly = info.kinds.has("integer") && !info.kinds.has("number");
    const value = Number(text);
    const valid = text !== "" && Number.isFinite(value) && typeMatches(value, schema);
    const delimiters = valid ? this.gemmaDelimiters(frame, true) : [];
    if ([...text].length >= 17) return sortedIds(delimiters);
    const ids = new Set<number>(delimiters);

    // Numeric enums: only continuations toward a member's literal rendering.
    const members = Array.isArray(s.enum)
      ? s.enum.filter((m): m is number => typeof m === "number").map((m) => String(m))
      : null;
    if (members?.length) {
      const remainders = members
        .filter((m) => m.startsWith(text) && m !== text)
        .map((m) => m.slice(text.length));
      for (const id of this.firstTokens(remainders)) ids.add(id);
      return sortedIds(ids);
    }

    const addViable = (tokens: Uint32Array, texts: string[]) => {
      for (let i = 0; i < tokens.length; i++) {
        const candidate = text + texts[i];
        if ([...candidate].length <= 17 && gemmaNumeralViable(candidate, s)) ids.add(tokens[i]);
      }
    };
    addViable(T.digits, T.digitTexts);
    if (!integerOnly && !text.includes(".")) addViable(T.dotted, T.dottedTexts);
    if (text === "" && T.minus !== undefined && gemmaNumeralViable("-", s)) ids.add(T.minus);
    return sortedIds(ids);
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

  /**
   * Incremental Gemma grammar walk.  `gemmaScan` re-derives the parse state
   * of the emitted body (nested objects, arrays, strings, bare scalars) and
   * this maps that state to the admissible next tokens: declared keys at
   * every depth, kind-correct value openers, enum-steered string content, and
   * delimiters gated on scalar validity and required/min-max completeness.
   * A structurally complete body admits only the native close token — or
   * nothing at all if it somehow fails the final schema check, which the
   * decode loop surfaces as a schema error instead of decoding on with no
   * way to ever close the call.
   */
  private gemmaAllowed(raw: string): TokenMask {
    const emitted = raw.trimStart();
    const targets = [...this.tools.keys()].map((name) => `call:${name}{`);
    const match = /^(?:call:)?\s*([A-Za-z_][\w.-]*)\s*\{/.exec(emitted);
    if (!match) return this.gemmaPrefix(emitted, targets);

    const schema = this.tools.get(match[1]);
    if (!schema) return this.allow(new Uint32Array());
    const objectStart = emitted.indexOf("{", match.index + match[0].length - 1);
    const state = gemmaScan(emitted.slice(objectStart), schema);

    switch (state.kind) {
      case "invalid":
        return this.allow(new Uint32Array());
      case "complete": {
        if (!typeMatches(parseGemmaArguments(emitted), schema)) return this.allow(new Uint32Array());
        return this.allow(this.closeId === undefined ? new Uint32Array() : Uint32Array.of(this.closeId));
      }
      case "free":
        return this.gemmaUnrestricted();
      case "key": {
        const remaining = [...state.frame.declared].filter((key) => !state.frame.seen.has(key));
        const keyTargets = remaining.map((key) => `${key}:`);
        const partial = state.frame.keyBuf.trim();
        if (!partial && [...state.frame.required].every((key) => state.frame.seen.has(key))) {
          keyTargets.push("}");
        }
        return this.gemmaPrefix(partial, keyTargets);
      }
      case "valueStart":
        return this.gemmaValueStart(state.schema, state.frame);
      case "inString":
        return this.gemmaInString(state.schema, state.content);
      case "inNumber":
        return this.allow(this.gemmaNumberTokens(state.text, state.schema, state.frame));
      case "inWord": {
        const words = gemmaWordTargets(state.schema);
        const matching = words.filter((word) => word.startsWith(state.text));
        if (!matching.length) return this.allow(new Uint32Array());
        const ids = new Set<number>(this.firstTokens(
          matching.filter((word) => word !== state.text).map((word) => word.slice(state.text.length))
        ));
        if (matching.includes(state.text) && typeMatches(parseToolValue(state.text), state.schema)) {
          for (const id of this.gemmaDelimiters(state.frame, true)) ids.add(id);
        }
        return this.allow(sortedIds(ids));
      }
      case "afterValue":
        return this.allow(sortedIds(this.gemmaDelimiters(state.frame, false)));
    }
  }

  /** Admissible openers for a value of `schema` (or a whole free span). */
  private gemmaValueStart(schema: unknown, frame: GemmaFrame | undefined): TokenMask {
    const info = gemmaSchemaKinds(schema);
    if (info.free) return this.gemmaUnrestricted();
    const T = this.gemmaTokenIds();
    const ids = new Set<number>();
    if (info.kinds.has("string") && T.quote !== undefined) ids.add(T.quote);
    if (info.kinds.has("number") || info.kinds.has("integer")) {
      for (const id of this.gemmaNumberTokens("", schema, undefined)) ids.add(id);
    }
    for (const id of this.firstTokens(gemmaWordTargets(schema))) ids.add(id);
    if (info.kinds.has("object") && T.openBrace !== undefined) ids.add(T.openBrace);
    if (info.kinds.has("array") && T.openBracket !== undefined) ids.add(T.openBracket);
    if (frame?.type === "arr" && frame.count === 0
        && (numberKeyword(frame.schema, "minItems") ?? 0) === 0
        && T.closeBracket !== undefined) {
      ids.add(T.closeBracket);
    }
    return this.allow(sortedIds(ids));
  }

  private gemmaInString(schema: unknown, content: string): TokenMask {
    const s = (schema && typeof schema === "object" && !Array.isArray(schema) ? schema : {}) as JsonSchema;
    const T = this.gemmaTokenIds();
    const members = Array.isArray(s.enum)
      ? s.enum.filter((member): member is string => typeof member === "string")
      : null;
    if (members?.length) {
      const matching = members.filter((member) => member.startsWith(content));
      if (!matching.length) return this.allow(new Uint32Array());
      const ids = new Set<number>(this.firstTokens(
        matching.filter((member) => member !== content).map((member) => member.slice(content.length))
      ));
      if (matching.includes(content) && T.quote !== undefined) ids.add(T.quote);
      return this.allow(sortedIds(ids));
    }
    const length = [...content].length;
    const maxLength = numberKeyword(s, "maxLength");
    if (maxLength !== undefined) {
      const ids = new Set<number>();
      const remaining = Math.max(0, Math.floor(maxLength) - length);
      for (let index = 0; index < T.stringIds.length; index++) {
        const id = T.stringIds[index];
        if (id !== this.closeId
            && !this.forbiddenIds.has(id)
            && T.stringLengths[index] <= remaining) ids.add(id);
      }
      // Structural closure is admitted only when the exact value satisfies
      // minLength/maxLength and any other scalar constraints.
      if (T.quote !== undefined && typeMatches(content, schema)) ids.add(T.quote);
      return this.allow(sortedIds(ids));
    }
    const minLength = numberKeyword(s, "minLength");
    if (minLength !== undefined && length < minLength) return this.gemmaUnrestricted(T.quote);
    return this.gemmaUnrestricted();
  }

  /** `,` / `}` / `]` for the enclosing frame; `pendingValue` counts an
   * in-progress scalar as one more completed entry. */
  private gemmaDelimiters(frame: GemmaFrame | undefined, pendingValue: boolean): number[] {
    if (!frame) return [];
    const T = this.gemmaTokenIds();
    const ids: number[] = [];
    if (frame.type === "obj") {
      if ([...frame.declared].some((key) => !frame.seen.has(key)) && T.comma !== undefined) ids.push(T.comma);
      if ([...frame.required].every((key) => frame.seen.has(key)) && T.closeBrace !== undefined) ids.push(T.closeBrace);
    } else if (frame.type === "arr") {
      const count = frame.count + (pendingValue ? 1 : 0);
      const maxItems = numberKeyword(frame.schema, "maxItems");
      if ((maxItems === undefined || count < maxItems) && T.comma !== undefined) ids.push(T.comma);
      if (count >= (numberKeyword(frame.schema, "minItems") ?? 0) && T.closeBracket !== undefined) ids.push(T.closeBracket);
    }
    return ids;
  }

  mask(raw: string, parseArguments: (raw: string) => { name: string; arguments: Record<string, unknown> }): TokenMask {
    // parseArguments remains part of the public adapter contract for callers
    // that need the finalized representation; prefix validation itself is
    // grammar-specific so Qwen and Gemma cannot accidentally share syntax.
    void parseArguments;
    return this.grammar === "qwen"
      ? this.allow(this.qwenAllowed(raw))
      : this.gemmaAllowed(raw);
  }

  /** Backward-compatible allow-list adapter. Gemma callers should use mask()
   * so unrestricted spans stay a tiny deny-list rather than a full vocab. */
  allowed(raw: string, parseArguments: (raw: string) => { name: string; arguments: Record<string, unknown> }): Uint32Array {
    const mask = this.mask(raw, parseArguments);
    if (mask.kind === "allow") return mask.tokenIds;
    if (mask.kind === "all") return this.allTokens();
    const denied = new Set(mask.tokenIds);
    return Uint32Array.from(Array.from(this.allTokens()).filter((id) => !denied.has(id)));
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

// ---------------------------------------------------------------------------
// Incremental Gemma grammar scanner.
//
// Gemma's native tool syntax is `call:name{key:value,...}` with unquoted
// keys, `<|"|>`-delimited strings, bare numbers/booleans, and nested
// objects/arrays.  `gemmaScan` replays the emitted body against the tool's
// JSON Schema and reports the exact parse state at the end of input, so the
// constraint can steer the next token instead of discovering violations only
// when the body closes.  Subschemas the walker cannot steer (compositions,
// additionalProperties, unknown kinds) become validated free spans.

interface GemmaTokenIds {
  quote?: number;
  openBrace?: number;
  closeBrace?: number;
  openBracket?: number;
  closeBracket?: number;
  comma?: number;
  minus?: number;
  digits: Uint32Array;
  digitTexts: string[];
  dotted: Uint32Array;
  dottedTexts: string[];
  /** Regular non-special, non-byte-fallback IDs and code-point lengths. */
  stringIds: Uint32Array;
  stringLengths: Uint32Array;
}

// Vocabulary classification scans every token; it depends only on the
// tokenizer, so share it across ToolConstraint instances (one per request).
const gemmaVocabularyCache = new WeakMap<ConstraintTokenizer, {
  stringIds: Uint32Array;
  stringLengths: Uint32Array;
  digits: [number, string][];
  dotted: [number, string][];
}>();

interface GemmaObjFrame {
  type: "obj";
  props: JsonSchema;
  declared: Set<string>;
  required: Set<string>;
  seen: Set<string>;
  phase: "key" | "value" | "after";
  keyBuf: string;
  valueSchema: unknown;
}

interface GemmaArrFrame {
  type: "arr";
  schema: JsonSchema;
  count: number;
  phase: "value" | "after";
}

interface GemmaFreeFrame {
  type: "free";
  schema: unknown;
  root: boolean;
  start: number;
  depth: number;
  quoted: boolean;
}

type GemmaFrame = GemmaObjFrame | GemmaArrFrame | GemmaFreeFrame;

type GemmaScanState =
  | { kind: "invalid" }
  | { kind: "complete" }
  | { kind: "free" }
  | { kind: "key"; frame: GemmaObjFrame }
  | { kind: "valueStart"; schema: unknown; frame: GemmaFrame | undefined }
  | { kind: "inString"; schema: unknown; content: string }
  | { kind: "inNumber"; schema: unknown; text: string; frame: GemmaFrame | undefined }
  | { kind: "inWord"; schema: unknown; text: string; frame: GemmaFrame | undefined }
  | { kind: "afterValue"; frame: GemmaFrame };

const sortedIds = (ids: Iterable<number>) => Uint32Array.from([...ids].sort((a, b) => a - b));

/** Which Gemma value kinds a subschema admits; `free` marks spans the
 * scanner validates as a whole instead of steering token-by-token. */
function gemmaSchemaKinds(schema: unknown): { free: boolean; kinds: Set<string> } {
  const free = { free: true, kinds: new Set<string>() };
  if (!schema || typeof schema !== "object" || Array.isArray(schema)) return free;
  const s = schema as JsonSchema;
  if (s.anyOf !== undefined || s.oneOf !== undefined || s.allOf !== undefined) return free;
  const kinds = new Set<string>();
  const declared = (Array.isArray(s.type) ? s.type : s.type !== undefined ? [s.type] : [])
    .filter((t): t is string => typeof t === "string");
  for (const t of declared) kinds.add(t);
  if (Array.isArray(s.enum)) {
    if (s.enum.some((member) => member !== null && typeof member === "object")) return free;
    if (!declared.length) {
      for (const member of s.enum) kinds.add(member === null ? "null" : typeof member);
    }
  } else if (!declared.length) {
    if (s.properties !== undefined || s.required !== undefined
        || s.additionalProperties !== undefined || s.propertyNames !== undefined) kinds.add("object");
    else if (s.items !== undefined || s.prefixItems !== undefined
        || s.minItems !== undefined || s.maxItems !== undefined) kinds.add("array");
    else if (s.minLength !== undefined || s.maxLength !== undefined) kinds.add("string");
    else if (s.minimum !== undefined || s.maximum !== undefined
        || s.exclusiveMinimum !== undefined || s.exclusiveMaximum !== undefined) kinds.add("number");
  }
  if (s.nullable === true) kinds.add("null");
  const known = new Set(["string", "number", "integer", "boolean", "null", "object", "array"]);
  for (const kind of kinds) if (!known.has(kind)) return free;
  return kinds.size ? { free: false, kinds } : free;
}

/** An object can be key-steered only when its keys are fully declared and
 * its required set is satisfiable. */
function gemmaObjectEnforceable(schema: unknown): boolean {
  if (!schema || typeof schema !== "object" || Array.isArray(schema)) return false;
  const s = schema as JsonSchema;
  if (s.additionalProperties !== undefined && s.additionalProperties !== false) return false;
  if (s.propertyNames !== undefined) return false;
  const props = s.properties;
  if (!props || typeof props !== "object" || Array.isArray(props)) return false;
  const declared = new Set(Object.keys(props as JsonSchema));
  const required = Array.isArray(s.required) ? s.required : [];
  return required.every((key) => typeof key === "string" && declared.has(key));
}

function gemmaWordTargets(schema: unknown): string[] {
  const info = gemmaSchemaKinds(schema);
  const words: string[] = [];
  if (info.kinds.has("boolean")) words.push("true", "false");
  if (info.kinds.has("null")) words.push("null");
  // An enum'd boolean must not steer the excluded literal into a dead end.
  return words.filter((word) => typeMatches(parseToolValue(word), schema));
}

/**
 * Can `text` (a partial numeral: optional sign, digits, at most one dot)
 * still be extended into a value within the schema's declared bounds?
 *
 * Inclusive and exclusive bounds are combined independently, so the prefix
 * is admitted only if a finite decimal continuation can intersect the exact
 * schema interval. Integer schemas use their exact effective integer range.
 */
function gemmaNumeralViable(text: string, s: JsonSchema): boolean {
  const info = gemmaSchemaKinds(s);
  const integerOnly = info.kinds.has("integer") && !info.kinds.has("number");
  const remaining = 17 - [...text].length;
  const integerRange = effectiveIntegerRange(s);
  const numberRange = effectiveNumberRange(s);
  if (!intervalIsNonEmpty(numberRange)
      || (integerOnly && integerRange.min > integerRange.max)) return false;

  if (text === "-") {
    const maxMagnitude = 10 ** Math.max(0, remaining) - 1;
    return integerOnly
      ? inclusiveRangesIntersect(-maxMagnitude, 0, integerRange.min, integerRange.max)
      : intervalsIntersect(
          { min: -maxMagnitude, minOpen: false, max: 0, maxOpen: false },
          numberRange,
        );
  }
  if (!/^-?(?:\d+(?:\.\d*)?|\.\d*)$/.test(text)) return false;
  const negative = text.startsWith("-");
  const body = (negative ? text.slice(1) : text) || "0";
  const hasDot = body.includes(".");
  const magnitude = Number(body.endsWith(".") ? `${body}0` : body);
  if (!Number.isFinite(magnitude)) return false;

  const exactValue = negative ? -magnitude : magnitude;
  const exactText = text !== "." && text !== "-.";
  if (exactText && typeMatches(exactValue, s)) return true;
  if (remaining <= 0) return false;

  if (integerOnly) {
    if (hasDot) return false;
    for (let appended = 1; appended <= remaining; appended++) {
      const prefix = integerContinuationRange(magnitude, negative, appended);
      if (inclusiveRangesIntersect(prefix.min, prefix.max, integerRange.min, integerRange.max)) return true;
    }
    return false;
  }

  if (hasDot) {
    const fractionalDigits = body.length - body.indexOf(".") - 1;
    const step = 10 ** -fractionalDigits;
    const prefix = negative
      ? { min: exactValue - step, minOpen: true, max: exactValue, maxOpen: false }
      : { min: exactValue, minOpen: false, max: exactValue + step, maxOpen: true };
    return intervalsIntersect(prefix, numberRange);
  }

  // Appending a dot plus at least one digit fills the interval between this
  // integer prefix and the next one. Appending integer digits first yields
  // both exact integer points and, when room remains, the corresponding
  // decimal intervals.
  for (let appended = 0; appended <= remaining; appended++) {
    const prefix = integerContinuationRange(magnitude, negative, appended);
    if (appended > 0
        && inclusiveRangesIntersect(prefix.min, prefix.max, integerRange.min, integerRange.max)) {
      return true;
    }
    if (remaining - appended >= 2) {
      const factor = 10 ** appended;
      const decimalPrefix = negative
        ? {
            min: -(magnitude + 1) * factor,
            minOpen: true,
            max: -magnitude * factor,
            maxOpen: false,
          }
        : {
            min: magnitude * factor,
            minOpen: false,
            max: (magnitude + 1) * factor,
            maxOpen: true,
          };
      if (intervalsIntersect(decimalPrefix, numberRange)) return true;
    }
  }
  return false;
}

interface NumericInterval {
  min: number;
  minOpen: boolean;
  max: number;
  maxOpen: boolean;
}

function effectiveNumberRange(s: JsonSchema): NumericInterval {
  let min = -Infinity;
  let minOpen = false;
  const tightenMin = (value: number | undefined, open: boolean) => {
    if (value === undefined) return;
    if (value > min) {
      min = value;
      minOpen = open;
    } else if (value === min) {
      minOpen ||= open;
    }
  };
  tightenMin(numberKeyword(s, "minimum"), false);
  tightenMin(numberKeyword(s, "exclusiveMinimum"), true);

  let max = Infinity;
  let maxOpen = false;
  const tightenMax = (value: number | undefined, open: boolean) => {
    if (value === undefined) return;
    if (value < max) {
      max = value;
      maxOpen = open;
    } else if (value === max) {
      maxOpen ||= open;
    }
  };
  tightenMax(numberKeyword(s, "maximum"), false);
  tightenMax(numberKeyword(s, "exclusiveMaximum"), true);
  return { min, minOpen, max, maxOpen };
}

function effectiveIntegerRange(s: JsonSchema): { min: number; max: number } {
  let min = -Infinity;
  let max = Infinity;
  const minimum = numberKeyword(s, "minimum");
  const exclusiveMinimum = numberKeyword(s, "exclusiveMinimum");
  const maximum = numberKeyword(s, "maximum");
  const exclusiveMaximum = numberKeyword(s, "exclusiveMaximum");
  if (minimum !== undefined) min = Math.max(min, Math.ceil(minimum));
  if (exclusiveMinimum !== undefined) min = Math.max(min, Math.floor(exclusiveMinimum) + 1);
  if (maximum !== undefined) max = Math.min(max, Math.floor(maximum));
  if (exclusiveMaximum !== undefined) max = Math.min(max, Math.ceil(exclusiveMaximum) - 1);
  return { min, max };
}

function intervalIsNonEmpty(interval: NumericInterval): boolean {
  return interval.min < interval.max
    || (interval.min === interval.max && !interval.minOpen && !interval.maxOpen);
}

function intervalsIntersect(a: NumericInterval, b: NumericInterval): boolean {
  const min = Math.max(a.min, b.min);
  const max = Math.min(a.max, b.max);
  if (min < max) return true;
  if (min !== max) return false;
  const minOpen = (a.min === min && a.minOpen) || (b.min === min && b.minOpen);
  const maxOpen = (a.max === max && a.maxOpen) || (b.max === max && b.maxOpen);
  return !minOpen && !maxOpen;
}

function inclusiveRangesIntersect(aMin: number, aMax: number, bMin: number, bMax: number): boolean {
  return Math.max(aMin, bMin) <= Math.min(aMax, bMax);
}

function integerContinuationRange(
  magnitude: number,
  negative: boolean,
  appendedDigits: number,
): { min: number; max: number } {
  const factor = 10 ** appendedDigits;
  return negative
    ? { min: -((magnitude + 1) * factor - 1), max: -(magnitude * factor) }
    : { min: magnitude * factor, max: (magnitude + 1) * factor - 1 };
}

function gemmaItemSchema(frame: GemmaArrFrame): unknown {
  const prefixItems = Array.isArray(frame.schema.prefixItems) ? frame.schema.prefixItems : [];
  if (frame.count < prefixItems.length) return prefixItems[frame.count];
  return frame.schema.items;
}

type GemmaScalar = { sort: "string" | "number" | "word"; schema: unknown; buf: string };

function gemmaScan(body: string, rootSchema: JsonSchema): GemmaScanState {
  const frames: GemmaFrame[] = [];
  // A property (not a plain let) so narrowing survives the closure writes in
  // beginValue below.
  const st: { scalar: GemmaScalar | null } = { scalar: null };
  let complete = false;
  let i = 0;
  const invalid: GemmaScanState = { kind: "invalid" };

  const finishValue = () => {
    const parent = frames[frames.length - 1];
    if (!parent) {
      complete = true;
      return;
    }
    if (parent.type === "obj") parent.phase = "after";
    else if (parent.type === "arr") {
      parent.count++;
      parent.phase = "after";
    }
  };

  // Begin the value whose first character sits at body[i].  Pushes a frame or
  // opens a scalar; consumes only what it must (string quote markers, the
  // composite openers).  False means the character cannot start any
  // admissible kind.
  const beginValue = (schema: unknown): boolean => {
    const info = gemmaSchemaKinds(schema);
    if (info.free) {
      frames.push({ type: "free", schema, root: false, start: i, depth: 0, quoted: false });
      return true;
    }
    if (body.startsWith(GEMMA_QUOTE, i)) {
      if (!info.kinds.has("string")) return false;
      st.scalar = { sort: "string", schema, buf: "" };
      i += GEMMA_QUOTE.length;
      return true;
    }
    const c = body[i];
    if (c === "{") {
      if (!info.kinds.has("object")) return false;
      if (!gemmaObjectEnforceable(schema)) {
        frames.push({ type: "free", schema, root: false, start: i, depth: 0, quoted: false });
        return true;
      }
      const s = schema as JsonSchema;
      const props = (s.properties ?? {}) as JsonSchema;
      frames.push({
        type: "obj",
        props,
        declared: new Set(Object.keys(props)),
        required: new Set((Array.isArray(s.required) ? s.required : [])
          .filter((key): key is string => typeof key === "string")),
        seen: new Set(),
        phase: "key",
        keyBuf: "",
        valueSchema: undefined,
      });
      i++;
      return true;
    }
    if (c === "[") {
      if (!info.kinds.has("array")) return false;
      frames.push({ type: "arr", schema: schema as JsonSchema, count: 0, phase: "value" });
      i++;
      return true;
    }
    if (/[-0-9.]/.test(c)) {
      if (!info.kinds.has("number") && !info.kinds.has("integer")) return false;
      st.scalar = { sort: "number", schema, buf: "" };
      return true;
    }
    if (/[a-z]/i.test(c)) {
      if (!info.kinds.has("boolean") && !info.kinds.has("null")) return false;
      st.scalar = { sort: "word", schema, buf: "" };
      return true;
    }
    return false;
  };

  while (i < body.length) {
    if (complete) return invalid;

    if (st.scalar) {
      if (st.scalar.sort === "string") {
        if (body.startsWith(GEMMA_QUOTE, i)) {
          if (!typeMatches(st.scalar.buf, st.scalar.schema)) return invalid;
          i += GEMMA_QUOTE.length;
          st.scalar = null;
          finishValue();
          continue;
        }
        st.scalar.buf += body[i];
        i++;
        continue;
      }
      const c = body[i];
      if (c === "," || c === "}" || c === "]") {
        if (!typeMatches(parseToolValue(st.scalar.buf), st.scalar.schema)) return invalid;
        st.scalar = null;
        finishValue();
        continue; // the delimiter belongs to the parent frame
      }
      if (body.startsWith(GEMMA_QUOTE, i)) return invalid;
      st.scalar.buf += c;
      i++;
      continue;
    }

    const top = frames[frames.length - 1];

    if (top?.type === "free") {
      if (body.startsWith(GEMMA_QUOTE, i)) {
        top.quoted = !top.quoted;
        i += GEMMA_QUOTE.length;
        continue;
      }
      const c = body[i];
      if (top.quoted) {
        i++;
        continue;
      }
      if (c === "{" || c === "[") {
        top.depth++;
        i++;
        continue;
      }
      const closing = c === "}" || c === "]";
      if (closing && (top.depth > 1 || (top.depth === 1 && !top.root))) {
        top.depth--;
        i++;
        continue;
      }
      if (closing && top.depth === 1) {
        // The free span is the whole root object; its close completes the call.
        i++;
        if (!typeMatches(parseToolValue(body.slice(top.start, i)), top.schema)) return invalid;
        frames.pop();
        complete = true;
        continue;
      }
      if (!closing && (c !== "," || top.depth > 0)) {
        i++;
        continue;
      }
      // The span ends just before c; validate it and let the parent frame
      // process the delimiter.
      if (!typeMatches(parseToolValue(body.slice(top.start, i)), top.schema)) return invalid;
      frames.pop();
      finishValue();
      continue;
    }

    const c = body[i];
    if (!top) {
      if (/\s/.test(c)) {
        i++;
        continue;
      }
      if (c !== "{") return invalid;
      const before = frames.length;
      if (!beginValue(rootSchema)) return invalid;
      const pushed = frames[frames.length - 1];
      if (frames.length > before && pushed.type === "free") pushed.root = true;
      continue;
    }

    if (top.type === "obj") {
      if (top.phase === "key") {
        if (c === "}") {
          if (top.keyBuf.trim()) return invalid;
          if (![...top.required].every((key) => top.seen.has(key))) return invalid;
          frames.pop();
          finishValue();
          i++;
          continue;
        }
        if (c === ":") {
          const key = top.keyBuf.trim();
          if (!key || !top.declared.has(key) || top.seen.has(key)) return invalid;
          top.seen.add(key);
          top.valueSchema = top.props[key];
          top.keyBuf = "";
          top.phase = "value";
          i++;
          continue;
        }
        if (c === ",") return invalid;
        top.keyBuf += c;
        i++;
        continue;
      }
      if (/\s/.test(c)) {
        i++;
        continue;
      }
      if (top.phase === "value") {
        if (!beginValue(top.valueSchema)) return invalid;
        continue;
      }
      if (c === ",") {
        if (![...top.declared].some((key) => !top.seen.has(key))) return invalid;
        top.phase = "key";
        i++;
        continue;
      }
      if (c === "}") {
        if (![...top.required].every((key) => top.seen.has(key))) return invalid;
        frames.pop();
        finishValue();
        i++;
        continue;
      }
      return invalid;
    }

    // Array frame.
    if (/\s/.test(c)) {
      i++;
      continue;
    }
    if (top.phase === "value") {
      if (c === "]") {
        if (top.count > 0) return invalid; // trailing comma
        if ((numberKeyword(top.schema, "minItems") ?? 0) > 0) return invalid;
        frames.pop();
        finishValue();
        i++;
        continue;
      }
      if (!beginValue(gemmaItemSchema(top))) return invalid;
      continue;
    }
    if (c === ",") {
      const maxItems = numberKeyword(top.schema, "maxItems");
      if (maxItems !== undefined && top.count >= maxItems) return invalid;
      top.phase = "value";
      i++;
      continue;
    }
    if (c === "]") {
      if (top.count < (numberKeyword(top.schema, "minItems") ?? 0)) return invalid;
      frames.pop();
      finishValue();
      i++;
      continue;
    }
    return invalid;
  }

  if (complete) return { kind: "complete" };
  const top = frames[frames.length - 1];
  if (st.scalar) {
    if (st.scalar.sort === "string") return { kind: "inString", schema: st.scalar.schema, content: st.scalar.buf };
    if (st.scalar.sort === "number") return { kind: "inNumber", schema: st.scalar.schema, text: st.scalar.buf, frame: top };
    return { kind: "inWord", schema: st.scalar.schema, text: st.scalar.buf, frame: top };
  }
  if (!top) return invalid;
  if (top.type === "free") return { kind: "free" };
  if (top.type === "obj") {
    if (top.phase === "key") return { kind: "key", frame: top };
    if (top.phase === "value") return { kind: "valueStart", schema: top.valueSchema, frame: top };
    return { kind: "afterValue", frame: top };
  }
  if (top.phase === "value") return { kind: "valueStart", schema: gemmaItemSchema(top), frame: top };
  return { kind: "afterValue", frame: top };
}
