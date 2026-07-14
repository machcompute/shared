const QUOTE = '<|"|>';
const BARE_NAME_RE = /^[A-Za-z_][\w.-]*$/;
const RESERVED = [
  QUOTE,
  "<|tool>",
  "<tool|>",
  "<|tool_call>",
  "<tool_call|>",
  "<|tool_response>",
  "<tool_response|>",
];

type Json = unknown;

function safeText(value: unknown): string {
  let text = String(value ?? "");
  // Gemma's sentinel-delimited strings have no native escape syntax. Render
  // the leading angle bracket as a visible Unicode escape so user/tool text
  // cannot terminate a declaration, call, or response block.
  for (const marker of RESERVED) {
    text = text.replaceAll(marker, `\\u003c${marker.slice(1)}`);
  }
  return text;
}

function quoted(value: unknown): string {
  return QUOTE + safeText(value) + QUOTE;
}

function bareName(value: string, label = "property name"): string {
  if (!BARE_NAME_RE.test(value)) {
    throw new Error(`${label} cannot be serialized in Gemma tool syntax: ${value}`);
  }
  return value;
}

function sortedKeys(obj: Record<string, unknown>): string[] {
  return Object.keys(obj).sort((a, b) => {
    const la = a.toLowerCase();
    const lb = b.toLowerCase();
    return la < lb ? -1 : la > lb ? 1 : 0;
  });
}

export function formatArgument(value: Json, escapeKeys: boolean): string {
  if (typeof value === "string") return quoted(value);
  if (typeof value === "boolean") return value ? "true" : "false";
  if (value === null || value === undefined) return "";
  if (Array.isArray(value)) {
    return "[" + value.map((item) => formatArgument(item, escapeKeys)).join(",") + "]";
  }
  if (typeof value === "object") {
    const obj = value as Record<string, unknown>;
    return (
      "{" +
      sortedKeys(obj)
        .map((key) => (escapeKeys ? quoted(key) : bareName(key)) + ":" + formatArgument(obj[key], escapeKeys))
        .join(",") +
      "}"
    );
  }
  return String(value);
}

function upper(value: unknown): string {
  return String(value ?? "").toUpperCase();
}

function quotedList(items: unknown[]): string {
  return "[" + items.map(quoted).join(",") + "]";
}

function formatType(value: unknown): string {
  if (typeof value === "string") return quoted(upper(value));
  if (Array.isArray(value) && value.every((item) => typeof item === "string")) {
    return formatArgument(value.map(upper), true);
  }
  throw new Error("Gemma tool schema type must be a string or an array of strings");
}

function formatParameters(
  properties: Record<string, unknown>,
  filterKeys = false
): string {
  const standardKeys = ["description", "type", "properties", "required", "nullable"];
  const out: string[] = [];
  for (const key of sortedKeys(properties)) {
    if (filterKeys && standardKeys.includes(key)) continue;
    const value = (properties[key] ?? {}) as Record<string, unknown>;
    const inner: string[] = [];
    const rawType = value["type"];
    const types = (Array.isArray(rawType) ? rawType : [rawType]).map(upper);

    if (value["description"]) inner.push("description:" + quoted(value["description"]));

    if (value["enum"]) inner.push("enum:" + formatArgument(value["enum"], true));
    if (types.includes("ARRAY") || value["items"] !== undefined) {
      const items = value["items"];
      if (items && typeof items === "object" && !Array.isArray(items)) {
        const itemsObj = items as Record<string, unknown>;
        const itemsInner: string[] = [];
        for (const itemKey of sortedKeys(itemsObj)) {
          const itemValue = itemsObj[itemKey];
          if (itemValue === null || itemValue === undefined) continue;
          if (itemKey === "properties" && typeof itemValue === "object") {
            itemsInner.push("properties:{" + formatParameters(itemValue as Record<string, unknown>) + "}");
          } else if (itemKey === "required" && Array.isArray(itemValue)) {
            itemsInner.push("required:" + quotedList(itemValue));
          } else if (itemKey === "type") {
            itemsInner.push("type:" + formatType(itemValue));
          } else {
            itemsInner.push(itemKey + ":" + formatArgument(itemValue, true));
          }
        }
        inner.push("items:{" + itemsInner.join(",") + "}");
      }
    }

    if (value["nullable"]) inner.push("nullable:true");

    if (types.includes("OBJECT") || value["properties"] !== undefined) {
      const props = value["properties"];
      if (props && typeof props === "object" && !Array.isArray(props)) {
        inner.push("properties:{" + formatParameters(props as Record<string, unknown>) + "}");
      }
      if (Array.isArray(value["required"])) inner.push("required:" + quotedList(value["required"] as unknown[]));
    }

    if (rawType !== undefined) inner.push("type:" + formatType(rawType));
    out.push(bareName(key) + ":{" + inner.join(",") + "}");
  }
  return out.join(",");
}

export interface ToolFunctionDef {
  function: { name: string; description?: string; parameters?: Record<string, unknown> };
}

export function formatToolDeclaration(tool: ToolFunctionDef): string {
  const fn = tool.function;
  let out = "declaration:" + bareName(fn.name, "tool function name") + "{description:" + quoted(fn.description ?? "");
  const params = fn.parameters;
  if (params) {
    const fields: string[] = [];
    const props = params["properties"];
    if (props && typeof props === "object" && !Array.isArray(props)) {
      fields.push("properties:{" + formatParameters(props as Record<string, unknown>) + "}");
    }
    if (Array.isArray(params["required"])) fields.push("required:" + quotedList(params["required"] as unknown[]));
    if (params["type"] !== undefined) fields.push("type:" + formatType(params["type"]));
    out += ",parameters:{" + fields.join(",") + "}";
  }
  out += "}";
  return out;
}

export function formatToolCall(name: string, args: Record<string, unknown>): string {
  const body = sortedKeys(args)
    .map((key) => bareName(key) + ":" + formatArgument(args[key], false))
    .join(",");
  return "<|tool_call>call:" + bareName(name, "tool function name") + "{" + body + "}<tool_call|>";
}

export function formatToolResponse(name: string, content: string): string {
  return "<|tool_response>response:" + bareName(name, "tool function name") + "{value:" + formatArgument(content, false) + "}<tool_response|>";
}

/** Parse one Gemma-format value (string/number/word/object/array) in
 * isolation, as the incremental tool constraint needs for free-form spans. */
export function parseToolValue(text: string): unknown {
  const p = { s: text, i: 0 };
  return parseValue(p);
}

export function parseToolArguments(body: string): Record<string, unknown> {
  const p = { s: body, i: 0 };
  skipWs(p);
  if (p.s[p.i] !== "{") return {};
  const value = parseObject(p);
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

interface Cursor {
  s: string;
  i: number;
}

function skipWs(p: Cursor) {
  while (p.i < p.s.length && /\s/.test(p.s[p.i])) p.i++;
}

function at(p: Cursor): string {
  return p.s[p.i];
}

function startsQuote(p: Cursor): boolean {
  return p.s.startsWith(QUOTE, p.i);
}

function parseString(p: Cursor): string {
  p.i += QUOTE.length;
  const end = p.s.indexOf(QUOTE, p.i);
  if (end < 0) {
    const rest = p.s.slice(p.i);
    p.i = p.s.length;
    return rest;
  }
  const value = p.s.slice(p.i, end);
  p.i = end + QUOTE.length;
  return value;
}

function parseKey(p: Cursor): string {
  if (startsQuote(p)) return parseString(p);
  const start = p.i;
  while (p.i < p.s.length && ![":", ",", "}"].includes(p.s[p.i])) p.i++;
  return p.s.slice(start, p.i).trim();
}

function parseValue(p: Cursor): Json {
  skipWs(p);
  if (startsQuote(p)) return parseString(p);
  const c = at(p);
  if (c === "{") return parseObject(p);
  if (c === "[") return parseArray(p);
  const start = p.i;
  while (p.i < p.s.length && ![",", "}", "]"].includes(p.s[p.i])) p.i++;
  const raw = p.s.slice(start, p.i).trim();
  if (raw === "true") return true;
  if (raw === "false") return false;
  if (raw === "null") return null;
  const n = Number(raw);
  return raw !== "" && Number.isFinite(n) ? n : raw;
}

function parseObject(p: Cursor): Record<string, unknown> {
  const obj: Record<string, unknown> = {};
  p.i++;
  skipWs(p);
  if (at(p) === "}") {
    p.i++;
    return obj;
  }
  for (;;) {
    skipWs(p);
    const key = parseKey(p);
    skipWs(p);
    if (at(p) !== ":") break;
    p.i++;
    obj[key] = parseValue(p);
    skipWs(p);
    const c = at(p);
    if (c === ",") {
      p.i++;
      continue;
    }
    if (c === "}") p.i++;
    break;
  }
  return obj;
}

function parseArray(p: Cursor): unknown[] {
  const arr: unknown[] = [];
  p.i++;
  skipWs(p);
  if (at(p) === "]") {
    p.i++;
    return arr;
  }
  for (;;) {
    arr.push(parseValue(p));
    skipWs(p);
    const c = at(p);
    if (c === ",") {
      p.i++;
      continue;
    }
    if (c === "]") p.i++;
    break;
  }
  return arr;
}
