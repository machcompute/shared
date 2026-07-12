const QUOTE = '<|"|>';

type Json = unknown;

function sortedKeys(obj: Record<string, unknown>): string[] {
  return Object.keys(obj).sort((a, b) => {
    const la = a.toLowerCase();
    const lb = b.toLowerCase();
    return la < lb ? -1 : la > lb ? 1 : 0;
  });
}

export function formatArgument(value: Json, escapeKeys: boolean): string {
  if (typeof value === "string") return QUOTE + value + QUOTE;
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
        .map((key) => (escapeKeys ? QUOTE + key + QUOTE : key) + ":" + formatArgument(obj[key], escapeKeys))
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
  return "[" + items.map((item) => QUOTE + String(item) + QUOTE).join(",") + "]";
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
    const type = upper(value["type"]);

    if (value["description"]) inner.push("description:" + QUOTE + value["description"] + QUOTE);

    if (type === "STRING") {
      if (value["enum"]) inner.push("enum:" + formatArgument(value["enum"], true));
    } else if (type === "ARRAY") {
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
            itemsInner.push(
              "type:" +
                (typeof itemValue === "string"
                  ? formatArgument(upper(itemValue), true)
                  : formatArgument((itemValue as unknown[]).map(upper), true))
            );
          } else {
            itemsInner.push(itemKey + ":" + formatArgument(itemValue, true));
          }
        }
        inner.push("items:{" + itemsInner.join(",") + "}");
      }
    }

    if (value["nullable"]) inner.push("nullable:true");

    if (type === "OBJECT") {
      const props = value["properties"];
      if (props && typeof props === "object" && !Array.isArray(props)) {
        inner.push("properties:{" + formatParameters(props as Record<string, unknown>) + "}");
      }
      if (Array.isArray(value["required"])) inner.push("required:" + quotedList(value["required"] as unknown[]));
    }

    inner.push("type:" + QUOTE + type + QUOTE);
    out.push(key + ":{" + inner.join(",") + "}");
  }
  return out.join(",");
}

export interface ToolFunctionDef {
  function: { name: string; description?: string; parameters?: Record<string, unknown> };
}

export function formatToolDeclaration(tool: ToolFunctionDef): string {
  const fn = tool.function;
  let out = "declaration:" + fn.name + "{description:" + QUOTE + (fn.description ?? "") + QUOTE;
  const params = fn.parameters;
  if (params) {
    out += ",parameters:{";
    const props = params["properties"];
    if (props && typeof props === "object") {
      out += "properties:{" + formatParameters(props as Record<string, unknown>) + "},";
    }
    if (Array.isArray(params["required"])) out += "required:" + quotedList(params["required"] as unknown[]) + ",";
    if (params["type"]) out += "type:" + QUOTE + upper(params["type"]) + QUOTE + "}";
  }
  out += "}";
  return out;
}

export function formatToolCall(name: string, args: Record<string, unknown>): string {
  const body = sortedKeys(args)
    .map((key) => key + ":" + formatArgument(args[key], false))
    .join(",");
  return "<|tool_call>call:" + name + "{" + body + "}<tool_call|>";
}

export function formatToolResponse(name: string, content: string): string {
  return "<|tool_response>response:" + name + "{value:" + formatArgument(content, false) + "}<tool_response|>";
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
