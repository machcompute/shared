import { describe, expect, it } from "vitest";
import { parseGemmaToolCallBody } from "../lib/webgpu-llm/gemma-tool-parser";
import {
  ToolConstraint,
  validateTools,
  type JsonSchema,
  type TokenMask,
} from "../lib/webgpu-llm/tool-constraint";
import { FakeTokenizer } from "./helpers/fake-tokenizer";

function tool(parameters: JsonSchema, name = "f") {
  return {
    type: "function" as const,
    function: { name, description: "", parameters },
  };
}

function propertyTool(name: string, schema: JsonSchema) {
  return tool({
    type: "object",
    properties: { [name]: schema },
    required: [name],
    additionalProperties: false,
  });
}

function constraintFor(parameters: JsonSchema) {
  const tokenizer = new FakeTokenizer();
  const constraint = new ToolConstraint([tool(parameters)], tokenizer, {
    closeToken: "<tool_call|>",
    grammar: "gemma",
    forbiddenTokenIds: [tokenizer.id("<eos>"), tokenizer.id("<turn>")],
  });
  return { constraint, tokenizer };
}

function permits(mask: TokenMask, id: number): boolean {
  if (mask.kind === "all") return true;
  const listed = Array.from(mask.tokenIds).includes(id);
  return mask.kind === "allow" ? listed : !listed;
}

function maskAt(constraint: ToolConstraint, raw: string): TokenMask {
  return constraint.mask(raw, parseGemmaToolCallBody);
}

describe("Gemma numeric reachability", () => {
  it("honors an exclusive integer maximum without steering into the boundary", () => {
    const { constraint, tokenizer } = constraintFor(propertyTool("n", {
      type: "integer",
      minimum: 10,
      exclusiveMaximum: 20,
    }).function.parameters);

    const start = maskAt(constraint, "call:f{n:");
    expect(permits(start, tokenizer.id("1"))).toBe(true);
    expect(permits(start, tokenizer.id("2"))).toBe(false);
    expect(permits(start, tokenizer.id("20"))).toBe(false);

    const afterOne = maskAt(constraint, "call:f{n:1");
    expect(permits(afterOne, tokenizer.id("0"))).toBe(true);
    expect(permits(afterOne, tokenizer.id("9"))).toBe(true);

    const valid = maskAt(constraint, "call:f{n:19");
    expect(permits(valid, tokenizer.id("}"))).toBe(true);
  });

  it("uses the stricter bound when inclusive and exclusive minima coexist", () => {
    const { constraint, tokenizer } = constraintFor(propertyTool("n", {
      type: "integer",
      minimum: 10,
      exclusiveMinimum: 10,
      maximum: 20,
    }).function.parameters);
    const afterOne = maskAt(constraint, "call:f{n:1");
    expect(permits(afterOne, tokenizer.id("0"))).toBe(false);
    expect(permits(afterOne, tokenizer.id("1"))).toBe(true);
  });

  it("mirrors exclusive bounds correctly for negative integers", () => {
    const { constraint, tokenizer } = constraintFor(propertyTool("n", {
      type: "integer",
      exclusiveMinimum: -20,
      maximum: -10,
    }).function.parameters);
    const start = maskAt(constraint, "call:f{n:");
    expect(permits(start, tokenizer.id("-"))).toBe(true);
    const afterMinus = maskAt(constraint, "call:f{n:-");
    expect(permits(afterMinus, tokenizer.id("1"))).toBe(true);
    expect(permits(afterMinus, tokenizer.id("2"))).toBe(false);
  });

  it("keeps decimal prefixes only when they intersect open number bounds", () => {
    const { constraint, tokenizer } = constraintFor(propertyTool("n", {
      type: "number",
      exclusiveMinimum: 0,
      exclusiveMaximum: 1,
    }).function.parameters);
    const start = maskAt(constraint, "call:f{n:");
    expect(permits(start, tokenizer.id("0"))).toBe(true);
    expect(permits(start, tokenizer.id("1"))).toBe(false);
    expect(permits(maskAt(constraint, "call:f{n:0"), tokenizer.id(".5"))).toBe(true);
    expect(permits(maskAt(constraint, "call:f{n:0.5"), tokenizer.id("}"))).toBe(true);
  });

  it("steers numeric and boolean enums toward reachable members only", () => {
    const numeric = constraintFor(propertyTool("n", { type: "integer", enum: [23] }).function.parameters);
    const numericStart = maskAt(numeric.constraint, "call:f{n:");
    expect(permits(numericStart, numeric.tokenizer.id("23"))).toBe(true);
    expect(permits(numericStart, numeric.tokenizer.id("20"))).toBe(false);

    const boolean = constraintFor(propertyTool("ok", { type: "boolean", enum: [false] }).function.parameters);
    const booleanStart = maskAt(boolean.constraint, "call:f{ok:");
    expect(permits(booleanStart, boolean.tokenizer.id("false"))).toBe(true);
    expect(permits(booleanStart, boolean.tokenizer.id("true"))).toBe(false);
  });
});

describe("Gemma string masks", () => {
  it("filters multi-character tokens before they can exceed maxLength", () => {
    const { constraint, tokenizer } = constraintFor(propertyTool("s", {
      type: "string",
      minLength: 2,
      maxLength: 3,
    }).function.parameters);

    const empty = maskAt(constraint, 'call:f{s:<|"|>');
    expect(empty.kind).toBe("allow");
    expect(permits(empty, tokenizer.id("ab"))).toBe(true);
    expect(permits(empty, tokenizer.id("abcd"))).toBe(false);
    expect(permits(empty, tokenizer.id("<byte>"))).toBe(false);
    expect(permits(empty, tokenizer.id('<|"|>'))).toBe(false);

    const twoChars = maskAt(constraint, 'call:f{s:<|"|>ab');
    expect(permits(twoChars, tokenizer.id("a"))).toBe(true);
    expect(permits(twoChars, tokenizer.id("ab"))).toBe(false);
    expect(permits(twoChars, tokenizer.id('<|"|>'))).toBe(true);

    const atLimit = maskAt(constraint, 'call:f{s:<|"|>abc');
    expect(permits(atLimit, tokenizer.id("a"))).toBe(false);
    expect(permits(atLimit, tokenizer.id('<|"|>'))).toBe(true);
  });

  it("uses a small deny-list for unrestricted strings", () => {
    const { constraint, tokenizer } = constraintFor(propertyTool("s", { type: "string" }).function.parameters);
    const mask = maskAt(constraint, 'call:f{s:<|"|>');
    expect(mask.kind).toBe("deny");
    if (mask.kind !== "deny") throw new Error("expected deny mask");
    expect(Array.from(mask.tokenIds)).toEqual([
      tokenizer.id("<tool_call|>"),
      tokenizer.id("<eos>"),
      tokenizer.id("<turn>"),
    ].sort((a, b) => a - b));
    expect(permits(mask, tokenizer.id("abcd"))).toBe(true);
    expect(permits(mask, tokenizer.id('<|"|>'))).toBe(true);
  });

  it("rejects an already-oversized string instead of relaxing maxLength", () => {
    const { constraint } = constraintFor(propertyTool("s", { type: "string", maxLength: 3 }).function.parameters);
    const mask = maskAt(constraint, 'call:f{s:<|"|>abcd<|"|>}');
    expect(mask.kind).toBe("allow");
    if (mask.kind !== "allow") throw new Error("expected allow mask");
    expect(mask.tokenIds).toHaveLength(0);
  });
});

describe("tool definition validation", () => {
  it("accepts an object schema whose root type is omitted", () => {
    expect(() => validateTools([tool({ properties: { s: { type: "string" } } })])).not.toThrow();
  });

  it("rejects invalid and duplicate function names", () => {
    expect(() => validateTools([tool({ type: "object" }, "bad:name")])).toThrow(/must match/);
    expect(() => validateTools([tool({ type: "object" }), tool({ type: "object" })])).toThrow(/Duplicate/);
  });

  it("rejects malformed parameter objects and unsafe property names", () => {
    expect(() => validateTools([{ type: "function", function: { name: "f", parameters: [] } }])).toThrow(/object schema/);
    expect(() => validateTools([tool({ type: "object", properties: { "bad:key": { type: "string" } } })])).toThrow(/cannot serialize/);
  });

  it("rejects enum values that the Gemma string grammar cannot represent", () => {
    const unsafe = propertyTool("s", { type: "string", enum: ['break <|"|> value'] });
    expect(() => validateTools([unsafe])).toThrow(/reserved Gemma control text/);
  });
});
