import { describe, expect, it } from "vitest";
import {
  formatArgument,
  formatToolCall,
  formatToolDeclaration,
} from "../lib/webgpu-llm/gemma-tool-format";

describe("Gemma tool declaration formatting", () => {
  it("closes parameters without a root type and emits no trailing comma", () => {
    const declaration = formatToolDeclaration({
      function: {
        name: "f",
        description: "",
        parameters: {
          properties: { s: { type: "string" } },
        },
      },
    });
    expect(declaration).toBe(
      'declaration:f{description:<|"|><|"|>,parameters:{properties:{s:{type:<|"|>STRING<|"|>}}}}',
    );
    expect(declaration).not.toContain(",}");
  });

  it("joins root fields without leaving a trailing comma", () => {
    const declaration = formatToolDeclaration({
      function: {
        name: "f",
        parameters: {
          type: "object",
          properties: { n: { type: "integer" } },
          required: ["n"],
        },
      },
    });
    expect(declaration).toContain(
      'parameters:{properties:{n:{type:<|"|>INTEGER<|"|>}},required:[<|"|>n<|"|>],type:<|"|>OBJECT<|"|>}',
    );
    expect(declaration).not.toContain(",}");
  });

  it("fails clearly on syntax-breaking names and escapes control markers", () => {
    expect(() => formatToolCall("bad:name", {})).toThrow(/cannot be serialized/);
    expect(formatArgument('unsafe <|"|> value', false)).toBe(
      '<|"|>unsafe \\u003c|"|> value<|"|>',
    );
  });
});
