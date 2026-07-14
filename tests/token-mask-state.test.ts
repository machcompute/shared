import { describe, expect, it, vi } from "vitest";
import { topk } from "../lib/webgpu-llm/kernels.js";
import { GemmaModel } from "../lib/webgpu-llm/gemma-model.js";
import { updateTokenMaskState } from "../lib/webgpu-llm/token-mask-state.js";

function fakeMaskState() {
  const writeBuffer = vi.fn();
  return {
    state: {
      cfg: { text: { vocab: 32 } },
      gpu: { device: { queue: { writeBuffer } } },
      b: { allowed: {}, uAllowed: {} },
      allowedCount: 0,
      tokenMaskKind: "all",
      tokenMaskIds: null as Uint32Array | null,
    },
    writeBuffer,
  };
}

describe("Gemma token-mask state", () => {
  it("does not upload the same immutable mask more than once", () => {
    const { state, writeBuffer } = fakeMaskState();
    const tokenIds = Uint32Array.of(1, 2, 3);
    const mask = { kind: "deny" as const, tokenIds };

    updateTokenMaskState(state, mask);
    expect(writeBuffer).toHaveBeenCalledTimes(2);

    updateTokenMaskState(state, mask);
    updateTokenMaskState(state, { kind: "deny", tokenIds });
    expect(writeBuffer).toHaveBeenCalledTimes(2);

    updateTokenMaskState(state, { kind: "deny", tokenIds: Uint32Array.of(1, 2, 3) });
    expect(writeBuffer).toHaveBeenCalledTimes(3);

    updateTokenMaskState(state, undefined);
    expect(writeBuffer).toHaveBeenCalledTimes(4);
    expect(state.tokenMaskKind).toBe("all");
  });

  it("preserves an empty allow mask as an empty constraint", () => {
    const { state, writeBuffer } = fakeMaskState();
    updateTokenMaskState(state, { kind: "allow", tokenIds: new Uint32Array() });
    expect(state.tokenMaskKind).toBe("allow");
    expect(writeBuffer).not.toHaveBeenCalled();
  });

  it("generates distinct allow and deny GPU filters", () => {
    const allow = topk({ VOCAB: 32, KTOP: 4, ALLOWED: true });
    const deny = topk({ VOCAB: 32, KTOP: 4, DENIED: true });
    expect(allow).toContain("&& isListed(g)");
    expect(deny).toContain("&& !isListed(g)");
    expect(() => topk({ VOCAB: 32, KTOP: 4, ALLOWED: true, DENIED: true })).toThrow();
  });
});

describe("Gemma speculative rewind", () => {
  it("rewinds cache positions and penalty tokens by their independent counts", () => {
    const model = Object.create(GemmaModel.prototype) as GemmaModel & {
      pos: number;
      genIds: number[];
    };
    model.pos = 12;
    model.genIds = [1, 2, 3, 4, 5];
    model.rewindDecode(3, 2);
    expect(model.pos).toBe(9);
    expect(model.genIds).toEqual([1, 2, 3]);
  });
});
