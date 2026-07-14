import { describe, expect, it } from "vitest";
import {
  canCommitCompletionPrefix,
  qwenDecodeBatchSize,
} from "../lib/completion-policy";

describe("completion cache policy", () => {
  it("forces single-token Qwen decoding whenever tools are available", () => {
    expect(qwenDecodeBatchSize(true, 8)).toBe(1);
    expect(qwenDecodeBatchSize(false, 8)).toBe(8);
  });

  it("does not reuse aborted or length-capped prefixes", () => {
    expect(canCommitCompletionPrefix({ aborted: false, stopped: false, hasToolCalls: false })).toBe(false);
    expect(canCommitCompletionPrefix({ aborted: true, stopped: true, hasToolCalls: false })).toBe(false);
  });

  it("reuses clean stop and completed tool-call prefixes", () => {
    expect(canCommitCompletionPrefix({ aborted: false, stopped: true, hasToolCalls: false })).toBe(true);
    expect(canCommitCompletionPrefix({ aborted: false, stopped: false, hasToolCalls: true })).toBe(true);
  });
});
