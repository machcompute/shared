import { describe, expect, it, vi } from "vitest";
import {
  buildBenchmarkPrompt,
  calculateRunMetrics,
  normalizeConfig,
  runSequentialModels,
  summarizeRuns,
} from "../bench/benchmark-core.js";

const model = (id: string) => ({ id, label: id, maxContext: 8192, batchSize: 4, mtp: false });
type TestModel = ReturnType<typeof model>;

describe("benchmark configuration", () => {
  it("normalizes the selected model settings", () => {
    expect(normalizeConfig({
      promptSeed: " test ",
      promptRepeats: "1024",
      maxTokens: "128",
      warmupTokens: "8",
      runs: "3",
      models: [
        { id: "a", label: "A", selected: true, maxContext: "4096", contextLimit: 8192, batchSize: "4", mtp: true },
        { id: "b", label: "B", selected: false, maxContext: "4096", contextLimit: 8192, batchSize: "4" },
      ],
    })).toEqual({
      promptSeed: "test",
      promptRepeats: 1024,
      maxTokens: 128,
      warmupTokens: 8,
      runs: 3,
      models: [{ id: "a", label: "A", maxContext: 4096, batchSize: 4, mtp: true }],
    });
  });

  it("rejects a context that cannot contain the requested benchmark", () => {
    expect(() => normalizeConfig({
      promptSeed: "test",
      promptRepeats: 4000,
      maxTokens: 256,
      warmupTokens: 0,
      runs: 1,
      models: [{ id: "a", label: "A", selected: true, maxContext: 4096, contextLimit: 8192, batchSize: 4 }],
    })).toThrow("needs a context above");
  });

  it("builds the deterministic repeated-token prompt", () => {
    const prompt = buildBenchmarkPrompt("test", 3);
    expect(prompt.startsWith("test test test ")).toBe(true);
    expect(prompt).toContain("outputting only test");
  });
});

describe("benchmark metrics", () => {
  it("separates first-token time from measured decode throughput", () => {
    const metrics = calculateRunMetrics({
      requestStart: 1000,
      firstTokenAt: 3000,
      end: 7000,
      usage: { prompt_tokens: 1000, completion_tokens: 201, total_tokens: 1201 },
      finishReason: "length",
    });

    expect(metrics.prefillSeconds).toBe(2);
    expect(metrics.decodeSeconds).toBe(4);
    expect(metrics.measuredDecodeTokens).toBe(200);
    expect(metrics.tps).toBe(50);
    expect(metrics.millisecondsPerToken).toBe(20);
    expect(metrics.promptTps).toBe(500);
  });

  it("summarizes repeated runs with medians and a range", () => {
    const summary = summarizeRuns([
      { tps: 40, millisecondsPerToken: 25, prefillSeconds: 2, promptTps: 500, totalSeconds: 5, usage: { prompt_tokens: 1000, completion_tokens: 100 } },
      { tps: 60, millisecondsPerToken: 16.67, prefillSeconds: 4, promptTps: 250, totalSeconds: 6, usage: { prompt_tokens: 1000, completion_tokens: 100 } },
      { tps: 50, millisecondsPerToken: 20, prefillSeconds: 3, promptTps: 333, totalSeconds: 4, usage: { prompt_tokens: 1000, completion_tokens: 100 } },
    ]);

    expect(summary).toMatchObject({ runs: 3, tps: 50, minTps: 40, maxTps: 60, prefillSeconds: 3 });
  });
});

describe("sequential model queue", () => {
  it("loads, benchmarks, and unloads each model before advancing", async () => {
    const calls: string[] = [];
    const entries = await runSequentialModels({
      models: [model("a"), model("b")],
      load: vi.fn(async (item) => { calls.push(`load:${item.id}`); return item.id; }),
      benchmark: vi.fn(async (item) => { calls.push(`bench:${item.id}`); return { tps: 1 }; }),
      unload: vi.fn(async (item) => { calls.push(`unload:${item.id}`); }),
    });

    expect(calls).toEqual(["load:a", "bench:a", "unload:a", "load:b", "bench:b", "unload:b"]);
    expect(entries.map((entry) => entry.status)).toEqual(["complete", "complete"]);
  });

  it("unloads a failed model and continues with the next one", async () => {
    const calls: string[] = [];
    const entries = await runSequentialModels({
      models: [model("a"), model("b")],
      load: async (item: TestModel) => { calls.push(`load:${item.id}`); },
      benchmark: async (item: TestModel) => {
        calls.push(`bench:${item.id}`);
        if (item.id === "a") throw new Error("benchmark failed");
        return { tps: 1 };
      },
      unload: async (item: TestModel) => { calls.push(`unload:${item.id}`); },
    });

    expect(calls).toEqual(["load:a", "bench:a", "unload:a", "load:b", "bench:b", "unload:b"]);
    expect(entries.map((entry) => entry.status)).toEqual(["error", "complete"]);
  });

  it("unloads after cancellation and does not start another model", async () => {
    const calls: string[] = [];
    const controller = new AbortController();
    const entries = await runSequentialModels({
      models: [model("a"), model("b")],
      signal: controller.signal,
      load: async (item: TestModel) => { calls.push(`load:${item.id}`); },
      benchmark: async (item: TestModel) => {
        calls.push(`bench:${item.id}`);
        controller.abort();
      },
      unload: async (item: TestModel) => { calls.push(`unload:${item.id}`); },
    });

    expect(calls).toEqual(["load:a", "bench:a", "unload:a"]);
    expect(entries).toHaveLength(1);
    expect(entries[0].status).toBe("cancelled");
  });
});
