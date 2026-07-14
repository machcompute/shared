export const MODEL_DEFAULTS = [
  {
    id: "Qwen/Qwen3.5-4B",
    label: "Qwen 3.5 4B",
    modalities: ["text"],
    maxContext: 65_536,
    context: 16_384,
    batchSize: 8,
    mtp: false,
  },
  {
    id: "google/gemma-4-E4B",
    label: "Gemma 4 E4B",
    modalities: ["text", "image", "audio", "video"],
    maxContext: 131_072,
    context: 8_192,
    batchSize: 4,
    mtp: false,
  },
  {
    id: "google/gemma-4-E2B",
    label: "Gemma 4 E2B",
    modalities: ["text", "image", "audio", "video"],
    maxContext: 131_072,
    context: 8_192,
    batchSize: 4,
    mtp: false,
  },
];

export const BENCHMARK_DEFAULTS = {
  promptSeed: "test",
  promptRepeats: 4_000,
  maxTokens: 256,
  warmupTokens: 16,
  runs: 1,
};

const finite = (value) => typeof value === "number" && Number.isFinite(value);

function integer(value, label, min, max) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < min || parsed > max) {
    throw new Error(`${label} must be a whole number from ${min.toLocaleString()} to ${max.toLocaleString()}.`);
  }
  return parsed;
}

export function normalizeConfig(input) {
  const promptSeed = String(input.promptSeed ?? "").trim();
  if (!promptSeed) throw new Error("Prompt text cannot be empty.");
  if (promptSeed.length > 80) throw new Error("Prompt text must be 80 characters or fewer.");

  const config = {
    promptSeed,
    promptRepeats: integer(input.promptRepeats, "Prompt repetitions", 1, 100_000),
    maxTokens: integer(input.maxTokens, "Output tokens", 2, 4_096),
    warmupTokens: integer(input.warmupTokens, "Warmup tokens", 0, 256),
    runs: integer(input.runs, "Runs per model", 1, 10),
    models: (input.models ?? []).filter((model) => model.selected !== false).map((model) => {
      const maxContext = integer(model.maxContext, `${model.label} context`, 1_024, model.contextLimit);
      const batchSize = integer(model.batchSize, `${model.label} batch size`, 1, 8);
      if (configPromptFloor(input.promptRepeats, input.maxTokens) > maxContext) {
        throw new Error(`${model.label} needs a context above the estimated prompt and output size.`);
      }
      return {
        id: model.id,
        label: model.label,
        maxContext,
        batchSize,
        mtp: !!model.mtp,
      };
    }),
  };

  if (!config.models.length) throw new Error("Select at least one model.");
  return config;
}

function configPromptFloor(repeats, maxTokens) {
  return Number(repeats) + Number(maxTokens) + 64;
}

export function buildBenchmarkPrompt(seed, repeats) {
  const word = String(seed).trim();
  return `${`${word} `.repeat(repeats)}\nContinue by outputting only ${word}, separated by spaces. Do not stop early.`;
}

export function calculateRunMetrics({ requestStart, firstTokenAt, end, usage, finishReason }) {
  const promptTokens = Number(usage?.prompt_tokens ?? 0);
  const completionTokens = Number(usage?.completion_tokens ?? 0);
  const measuredDecodeTokens = Math.max(0, completionTokens - 1);
  const totalSeconds = Math.max(0, end - requestStart) / 1_000;
  const prefillSeconds = firstTokenAt == null
    ? null
    : Math.max(0, firstTokenAt - requestStart) / 1_000;
  const decodeSeconds = firstTokenAt == null
    ? null
    : Math.max(0, end - firstTokenAt) / 1_000;
  const tps = decodeSeconds && measuredDecodeTokens
    ? measuredDecodeTokens / decodeSeconds
    : null;

  return {
    usage: {
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      total_tokens: Number(usage?.total_tokens ?? promptTokens + completionTokens),
    },
    finishReason: finishReason ?? null,
    totalSeconds,
    prefillSeconds,
    promptTps: prefillSeconds && promptTokens ? promptTokens / prefillSeconds : null,
    decodeSeconds,
    measuredDecodeTokens,
    tps,
    millisecondsPerToken: tps ? 1_000 / tps : null,
  };
}

export function median(values) {
  const sorted = values.filter(finite).toSorted((a, b) => a - b);
  if (!sorted.length) return null;
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}

export function summarizeRuns(runs) {
  const tpsValues = runs.map((run) => run.tps).filter(finite);
  return {
    runs: runs.length,
    tps: median(tpsValues),
    minTps: tpsValues.length ? Math.min(...tpsValues) : null,
    maxTps: tpsValues.length ? Math.max(...tpsValues) : null,
    millisecondsPerToken: median(runs.map((run) => run.millisecondsPerToken)),
    prefillSeconds: median(runs.map((run) => run.prefillSeconds)),
    promptTps: median(runs.map((run) => run.promptTps)),
    promptTokens: median(runs.map((run) => run.usage?.prompt_tokens)),
    completionTokens: median(runs.map((run) => run.usage?.completion_tokens)),
    totalSeconds: median(runs.map((run) => run.totalSeconds)),
  };
}

export function abortError() {
  return new DOMException("Benchmark stopped.", "AbortError");
}

export function throwIfAborted(signal) {
  if (signal?.aborted) throw abortError();
}

export function isAbortError(error) {
  return error?.name === "AbortError";
}

/**
 * @param {{
 *   models: Array<any>,
 *   load: (model: any) => Promise<any>,
 *   benchmark: (model: any, loadResult: any) => Promise<any>,
 *   unload: (model: any, loadResult: any, benchmarkResult: any) => Promise<any>,
 *   signal?: AbortSignal | null,
 *   onPhase?: (model: any, phase: "loading" | "benchmarking" | "unloading") => void,
 *   onModelComplete?: (entry: any, entries: Array<any>) => void,
 * }} options
 */
export async function runSequentialModels({
  models,
  load,
  benchmark,
  unload,
  signal = null,
  onPhase = () => {},
  onModelComplete = () => {},
}) {
  const entries = [];

  for (const model of models) {
    if (signal?.aborted) break;
    const entry = { model, status: "running", load: null, result: null, unload: null, error: null };

    try {
      onPhase(model, "loading");
      entry.load = await load(model);
      throwIfAborted(signal);

      onPhase(model, "benchmarking");
      entry.result = await benchmark(model, entry.load);
      throwIfAborted(signal);
      entry.status = "complete";
    } catch (error) {
      entry.error = error;
      entry.status = isAbortError(error) ? "cancelled" : "error";
    } finally {
      try {
        onPhase(model, "unloading");
        entry.unload = await unload(model, entry.load, entry.result);
      } catch (error) {
        entry.unloadError = error;
        if (entry.status === "complete") {
          entry.status = "error";
          entry.error = error;
        }
      }
    }

    entries.push(entry);
    onModelComplete(entry, entries);
    if (entry.status === "cancelled" || signal?.aborted) break;
  }

  return entries;
}
