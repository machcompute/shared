import {
  BENCHMARK_DEFAULTS,
  MODEL_DEFAULTS,
  buildBenchmarkPrompt,
  calculateRunMetrics,
  isAbortError,
  normalizeConfig,
  runSequentialModels,
  summarizeRuns,
  throwIfAborted,
} from "./benchmark-core.js";

const $ = (selector) => document.querySelector(selector);
const form = $("#benchmark-form");
const modelList = $("#model-list");
const runButton = $("#run-button");
const stopButton = $("#stop-button");
const connectButton = $("#connect-button");
const restoreButton = $("#restore-button");
const connectionPill = $("#connection-pill");
const connectionLabel = $("#connection-label");
const runStatusTitle = $("#run-status-title");
const runStatusDetail = $("#run-status-detail");
const overallProgressLabel = $("#overall-progress-label");
const overallProgressBar = $("#overall-progress-bar");
const metricGrid = $("#metric-grid");
const resultsBody = $("#results-body");
const throughputChart = $("#throughput-chart");
const rawOutput = $("#raw-output");
const copyButton = $("#copy-button");
const downloadButton = $("#download-button");
const activityLog = $("#activity-log");
const logCount = $("#log-count");
const toast = $("#toast");

const DEFAULT_ENGINE_URL = ["localhost", "127.0.0.1", "::1"].includes(location.hostname)
  ? "http://localhost:3001"
  : "https://shared.machcomputing.com";

const state = {
  client: null,
  connectedUrl: null,
  connectPromise: null,
  initialStatus: null,
  models: MODEL_DEFAULTS.map((model) => ({ ...model, selected: true, cached: null })),
  running: false,
  abortController: null,
  activeModelId: null,
  activePhase: null,
  startedAt: null,
  report: null,
  logs: [],
  toastTimer: null,
};

window.__modelBenchmark = { pending: true };

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function format(value, digits = 1) {
  return Number.isFinite(value)
    ? new Intl.NumberFormat(undefined, { maximumFractionDigits: digits }).format(value)
    : "—";
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds)) return "—";
  if (seconds < 1) return `${format(seconds * 1_000, 0)} ms`;
  if (seconds < 60) return `${format(seconds, 2)} s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${format(seconds % 60, 0)}s`;
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "—";
  const gib = bytes / 1024 ** 3;
  return `${format(gib, 2)} GB`;
}

function errorMessage(error) {
  return error?.message || String(error);
}

function showToast(message, kind = "info") {
  clearTimeout(state.toastTimer);
  toast.textContent = message;
  toast.className = `toast visible${kind === "error" ? " error" : ""}`;
  state.toastTimer = setTimeout(() => {
    toast.className = "toast";
  }, 4_500);
}

function appendLog(stage, message) {
  const event = {
    at: new Date().toISOString(),
    stage: String(stage || "event"),
    message: String(message || ""),
  };
  state.logs.push(event);
  if (state.logs.length > 200) state.logs.shift();

  const item = document.createElement("li");
  item.className = "log-entry";
  const time = document.createElement("time");
  time.dateTime = event.at;
  time.textContent = new Date(event.at).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  const stageLabel = document.createElement("b");
  stageLabel.textContent = event.stage;
  const detail = document.createElement("span");
  detail.textContent = event.message;
  item.append(time, stageLabel, detail);
  activityLog.append(item);
  while (activityLog.children.length > 200) activityLog.firstElementChild.remove();
  logCount.textContent = `${state.logs.length} event${state.logs.length === 1 ? "" : "s"}`;
}

function setConnectionState(connectionState, label) {
  connectionPill.dataset.state = connectionState;
  connectionLabel.textContent = label;
}

function captureModelPreferences() {
  return new Map([...modelList.querySelectorAll(".model-card")].map((card) => [
    card.dataset.modelId,
    {
      selected: card.querySelector("[data-role='selected']")?.checked ?? true,
      context: Number(card.querySelector("[data-role='context']")?.value),
      batchSize: Number(card.querySelector("[data-role='batch']")?.value),
      mtp: card.querySelector("[data-role='mtp']")?.checked ?? false,
    },
  ]));
}

function renderModels(models, preserve = true) {
  const preferences = preserve ? captureModelPreferences() : new Map();
  state.models = models.map((model) => ({
    ...model,
    ...(preferences.get(model.id) ?? {}),
  }));

  modelList.innerHTML = state.models.map((model, index) => {
    const selected = model.selected !== false;
    const cacheLabel = model.cached === true ? "Cached" : model.cached === false ? "Not cached" : "Cache unknown";
    const batchOptions = [1, 2, 4, 8].map((size) =>
      `<option value="${size}"${Number(model.batchSize) === size ? " selected" : ""}>${size}</option>`
    ).join("");
    const mtpControl = model.id === "Qwen/Qwen3.5-4B"
      ? `<label class="mtp-control">
          <span>Multi-token prediction (MTP)</span>
          <input data-role="mtp" type="checkbox"${model.mtp ? " checked" : ""}>
          <span class="switch-ui" aria-hidden="true"></span>
        </label>`
      : `<div class="mtp-control" aria-disabled="true"><span>MTP unavailable for this runtime</span></div>`;

    return `<article class="model-card" data-model-id="${escapeHtml(model.id)}" data-selected="${selected}" data-state="idle">
      <div class="model-card-top">
        <span class="queue-index">QUEUE ${String(index + 1).padStart(2, "0")}</span>
        <label class="select-control">
          <input data-role="selected" type="checkbox"${selected ? " checked" : ""}>
          <span class="check-ui" aria-hidden="true"></span>
          <span>Include</span>
        </label>
      </div>
      <div class="model-title-row">
        <div>
          <h3>${escapeHtml(model.label)}</h3>
          <span class="model-id" title="${escapeHtml(model.id)}">${escapeHtml(model.id)}</span>
        </div>
        <span class="cache-badge" data-cached="${model.cached === true}">${cacheLabel}</span>
      </div>
      <div class="modality-list">
        ${(model.modalities ?? ["text"]).map((modality) => `<span class="modality-tag">${escapeHtml(modality)}</span>`).join("")}
      </div>
      <div class="model-config">
        <label class="model-field">
          <span>Max context</span>
          <input data-role="context" type="number" value="${Number(model.context)}" min="1024" max="${Number(model.maxContext)}" step="1024">
        </label>
        <label class="model-field">
          <span>Batch</span>
          <select data-role="batch">${batchOptions}</select>
        </label>
        ${mtpControl}
      </div>
      <div class="model-status">
        <div class="model-status-line">
          <span class="model-state-label" data-role="state-label">Queued</span>
          <span class="model-detail" data-role="detail">Waiting</span>
        </div>
        <div class="model-progress"><span data-role="progress"></span></div>
      </div>
    </article>`;
  }).join("");
  updateSelectedCount();
}

function mergeRemoteModels(remoteModels) {
  const existing = new Map(state.models.map((model) => [model.id, model]));
  const defaults = new Map(MODEL_DEFAULTS.map((model) => [model.id, model]));
  return remoteModels.map((remote) => {
    const fallback = defaults.get(remote.id) ?? {};
    const current = existing.get(remote.id) ?? {};
    return {
      ...fallback,
      ...remote,
      context: current.context ?? fallback.context ?? Math.min(8_192, remote.maxContext),
      batchSize: current.batchSize ?? fallback.batchSize ?? 4,
      mtp: current.mtp ?? false,
      selected: current.selected ?? true,
    };
  });
}

function selectedCards() {
  return [...modelList.querySelectorAll(".model-card")].filter(
    (card) => card.querySelector("[data-role='selected']")?.checked
  );
}

function updateSelectedCount() {
  const count = selectedCards().length;
  overallProgressLabel.textContent = `0 / ${count}`;
  if (!state.running) overallProgressBar.style.width = "0%";
}

function modelCard(modelId) {
  return [...modelList.querySelectorAll(".model-card")].find((card) => card.dataset.modelId === modelId);
}

function setModelState(modelId, modelState, label, progress, detail = "") {
  const card = modelCard(modelId);
  if (!card) return;
  card.dataset.state = modelState;
  card.querySelector("[data-role='state-label']").textContent = label;
  card.querySelector("[data-role='detail']").textContent = detail;
  card.querySelector("[data-role='progress']").style.width = `${Math.max(0, Math.min(100, progress))}%`;
}

function resetModelStates() {
  for (const card of modelList.querySelectorAll(".model-card")) {
    const selected = card.querySelector("[data-role='selected']")?.checked;
    setModelState(
      card.dataset.modelId,
      "idle",
      selected ? "Queued" : "Skipped",
      0,
      selected ? "Waiting" : "Not selected"
    );
  }
}

function handleEngineProgress(event) {
  appendLog(event.stage, event.message);
  if (!state.activeModelId) return;

  if (state.activePhase === "loading") {
    const fraction = Number.isFinite(event.progress) ? event.progress : 0.08;
    const progress = event.stage === "ready" ? 30 : 3 + fraction * 25;
    setModelState(state.activeModelId, "loading", "Loading", progress, event.message);
  } else if (state.activePhase === "benchmarking" && event.stage === "prefill") {
    const card = modelCard(state.activeModelId);
    const progress = Number.parseFloat(card?.querySelector("[data-role='progress']")?.style.width) || 38;
    setModelState(state.activeModelId, "benchmarking", "Benchmarking", progress, event.message);
  }
}

async function connectEngine(force = false) {
  const requestedUrl = new URL($("#engine-url").value.trim(), location.href).origin;
  if (!force && state.client && state.connectedUrl === requestedUrl) return state.client;
  if (state.connectPromise) return state.connectPromise;

  state.connectPromise = (async () => {
    setConnectionState("connecting", "Connecting…");
    connectButton.disabled = true;
    connectButton.querySelector("span").textContent = "Connecting…";
    appendLog("connect", `Opening ${requestedUrl}`);

    try {
      state.client?.close();
      state.client = null;
      const clientUrl = new URL("/client.js", requestedUrl).href;
      const { MachLLM } = await import(clientUrl);
      const client = await MachLLM.connect({
        engineUrl: requestedUrl,
        timeoutMs: 60_000,
        onProgress: handleEngineProgress,
      });
      const [status, listing] = await Promise.all([client.status(), client.models.list()]);
      if (!status.webgpu || !status.adapter) {
        client.close();
        throw new Error("A usable WebGPU adapter was not found in this browser.");
      }

      state.client = client;
      state.connectedUrl = requestedUrl;
      state.initialStatus = status;
      renderModels(mergeRemoteModels(listing.data), true);
      setConnectionState("connected", `${status.webgpu ? "WebGPU" : "Engine"} ready`);
      appendLog("connect", `Connected; ${listing.data.length} models registered.`);
      return client;
    } catch (error) {
      state.client = null;
      state.connectedUrl = null;
      setConnectionState("error", "Connection failed");
      appendLog("error", errorMessage(error));
      throw error;
    } finally {
      connectButton.disabled = false;
      connectButton.querySelector("span").textContent = "Reconnect";
      state.connectPromise = null;
    }
  })();

  return state.connectPromise;
}

function collectConfig() {
  const models = [...modelList.querySelectorAll(".model-card")].map((card) => {
    const profile = state.models.find((model) => model.id === card.dataset.modelId);
    return {
      id: profile.id,
      label: profile.label,
      selected: card.querySelector("[data-role='selected']").checked,
      maxContext: card.querySelector("[data-role='context']").value,
      contextLimit: profile.maxContext,
      batchSize: card.querySelector("[data-role='batch']").value,
      mtp: card.querySelector("[data-role='mtp']")?.checked ?? false,
    };
  });

  return normalizeConfig({
    promptSeed: $("#prompt-seed").value,
    promptRepeats: $("#prompt-repeats").value,
    maxTokens: $("#output-tokens").value,
    warmupTokens: $("#warmup-tokens").value,
    runs: $("#runs-per-model").value,
    models,
  });
}

async function consumeCompletion(modelId, prompt, maxTokens, signal) {
  const stream = await state.client.chat.completions.create({
    model: modelId,
    messages: [{ role: "user", content: prompt }],
    stream: true,
    max_tokens: maxTokens,
    temperature: 0,
    top_p: 1,
    top_k: 1,
    presence_penalty: 0,
    thinking: false,
  }, { signal });

  for await (const _chunk of stream) {
    // Consuming the stream is the benchmark; output text is intentionally discarded.
  }
  throwIfAborted(signal);
  if (!stream.completion) throw new Error("The completion stream ended without a final result.");
  return stream.completion;
}

async function benchmarkOnce(modelId, prompt, maxTokens, signal) {
  const requestStart = performance.now();
  const stream = await state.client.chat.completions.create({
    model: modelId,
    messages: [{ role: "user", content: prompt }],
    stream: true,
    max_tokens: maxTokens,
    temperature: 0,
    top_p: 1,
    top_k: 1,
    presence_penalty: 0,
    thinking: false,
  }, { signal });

  let firstTokenAt = null;
  for await (const chunk of stream) {
    const delta = chunk.choices?.[0]?.delta;
    if (firstTokenAt == null && (delta?.content || delta?.reasoning_content)) {
      firstTokenAt = performance.now();
    }
  }
  const end = performance.now();
  throwIfAborted(signal);
  const completion = stream.completion;
  if (!completion) throw new Error("The completion stream ended without a final result.");

  return calculateRunMetrics({
    requestStart,
    firstTokenAt,
    end,
    usage: completion.usage,
    finishReason: completion.choices?.[0]?.finish_reason,
  });
}

function serializeEntry(entry) {
  const load = entry.load;
  const result = entry.result;
  return {
    model: entry.model.id,
    label: entry.model.label,
    status: entry.status,
    error: entry.error ? errorMessage(entry.error) : null,
    unloadError: entry.unloadError ? errorMessage(entry.unloadError) : null,
    settings: {
      maxContext: entry.model.maxContext,
      batchSize: entry.model.batchSize,
      mtp: entry.model.mtp,
    },
    loadSeconds: load?.seconds ?? null,
    unloadSeconds: entry.unload?.seconds ?? null,
    device: load?.status?.device ?? null,
    hasMtp: load?.status?.hasMtp ?? false,
    mtpActive: !!entry.model.mtp && !!load?.status?.hasMtp,
    finalLoadedStatus: result?.status ?? null,
    runs: result?.runs ?? [],
    summary: result?.summary ?? null,
  };
}

function createReport(entries, config, startedAt, finishedAt = null) {
  const serialized = entries.map(serializeEntry);
  const endTime = finishedAt ?? new Date();
  const report = {
    name: "Mach WebGPU model benchmark",
    version: 1,
    startedAt: startedAt.toISOString(),
    finishedAt: finishedAt?.toISOString() ?? null,
    elapsedSeconds: Math.max(0, endTime.getTime() - startedAt.getTime()) / 1_000,
    engineUrl: state.connectedUrl,
    environment: {
      userAgent: navigator.userAgent,
      language: navigator.language,
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      initialStatus: state.initialStatus,
    },
    config,
    models: serialized,
  };
  return report;
}

function renderReport(report) {
  state.report = report;
  window.__modelBenchmark = report;
  window.__fullModelBench = report;
  rawOutput.textContent = JSON.stringify(report, null, 2);
  copyButton.disabled = false;
  downloadButton.disabled = false;

  const completed = report.models.filter((model) => model.status === "complete");
  const fastest = completed.reduce((best, model) =>
    (model.summary?.tps ?? -Infinity) > (best?.summary?.tps ?? -Infinity) ? model : best
  , null);
  const device = report.models.find((model) => model.device)?.device;
  const adapterName = device
    ? [device.vendor, device.architecture].filter(Boolean).join(" ")
    : "—";

  metricGrid.innerHTML = `
    <article class="metric-card"><span>Completed</span><strong>${completed.length} / ${report.config.models.length}</strong><small>models</small></article>
    <article class="metric-card"><span>Fastest decode</span><strong>${fastest ? format(fastest.summary.tps, 2) : "—"}</strong><small>${fastest ? escapeHtml(fastest.label) : "tokens / second"}</small></article>
    <article class="metric-card"><span>Elapsed</span><strong>${formatDuration(report.elapsedSeconds)}</strong><small>wall clock</small></article>
    <article class="metric-card"><span>Adapter</span><strong title="${escapeHtml(adapterName)}">${escapeHtml(adapterName)}</strong><small>WebGPU device</small></article>`;

  if (!report.models.length) {
    resultsBody.innerHTML = `<tr class="empty-row"><td colspan="8">Results will appear here as each model finishes.</td></tr>`;
  } else {
    resultsBody.innerHTML = report.models.map((model) => {
      const summary = model.summary;
      const title = model.error ? ` title="${escapeHtml(model.error)}"` : "";
      return `<tr${title}>
        <td class="table-model"><strong>${escapeHtml(model.label)}</strong><small>${summary?.runs ?? 0} measured run${summary?.runs === 1 ? "" : "s"}${model.mtpActive ? " · MTP" : ""}</small></td>
        <td><span class="status-chip ${escapeHtml(model.status)}">${escapeHtml(model.status)}</span></td>
        <td class="table-value"><b>${format(model.loadSeconds, 2)}</b><small>s</small></td>
        <td class="table-value"><b>${format(summary?.promptTokens, 0)}</b><small>tok</small></td>
        <td class="table-value"><b>${format(summary?.promptTps, 1)}</b><small>tok/s</small></td>
        <td class="table-value"><b>${format(summary?.tps, 2)}</b><small>tok/s</small></td>
        <td class="table-value"><b>${format(summary?.millisecondsPerToken, 2)}</b><small>ms/tok</small></td>
        <td class="table-value"><b>${formatBytes(model.device?.vramBytes)}</b></td>
      </tr>`;
    }).join("");
  }

  const chartModels = completed.filter((model) => Number.isFinite(model.summary?.tps));
  const maxTps = Math.max(0, ...chartModels.map((model) => model.summary.tps));
  throughputChart.innerHTML = chartModels.length
    ? chartModels.map((model) => `<div class="bar-row">
        <div class="bar-label"><span>${escapeHtml(model.label)}</span><b>${format(model.summary.tps, 2)}</b></div>
        <div class="bar-track"><span style="width:${maxTps ? (model.summary.tps / maxTps) * 100 : 0}%"></span></div>
      </div>`).join("")
    : `<div class="chart-empty">No completed measurements yet.</div>`;
}

function setRunning(running) {
  state.running = running;
  document.body.dataset.running = String(running);
  runButton.disabled = running;
  stopButton.disabled = !running;
}

async function runBenchmark(event) {
  event.preventDefault();
  if (state.running) return;

  let config;
  try {
    config = collectConfig();
    await connectEngine(false);
  } catch (error) {
    showToast(errorMessage(error), "error");
    return;
  }

  setRunning(true);
  resetModelStates();
  const controller = new AbortController();
  state.abortController = controller;
  state.startedAt = new Date();
  state.logs = [];
  activityLog.replaceChildren();
  logCount.textContent = "0 events";
  appendLog("queue", `Starting ${config.models.length} model${config.models.length === 1 ? "" : "s"}.`);
  runStatusTitle.textContent = "Preparing clean runtime";
  runStatusDetail.textContent = "Checking for a previously loaded model.";
  overallProgressLabel.textContent = `0 / ${config.models.length}`;
  overallProgressBar.style.width = "0%";

  try {
    const status = await state.client.status();
    if (status.loaded) {
      appendLog("unload", `Releasing previously loaded ${status.activeModel}.`);
      await state.client.unload();
    }

    const prompt = buildBenchmarkPrompt(config.promptSeed, config.promptRepeats);
    const entries = await runSequentialModels({
      models: config.models,
      signal: controller.signal,
      onPhase(model, phase) {
        state.activeModelId = model.id;
        state.activePhase = phase;
        if (phase === "loading") {
          setModelState(model.id, phase, "Loading", 3, "Opening cached weights");
          runStatusTitle.textContent = `Loading ${model.label}`;
          runStatusDetail.textContent = "Preparing tokenizer, weights, and WebGPU pipelines.";
        } else if (phase === "benchmarking") {
          setModelState(model.id, phase, "Benchmarking", 34, "Warming up");
          runStatusTitle.textContent = `Measuring ${model.label}`;
          runStatusDetail.textContent = "Timing prefill and visible-token decode throughput.";
        } else {
          setModelState(model.id, phase, "Unloading", 96, "Releasing WebGPU runtime");
          runStatusTitle.textContent = `Unloading ${model.label}`;
          runStatusDetail.textContent = "Releasing model buffers before the next queue item.";
        }
      },
      async load(model) {
        const start = performance.now();
        const status = await state.client.load({
          model: model.id,
          maxContext: model.maxContext,
          batchSize: model.batchSize,
          mtp: model.mtp,
        });
        const seconds = (performance.now() - start) / 1_000;
        appendLog("load", `${model.label} ready in ${format(seconds, 2)} s.`);
        return { seconds, status };
      },
      async benchmark(model) {
        if (config.warmupTokens > 0) {
          setModelState(model.id, "benchmarking", "Warming up", 35, `${config.warmupTokens} generated tokens`);
          await consumeCompletion(
            model.id,
            "Warm up the inference pipeline by repeating the word warmup separated by spaces.",
            config.warmupTokens,
            controller.signal
          );
        }
        throwIfAborted(controller.signal);

        const runs = [];
        for (let index = 0; index < config.runs; index++) {
          const startProgress = 40 + (index / config.runs) * 50;
          setModelState(
            model.id,
            "benchmarking",
            "Benchmarking",
            startProgress,
            `Measured run ${index + 1} of ${config.runs}`
          );
          const metrics = await benchmarkOnce(model.id, prompt, config.maxTokens, controller.signal);
          runs.push(metrics);
          setModelState(
            model.id,
            "benchmarking",
            "Benchmarking",
            40 + ((index + 1) / config.runs) * 50,
            `${format(metrics.tps, 2)} tokens/s`
          );
          appendLog("result", `${model.label} run ${index + 1}: ${format(metrics.tps, 2)} tokens/s.`);
        }
        return {
          runs,
          summary: summarizeRuns(runs),
          status: await state.client.status(),
        };
      },
      async unload(model) {
        const start = performance.now();
        const status = await state.client.unload();
        const seconds = (performance.now() - start) / 1_000;
        appendLog("unload", `${model.label} released in ${format(seconds, 2)} s.`);
        return { seconds, status };
      },
      onModelComplete(entry, processed) {
        const detail = entry.status === "complete"
          ? `${format(entry.result?.summary?.tps, 2)} tokens/s`
          : errorMessage(entry.error ?? entry.unloadError ?? "Not completed");
        setModelState(entry.model.id, entry.status, entry.status, 100, detail);
        state.activeModelId = null;
        state.activePhase = null;
        overallProgressLabel.textContent = `${processed.length} / ${config.models.length}`;
        overallProgressBar.style.width = `${(processed.length / config.models.length) * 100}%`;
        renderReport(createReport(processed, config, state.startedAt));
      },
    });

    const finishedAt = new Date();
    const report = createReport(entries, config, state.startedAt, finishedAt);
    renderReport(report);
    const errors = entries.filter((entry) => entry.status === "error").length;
    const cancelled = entries.some((entry) => entry.status === "cancelled") || controller.signal.aborted;

    if (cancelled) {
      runStatusTitle.textContent = "Benchmark stopped";
      runStatusDetail.textContent = "The active model was unloaded; completed measurements are preserved.";
      appendLog("queue", "Benchmark stopped by user.");
    } else if (errors) {
      runStatusTitle.textContent = "Queue finished with errors";
      runStatusDetail.textContent = `${entries.length - errors} completed, ${errors} failed. Open the log for details.`;
      appendLog("queue", `Finished with ${errors} error${errors === 1 ? "" : "s"}.`);
    } else {
      runStatusTitle.textContent = "Benchmark complete";
      runStatusDetail.textContent = `All ${entries.length} models measured and unloaded cleanly.`;
      appendLog("queue", `Completed in ${formatDuration(report.elapsedSeconds)}.`);
    }
  } catch (error) {
    if (!isAbortError(error)) {
      appendLog("error", errorMessage(error));
      showToast(errorMessage(error), "error");
    }
    runStatusTitle.textContent = isAbortError(error) ? "Benchmark stopped" : "Benchmark failed";
    runStatusDetail.textContent = errorMessage(error);
  } finally {
    state.activeModelId = null;
    state.activePhase = null;
    state.abortController = null;
    setRunning(false);
  }
}

function restoreDefaults() {
  $("#engine-url").value = DEFAULT_ENGINE_URL;
  $("#prompt-seed").value = BENCHMARK_DEFAULTS.promptSeed;
  $("#prompt-repeats").value = BENCHMARK_DEFAULTS.promptRepeats;
  $("#output-tokens").value = BENCHMARK_DEFAULTS.maxTokens;
  $("#warmup-tokens").value = BENCHMARK_DEFAULTS.warmupTokens;
  $("#runs-per-model").value = BENCHMARK_DEFAULTS.runs;
  const remote = new Map(state.models.map((model) => [model.id, model]));
  renderModels(MODEL_DEFAULTS.map((defaults) => ({
    ...defaults,
    cached: remote.get(defaults.id)?.cached ?? null,
    selected: true,
  })), false);
  resetModelStates();
  showToast("Benchmark defaults restored.");
}

async function copyReport() {
  if (!state.report) return;
  try {
    await navigator.clipboard.writeText(JSON.stringify(state.report, null, 2));
    showToast("Benchmark JSON copied.");
  } catch {
    showToast("Clipboard access was denied by the browser.", "error");
  }
}

function downloadReport() {
  if (!state.report) return;
  const blob = new Blob([JSON.stringify(state.report, null, 2)], { type: "application/json" });
  const href = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = href;
  link.download = `mach-model-benchmark-${new Date().toISOString().replaceAll(":", "-")}.json`;
  link.click();
  URL.revokeObjectURL(href);
}

function applyQueryConfig() {
  const query = new URLSearchParams(location.search);
  $("#engine-url").value = query.get("engine") || DEFAULT_ENGINE_URL;
  if (query.has("repeats")) $("#prompt-repeats").value = query.get("repeats");
  if (query.has("max_tokens")) $("#output-tokens").value = query.get("max_tokens");
  if (query.has("warmup")) $("#warmup-tokens").value = query.get("warmup");
  if (query.has("runs")) $("#runs-per-model").value = query.get("runs");
}

form.addEventListener("submit", runBenchmark);
stopButton.addEventListener("click", () => {
  if (!state.abortController) return;
  stopButton.disabled = true;
  runStatusTitle.textContent = "Stop requested";
  runStatusDetail.textContent = "Finishing the current operation, then releasing the model.";
  appendLog("queue", "Stop requested by user.");
  state.abortController.abort();
});
connectButton.addEventListener("click", () => {
  connectEngine(true).catch((error) => showToast(errorMessage(error), "error"));
});
restoreButton.addEventListener("click", restoreDefaults);
copyButton.addEventListener("click", copyReport);
downloadButton.addEventListener("click", downloadReport);
modelList.addEventListener("change", (event) => {
  const card = event.target.closest(".model-card");
  if (!card) return;
  if (event.target.matches("[data-role='selected']")) {
    card.dataset.selected = String(event.target.checked);
    setModelState(
      card.dataset.modelId,
      "idle",
      event.target.checked ? "Queued" : "Skipped",
      0,
      event.target.checked ? "Waiting" : "Not selected"
    );
    updateSelectedCount();
  }
});

renderModels(state.models, false);
applyQueryConfig();
resetModelStates();
connectEngine(false).catch((error) => {
  showToast(`${errorMessage(error)} Start the engine and press Reconnect.`, "error");
});
