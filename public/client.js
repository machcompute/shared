const NS = "mach-llm";
const DEFAULT_ENGINE_URL = "https://shared.machcomputing.com";

let counter = 0;
const nextId = () => `req-${++counter}-${Math.random().toString(36).slice(2)}`;

export class MachLLM {
  #iframe;
  #engineOrigin;
  #pending = new Map();
  #progressHandlers = new Set();
  #listener;
  #closed = false;

  constructor(iframe, engineOrigin) {
    this.#iframe = iframe;
    this.#engineOrigin = engineOrigin;
    this.#listener = (event) => {
      if (event.origin !== this.#engineOrigin) return;
      if (!this.#iframe.contentWindow || event.source !== this.#iframe.contentWindow) return;
      const data = event.data;
      if (!data || data.ns !== NS || typeof data.id !== "string") return;
      const pending = this.#pending.get(data.id);
      if (!pending) return;
      if (data.type === "chunk") {
        if (data.data?.event === "progress") {
          for (const handler of this.#progressHandlers) handler(data.data.progress);
        } else {
          pending.onChunk?.(data.data);
        }
      } else if (data.type === "result") {
        this.#pending.delete(data.id);
        pending.resolve(data.data);
      } else if (data.type === "error") {
        this.#pending.delete(data.id);
        const error = new Error(data.data?.message ?? "Engine error");
        error.code = data.data?.code;
        pending.reject(error);
      }
    };
    window.addEventListener("message", this.#listener);
    this.chat = {
      completions: {
        create: (params, options) => this.#createCompletion(params, options),
      },
    };
    this.models = {
      list: () => this.#request("models.list"),
    };
  }

  static async connect(options = {}) {
    const { engineUrl = DEFAULT_ENGINE_URL, onProgress, timeoutMs = 20000 } = options;
    const url = new URL(engineUrl, location.href);
    const iframe = document.createElement("iframe");
    iframe.setAttribute("allow", "webgpu");
    iframe.setAttribute("aria-hidden", "true");
    iframe.style.display = "none";
    iframe.src = url.href;

    const ready = new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        cleanup();
        iframe.remove();
        reject(new Error(`Timed out connecting to the LLM engine at ${url.origin}`));
      }, timeoutMs);
      const onMessage = (event) => {
        if (event.origin !== url.origin || event.source !== iframe.contentWindow) return;
        if (event.data?.ns === NS && event.data.type === "ready") {
          cleanup();
          resolve();
        }
      };
      const cleanup = () => {
        clearTimeout(timer);
        window.removeEventListener("message", onMessage);
      };
      window.addEventListener("message", onMessage);
    });

    document.body.appendChild(iframe);
    await ready;

    const llm = new MachLLM(iframe, url.origin);
    if (onProgress) llm.on("progress", onProgress);
    return llm;
  }

  on(event, handler) {
    if (event === "progress") this.#progressHandlers.add(handler);
    return this;
  }

  off(event, handler) {
    if (event === "progress") this.#progressHandlers.delete(handler);
    return this;
  }

  status() {
    return this.#request("status");
  }

  load(options) {
    return this.#request("load", options);
  }

  unload() {
    return this.#request("unload");
  }

  updateSettings(options) {
    return this.#request("settings.update", options);
  }

  wipeCache(options) {
    return this.#request("cache.wipe", options);
  }

  close() {
    if (this.#closed) return;
    this.#closed = true;
    window.removeEventListener("message", this.#listener);
    this.#iframe.remove();
    const closedError = new Error("MachLLM connection is closed");
    for (const pending of this.#pending.values()) pending.reject(closedError);
    this.#pending.clear();
  }

  #request(method, params, { signal, onChunk, abortable = false } = {}) {
    return new Promise((resolve, reject) => {
      if (this.#closed) return reject(new Error("MachLLM connection is closed"));
      const id = nextId();
      this.#pending.set(id, { resolve, reject, onChunk });
      this.#iframe.contentWindow.postMessage({ ns: NS, id, method, params }, this.#engineOrigin);
      if (signal && abortable) {
        const sendAbort = () => {
          if (this.#closed || !this.#pending.has(id)) return;
          this.#iframe.contentWindow.postMessage(
            { ns: NS, id: nextId(), method: "abort", params: { targetId: id } },
            this.#engineOrigin
          );
        };
        if (signal.aborted) sendAbort();
        else signal.addEventListener("abort", sendAbort, { once: true });
      }
    });
  }

  async #createCompletion(params = {}, options = {}) {
    const controller = new AbortController();
    if (options.signal) {
      if (options.signal.aborted) controller.abort();
      else options.signal.addEventListener("abort", () => controller.abort(), { once: true });
    }
    const { stream: wantStream, ...request } = params;
    const requestOptions = { signal: controller.signal, abortable: true };

    if (!wantStream) {
      return this.#request("chat.completions.create", request, requestOptions);
    }

    const queue = [];
    let notify = null;
    let done = false;
    let error = null;
    const wake = () => {
      notify?.();
      notify = null;
    };

    const stream = {
      controller,
      completion: null,
      async *[Symbol.asyncIterator]() {
        while (true) {
          while (queue.length) yield queue.shift();
          if (done) {
            if (error) throw error;
            return;
          }
          await new Promise((resolve) => {
            notify = resolve;
          });
        }
      },
    };

    this.#request("chat.completions.create", request, {
      ...requestOptions,
      onChunk: (data) => {
        if (data?.event === "chunk") {
          queue.push(data.chunk);
          wake();
        }
      },
    })
      .then((completion) => {
        stream.completion = completion;
        done = true;
        wake();
      })
      .catch((e) => {
        error = e;
        done = true;
        wake();
      });

    return stream;
  }
}
