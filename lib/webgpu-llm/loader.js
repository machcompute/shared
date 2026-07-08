// Weight loader: streams bf16 safetensors from HuggingFace via HTTP Range
// requests (text model only — vision tower and MTP head are skipped),
// quantizes to Q4 in a worker pool, caches the quantized result in OPFS.
import { CFG, isFullAttn } from './config.js';

const HF = `https://huggingface.co/${CFG.repo}/resolve/main/`;
const SHARDS = [
  'model.safetensors-00001-of-00002.safetensors',
  'model.safetensors-00002-of-00002.safetensors',
];
const CACHE_VERSION = 2;
const PFX = 'model.language_model.';

// ---------------------------------------------------------------------------
const bf16ToF32 = (u16) => {
  const u32 = new Uint32Array(u16.length);
  for (let i = 0; i < u16.length; i++) u32[i] = u16[i] << 16;
  return new Float32Array(u32.buffer);
};

class WorkerPool {
  constructor(n = Math.min(6, navigator.hardwareConcurrency || 4)) {
    this.pending = new Map();
    this.nextId = 0;
    this.rr = 0;
    this.workers = Array.from({ length: n }, () => this.#spawn());
  }
  // A worker that fails to load or throws must still fail any job posted to
  // it, or a bad worker silently deadlocks Promise.all(jobs) in runQ4 forever
  // — that read as "download gets stuck" / "finishes but never quantizes",
  // since download progress is driven by network bytes, not quantize jobs.
  #spawn() {
    const w = new Worker('/webgpu-llm/quant-worker.js');
    w.onmessage = (ev) => {
      const { id, qdata, scales } = ev.data;
      const p = this.pending.get(id);
      if (!p) return;
      this.pending.delete(id);
      p.resolve({ qdata: new Uint32Array(qdata), scales: new Uint32Array(scales) });
    };
    w.onerror = (ev) => {
      const err = new Error(`Quantization worker failed: ${ev.message || 'unknown error'}`);
      for (const [id, p] of this.pending) {
        p.reject(err);
        this.pending.delete(id);
      }
    };
    return w;
  }
  quantize(u16, rows, K) {
    const id = this.nextId++;
    const w = this.workers[this.rr++ % this.workers.length];
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      w.postMessage({ id, u16: u16.buffer, rows, K }, [u16.buffer]);
    });
  }
  destroy() { for (const w of this.workers) w.terminate(); }
}

// A stalled connection (dropped/idle socket) otherwise hangs fetch()/reader.read()
// forever with no error — this bounds both the initial connect and each read.
const STALL_TIMEOUT_MS = 20000;

async function fetchWithTimeout(url, opts, ms = STALL_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ms);
  try {
    return await fetch(url, { ...opts, signal: controller.signal });
  } catch (e) {
    if (e.name === 'AbortError') throw new Error('Connecting to Hugging Face timed out. Check your connection and try again.');
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

async function readChunk(reader, ms = STALL_TIMEOUT_MS) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error('Network read stalled — no data received in 20s. Check your connection and try again.')), ms);
  });
  try {
    return await Promise.race([reader.read(), timeout]);
  } finally {
    clearTimeout(timer);
  }
}

// ---------------------------------------------------------------------------
async function fetchShardHeaders() {
  const infos = {};
  for (let s = 0; s < SHARDS.length; s++) {
    const url = HF + SHARDS[s];
    const r0 = await fetchWithTimeout(url, { headers: { Range: 'bytes=0-7' } });
    if (!r0.ok) throw new Error(`HF fetch failed (${r0.status}) for ${SHARDS[s]}`);
    const hlen = Number(new DataView(await r0.arrayBuffer()).getBigUint64(0, true));
    const r1 = await fetchWithTimeout(url, { headers: { Range: `bytes=8-${7 + hlen}` } });
    if (!r1.ok) throw new Error(`HF fetch failed (${r1.status}) for ${SHARDS[s]}`);
    const header = JSON.parse(await r1.text());
    for (const [name, t] of Object.entries(header)) {
      if (name === '__metadata__') continue;
      infos[name] = { shard: s, dtype: t.dtype, shape: t.shape, start: 8 + hlen + t.data_offsets[0], end: 8 + hlen + t.data_offsets[1] };
    }
  }
  return infos;
}

// Build the list of GPU-tensor specs. Q4 specs may concatenate several
// checkpoint tensors row-wise (order matters — kernels assume it).
function buildSpecs() {
  const q4 = [], f32 = [];
  for (let i = 0; i < CFG.layers; i++) {
    const L = (n) => `${PFX}layers.${i}.${n}`;
    f32.push({ name: `L${i}.ln1`, parts: [L('input_layernorm.weight')], transform: 'add1' });
    f32.push({ name: `L${i}.ln2`, parts: [L('post_attention_layernorm.weight')], transform: 'add1' });
    if (isFullAttn(i)) {
      q4.push({ name: `L${i}.in`, parts: [L('self_attn.q_proj.weight'), L('self_attn.k_proj.weight'), L('self_attn.v_proj.weight')], N: CFG.inF, K: 2560 });
      q4.push({ name: `L${i}.o`, parts: [L('self_attn.o_proj.weight')], N: 2560, K: 4096 });
      f32.push({ name: `L${i}.qnw`, parts: [L('self_attn.q_norm.weight')], transform: 'add1' });
      f32.push({ name: `L${i}.knw`, parts: [L('self_attn.k_norm.weight')], transform: 'add1' });
    } else {
      q4.push({
        name: `L${i}.in`,
        parts: [L('linear_attn.in_proj_qkv.weight'), L('linear_attn.in_proj_z.weight'), L('linear_attn.in_proj_b.weight'), L('linear_attn.in_proj_a.weight')],
        N: CFG.inL, K: 2560,
      });
      q4.push({ name: `L${i}.out`, parts: [L('linear_attn.out_proj.weight')], N: 2560, K: 4096 });
      f32.push({ name: `L${i}.convw`, parts: [L('linear_attn.conv1d.weight')], transform: 'raw' });
      f32.push({ name: `L${i}.adt`, parts: [L('linear_attn.A_log'), L('linear_attn.dt_bias')], transform: 'adt' });
      f32.push({ name: `L${i}.gnw`, parts: [L('linear_attn.norm.weight')], transform: 'raw' });
    }
    q4.push({ name: `L${i}.gateup`, parts: [L('mlp.gate_proj.weight'), L('mlp.up_proj.weight')], N: 18432, K: 2560 });
    q4.push({ name: `L${i}.down`, parts: [L('mlp.down_proj.weight')], N: 2560, K: 9216 });
  }
  f32.push({ name: 'norm', parts: [`${PFX}norm.weight`], transform: 'add1' });
  q4.push({ name: 'emb', parts: [`${PFX}embed_tokens.weight`], N: CFG.vocab, K: 2560 });
  // MTP head (self-speculative decoding): fc + one full-attention decoder layer
  const M = (n) => `mtp.${n}`;
  q4.push({ name: 'mtp.in', parts: [M('layers.0.self_attn.q_proj.weight'), M('layers.0.self_attn.k_proj.weight'), M('layers.0.self_attn.v_proj.weight')], N: CFG.inF, K: 2560 });
  q4.push({ name: 'mtp.o', parts: [M('layers.0.self_attn.o_proj.weight')], N: 2560, K: 4096 });
  q4.push({ name: 'mtp.gateup', parts: [M('layers.0.mlp.gate_proj.weight'), M('layers.0.mlp.up_proj.weight')], N: 18432, K: 2560 });
  q4.push({ name: 'mtp.down', parts: [M('layers.0.mlp.down_proj.weight')], N: 2560, K: 9216 });
  q4.push({ name: 'mtp.fc', parts: [M('fc.weight')], N: 2560, K: 5120 });
  f32.push({ name: 'mtp.ln1', parts: [M('layers.0.input_layernorm.weight')], transform: 'add1' });
  f32.push({ name: 'mtp.ln2', parts: [M('layers.0.post_attention_layernorm.weight')], transform: 'add1' });
  f32.push({ name: 'mtp.qnw', parts: [M('layers.0.self_attn.q_norm.weight')], transform: 'add1' });
  f32.push({ name: 'mtp.knw', parts: [M('layers.0.self_attn.k_norm.weight')], transform: 'add1' });
  f32.push({ name: 'mtp.norm', parts: [M('norm.weight')], transform: 'add1' });
  f32.push({ name: 'mtp.preE', parts: [M('pre_fc_norm_embedding.weight')], transform: 'add1' });
  f32.push({ name: 'mtp.preH', parts: [M('pre_fc_norm_hidden.weight')], transform: 'add1' });
  return { q4, f32 };
}

const q4Bytes = (s) => ({ q: s.N * s.K / 2, s: s.N * s.K / 8 });

// ---------------------------------------------------------------------------
export class Loader {
  constructor(gpu, status) {
    this.gpu = gpu;
    this.status = status; // (msg, phase, frac) callback
    this.specs = buildSpecs();
  }

  async opfsDir() {
    // best-effort: without persistence Chrome may evict the 3GB cache under disk pressure
    navigator.storage.persist?.().catch(() => {});
    const root = await navigator.storage.getDirectory();
    return root.getDirectoryHandle('qwen35-4b', { create: true });
  }

  async cacheValid() {
    try {
      const dir = await this.opfsDir();
      const mf = await (await (await dir.getFileHandle('manifest.json')).getFile()).text();
      const m = JSON.parse(mf);
      return m.version === CACHE_VERSION && m.complete === true ? m : null;
    } catch { return null; }
  }

  async clearCache() {
    const root = await navigator.storage.getDirectory();
    try { await root.removeEntry('qwen35-4b', { recursive: true }); } catch {}
  }

  // Create GPU buffers for a q4 spec (embedding may be sharded to fit binding limits).
  #makeQ4Buffers(spec, weights) {
    const g = this.gpu;
    if (spec.name !== 'emb') {
      const b = q4Bytes(spec);
      weights[spec.name] = { q: g.storage(b.q, spec.name + '.q'), s: g.storage(b.s, spec.name + '.s'), N: spec.N, K: spec.K };
      return;
    }
    const limit = Math.min(g.limits.maxStorageBufferBindingSize, g.limits.maxBufferSize);
    const rowQ = spec.K / 2, rowS = spec.K / 8;
    let shardRows = Math.floor((limit * 0.95) / rowQ);
    shardRows = Math.min(spec.N, Math.floor(shardRows / 32) * 32);
    const shards = [];
    for (let start = 0; start < spec.N; start += shardRows) {
      const rows = Math.min(shardRows, spec.N - start);
      shards.push({
        start, rows,
        q: g.storage(rows * rowQ, `emb.q.${start}`),
        s: g.storage(rows * rowS, `emb.s.${start}`),
      });
    }
    weights.embShards = shards;
  }

  #uploadQ4(spec, weights, qdata, scales) {
    const g = this.gpu;
    if (spec.name !== 'emb') {
      g.upload(weights[spec.name].q, qdata);
      g.upload(weights[spec.name].s, scales);
      return;
    }
    const rowQw = spec.K / 8, rowSw = spec.K / 32; // u32 words per row
    for (const sh of weights.embShards) {
      g.upload(sh.q, qdata.subarray(sh.start * rowQw, (sh.start + sh.rows) * rowQw));
      g.upload(sh.s, scales.subarray(sh.start * rowSw, (sh.start + sh.rows) * rowSw));
    }
  }

  #transformF32(spec, partArrays) {
    // partArrays: Float32Array per part (already dtype-converted)
    if (spec.transform === 'add1') {
      const a = partArrays[0].slice();
      for (let i = 0; i < a.length; i++) a[i] += 1;
      return a;
    }
    if (spec.transform === 'adt') {
      const [alog, dt] = partArrays;
      const out = new Float32Array(64);
      for (let i = 0; i < 32; i++) out[i] = -Math.exp(alog[i]);
      out.set(dt, 32);
      return out;
    }
    const total = partArrays.reduce((n, a) => n + a.length, 0);
    const out = new Float32Array(total);
    let o = 0;
    for (const a of partArrays) { out.set(a, o); o += a.length; }
    return out;
  }

  async #fetchTensor(info, onBytes) {
    const url = HF + SHARDS[info.shard];
    const res = await fetchWithTimeout(url, { headers: { Range: `bytes=${info.start}-${info.end - 1}` } });
    if (!res.ok && res.status !== 206) throw new Error(`Range fetch failed: ${res.status}`);
    const reader = res.body.getReader();
    const chunks = [];
    try {
      for (;;) {
        const { done, value } = await readChunk(reader);
        if (done) break;
        chunks.push(value);
        onBytes?.(value.byteLength);
      }
    } catch (e) {
      await reader.cancel().catch(() => {});
      throw e;
    }
    const total = chunks.reduce((n, c) => n + c.byteLength, 0);
    const buf = new Uint8Array(total);
    let o = 0;
    for (const c of chunks) { buf.set(c, o); o += c.byteLength; }
    return buf;
  }

  // Streaming variant: yields row-aligned bf16 pieces as they arrive.
  async #fetchTensorRows(info, rowBytes, onRows, onBytes) {
    const url = HF + SHARDS[info.shard];
    const res = await fetchWithTimeout(url, { headers: { Range: `bytes=${info.start}-${info.end - 1}` } });
    if (!res.ok && res.status !== 206) throw new Error(`Range fetch failed: ${res.status}`);
    const reader = res.body.getReader();
    let pend = new Uint8Array(0);
    let rowOff = 0;
    try {
      for (;;) {
        const { done, value } = await readChunk(reader);
        if (done) break;
        onBytes?.(value.byteLength);
        let buf;
        if (pend.length) {
          buf = new Uint8Array(pend.length + value.byteLength);
          buf.set(pend, 0); buf.set(value, pend.length);
        } else buf = value;
        const nrows = Math.floor(buf.byteLength / rowBytes);
        if (nrows > 0) {
          const take = nrows * rowBytes;
          // copy to an aligned, transferable buffer
          const piece = new Uint8Array(buf.slice(0, take));
          pend = buf.slice(take);
          await onRows(new Uint16Array(piece.buffer), rowOff, nrows);
          rowOff += nrows;
        } else pend = buf;
      }
    } catch (e) {
      await reader.cancel().catch(() => {});
      throw e;
    }
    if (pend.length) throw new Error('tensor stream ended mid-row');
  }

  /** Main entry: returns weights map (GPU buffers uploaded). Cached specs come
   * from OPFS; any specs missing from the cache (e.g. newly added tensors) are
   * downloaded and appended to it. */
  async load() {
    const dev = this.gpu.device;
    dev.pushErrorScope('out-of-memory');
    dev.pushErrorScope('validation');
    const weights = {};
    let vErr, mErr;
    try {
      for (const spec of this.specs.q4) this.#makeQ4Buffers(spec, weights);

      const cached = await this.cacheValid();
      const have = new Set((cached?.entries ?? []).map((e) => e.name));
      const missQ4 = this.specs.q4.filter((s) => !have.has(s.name));
      const missF32 = this.specs.f32.filter((s) => !have.has(s.name));
      if (cached) await this.#loadFromCache(cached, weights);
      if (!cached || missQ4.length || missF32.length) {
        await this.#downloadAndQuantize(weights, cached, missQ4, missF32);
      }
    } finally {
      // Always pop the scopes — leaving them pushed after a throw would
      // swallow every later error on this device.
      vErr = await dev.popErrorScope().catch(() => null);
      mErr = await dev.popErrorScope().catch(() => null);
    }
    if (mErr) throw new Error('Out of GPU memory while loading weights (~3 GB needed). Close other GPU-heavy tabs/apps and reload the page.');
    if (vErr) throw new Error('GPU error while loading weights: ' + vErr.message);
    return weights;
  }

  // Flush queued writeBuffer uploads periodically: Dawn otherwise accumulates
  // staging memory for every pending write and can double peak VRAM.
  async #flush(force = false) {
    this._pend = (this._pend ?? 0);
    if (force || this._pend > 192 * 2 ** 20) {
      await this.gpu.device.queue.onSubmittedWorkDone();
      this._pend = 0;
    }
  }
  #track(bytes) { this._pend = (this._pend ?? 0) + bytes; }

  async #loadFromCache(manifest, weights) {
    const dir = await this.opfsDir();
    const file = await (await dir.getFileHandle('weights.bin')).getFile();
    let done = 0;
    const total = file.size;
    for (const e of manifest.entries) {
      if (e.kind === 'q4') {
        const spec = this.specs.q4.find((s) => s.name === e.name);
        if (!spec) continue;
        const qbuf = new Uint32Array(await file.slice(e.off, e.off + e.qbytes).arrayBuffer());
        const sbuf = new Uint32Array(await file.slice(e.off + e.qbytes, e.off + e.qbytes + e.sbytes).arrayBuffer());
        this.#uploadQ4(spec, weights, qbuf, sbuf);
        done += e.qbytes + e.sbytes;
        this.#track(e.qbytes + e.sbytes);
      } else {
        if (!this.specs.f32.some((s) => s.name === e.name)) continue;
        const a = new Float32Array(await file.slice(e.off, e.off + e.bytes).arrayBuffer());
        weights[e.name] = this.gpu.storage(a.byteLength, e.name);
        this.gpu.upload(weights[e.name], a);
        done += e.bytes;
        this.#track(e.bytes);
      }
      await this.#flush();
      this.status(`Loading cached weights… ${(done / 1e9).toFixed(2)} / ${(total / 1e9).toFixed(2)} GB`, 'cache', done / total);
    }
    await this.#flush(true);
  }

  // Download and quantize the given specs; existing !== null means append to
  // the current cache file (used to fetch newly added tensors incrementally).
  async #downloadAndQuantize(weights, existing, q4Specs = this.specs.q4, f32Specs = this.specs.f32) {
    const infos = await fetchShardHeaders();
    const missing = [...q4Specs, ...f32Specs].flatMap((s) => s.parts).filter((p) => !infos[p]);
    if (missing.length) throw new Error('Missing tensors in checkpoint: ' + missing.slice(0, 3).join(', '));

    const totalDl = [...q4Specs, ...f32Specs]
      .flatMap((s) => s.parts).reduce((n, p) => n + (infos[p].end - infos[p].start), 0);
    let dlBytes = 0;
    const t0 = performance.now();
    const onBytes = (n) => {
      dlBytes += n;
      const mbps = dlBytes / 1e6 / ((performance.now() - t0) / 1000);
      this.status(`Downloading + quantizing… ${(dlBytes / 1e9).toFixed(2)} / ${(totalDl / 1e9).toFixed(2)} GB  (${mbps.toFixed(0)} MB/s)`, 'download', dlBytes / totalDl);
    };

    // The existing manifest stays in place until the new one replaces it at
    // commit time: createWritable() writes to a swap file that only lands on
    // close(), so a failure mid-download leaves the old cache fully valid.
    const pool = new WorkerPool();
    const dir = await this.opfsDir();
    const fh = await dir.getFileHandle('weights.bin', { create: true });
    const writable = await fh.createWritable({ keepExistingData: !!existing });
    // serialize positional writes — WritableStream does not allow concurrent write()
    let wlock = Promise.resolve();
    const wwrite = (args) => { const p = wlock.then(() => writable.write(args)); wlock = p.catch(() => {}); return p; };
    const manifest = { version: CACHE_VERSION, complete: false, entries: [...(existing?.entries ?? [])] };

    let off = existing
      ? Math.max(0, ...existing.entries.map((e) => e.off + (e.kind === 'q4' ? e.qbytes + e.sbytes : e.bytes)))
      : 0;
    const q4Offsets = new Map();
    for (const s of q4Specs) {
      const b = q4Bytes(s);
      q4Offsets.set(s.name, off);
      manifest.entries.push({ name: s.name, kind: 'q4', off, qbytes: b.q, sbytes: b.s });
      off += b.q + b.s;
    }

    // ---- q4 specs (3 concurrent) ----
    try {
      const runQ4 = async (spec) => {
        const rowQw = spec.K / 8, rowSw = spec.K / 32;
        const qdata = new Uint32Array(spec.N * rowQw);
        const scales = new Uint32Array(spec.N * rowSw);
        const jobs = [];
        let row = 0;
        for (const part of spec.parts) {
          const info = infos[part];
          if (info.dtype !== 'BF16') throw new Error(`${part}: expected BF16, got ${info.dtype}`);
          const partRow0 = row;
          await this.#fetchTensorRows(info, spec.K * 2, async (u16, rowOff, nrows) => {
            const r0 = partRow0 + rowOff;
            jobs.push(pool.quantize(u16, nrows, spec.K).then(({ qdata: q, scales: s }) => {
              qdata.set(q, r0 * rowQw);
              scales.set(s, r0 * rowSw);
            }));
          }, onBytes);
          row += info.shape[0];
        }
        await Promise.all(jobs);
        this.#uploadQ4(spec, weights, qdata, scales);
        this.#track(qdata.byteLength + scales.byteLength);
        await this.#flush();
        const o = q4Offsets.get(spec.name);
        await wwrite({ type: 'write', position: o, data: qdata });
        await wwrite({ type: 'write', position: o + qdata.byteLength, data: scales });
      };

      const queue = [...q4Specs];
      const lanes = Array.from({ length: 3 }, async () => {
        while (queue.length) await runQ4(queue.shift());
      });
      await Promise.all(lanes);

      // ---- f32 specs (small; sequential) ----
      for (const spec of f32Specs) {
        const parts = [];
        for (const p of spec.parts) {
          const info = infos[p];
          const raw = await this.#fetchTensor(info, onBytes);
          parts.push(info.dtype === 'F32'
            ? new Float32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength))
            : bf16ToF32(new Uint16Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength))));
        }
        const a = this.#transformF32(spec, parts);
        weights[spec.name] = this.gpu.storage(a.byteLength, spec.name);
        this.gpu.upload(weights[spec.name], a);
        manifest.entries.push({ name: spec.name, kind: 'f32', off, bytes: a.byteLength });
        await wwrite({ type: 'write', position: off, data: a });
        off += a.byteLength;
      }

      manifest.complete = true;
      await writable.close(); // commit the swap file before the manifest points at it
      const mfh = await dir.getFileHandle('manifest.json', { create: true });
      const mw = await mfh.createWritable();
      await mw.write(JSON.stringify(manifest));
      await mw.close();
    } catch (e) {
      await writable.abort().catch(() => {}); // discard the swap; the old cache stays intact
      throw e;
    } finally {
      pool.destroy();
    }
    await this.gpu.device.queue.onSubmittedWorkDone();
    this.status('Weights ready.', 'done', 1);
  }
}
