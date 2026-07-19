import { GGUFNeedMoreDataError, nativeWeightDescriptor, parseGGUF } from './parser.js';
import { SHA256 } from './sha256.js';
import { validateGGUFSource } from './sources.js';

const CACHE_SCHEMA = 1;
const HEADER_START_BYTES = 1 << 20;
const HEADER_MAX_BYTES = 32 << 20;
const UPLOAD_CHUNK_BYTES = 16 << 20;
const STALL_TIMEOUT_MS = 20_000;

async function fetchWithTimeout(url, options, ms = STALL_TIMEOUT_MS, fetchImpl = fetch) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ms);
  try {
    return await fetchImpl(url, { ...options, signal: controller.signal });
  } catch (error) {
    if (error?.name === 'AbortError') throw new Error(`Timed out fetching ${url}`);
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

async function readWithTimeout(reader, ms = STALL_TIMEOUT_MS) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error('GGUF download stalled for 20 seconds')), ms);
  });
  try { return await Promise.race([reader.read(), timeout]); }
  finally { clearTimeout(timer); }
}

function validateRange(response, start, end, source) {
  if (response.status !== 206) throw new Error(`${source.filename}: server did not honor the required byte range`);
  const contentRange = response.headers.get('content-range');
  const expected = `bytes ${start}-${end}/`;
  if (!contentRange?.startsWith(expected)) throw new Error(`${source.filename}: invalid Content-Range response`);
}

export async function fetchGGUFHeader(source, fetchImpl = fetch) {
  let bytes = HEADER_START_BYTES;
  for (;;) {
    const end = Math.min(source.byteLength, bytes) - 1;
    const response = await fetchWithTimeout(source.url, { headers: { Range: `bytes=0-${end}` } }, STALL_TIMEOUT_MS, fetchImpl);
    validateRange(response, 0, end, source);
    const buffer = await response.arrayBuffer();
    if (buffer.byteLength !== end + 1) throw new Error(`${source.filename}: truncated header range`);
    try {
      return parseGGUF(buffer, { fileSize: source.byteLength });
    } catch (error) {
      if (!(error instanceof GGUFNeedMoreDataError)) throw error;
      if (bytes >= HEADER_MAX_BYTES) throw new Error(`${source.filename}: GGUF tables exceed ${HEADER_MAX_BYTES} bytes`);
      bytes = Math.min(HEADER_MAX_BYTES, Math.max(bytes * 2, error.requiredBytes));
    }
  }
}

async function readJson(dir, name) {
  const file = await (await dir.getFileHandle(name)).getFile();
  return JSON.parse(await file.text());
}

async function writeJson(dir, name, value) {
  const handle = await dir.getFileHandle(name, { create: true });
  const writable = await handle.createWritable();
  try {
    await writable.write(JSON.stringify(value));
    await writable.close();
  } catch (error) {
    await writable.abort().catch(() => {});
    throw error;
  }
}

export function cacheManifestMatches(modelSource, manifest) {
  return manifest?.schema === CACHE_SCHEMA
    && manifest.complete === true
    && manifest.namespace === modelSource.cacheNamespace
    && Array.isArray(manifest.files)
    && manifest.files.length === modelSource.files.length
    && modelSource.files.every((source, i) => {
      const file = manifest.files[i];
      return file?.filename === source.filename
        && file.revision === source.revision
        && file.byteLength === source.byteLength
        && file.sha256 === source.sha256
        && Number.isSafeInteger(file.dataOffset);
    });
}

export class GGUFCache {
  constructor(modelSource, status = () => {}, dependencies = {}) {
    this.modelSource = modelSource;
    this.status = status;
    this.headers = new Map();
    this._legacyCleaned = false;
    this._pendingUploadBytes = 0;
    this.fetch = dependencies.fetch ?? fetch;
    this.validateSource = dependencies.validateSource ?? validateGGUFSource;
  }

  async root() {
    navigator.storage.persist?.().catch(() => {});
    return navigator.storage.getDirectory();
  }

  async dir(create = true) {
    return (await this.root()).getDirectoryHandle(this.modelSource.cacheNamespace, { create });
  }

  async cleanupLegacy() {
    if (this._legacyCleaned) return;
    const root = await this.root();
    for (const name of this.modelSource.legacyNamespaces) {
      try { await root.removeEntry(name, { recursive: true }); } catch {}
    }
    this._legacyCleaned = true;
  }

  async cacheValid() {
    await this.cleanupLegacy();
    try {
      const dir = await this.dir(false);
      const manifest = await readJson(dir, 'manifest.json');
      if (!cacheManifestMatches(this.modelSource, manifest)) return null;
      for (const source of this.modelSource.files) {
        const file = await (await dir.getFileHandle(source.filename)).getFile();
        if (file.size !== source.byteLength) return null;
      }
      return manifest;
    } catch { return null; }
  }

  async clear() {
    const root = await this.root();
    for (const name of [this.modelSource.cacheNamespace, ...this.modelSource.legacyNamespaces]) {
      try { await root.removeEntry(name, { recursive: true }); } catch {}
    }
    this.headers.clear();
    this._legacyCleaned = true;
  }

  async #validatedRemoteHeaders() {
    const result = [];
    for (const source of this.modelSource.files) {
      this.status(`Validating ${source.filename} GGUF tables…`, 'download', null);
      const gguf = this.validateSource(source, await fetchGGUFHeader(source, this.fetch), this.modelSource.config);
      this.headers.set(source.filename, gguf);
      result.push(gguf);
    }
    return result;
  }

  async #download(source, dir, index, count) {
    const response = await fetchWithTimeout(source.url, {}, STALL_TIMEOUT_MS, this.fetch);
    if (!response.ok) throw new Error(`${source.filename}: download failed (${response.status})`);
    const declared = Number(response.headers.get('content-length'));
    if (Number.isFinite(declared) && declared !== source.byteLength) {
      throw new Error(`${source.filename}: expected ${source.byteLength} bytes, server declared ${declared}`);
    }
    const reader = response.body?.getReader();
    if (!reader) throw new Error(`${source.filename}: response has no readable body`);
    const handle = await dir.getFileHandle(source.filename, { create: true });
    const writable = await handle.createWritable();
    const digest = new SHA256();
    let received = 0;
    const started = performance.now();
    try {
      for (;;) {
        const { done, value } = await readWithTimeout(reader);
        if (done) break;
        received += value.byteLength;
        if (received > source.byteLength) throw new Error(`${source.filename}: download exceeds expected length`);
        digest.update(value);
        await writable.write(value);
        const elapsed = Math.max((performance.now() - started) / 1000, 0.001);
        const overall = (index + received / source.byteLength) / count;
        this.status(
          `Downloading GGUF ${index + 1}/${count}: ${(received / 1e9).toFixed(2)} / ${(source.byteLength / 1e9).toFixed(2)} GB (${(received / 1e6 / elapsed).toFixed(0)} MB/s)`,
          'download', overall,
        );
      }
      if (received !== source.byteLength) throw new Error(`${source.filename}: received ${received} of ${source.byteLength} bytes`);
      this.status(`Verifying ${source.filename} SHA-256…`, 'download', (index + 0.99) / count);
      const actual = digest.digestHex();
      if (actual !== source.sha256) throw new Error(`${source.filename}: SHA-256 mismatch (expected ${source.sha256}, got ${actual})`);
      await writable.close();
    } catch (error) {
      await reader.cancel().catch(() => {});
      await writable.abort().catch(() => {});
      try { await dir.removeEntry(source.filename); } catch {}
      throw error;
    }
  }

  async ensure() {
    const valid = await this.cacheValid();
    if (valid) return valid;
    const headers = await this.#validatedRemoteHeaders();
    const dir = await this.dir(true);
    try { await dir.removeEntry('manifest.json'); } catch {}
    const files = [];
    try {
      for (let i = 0; i < this.modelSource.files.length; i++) {
        const source = this.modelSource.files[i];
        let existing = null;
        try { existing = await (await dir.getFileHandle(source.filename)).getFile(); } catch {}
        if (!existing || existing.size !== source.byteLength) {
          if (existing) try { await dir.removeEntry(source.filename); } catch {}
          await this.#download(source, dir, i, this.modelSource.files.length);
        } else {
          // A file without the committed manifest is not trusted. Rehashing it
          // would cost the same I/O as a fresh download and cannot prove its
          // origin, so rebuild it deterministically.
          try { await dir.removeEntry(source.filename); } catch {}
          await this.#download(source, dir, i, this.modelSource.files.length);
        }
        files.push({
          filename: source.filename,
          revision: source.revision,
          byteLength: source.byteLength,
          sha256: source.sha256,
          dataOffset: headers[i].dataOffset,
        });
      }
      const manifest = { schema: CACHE_SCHEMA, namespace: this.modelSource.cacheNamespace, complete: true, files };
      await writeJson(dir, 'manifest.json', manifest);
      return manifest;
    } catch (error) {
      for (const source of this.modelSource.files) {
        try { await dir.removeEntry(source.filename); } catch {}
      }
      throw error;
    }
  }

  async header(filename) {
    const known = this.headers.get(filename);
    if (known) return known;
    const source = this.modelSource.files.find((entry) => entry.filename === filename);
    if (!source) throw new Error(`Unknown GGUF cache file ${filename}`);
    const dir = await this.dir(false);
    const file = await (await dir.getFileHandle(filename)).getFile();
    let bytes = HEADER_START_BYTES;
    for (;;) {
      try {
        const gguf = this.validateSource(source, parseGGUF(await file.slice(0, bytes).arrayBuffer(), { fileSize: file.size }), this.modelSource.config);
        this.headers.set(filename, gguf);
        return gguf;
      } catch (error) {
        if (!(error instanceof GGUFNeedMoreDataError) || bytes >= HEADER_MAX_BYTES) throw error;
        bytes = Math.min(HEADER_MAX_BYTES, Math.max(bytes * 2, error.requiredBytes));
      }
    }
  }

  async file(filename) {
    const dir = await this.dir(false);
    return (await dir.getFileHandle(filename)).getFile();
  }

  async uploadTensor(gpu, filename, tensorName, { label = tensorName, shardRows = true } = {}) {
    const gguf = await this.header(filename);
    const tensor = gguf.tensorsByName.get(tensorName);
    if (!tensor) throw new Error(`${filename}: missing tensor ${tensorName}`);
    const file = await this.file(filename);
    const layout = nativeWeightDescriptor(tensor, null);
    const limit = Math.min(gpu.limits.maxStorageBufferBindingSize, gpu.limits.maxBufferSize);
    let rowsPerShard = layout.N;
    if (tensor.byteLength > limit) {
      if (!shardRows || tensor.dimensions.length < 2) throw new Error(`${tensorName} exceeds this GPU's storage buffer limit`);
      rowsPerShard = Math.max(1, Math.floor((limit * 0.9) / layout.rowStride));
      rowsPerShard = Math.max(1, Math.floor(rowsPerShard / 32) * 32);
    }
    const shards = [];
    for (let start = 0; start < layout.N; start += rowsPerShard) {
      const rows = Math.min(rowsPerShard, layout.N - start);
      const byteLength = rows * layout.rowStride;
      const buffer = gpu.storage(byteLength, `${label}.${start}`);
      const fileStart = tensor.offset + start * layout.rowStride;
      for (let offset = 0; offset < byteLength; offset += UPLOAD_CHUNK_BYTES) {
        const size = Math.min(UPLOAD_CHUNK_BYTES, byteLength - offset);
        const data = new Uint8Array(await file.slice(fileStart + offset, fileStart + offset + size).arrayBuffer());
        if (data.byteLength !== size) throw new Error(`${tensorName}: short cached tensor read`);
        gpu.upload(buffer, data, offset);
        this._pendingUploadBytes += size;
        if (this._pendingUploadBytes >= 128 * 2 ** 20) {
          await gpu.device.queue.onSubmittedWorkDone();
          this._pendingUploadBytes = 0;
        }
      }
      shards.push(nativeWeightDescriptor({ ...tensor, dimensions: [layout.K, rows], byteLength }, buffer, { start, rows }));
    }
    if (shards.length === 1) return nativeWeightDescriptor(tensor, shards[0].buffer);
    return Object.freeze({ ...nativeWeightDescriptor(tensor, null), buffer: undefined, shards: Object.freeze(shards) });
  }
}
