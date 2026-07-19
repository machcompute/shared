import { describe, expect, it } from 'vitest';
import { GGUFCache, cacheManifestMatches } from '../lib/webgpu-llm/gguf/cache.js';
import { GGML_TYPE } from '../lib/webgpu-llm/gguf/parser.js';
import { sha256Hex } from '../lib/webgpu-llm/gguf/sha256.js';

const te = new TextEncoder();
const u32 = (n: number) => { const b = new Uint8Array(4); new DataView(b.buffer).setUint32(0, n, true); return b; };
const u64 = (n: number) => { const b = new Uint8Array(8); new DataView(b.buffer).setBigUint64(0, BigInt(n), true); return b; };
const concat = (...parts: Uint8Array[]) => {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0; for (const part of parts) { out.set(part, at); at += part.length; }
  return out;
};
const str = (value: string) => concat(u64(value.length), te.encode(value));

function ggufFixture() {
  const metadata = concat(str('general.architecture'), u32(8), str('fixture'));
  const tensor = concat(str('weight'), u32(2), u64(32), u64(2), u32(GGML_TYPE.Q4_0), u64(0));
  const header = concat(te.encode('GGUF'), u32(3), u64(1), u64(1), metadata, tensor);
  const dataOffset = Math.ceil(header.length / 32) * 32;
  return concat(header, new Uint8Array(dataOffset - header.length), new Uint8Array(36));
}

class MemoryFileHandle {
  constructor(private dir: MemoryDirectory, private name: string) {}
  async getFile() {
    if (!this.dir.files.has(this.name)) throw new Error('NotFound');
    return new File([this.dir.files.get(this.name)!.slice().buffer as ArrayBuffer], this.name);
  }
  async createWritable() {
    let value = new Uint8Array();
    return {
      write: async (chunk: string | Uint8Array) => {
        const next = typeof chunk === 'string' ? te.encode(chunk) : new Uint8Array(chunk);
        value = concat(value, next);
      },
      close: async () => { this.dir.files.set(this.name, value); },
      abort: async () => {},
    };
  }
}

class MemoryDirectory {
  files = new Map<string, Uint8Array>();
  dirs = new Map<string, MemoryDirectory>();
  async getFileHandle(name: string, options: { create?: boolean } = {}) {
    if (!options.create && !this.files.has(name)) throw new Error('NotFound');
    if (options.create && !this.files.has(name)) this.files.set(name, new Uint8Array());
    return new MemoryFileHandle(this, name);
  }
  async getDirectoryHandle(name: string, options: { create?: boolean } = {}) {
    if (!this.dirs.has(name)) {
      if (!options.create) throw new Error('NotFound');
      this.dirs.set(name, new MemoryDirectory());
    }
    return this.dirs.get(name)!;
  }
  async removeEntry(name: string) { this.files.delete(name); this.dirs.delete(name); }
}

function setup(bytes = ggufFixture(), body = bytes) {
  const source = {
    filename: 'fixture.gguf', revision: 'commit', byteLength: bytes.length,
    sha256: sha256Hex(bytes), url: 'https://fixture/fixture.gguf',
  };
  const modelSource = { cacheNamespace: 'fixture-gguf-v1', legacyNamespaces: ['legacy'], config: null, files: [source] };
  const fetch = async (_url: string, options: RequestInit = {}) => {
    const range = (options.headers as Record<string, string> | undefined)?.Range;
    if (range) {
      return new Response(bytes, { status: 206, headers: { 'content-range': `bytes 0-${bytes.length - 1}/${bytes.length}` } });
    }
    return new Response(body, { status: 200, headers: { 'content-length': String(bytes.length) } });
  };
  const root = new MemoryDirectory();
  root.dirs.set('legacy', new MemoryDirectory());
  const cache = new GGUFCache(modelSource, () => {}, { fetch, validateSource: (_source: unknown, gguf: unknown) => gguf });
  Object.defineProperty(cache, 'root', { value: async () => root });
  return { cache, modelSource, root, source };
}

describe('atomic GGUF cache', () => {
  it('commits a complete manifest only after a verified file and removes legacy data', async () => {
    const { cache, modelSource, root } = setup();
    const manifest = await cache.ensure();
    expect(cacheManifestMatches(modelSource, manifest)).toBe(true);
    expect(root.dirs.has('legacy')).toBe(false);
    expect(await cache.cacheValid()).toEqual(manifest);
  });

  it('deletes bytes and leaves no manifest after a SHA mismatch', async () => {
    const bytes = ggufFixture(); const corrupt = bytes.slice(); corrupt[corrupt.length - 1] ^= 1;
    const { cache, root } = setup(bytes, corrupt);
    await expect(cache.ensure()).rejects.toThrow(/SHA-256 mismatch/);
    const dir = root.dirs.get('fixture-gguf-v1')!;
    expect(dir.files.has('fixture.gguf')).toBe(false);
    expect(dir.files.has('manifest.json')).toBe(false);
  });

  it('recovers atomically after an interrupted body stream', async () => {
    const bytes = ggufFixture(); const { cache, root } = setup(bytes);
    let calls = 0;
    cache.fetch = async (_url: string, options: RequestInit = {}) => {
      const range = (options.headers as Record<string, string> | undefined)?.Range;
      if (range) return new Response(bytes, { status: 206, headers: { 'content-range': `bytes 0-${bytes.length - 1}/${bytes.length}` } });
      calls++;
      if (calls === 1) return new Response(new ReadableStream({ start(controller) { controller.enqueue(bytes.slice(0, 32)); controller.error(new Error('interrupted')); } }), { headers: { 'content-length': String(bytes.length) } });
      return new Response(bytes, { headers: { 'content-length': String(bytes.length) } });
    };
    await expect(cache.ensure()).rejects.toThrow(/interrupted/);
    expect(root.dirs.get('fixture-gguf-v1')!.files.has('manifest.json')).toBe(false);
    await expect(cache.ensure()).resolves.toMatchObject({ complete: true });
  });
});
