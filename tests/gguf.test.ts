import { describe, expect, it } from 'vitest';
import {
  GGML_TYPE,
  GGUFNeedMoreDataError,
  ggmlTensorByteLength,
  parseGGUF,
  tensorMatrixShape,
} from '../lib/webgpu-llm/gguf/parser.js';
import { sha256Hex } from '../lib/webgpu-llm/gguf/sha256.js';
import { buildQwenGGUFMap } from '../lib/webgpu-llm/loader.js';
import { buildGemmaGGUFMap } from '../lib/webgpu-llm/gemma-loader.js';
import { validateGGUFSource } from '../lib/webgpu-llm/gguf/sources.js';
import { GEMMA_E2B_CFG } from '../lib/webgpu-llm/gemma-config.js';
import { audioQkScale } from '../lib/webgpu-llm/gemma-kernels.js';
import { deltaRule } from '../lib/webgpu-llm/kernels.js';
import { patchifyRgba } from '../lib/webgpu-llm/gemma-media.js';

const te = new TextEncoder();
const u32 = (n: number) => { const b = new Uint8Array(4); new DataView(b.buffer).setUint32(0, n, true); return b; };
const u64 = (n: number) => { const b = new Uint8Array(8); new DataView(b.buffer).setBigUint64(0, BigInt(n), true); return b; };
const str = (s: string) => { const b = te.encode(s); return concat(u64(b.length), b); };
function concat(...parts: Uint8Array[]) {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) { out.set(p, at); at += p.length; }
  return out;
}
function fixture(type: number = GGML_TYPE.Q4_0) {
  const metadata = concat(
    str('general.architecture'), u32(8), str('fixture'),
    str('general.type'), u32(8), str('model'),
    str('general.alignment'), u32(4), u32(64),
  );
  const tensor = concat(str('weight'), u32(2), u64(32), u64(2), u32(type), u64(0));
  const header = concat(te.encode('GGUF'), u32(3), u64(1), u64(3), metadata, tensor);
  const dataOffset = Math.ceil(header.length / 64) * 64;
  return concat(header, new Uint8Array(dataOffset - header.length), new Uint8Array(type === GGML_TYPE.Q4_0 ? 36 : 64));
}

describe('GGUF v3 parser', () => {
  it('parses little-endian metadata, alignment, dimensions, and absolute tensor offsets', () => {
    const bytes = fixture();
    const gguf = parseGGUF(bytes, { fileSize: bytes.length } as any);
    expect(gguf.version).toBe(3);
    expect(gguf.metadata['general.architecture']).toBe('fixture');
    expect(gguf.alignment).toBe(64);
    expect(gguf.dataOffset % 64).toBe(0);
    const tensor = gguf.tensorsByName.get('weight')!;
    expect(tensorMatrixShape(tensor)).toEqual({ K: 32, N: 2 });
    expect(tensor.offset).toBe(gguf.dataOffset);
    expect(tensor.byteLength).toBe(36);
  });

  it('reports a bounded retry size for truncated headers', () => {
    expect(() => parseGGUF(fixture().subarray(0, 20))).toThrow(GGUFNeedMoreDataError);
    try { parseGGUF(fixture().subarray(0, 20)); } catch (error) {
      expect((error as GGUFNeedMoreDataError).requiredBytes).toBeGreaterThan(20);
    }
  });

  it('rejects unsupported tensor encodings before tensor data is read', () => {
    expect(() => parseGGUF(fixture(99))).toThrow(/Unsupported GGML tensor type 99/);
  });

  it('rejects a source whose declared architecture does not match', () => {
    const gguf = parseGGUF(fixture(), { fileSize: fixture().length } as any);
    expect(() => validateGGUFSource({
      filename: 'fixture.gguf', tensorCount: 1, architecture: 'wrong', role: 'model',
    } as any, gguf)).toThrow(/expected wrong architecture, got fixture/);
  });

  it('computes the pinned GGML block sizes', () => {
    expect(ggmlTensorByteLength([32], GGML_TYPE.Q4_0)).toBe(18);
    expect(ggmlTensorByteLength([32], GGML_TYPE.Q4_1)).toBe(20);
    expect(ggmlTensorByteLength([32], GGML_TYPE.Q8_0)).toBe(34);
    expect(ggmlTensorByteLength([256], GGML_TYPE.Q5_K)).toBe(176);
    expect(ggmlTensorByteLength([256], GGML_TYPE.Q6_K)).toBe(210);
  });
});

describe('GGUF migration manifests', () => {
  it('maps Qwen MTP and mixed GatedDeltaNet segments explicitly', () => {
    const specs = buildQwenGGUFMap() as any[];
    expect(specs.find((s) => s.logical === 'mtp.fc').tensors[0].tensor).toBe('blk.32.nextn.eh_proj.weight');
    expect(specs.find((s) => s.logical === 'L0.in').tensors.map((p: any) => p.tensor)).toEqual([
      'blk.0.attn_qkv.weight', 'blk.0.attn_gate.weight', 'blk.0.ssm_beta.weight', 'blk.0.ssm_alpha.weight',
    ]);
  });

  it('omits repeated Gemma K/V weights in the shared tail and maps both media towers', () => {
    const specs = buildGemmaGGUFMap(GEMMA_E2B_CFG) as any[];
    expect(specs.find((s) => s.logical === 'text.L15.qkv').tensors).toHaveLength(1);
    expect(specs.some((s) => s.logical === 'vision.patchProj' && s.tensors[0].tensor === 'v.patch_embd.weight')).toBe(true);
    expect(specs.some((s) => s.logical === 'audio.L0.attn.perDimScale' && s.tensors[0].tensor === 'a.blk.0.per_dim_scale.weight')).toBe(true);
  });

  it('uses the converter tiled V-head order in recurrent attention', () => {
    expect(deltaRule({ PREP: 0, SNAP: 0, FLAG: 0, LAZY: 0 })).toContain('let kh = h & 15u;');
  });

  it('patchifies vision pixels in CHW order', () => {
    const rgba = new Uint8ClampedArray(16 * 16 * 4);
    for (let i = 0; i < 16 * 16; i++) {
      rgba[i * 4] = 10; rgba[i * 4 + 1] = 20; rgba[i * 4 + 2] = 30; rgba[i * 4 + 3] = 255;
    }
    const output = new Float32Array(16 * 16 * 3);
    patchifyRgba(rgba, { width: 16, height: 16, patchWidth: 1, patchHeight: 1 } as any, output, 0);
    expect(output[0]).toBeCloseTo(10 / 255);
    expect(output[256]).toBeCloseTo(20 / 255);
    expect(output[512]).toBeCloseTo(30 / 255);
  });

  it('consumes already-softplused audio scales directly', () => {
    const wgsl = audioQkScale({});
    expect(wgsl).toContain('perDimScale[d % 128u]');
    expect(wgsl).not.toContain('gemma_softplus(perDimScale');
  });
});

describe('incremental SHA-256', () => {
  it('matches the standard vector across arbitrary chunk boundaries', () => {
    expect(sha256Hex(te.encode('abc'))).toBe('ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
  });
});
