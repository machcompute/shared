// Verification-only scalar decoder. Runtime model loaders and kernels must not
// import this module: production consumes packed GGUF bytes directly on GPU.
import { GGML_LAYOUT, GGML_TYPE } from '../../lib/webgpu-llm/gguf/parser.js';

function f16(value) {
  const sign = (value & 0x8000) << 16;
  let exponent = (value >>> 10) & 0x1f;
  let mantissa = value & 0x3ff;
  let bits;
  if (exponent === 0) {
    if (mantissa === 0) bits = sign;
    else {
      exponent = 127 - 15 + 1;
      while ((mantissa & 0x400) === 0) { mantissa <<= 1; exponent--; }
      bits = sign | (exponent << 23) | ((mantissa & 0x3ff) << 13);
    }
  } else if (exponent === 31) bits = sign | 0x7f800000 | (mantissa << 13);
  else bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
  const u = new Uint32Array([bits >>> 0]);
  return new Float32Array(u.buffer)[0];
}

const u16 = (b, o) => b[o] | (b[o + 1] << 8);
const f32 = (b, o) => new DataView(b.buffer, b.byteOffset + o, 4).getFloat32(0, true);
const i8 = (v) => (v << 24) >> 24;

function scaleMinK4(scales, j) {
  if (j < 4) return [scales[j] & 63, scales[j + 4] & 63];
  return [
    (scales[j + 4] & 15) | ((scales[j - 4] >>> 6) << 4),
    (scales[j + 4] >>> 4) | ((scales[j] >>> 6) << 4),
  ];
}

export function dequantizeGGMLValue(input, type, index) {
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(input.buffer ?? input, input.byteOffset ?? 0, input.byteLength);
  const layout = GGML_LAYOUT[type];
  if (!layout) throw new Error(`Unsupported GGML type ${type}`);
  const block = Math.floor(index / layout.blockSize);
  const i = index % layout.blockSize;
  const base = block * layout.typeSize;
  switch (type) {
    case GGML_TYPE.F32: return f32(bytes, base);
    case GGML_TYPE.BF16: {
      const bits = new Uint32Array([u16(bytes, base) << 16]);
      return new Float32Array(bits.buffer)[0];
    }
    case GGML_TYPE.Q4_0: {
      const d = f16(u16(bytes, base));
      const q = bytes[base + 2 + (i & 15)];
      return d * (((i < 16 ? q : q >>> 4) & 15) - 8);
    }
    case GGML_TYPE.Q4_1: {
      const d = f16(u16(bytes, base));
      const m = f16(u16(bytes, base + 2));
      const q = bytes[base + 4 + (i & 15)];
      return d * ((i < 16 ? q : q >>> 4) & 15) + m;
    }
    case GGML_TYPE.Q8_0: return f16(u16(bytes, base)) * i8(bytes[base + 2 + i]);
    case GGML_TYPE.Q5_K: {
      const d = f16(u16(bytes, base));
      const dmin = f16(u16(bytes, base + 2));
      const group = i >>> 5;
      const l = i & 31;
      const pair = group >>> 1;
      const [sc, min] = scaleMinK4(bytes.subarray(base + 4, base + 16), group);
      const packed = bytes[base + 48 + pair * 32 + l];
      const low = (group & 1) === 0;
      const qhMask = (low ? 1 : 2) << (pair * 2);
      const q = (low ? packed & 15 : packed >>> 4) + ((bytes[base + 16 + l] & qhMask) ? 16 : 0);
      return d * sc * q - dmin * min;
    }
    case GGML_TYPE.Q6_K: {
      const d = f16(u16(bytes, base + 208));
      const half = i >>> 7;
      const p = i & 127;
      const quarter = p >>> 5;
      const l = p & 31;
      const qlBase = base + half * 64;
      const qh = bytes[base + 128 + half * 32 + l];
      const qlOffset = (quarter & 1) ? 32 : 0;
      const nibble = quarter < 2 ? bytes[qlBase + qlOffset + l] & 15 : bytes[qlBase + qlOffset + l] >>> 4;
      const high = (qh >>> (quarter * 2)) & 3;
      const scaleIndex = half * 8 + (l >>> 4) + quarter * 2;
      return d * i8(bytes[base + 192 + scaleIndex]) * ((nibble | (high << 4)) - 32);
    }
    default: throw new Error(`Unsupported GGML type ${type}`);
  }
}

export function dequantizeGGML(input, type, elements) {
  const out = new Float32Array(elements);
  for (let i = 0; i < elements; i++) out[i] = dequantizeGGMLValue(input, type, i);
  return out;
}
