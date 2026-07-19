import { describe, expect, it } from 'vitest';
import { GGML_TYPE } from '../lib/webgpu-llm/gguf/parser.js';
import { dequantizeGGML } from '../tools/reference/ggml.js';
import { ggmlLoadWGSL } from '../lib/webgpu-llm/ggml/kernels.js';

const setHalf = (view: DataView, offset: number, bits: number) => view.setUint16(offset, bits, true);

describe('native packed GGML CPU references', () => {
  it('decodes Q4_0 low and high halves with the symmetric zero point', () => {
    const block = new Uint8Array(18);
    setHalf(new DataView(block.buffer), 0, 0x3c00); // 1.0
    for (let i = 0; i < 16; i++) block[2 + i] = i | ((15 - i) << 4);
    const out = dequantizeGGML(block, GGML_TYPE.Q4_0, 32);
    expect(out[0]).toBe(-8);
    expect(out[15]).toBe(7);
    expect(out[16]).toBe(7);
    expect(out[31]).toBe(-8);
  });

  it('decodes affine Q4_1, signed Q8_0, and BF16', () => {
    const q41 = new Uint8Array(20);
    const q41v = new DataView(q41.buffer);
    setHalf(q41v, 0, 0x4000); // d=2
    setHalf(q41v, 2, 0x3c00); // m=1
    q41.fill(0x21, 4);
    expect(Array.from(dequantizeGGML(q41, GGML_TYPE.Q4_1, 32).slice(0, 2))).toEqual([3, 3]);
    expect(dequantizeGGML(q41, GGML_TYPE.Q4_1, 32)[16]).toBe(5);

    const q8 = new Uint8Array(34);
    setHalf(new DataView(q8.buffer), 0, 0x3800); // .5
    q8[2] = 0xfe;
    expect(dequantizeGGML(q8, GGML_TYPE.Q8_0, 32)[0]).toBe(-1);

    const bf16 = new Uint8Array([0x80, 0x3f]);
    expect(dequantizeGGML(bf16, GGML_TYPE.BF16, 1)[0]).toBe(1);
  });

  it('decodes K-quant blocks and generates a loader for every accepted type', () => {
    const q5 = new Uint8Array(176);
    const q5v = new DataView(q5.buffer);
    setHalf(q5v, 0, 0x3c00);
    setHalf(q5v, 2, 0x0000);
    q5.fill(1, 4, 16); // scale/min packing gives scale 1
    expect(dequantizeGGML(q5, GGML_TYPE.Q5_K, 256)[0]).toBe(0);

    const q6 = new Uint8Array(210);
    setHalf(new DataView(q6.buffer), 208, 0x3c00);
    q6.fill(1, 192, 208);
    q6.fill(0, 0, 128);
    q6.fill(2, 128, 192); // high bits make q=32
    expect(dequantizeGGML(q6, GGML_TYPE.Q6_K, 256)[0]).toBe(0);

    for (const type of [GGML_TYPE.F32, GGML_TYPE.BF16, GGML_TYPE.Q4_0, GGML_TYPE.Q4_1, GGML_TYPE.Q5_K, GGML_TYPE.Q6_K, GGML_TYPE.Q8_0]) {
      expect(ggmlLoadWGSL(type, type === GGML_TYPE.Q5_K || type === GGML_TYPE.Q6_K ? 256 : 32)).toContain('fn loadWeight');
    }
  });
});
