#!/usr/bin/env node

import { GPU } from '../lib/webgpu-llm/gpu.js';
import { GGML_LAYOUT, GGML_TYPE, GGML_TYPE_NAME } from '../lib/webgpu-llm/gguf/parser.js';
import { dequantizeGGML } from './reference/ggml.js';
import {
  gatherGGML,
  gemmGGML,
  gemvGGML,
  gemvSubgroupGGML,
  hasNativeSubgroupGemv,
} from '../lib/webgpu-llm/ggml/kernels.js';
import { gemvQ81GGML, q81ByteLength, quantizeQ81, supportsQ81Gemv } from '../lib/webgpu-llm/ggml/q8.js';
import { installNativeWebGPU } from './webgpu-model-throughput.mjs';

const halfBits = (value) => {
  // All generated test scales are exactly representable normal values.
  if (value === 1) return 0x3c00;
  if (value === 0.5) return 0x3800;
  if (value === 2) return 0x4000;
  throw new Error(`unsupported fixture half ${value}`);
};

function fixture(type) {
  const { blockSize, typeSize } = GGML_LAYOUT[type];
  const K = type === GGML_TYPE.F32 || type === GGML_TYPE.BF16 ? 32 : blockSize;
  const b = new Uint8Array(K / blockSize * typeSize);
  const v = new DataView(b.buffer);
  if (type === GGML_TYPE.F32) for (let i = 0; i < K; i++) v.setFloat32(i * 4, 1.25 - i * 0.01, true);
  else if (type === GGML_TYPE.BF16) for (let i = 0; i < K; i++) v.setUint16(i * 2, i & 1 ? 0xbf80 : 0x3fa0, true);
  else if (type === GGML_TYPE.Q4_0) {
    v.setUint16(0, halfBits(0.5), true);
    for (let i = 0; i < 16; i++) b[2 + i] = ((i * 7) & 15) | ((((15 - i) * 5) & 15) << 4);
  } else if (type === GGML_TYPE.Q4_1) {
    v.setUint16(0, halfBits(0.5), true); v.setUint16(2, halfBits(1), true);
    for (let i = 0; i < 16; i++) b[4 + i] = ((i * 3) & 15) | ((((i + 5) * 7) & 15) << 4);
  } else if (type === GGML_TYPE.Q8_0) {
    v.setUint16(0, halfBits(0.5), true);
    for (let i = 0; i < 32; i++) b[2 + i] = (i * 11 - 120) & 255;
  } else if (type === GGML_TYPE.Q5_K) {
    v.setUint16(0, halfBits(0.5), true); v.setUint16(2, halfBits(0.5), true);
    for (let i = 0; i < 12; i++) b[4 + i] = (i * 9 + 3) & 63;
    for (let i = 0; i < 32; i++) b[16 + i] = (i * 13) & 255;
    for (let i = 0; i < 128; i++) b[48 + i] = (i * 17 + 5) & 255;
  } else if (type === GGML_TYPE.Q6_K) {
    for (let i = 0; i < 128; i++) b[i] = (i * 19 + 7) & 255;
    for (let i = 0; i < 64; i++) b[128 + i] = (i * 23 + 11) & 255;
    for (let i = 0; i < 16; i++) b[192 + i] = (i * 7 - 50) & 255;
    v.setUint16(208, halfBits(0.5), true);
  }
  return { bytes: b, K };
}

const dot = (a, x) => a.reduce((sum, value, i) => sum + value * x[i], 0);
const rel = (actual, expected) => Math.abs(actual - expected) / Math.max(1, Math.abs(expected));
const assertClose = (label, actual, expected) => {
  const error = rel(actual, expected);
  if (!(error <= 1e-3)) throw new Error(`${label}: ${actual} != ${expected} (relative error ${error})`);
  return error;
};

function f16Bits(value) {
  const sign = (value & 0x8000) << 16; let exponent = (value >>> 10) & 31; let mantissa = value & 1023; let bits;
  if (!exponent) { if (!mantissa) bits = sign; else { exponent = 113; while (!(mantissa & 1024)) { mantissa <<= 1; exponent--; } bits = sign | (exponent << 23) | ((mantissa & 1023) << 13); } }
  else if (exponent === 31) bits = sign | 0x7f800000 | (mantissa << 13);
  else bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
  return new Float32Array(new Uint32Array([bits >>> 0]).buffer)[0];
}

function q81DotReference(weightBytes, type, q8Bytes, K) {
  if (type === GGML_TYPE.Q5_K || type === GGML_TYPE.Q6_K) {
    const weights = dequantizeGGML(weightBytes, type, K);
    const q = new DataView(q8Bytes.buffer, q8Bytes.byteOffset, q8Bytes.byteLength);
    let total = 0;
    for (let i = 0; i < K; i++) {
      const block = i >> 5; const d8 = f16Bits(q.getUint16(block * 36, true));
      total += weights[i] * d8 * q.getInt8(block * 36 + 4 + (i & 31));
    }
    return total;
  }
  const w = new DataView(weightBytes.buffer, weightBytes.byteOffset, weightBytes.byteLength);
  const q = new DataView(q8Bytes.buffer, q8Bytes.byteOffset, q8Bytes.byteLength);
  let total = 0;
  for (let block = 0; block < K / 32; block++) {
    const wb = block * GGML_LAYOUT[type].typeSize; const qb = block * 36;
    const d8 = f16Bits(q.getUint16(qb, true)); const s8 = f16Bits(q.getUint16(qb + 2, true));
    let rawDot = 0;
    for (let i = 0; i < 32; i++) {
      const q8 = q.getInt8(qb + 4 + i); let qw;
      if (type === GGML_TYPE.Q8_0) qw = w.getInt8(wb + 2 + i);
      else { const packed = w.getUint8(wb + (type === GGML_TYPE.Q4_0 ? 2 : 4) + (i & 15)); qw = (i < 16 ? packed : packed >>> 4) & 15; }
      rawDot += qw * q8;
    }
    const d = f16Bits(w.getUint16(wb, true));
    if (type === GGML_TYPE.Q4_0) total += d * (d8 * rawDot - 8 * s8);
    else if (type === GGML_TYPE.Q4_1) total += d * d8 * rawDot + f16Bits(w.getUint16(wb + 2, true)) * s8;
    else total += d * d8 * rawDot;
  }
  return total;
}

async function main() {
  const restoreWebGPU = installNativeWebGPU({});
  let gpu = null;
  try {
    gpu = await new GPU().init();
    const results = [];
    for (const type of [GGML_TYPE.F32, GGML_TYPE.BF16, GGML_TYPE.Q4_0, GGML_TYPE.Q4_1, GGML_TYPE.Q5_K, GGML_TYPE.Q6_K, GGML_TYPE.Q8_0]) {
    const { bytes, K } = fixture(type);
    const x = Float32Array.from({ length: K }, (_, i) => Math.sin(i * 0.17) * 0.25 + Math.cos(i * 0.03));
    const expected = dot(dequantizeGGML(bytes, type, K), x);
    const wb = gpu.storage(bytes.byteLength, `verify.${type}.w`);
    const xb = gpu.storage(x.byteLength, `verify.${type}.x`);
    const ob = gpu.storage(4, `verify.${type}.out`);
    gpu.upload(wb, bytes); gpu.upload(xb, x);
    const paths = [0, ...(gpu.subgroups ? [1] : [])];
    let maxError = 0;
    for (const subgroups of paths) {
      const pipeline = gpu.pipeline(`verify.ggml.${type}.${subgroups}`, () => gemvGGML({ N: 1, K, TYPE: type, SUBGROUPS: subgroups }));
      const bind = gpu.bind(pipeline, [wb, xb, ob]);
      const enc = gpu.device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(1); pass.end();
      gpu.device.queue.submit([enc.finish()]);
      const actual = new Float32Array(await gpu.readback(ob, 4))[0];
      maxError = Math.max(maxError, assertClose(`${GGML_TYPE_NAME[type]} GEMV subgroup=${subgroups}`, actual, expected));
    }
    if (hasNativeSubgroupGemv(type) && gpu.subgroups) {
      const fast = gpu.pipeline(`verify.ggml.native-subgroup.${type}`, () => gemvSubgroupGGML({ TYPE: type, N: 1, K }));
      const fastBind = gpu.bind(fast, [wb, xb, ob]);
      const fastEncoder = gpu.device.createCommandEncoder(); const fastPass = fastEncoder.beginComputePass();
      fastPass.setPipeline(fast); fastPass.setBindGroup(0, fastBind); fastPass.dispatchWorkgroups(1); fastPass.end();
      gpu.device.queue.submit([fastEncoder.finish()]);
      const actual = new Float32Array(await gpu.readback(ob, 4))[0];
      maxError = Math.max(maxError, assertClose(`${GGML_TYPE_NAME[type]} packed subgroup GEMV`, actual, expected));
    }
    if (gpu.dp4a && gpu.subgroups && supportsQ81Gemv(type)) {
      const q8b = gpu.storage(q81ByteLength(1, K), `verify.${type}.q8_1`);
      const quant = gpu.pipeline(`verify.ggml.q8_1.quant.${K}`, () => quantizeQ81({ K }));
      const quantBind = gpu.bind(quant, [xb, q8b]);
      const fast = gpu.pipeline(`verify.ggml.q8_1.gemv.${type}`, () => gemvQ81GGML({ TYPE: type, N: 1, K }));
      const fastBind = gpu.bind(fast, [wb, q8b, ob]);
      const fastEncoder = gpu.device.createCommandEncoder(); const fastPass = fastEncoder.beginComputePass();
      fastPass.setPipeline(quant); fastPass.setBindGroup(0, quantBind); fastPass.dispatchWorkgroups(1, 1);
      fastPass.setPipeline(fast); fastPass.setBindGroup(0, fastBind); fastPass.dispatchWorkgroups(1, 1); fastPass.end();
      gpu.device.queue.submit([fastEncoder.finish()]);
      const actual = new Float32Array(await gpu.readback(ob, 4))[0];
      const q8Bytes = new Uint8Array(await gpu.readback(q8b, q81ByteLength(1, K)));
      const qExpected = q81DotReference(bytes, type, q8Bytes, K);
      maxError = Math.max(maxError, assertClose(`${GGML_TYPE_NAME[type]} Q8_1 packed GEMV`, actual, qExpected));

      const qSegmentedOut = gpu.storage(12, `verify.${type}.q8_1.segmented`);
      gpu.upload(qSegmentedOut, new Float32Array([2, 2, 2]));
      const qSegmented = gpu.pipeline(
        `verify.ggml.q8_1.segmented.${type}`,
        () => gemvQ81GGML({ TYPE: type, N: 1, K, OSTRIDE: 3, OUTOFF: 1, RESIDUAL: 1 }),
      );
      const qSegmentedBind = gpu.bind(qSegmented, [wb, q8b, qSegmentedOut]);
      const qSegmentedEncoder = gpu.device.createCommandEncoder();
      const qSegmentedPass = qSegmentedEncoder.beginComputePass();
      qSegmentedPass.setPipeline(qSegmented); qSegmentedPass.setBindGroup(0, qSegmentedBind);
      qSegmentedPass.dispatchWorkgroups(1, 1); qSegmentedPass.end();
      gpu.device.queue.submit([qSegmentedEncoder.finish()]);
      const qSegmentedValues = new Float32Array(await gpu.readback(qSegmentedOut, 12));
      maxError = Math.max(maxError, assertClose(`${GGML_TYPE_NAME[type]} Q8_1 segmented residual`, qSegmentedValues[1], qExpected + 2));
      assertClose(`${GGML_TYPE_NAME[type]} Q8_1 segmented left guard`, qSegmentedValues[0], 2);
      assertClose(`${GGML_TYPE_NAME[type]} Q8_1 segmented right guard`, qSegmentedValues[2], 2);
      qSegmentedOut.destroy();
      q8b.destroy();
    }
    // Segmented projections write at a fixed output offset and can accumulate
    // a later segment without disturbing adjacent output columns.
    const segmentedOut = gpu.storage(12, `verify.${type}.segmented`);
    gpu.upload(segmentedOut, new Float32Array([2, 2, 2]));
    const segmentedPipeline = gpu.pipeline(
      `verify.ggml.segmented.${type}`,
      () => gemvGGML({ N: 1, K, TYPE: type, OSTRIDE: 3, OUTOFF: 1, RESIDUAL: 1 }),
    );
    const segmentedBind = gpu.bind(segmentedPipeline, [wb, xb, segmentedOut]);
    const segmentedEncoder = gpu.device.createCommandEncoder(); const segmentedPass = segmentedEncoder.beginComputePass();
    segmentedPass.setPipeline(segmentedPipeline); segmentedPass.setBindGroup(0, segmentedBind); segmentedPass.dispatchWorkgroups(1); segmentedPass.end();
    gpu.device.queue.submit([segmentedEncoder.finish()]);
    const segmented = new Float32Array(await gpu.readback(segmentedOut, 12));
    maxError = Math.max(maxError, assertClose(`${GGML_TYPE_NAME[type]} segmented residual`, segmented[1], expected + 2));
    assertClose(`${GGML_TYPE_NAME[type]} segmented left guard`, segmented[0], 2);
    assertClose(`${GGML_TYPE_NAME[type]} segmented right guard`, segmented[2], 2);

    // Tiled prefill GEMM with two activation rows.
    const x2 = new Float32Array(K * 2); x2.set(x); x2.set(x.map((value) => value * -0.75), K);
    const x2b = gpu.storage(x2.byteLength, `verify.${type}.x2`);
    const o2b = gpu.storage(8, `verify.${type}.o2`);
    const ub = gpu.uniform(16, `verify.${type}.u`);
    gpu.upload(x2b, x2); gpu.upload(ub, new Uint32Array([2, 0, 0, 0]));
    const gp = gpu.pipeline(`verify.ggml.gemm.${type}`, () => gemmGGML({ N: 1, K, TYPE: type }));
    const gb = gpu.bind(gp, [wb, x2b, o2b, ub]);
    const ge = gpu.device.createCommandEncoder(); const pass = ge.beginComputePass();
    pass.setPipeline(gp); pass.setBindGroup(0, gb); pass.dispatchWorkgroups(1, 1); pass.end(); gpu.device.queue.submit([ge.finish()]);
    const gout = new Float32Array(await gpu.readback(o2b, 8));
    maxError = Math.max(maxError, assertClose(`${GGML_TYPE_NAME[type]} GEMM row0`, gout[0], expected));
    maxError = Math.max(maxError, assertClose(`${GGML_TYPE_NAME[type]} GEMM row1`, gout[1], expected * -0.75));

    // Native embedding gather uses the same type decoder.
    const token = gpu.storage(4, `verify.${type}.token`); const gatherOut = gpu.storage(K * 4, `verify.${type}.gather`);
    gpu.upload(token, new Uint32Array([0]));
    const gather = gpu.pipeline(`verify.ggml.gather.${type}`, () => gatherGGML({ NUM: 1, K, TYPE: type, UNIFORM: 0 }));
    const gatherBind = gpu.bind(gather, [token, wb, gatherOut]);
    const gatherEncoder = gpu.device.createCommandEncoder(); const gatherPass = gatherEncoder.beginComputePass();
    gatherPass.setPipeline(gather); gatherPass.setBindGroup(0, gatherBind); gatherPass.dispatchWorkgroups(Math.ceil(K / 256), 1); gatherPass.end(); gpu.device.queue.submit([gatherEncoder.finish()]);
    const gathered = new Float32Array(await gpu.readback(gatherOut, K * 4));
    const reference = dequantizeGGML(bytes, type, K);
    for (let i = 0; i < K; i++) maxError = Math.max(maxError, assertClose(`${GGML_TYPE_NAME[type]} gather[${i}]`, gathered[i], reference[i]));

    for (const buffer of [wb, xb, ob, segmentedOut, x2b, o2b, ub, token, gatherOut]) buffer.destroy();
      results.push({ type: GGML_TYPE_NAME[type], maxRelativeError: maxError });
    }
    process.stdout.write(`${JSON.stringify({ device: gpu.info, subgroups: gpu.subgroups, results }, null, 2)}\n`);
  } finally {
    gpu?.destroy();
    gpu = null;
    restoreWebGPU();
  }
}

main().catch((error) => {
  process.stderr.write(`${error?.stack ?? error}\n`);
  process.exitCode = 1;
});
