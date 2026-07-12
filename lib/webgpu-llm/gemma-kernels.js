// Stand-alone WGSL generators for the Google Gemma 4 E4B runtime.
//
// This module deliberately does not import the Qwen kernels.  Gemma 4 has a
// different decoder (sliding/global attention rather than DeltaNet), direct
// RMSNorm scales, GELU-gated MLPs, per-layer embeddings, and image/audio
// encoders.  Keeping its generators here prevents model-specific layouts from
// leaking into the established Qwen execution path.
//
// Matrix convention used throughout this file:
//   * Q4 matrices are row-major [N, K], encoded in 32-value affine-Q4 blocks.
//   * activations are f32, row-major [T, K].
//   * Gemma E4B decoder QKV rows are [Q heads | K heads | V heads].
//   * text KV cache buffers contain emulated FP8 E4M3FN values packed four per
//     u32, followed by one f32 scale (bitcast to u32) per (position, KV head).
//
// The generators favor clear, parameterized reference kernels.  They are
// intended to be correct on every WebGPU implementation first; a Gemma model
// runtime can replace individual hot paths with tuned variants without
// changing these binding contracts.

const requireInt = (name, value, min = 1) => {
  if (!Number.isInteger(value) || value < min) {
    throw new Error(`${name} must be an integer >= ${min}; got ${value}`);
  }
};

const assertQ4Shape = (N, K) => {
  requireInt('N', N);
  requireInt('K', K);
  if (K % 32) throw new Error(`Q4 K must be divisible by 32; got ${K}`);
};

const FP8_E4M3FN = /* wgsl */`
// FP8 E4M3FN is stored in its native byte representation, but WebGPU has no
// portable storage-fp8 type.  E=15/M=7 is the NaN encoding; the finite range
// is [-448, 448].  A separate f32 scale is kept per token/KV head.
fn gemma_fp8e4m3fn_encode(x: f32, invScale: f32) -> u32 {
  let sign = select(0u, 0x80u, x < 0.0);
  let a = min(abs(x) * invScale, 448.0);
  if (a == 0.0) { return sign; }

  // E=0 represents subnormals M * 2^-9.  Rounding may promote the largest
  // subnormal into the smallest normal, which the normal path handles.
  if (a < 0.015625) {
    let m = u32(round(a * 512.0));
    if (m < 8u) { return sign | m; }
  }

  var e = i32(floor(log2(a))) + 7;
  e = clamp(e, 1, 15);
  let unit = exp2(f32(e - 7));
  var m = i32(round((a / unit - 1.0) * 8.0));
  if (m >= 8) {
    if (e < 15) {
      e += 1;
      m = 0;
    } else {
      // E=15/M=7 is NaN, so saturate at the largest finite value, 448.
      m = 6;
    }
  }
  if (e == 15 && m > 6) { m = 6; }
  m = max(m, 0);
  return sign | (u32(e) << 3u) | u32(m);
}

fn gemma_fp8e4m3fn_decode_lane(w: u32, lane: u32) -> f32 {
  let byte = (w >> ((lane & 3u) * 8u)) & 0xFFu;
  let sign = select(1.0, -1.0, (byte & 0x80u) != 0u);
  let e = (byte >> 3u) & 0xFu;
  let m = byte & 0x7u;
  if (e == 0u) { return sign * f32(m) * 0.001953125; }
  // Never emit the reserved NaN code.  Decoding corrupt data as the largest
  // finite mantissa keeps cache reads finite rather than poisoning attention.
  let finiteM = select(m, 6u, e == 15u && m == 7u);
  return sign * (1.0 + f32(finiteM) * 0.125) * exp2(f32(e) - 7.0);
}

fn gemma_fp8e4m3fn_pack4(v: vec4f, invScale: f32) -> u32 {
  return gemma_fp8e4m3fn_encode(v.x, invScale)
       | (gemma_fp8e4m3fn_encode(v.y, invScale) << 8u)
       | (gemma_fp8e4m3fn_encode(v.z, invScale) << 16u)
       | (gemma_fp8e4m3fn_encode(v.w, invScale) << 24u);
}
`;

const GELU = /* wgsl */`
fn gemma_gelu_tanh(x: f32) -> f32 {
  // PyTorch's approximate='tanh' GELU used by Gemma 4.
  return 0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
}

fn gemma_softplus(x: f32) -> f32 {
  // Stable enough for the trained per-dimension audio scales and avoids
  // unnecessary overflow for arbitrary user-provided test tensors.
  return max(x, 0.0) + log(1.0 + exp(-abs(x)));
}
`;

/**
 * Decode-style affine-Q4 matrix-vector multiply.
 *
 * Bindings: qdata, scales, x[T,K], out[T,OSTRIDE].  `RESIDUAL` controls
 * assignment vs +=.  One workgroup produces four output rows for one input
 * row; dispatch `(ceil(N / 4), T)`.
 */
export const gemvQ4 = ({ N, K, RESIDUAL = 0, OSTRIDE = N, OUTOFF = 0 }) => {
  assertQ4Shape(N, K);
  requireInt('OSTRIDE', OSTRIDE);
  requireInt('OUTOFF', OUTOFF, 0);
  return /* wgsl */`
@group(0) @binding(0) var<storage, read> qdata: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
var<workgroup> red: array<f32, 256>;
const WPR = ${K / 8}u;
const BPR = ${K / 32}u;
const XSTRIDE = ${K / 4}u;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let rowIn = wid.y;
  let rowInBase = rowIn * XSTRIDE;
  let localRow = lid.x >> 6u;
  let lane = lid.x & 63u;
  let row = wid.x * 4u + localRow;
  var acc = 0.0;
  if (row < ${N}u) {
    let wb = row * WPR;
    let sb = row * BPR;
    var w = lane * 2u;
    loop {
      if (w >= WPR) { break; }
      let a = qdata[wb + w];
      let b = qdata[wb + w + 1u];
      let sm = unpack2x16float(scales[sb + (w >> 2u)]);
      let x0 = x[rowInBase + w * 2u];
      let x1 = x[rowInBase + w * 2u + 1u];
      let x2 = x[rowInBase + w * 2u + 2u];
      let x3 = x[rowInBase + w * 2u + 3u];
      let qsum =
          dot(vec4f(f32(a & 15u), f32((a >> 4u) & 15u), f32((a >> 8u) & 15u), f32((a >> 12u) & 15u)), x0)
        + dot(vec4f(f32((a >> 16u) & 15u), f32((a >> 20u) & 15u), f32((a >> 24u) & 15u), f32((a >> 28u) & 15u)), x1)
        + dot(vec4f(f32(b & 15u), f32((b >> 4u) & 15u), f32((b >> 8u) & 15u), f32((b >> 12u) & 15u)), x2)
        + dot(vec4f(f32((b >> 16u) & 15u), f32((b >> 20u) & 15u), f32((b >> 24u) & 15u), f32((b >> 28u) & 15u)), x3);
      let xsum = dot(x0 + x1 + x2 + x3, vec4f(1.0));
      acc += sm.x * qsum + sm.y * xsum;
      w += 128u;
    }
  }
  red[lid.x] = acc;
  workgroupBarrier();
  var step = 32u;
  loop {
    if (step == 0u) { break; }
    if (lane < step) { red[lid.x] += red[lid.x + step]; }
    workgroupBarrier();
    step = step >> 1u;
  }
  if (lane == 0u && row < ${N}u) {
    let dst = rowIn * ${OSTRIDE}u + ${OUTOFF}u + row;
    ${RESIDUAL ? 'out[dst] += red[lid.x];' : 'out[dst] = red[lid.x];'}
  }
}`;
};

/**
 * Simple batch-friendly affine-Q4 GEMM reference kernel.
 *
 * Bindings: qdata, scales, x[T,K], out[T,OSTRIDE], u where u.rows is T.
 * Dispatch `(ceil(N / 64), ceil(T / 4))`.  Each thread computes one output
 * element.  It is intentionally conservative and makes an excellent oracle
 * for optimized tiled implementations.
 */
export const gemmQ4 = ({ N, K, RESIDUAL = 0, OSTRIDE = N, OUTOFF = 0 }) => {
  assertQ4Shape(N, K);
  requireInt('OSTRIDE', OSTRIDE);
  requireInt('OUTOFF', OUTOFF, 0);
  return /* wgsl */`
struct U { rows: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> qdata: array<u32>;
@group(0) @binding(1) var<storage, read> scales: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> u: U;
const WPR = ${K / 8}u;
const BPR = ${K / 32}u;
const XSTRIDE = ${K / 4}u;

@compute @workgroup_size(64, 4)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let n = wid.x * 64u + lid.x;
  let t = wid.y * 4u + lid.y;
  if (n >= ${N}u || t >= u.rows) { return; }
  let wb = n * WPR;
  let sb = n * BPR;
  let xb = t * XSTRIDE;
  var acc = 0.0;
  for (var w = 0u; w < WPR; w += 2u) {
    let a = qdata[wb + w];
    let b = qdata[wb + w + 1u];
    let sm = unpack2x16float(scales[sb + (w >> 2u)]);
    let x0 = x[xb + w * 2u];
    let x1 = x[xb + w * 2u + 1u];
    let x2 = x[xb + w * 2u + 2u];
    let x3 = x[xb + w * 2u + 3u];
    let qsum =
        dot(vec4f(f32(a & 15u), f32((a >> 4u) & 15u), f32((a >> 8u) & 15u), f32((a >> 12u) & 15u)), x0)
      + dot(vec4f(f32((a >> 16u) & 15u), f32((a >> 20u) & 15u), f32((a >> 24u) & 15u), f32((a >> 28u) & 15u)), x1)
      + dot(vec4f(f32(b & 15u), f32((b >> 4u) & 15u), f32((b >> 8u) & 15u), f32((b >> 12u) & 15u)), x2)
      + dot(vec4f(f32((b >> 16u) & 15u), f32((b >> 20u) & 15u), f32((b >> 24u) & 15u), f32((b >> 28u) & 15u)), x3);
    let xsum = dot(x0 + x1 + x2 + x3, vec4f(1.0));
    acc += sm.x * qsum + sm.y * xsum;
  }
  let dst = t * ${OSTRIDE}u + ${OUTOFF}u + n;
  ${RESIDUAL ? 'out[dst] += acc;' : 'out[dst] = acc;'}
}`;
};

/**
 * Dequantized embedding lookup, suitable for both the regular embedding table
 * and Gemma's [vocab, 42 * 256] PLE table.  The table can be row-sharded:
 * `START` is the global first vocabulary id, `NUM` is this shard's row count.
 *
 * Bindings: tokens[T], qdata, scales, out[T,K], u where u.rows is T.
 * The caller must clear `out` before dispatching a set of shards.  Dispatch
 * `(ceil(K / 256), T)` for every shard; only matching token ids write.
 */
export const gatherQ4 = ({ START = 0, NUM, K, SCALE = 1 }) => {
  requireInt('START', START, 0);
  requireInt('NUM', NUM);
  assertQ4Shape(NUM, K);
  return /* wgsl */`
struct U { rows: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> tokens: array<u32>;
@group(0) @binding(1) var<storage, read> qdata: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> u: U;
const WPR = ${K / 8}u;
const BPR = ${K / 32}u;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let t = wid.y;
  let d = wid.x * 256u + lid.x;
  if (t >= u.rows || d >= ${K}u) { return; }
  let id = tokens[t];
  if (id < ${START}u || id >= ${START + NUM}u) { return; }
  let row = id - ${START}u;
  let word = d >> 3u;
  let nibble = (d & 7u) * 4u;
  let q = f32((qdata[row * WPR + word] >> nibble) & 15u);
  let sm = unpack2x16float(scales[row * BPR + (d >> 5u)]);
  out[t * ${K}u + d] = (sm.x * q + sm.y) * ${Number(SCALE)};
}`;
};

/**
 * Direct-scale RMSNorm (Gemma's norm weights are not zero-centred).
 * Bindings: x[rows,K], gamma[K], out[rows,OSTRIDE], u.rows.  `WITH_GAMMA=0`
 * makes it the unit-scale RMSNorm used by V heads and multimodal projectors.
 */
export const rmsnorm = ({ K, OSTRIDE = K, OUTOFF = 0, WITH_GAMMA = 1, EPS = 1e-6 }) => {
  requireInt('K', K);
  if (K % 4) throw new Error(`RMSNorm K must be divisible by 4; got ${K}`);
  requireInt('OSTRIDE', OSTRIDE);
  requireInt('OUTOFF', OUTOFF, 0);
  return /* wgsl */`
struct U { rows: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> x: array<vec4f>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> u: U;
var<workgroup> red: array<f32, 256>;
const VEC = ${K / 4}u;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let r = wid.x;
  if (r >= u.rows) { return; }
  let xb = r * VEC;
  var ss = 0.0;
  for (var i = lid.x; i < VEC; i += 256u) { ss += dot(x[xb + i], x[xb + i]); }
  red[lid.x] = ss;
  workgroupBarrier();
  var step = 128u;
  loop {
    if (step == 0u) { break; }
    if (lid.x < step) { red[lid.x] += red[lid.x + step]; }
    workgroupBarrier();
    step = step >> 1u;
  }
  let ri = inverseSqrt(red[0] / ${K}.0 + ${Number(EPS)});
  for (var i = lid.x; i < ${K}u; i += 256u) {
    // Keep binding(1) present for the unit-scale variant too.  The harmless
    // NaN-preserving zero multiply prevents WebGPU's reflection from dropping
    // gamma and changing the bind-group layout at runtime.
    let g = ${WITH_GAMMA ? 'gamma[i]' : '1.0 + 0.0 * gamma[i]'};
    out[r * ${OSTRIDE}u + ${OUTOFF}u + i] = x[xb + (i >> 2u)][i & 3u] * ri * g;
  }
}`;
};

/** LayerNorm with learned scale and no bias (needed by audio SSCP). */
export const layernorm = ({ K, EPS = 1e-6 }) => {
  requireInt('K', K);
  return /* wgsl */`
struct U { rows: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> u: U;
var<workgroup> red: array<f32, 256>;
var<workgroup> mean: f32;
var<workgroup> invStd: f32;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let r = wid.x;
  if (r >= u.rows) { return; }
  let b = r * ${K}u;
  var sum = 0.0;
  for (var i = lid.x; i < ${K}u; i += 256u) { sum += x[b + i]; }
  red[lid.x] = sum;
  workgroupBarrier();
  var step = 128u;
  loop { if (step == 0u) { break; }
    if (lid.x < step) { red[lid.x] += red[lid.x + step]; }
    workgroupBarrier(); step = step >> 1u;
  }
  if (lid.x == 0u) { mean = red[0] / ${K}.0; }
  workgroupBarrier();
  var sq = 0.0;
  for (var i = lid.x; i < ${K}u; i += 256u) { let d = x[b + i] - mean; sq += d * d; }
  red[lid.x] = sq;
  workgroupBarrier();
  step = 128u;
  loop { if (step == 0u) { break; }
    if (lid.x < step) { red[lid.x] += red[lid.x + step]; }
    workgroupBarrier(); step = step >> 1u;
  }
  if (lid.x == 0u) { invStd = inverseSqrt(red[0] / ${K}.0 + ${Number(EPS)}); }
  workgroupBarrier();
  for (var i = lid.x; i < ${K}u; i += 256u) { out[b + i] = (x[b + i] - mean) * invStd * gamma[i]; }
}`;
};

/** Clear a contiguous f32 buffer. Bindings: dst, u.elements; dispatch ceil(elements/256). */
export const clearF32 = () => /* wgsl */`
struct U { elements: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x < u.elements) { dst[gid.x] = 0.0; }
}`;

/** dst = a + b. Bindings: a, b, dst, u.elements. */
export const add = () => /* wgsl */`
struct U { elements: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x < u.elements) { dst[gid.x] = a[gid.x] + b[gid.x]; }
}`;

/**
 * Add a learned bias to every row, used by Gemma E4B's audio output
 * projection. Bindings: x[rows,K], bias[K], out[rows,K], u.rows. Dispatch
 * `(ceil(K / 256), rows)`.
 */
export const addBias = ({ K }) => {
  requireInt('K', K);
  return /* wgsl */`
struct U { rows: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let r = wid.y;
  let d = wid.x * 256u + lid.x;
  if (r < u.rows && d < ${K}u) { out[r * ${K}u + d] = x[r * ${K}u + d] + bias[d]; }
}`;
};

/** dst = src * SCALE. Bindings: src, dst, u.elements. */
export const scale = ({ SCALE = 1 }) => /* wgsl */`
struct U { elements: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x < u.elements) { dst[gid.x] = src[gid.x] * ${Number(SCALE)}; }
}`;

/**
 * GELU(gate) * up for a fused [rows, 2*K] buffer.  Bindings: gateUp, out,
 * u.rows. Dispatch `(ceil(K/256), rows)`.
 */
export const geluMul = ({ K }) => {
  requireInt('K', K);
  return /* wgsl */`
${GELU}
struct U { rows: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> gateUp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let r = wid.y;
  let d = wid.x * 256u + lid.x;
  if (r < u.rows && d < ${K}u) {
    let b = r * ${2 * K}u;
    out[r * ${K}u + d] = gemma_gelu_tanh(gateUp[b + d]) * gateUp[b + ${K}u + d];
  }
}`;
};

/** GELU(a) * b for two independent [rows,K] buffers (PLE injection). */
export const geluMulPair = ({ K }) => {
  requireInt('K', K);
  return /* wgsl */`
${GELU}
struct U { rows: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let n = u.rows * ${K}u;
  if (gid.x < n) { out[gid.x] = gemma_gelu_tanh(a[gid.x]) * b[gid.x]; }
}`;
};

/**
 * Fuse a Gemma per-layer embedding slice into its decoder-layer gate without
 * materializing a non-contiguous `[T,DIM]` PLE view.
 *
 * `ple` is packed `[T][LAYERS][DIM]`; `LAYER` selects the layer's 256-vector.
 * Bindings: gate[T,DIM], ple, out[T,DIM], u.rows. Dispatch
 * `(ceil(DIM / 256), rows)`.
 */
export const pleGateMul = ({ LAYERS = 42, DIM = 256, LAYER }) => {
  requireInt('LAYERS', LAYERS);
  requireInt('DIM', DIM);
  requireInt('LAYER', LAYER, 0);
  if (LAYER >= LAYERS) throw new Error(`PLE LAYER ${LAYER} must be < ${LAYERS}`);
  return /* wgsl */`
${GELU}
struct U { rows: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> ple: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let t = wid.y;
  let d = wid.x * 256u + lid.x;
  if (t < u.rows && d < ${DIM}u) {
    out[t * ${DIM}u + d] = gemma_gelu_tanh(gate[t * ${DIM}u + d])
      * ple[(t * ${LAYERS}u + ${LAYER}u) * ${DIM}u + d];
  }
}`;
};

/** Gemma final logit soft-cap: dst = CAP * tanh(src / CAP). */
export const softcap = ({ CAP = 30 }) => /* wgsl */`
struct U { elements: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x < u.elements) { dst[gid.x] = ${Number(CAP)} * tanh(src[gid.x] / ${Number(CAP)}); }
}`;

/**
 * Combine Gemma PLE identity and normalized context streams.
 * Both bindings are [tokens * layers, dim]; output is `(a + b) / sqrt(2)`.
 */
export const pleCombine = ({ LAYERS = 42, DIM = 256 }) => {
  requireInt('LAYERS', LAYERS);
  requireInt('DIM', DIM);
  return /* wgsl */`
struct U { tokens: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> identity: array<f32>;
@group(0) @binding(1) var<storage, read> context: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let n = u.tokens * ${LAYERS * DIM}u;
  if (gid.x < n) { out[gid.x] = (identity[gid.x] + context[gid.x]) * 0.7071067811865476; }
}`;
};

/**
 * PLE combination with an explicit text-token mask.  Text positions combine
 * identity/context streams and scale by 1/sqrt(2); multimodal soft-token
 * positions have no token identity and retain the context-aware stream alone.
 * Bindings: identity, context, textMask[T] (1=text, 0=soft token), output,
 * u.tokens. Dispatch `ceil(tokens*layers*dim/256)`.
 */
export const pleCombineMasked = ({ LAYERS = 42, DIM = 256 }) => {
  requireInt('LAYERS', LAYERS);
  requireInt('DIM', DIM);
  return /* wgsl */`
struct U { tokens: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> identity: array<f32>;
@group(0) @binding(1) var<storage, read> context: array<f32>;
@group(0) @binding(2) var<storage, read> textMask: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let n = u.tokens * ${LAYERS * DIM}u;
  if (gid.x >= n) { return; }
  let token = gid.x / ${LAYERS * DIM}u;
  let combined = (identity[gid.x] + context[gid.x]) * 0.7071067811865476;
  out[gid.x] = select(context[gid.x], combined, textMask[token] != 0u);
}`;
};

/**
 * Scatter rows into a sequence.  Bindings: source[rows,H], destination[T,H],
 * targetRows[rows] (absolute destination row), u.rows.  Dispatch
 * `(ceil(H/256), rows)`.
 */
export const scatterRows = ({ H }) => {
  requireInt('H', H);
  return /* wgsl */`
struct U { rows: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read_write> destination: array<f32>;
@group(0) @binding(2) var<storage, read> targetRows: array<u32>;
@group(0) @binding(3) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let r = wid.y;
  let d = wid.x * 256u + lid.x;
  if (r < u.rows && d < ${H}u) { destination[targetRows[r] * ${H}u + d] = source[r * ${H}u + d]; }
}`;
};

// ---------------------------------------------------------------------------
// Gemma text attention
//
// E4B uses eight Q heads and two KV heads.  Sliding layers use 256-wide heads
// and ordinary 256-d RoPE; global layers use 512-wide heads and proportional
// RoPE, which rotates only 64 pairs (128 dimensions) of the two 256-d halves.
// The parameters below intentionally make both cases a single binding layout.

const textAttentionShape = ({ HEADS, KV_HEADS, HEAD_DIM, ROTARY_PAIRS, MAXCTX }) => {
  requireInt('HEADS', HEADS);
  requireInt('KV_HEADS', KV_HEADS);
  requireInt('HEAD_DIM', HEAD_DIM);
  requireInt('ROTARY_PAIRS', ROTARY_PAIRS, 0);
  requireInt('MAXCTX', MAXCTX);
  if (HEADS % KV_HEADS) throw new Error('HEADS must be divisible by KV_HEADS');
  if (HEAD_DIM > 512 || HEAD_DIM % 4) {
    throw new Error(`text attention supports an f32/FP8 cache head width divisible by 4 up to 512; got ${HEAD_DIM}`);
  }
  if (ROTARY_PAIRS > HEAD_DIM / 2) {
    throw new Error(`ROTARY_PAIRS (${ROTARY_PAIRS}) exceeds HEAD_DIM / 2 (${HEAD_DIM / 2})`);
  }
  return {
    GROUPS: HEADS / KV_HEADS,
    QDIM: HEADS * HEAD_DIM,
    KVDIM: KV_HEADS * HEAD_DIM,
    QKV: (HEADS + 2 * KV_HEADS) * HEAD_DIM,
    WORDS: HEAD_DIM / 4,
    CACHE_WORDS: MAXCTX * KV_HEADS * (HEAD_DIM / 4),
  };
};

/**
 * Prepare Gemma text Q/K/V and append K/V to an emulated FP8 E4M3FN cache.
 *
 * QKV input is `[T][Q | K | V]`, with all vectors output by a fused Q4
 * projection.  `qNorm` and `kNorm` are direct Gemma RMS scales of length
 * `HEAD_DIM`; V uses unit-scale RMSNorm.  `rope` is laid out as
 * `[MAXCTX][cos[ROTARY_PAIRS] | sin[ROTARY_PAIRS]]`.
 *
 * Cache layout for each of K and V: data `[MAXCTX][KV_HEADS][HEAD_DIM/4]`,
 * then bitcast-f32 scales `[MAXCTX][KV_HEADS]`.  Dispatch `(T, KV_HEADS)`.
 * The same generator covers E4B local `{8,2,256,128}` and global
 * `{8,2,512,64}` heads.
 */
export const textKvPrep = ({ HEADS = 8, KV_HEADS = 2, HEAD_DIM, ROTARY_PAIRS, MAXCTX }) => {
  const { GROUPS, QDIM, KVDIM, QKV, WORDS, CACHE_WORDS } = textAttentionShape({
    HEADS, KV_HEADS, HEAD_DIM, ROTARY_PAIRS, MAXCTX,
  });
  return /* wgsl */`
${FP8_E4M3FN}
struct U { basePos: u32, T: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read> qNorm: array<f32>;
@group(0) @binding(2) var<storage, read> kNorm: array<f32>;
@group(0) @binding(3) var<storage, read> rope: array<f32>;
@group(0) @binding(4) var<storage, read_write> qOut: array<f32>;
@group(0) @binding(5) var<storage, read_write> kCache: array<u32>;
@group(0) @binding(6) var<storage, read_write> vCache: array<u32>;
@group(0) @binding(7) var<uniform> u: U;
var<workgroup> qsh: array<f32, ${GROUPS * HEAD_DIM}>;
var<workgroup> ksh: array<f32, ${HEAD_DIM}>;
var<workgroup> krot: array<f32, ${HEAD_DIM}>;
var<workgroup> vsh: array<f32, ${HEAD_DIM}>;
var<workgroup> red: array<f32, 256>;
var<workgroup> kInv: f32;
var<workgroup> vInv: f32;
const HALF = ${HEAD_DIM / 2}u;
const ROPE = ${ROTARY_PAIRS}u;
const QDIM = ${QDIM}u;
const KVDIM = ${KVDIM}u;
const QKV = ${QKV}u;
const WORDS = ${WORDS}u;
const CACHE_WORDS = ${CACHE_WORDS}u;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let t = wid.x;
  let kvh = wid.y;
  if (t >= u.T || kvh >= ${KV_HEADS}u) { return; }
  let pos = u.basePos + t;
  let qb = t * QKV;
  let qob = t * QDIM;
  let ropeBase = pos * (ROPE * 2u);

  // Four Q heads share one KV head for E4B.  Keep normalized, unrotated Q in
  // workgroup memory so rotating the paired halves never aliases a read.
  for (var g = 0u; g < ${GROUPS}u; g++) {
    let qh = kvh * ${GROUPS}u + g;
    let src = qb + qh * ${HEAD_DIM}u;
    var ss = 0.0;
    for (var d = lid.x; d < ${HEAD_DIM}u; d += 256u) {
      let v = qkv[src + d];
      ss += v * v;
    }
    red[lid.x] = ss;
    workgroupBarrier();
    var step = 128u;
    loop { if (step == 0u) { break; }
      if (lid.x < step) { red[lid.x] += red[lid.x + step]; }
      workgroupBarrier(); step = step >> 1u;
    }
    let ri = inverseSqrt(red[0] / ${HEAD_DIM}.0 + 1e-6);
    for (var d = lid.x; d < ${HEAD_DIM}u; d += 256u) {
      qsh[g * ${HEAD_DIM}u + d] = qkv[src + d] * ri * qNorm[d];
    }
    workgroupBarrier();
    for (var d = lid.x; d < ${HEAD_DIM}u; d += 256u) {
      var v = qsh[g * ${HEAD_DIM}u + d];
      if (d < ROPE) {
        let c = rope[ropeBase + d];
        let s = rope[ropeBase + ROPE + d];
        v = qsh[g * ${HEAD_DIM}u + d] * c - qsh[g * ${HEAD_DIM}u + d + HALF] * s;
      } else if (d >= HALF && d < HALF + ROPE) {
        let p = d - HALF;
        let c = rope[ropeBase + p];
        let s = rope[ropeBase + ROPE + p];
        v = qsh[g * ${HEAD_DIM}u + d] * c + qsh[g * ${HEAD_DIM}u + p] * s;
      }
      qOut[qob + qh * ${HEAD_DIM}u + d] = v;
    }
    workgroupBarrier();
  }

  // K RMSNorm (with learned scale) and RoPE.
  let ksrc = qb + QDIM + kvh * ${HEAD_DIM}u;
  var kss = 0.0;
  for (var d = lid.x; d < ${HEAD_DIM}u; d += 256u) { let v = qkv[ksrc + d]; kss += v * v; }
  red[lid.x] = kss;
  workgroupBarrier();
  var step = 128u;
  loop { if (step == 0u) { break; }
    if (lid.x < step) { red[lid.x] += red[lid.x + step]; }
    workgroupBarrier(); step = step >> 1u;
  }
  let kri = inverseSqrt(red[0] / ${HEAD_DIM}.0 + 1e-6);
  for (var d = lid.x; d < ${HEAD_DIM}u; d += 256u) { ksh[d] = qkv[ksrc + d] * kri * kNorm[d]; }
  workgroupBarrier();
  for (var d = lid.x; d < ${HEAD_DIM}u; d += 256u) {
    var v = ksh[d];
    if (d < ROPE) {
      let c = rope[ropeBase + d]; let s = rope[ropeBase + ROPE + d];
      v = ksh[d] * c - ksh[d + HALF] * s;
    } else if (d >= HALF && d < HALF + ROPE) {
      let p = d - HALF; let c = rope[ropeBase + p]; let s = rope[ropeBase + ROPE + p];
      v = ksh[d] * c + ksh[p] * s;
    }
    krot[d] = v;
  }
  workgroupBarrier();

  // V uses an RMSNorm without learned scale in Gemma 4.
  let vsrc = qb + QDIM + KVDIM + kvh * ${HEAD_DIM}u;
  var vss = 0.0;
  for (var d = lid.x; d < ${HEAD_DIM}u; d += 256u) { let v = qkv[vsrc + d]; vss += v * v; }
  red[lid.x] = vss;
  workgroupBarrier();
  step = 128u;
  loop { if (step == 0u) { break; }
    if (lid.x < step) { red[lid.x] += red[lid.x + step]; }
    workgroupBarrier(); step = step >> 1u;
  }
  let vri = inverseSqrt(red[0] / ${HEAD_DIM}.0 + 1e-6);
  for (var d = lid.x; d < ${HEAD_DIM}u; d += 256u) { vsh[d] = qkv[vsrc + d] * vri; }
  workgroupBarrier();

  // Per-token/head FP8 E4M3FN K cache.  448 is its largest finite magnitude;
  // cache scales stay f32 so a token/head can use the full FP8 dynamic range.
  var km = 0.0;
  for (var d = lid.x; d < ${HEAD_DIM}u; d += 256u) { km = max(km, abs(krot[d])); }
  red[lid.x] = km;
  workgroupBarrier();
  step = 128u;
  loop { if (step == 0u) { break; }
    if (lid.x < step) { red[lid.x] = max(red[lid.x], red[lid.x + step]); }
    workgroupBarrier(); step = step >> 1u;
  }
  if (lid.x == 0u) { kInv = select(0.0, 448.0 / red[0], red[0] > 0.0); }
  workgroupBarrier();
  let dataBase = (pos * ${KV_HEADS}u + kvh) * WORDS;
  if (lid.x < WORDS) {
    let d = lid.x * 4u;
    kCache[dataBase + lid.x] = gemma_fp8e4m3fn_pack4(vec4f(krot[d], krot[d + 1u], krot[d + 2u], krot[d + 3u]), kInv);
  }
  if (lid.x == 0u) { kCache[CACHE_WORDS + pos * ${KV_HEADS}u + kvh] = bitcast<u32>(select(0.0, 1.0 / kInv, kInv > 0.0)); }

  // And the V cache, using an independent scale.
  var vm = 0.0;
  for (var d = lid.x; d < ${HEAD_DIM}u; d += 256u) { vm = max(vm, abs(vsh[d])); }
  red[lid.x] = vm;
  workgroupBarrier();
  step = 128u;
  loop { if (step == 0u) { break; }
    if (lid.x < step) { red[lid.x] = max(red[lid.x], red[lid.x + step]); }
    workgroupBarrier(); step = step >> 1u;
  }
  if (lid.x == 0u) { vInv = select(0.0, 448.0 / red[0], red[0] > 0.0); }
  workgroupBarrier();
  if (lid.x < WORDS) {
    let d = lid.x * 4u;
    vCache[dataBase + lid.x] = gemma_fp8e4m3fn_pack4(vec4f(vsh[d], vsh[d + 1u], vsh[d + 2u], vsh[d + 3u]), vInv);
  }
  if (lid.x == 0u) { vCache[CACHE_WORDS + pos * ${KV_HEADS}u + kvh] = bitcast<u32>(select(0.0, 1.0 / vInv, vInv > 0.0)); }
}`;
};

/**
 * Prepare Q only for Gemma's declared shared-KV tail.  E4B shares the K/V
 * states from layer 22 (sliding) and layer 23 (global) across its final 18
 * decoder layers, so those layers still apply their own Q projection/norm/RoPE
 * but must not append checkpoint-local K/V values to the cache.
 *
 * Bindings: q[T,QDIM], qNorm[HEAD_DIM], rope, qOut[T,QDIM], u(basePos,T).
 * Dispatch `(T, KV_HEADS)`.
 */
export const textQPrep = ({ HEADS = 8, KV_HEADS = 2, HEAD_DIM, ROTARY_PAIRS, MAXCTX }) => {
  const { GROUPS, QDIM } = textAttentionShape({
    HEADS, KV_HEADS, HEAD_DIM, ROTARY_PAIRS, MAXCTX,
  });
  return /* wgsl */`
struct U { basePos: u32, T: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read> qIn: array<f32>;
@group(0) @binding(1) var<storage, read> qNorm: array<f32>;
@group(0) @binding(2) var<storage, read> rope: array<f32>;
@group(0) @binding(3) var<storage, read_write> qOut: array<f32>;
@group(0) @binding(4) var<uniform> u: U;
var<workgroup> qsh: array<f32, ${GROUPS * HEAD_DIM}>;
var<workgroup> red: array<f32, 256>;
const HALF = ${HEAD_DIM / 2}u;
const ROPE = ${ROTARY_PAIRS}u;
const QDIM = ${QDIM}u;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let t = wid.x;
  let kvh = wid.y;
  if (t >= u.T || kvh >= ${KV_HEADS}u) { return; }
  let pos = u.basePos + t;
  let qbase = t * QDIM;
  let ropeBase = pos * (ROPE * 2u);
  for (var g = 0u; g < ${GROUPS}u; g++) {
    let qh = kvh * ${GROUPS}u + g;
    let src = qbase + qh * ${HEAD_DIM}u;
    var ss = 0.0;
    for (var d = lid.x; d < ${HEAD_DIM}u; d += 256u) {
      let v = qIn[src + d];
      ss += v * v;
    }
    red[lid.x] = ss;
    workgroupBarrier();
    var step = 128u;
    loop { if (step == 0u) { break; }
      if (lid.x < step) { red[lid.x] += red[lid.x + step]; }
      workgroupBarrier(); step = step >> 1u;
    }
    let ri = inverseSqrt(red[0] / ${HEAD_DIM}.0 + 1e-6);
    for (var d = lid.x; d < ${HEAD_DIM}u; d += 256u) {
      qsh[g * ${HEAD_DIM}u + d] = qIn[src + d] * ri * qNorm[d];
    }
    workgroupBarrier();
    for (var d = lid.x; d < ${HEAD_DIM}u; d += 256u) {
      var value = qsh[g * ${HEAD_DIM}u + d];
      if (d < ROPE) {
        let c = rope[ropeBase + d];
        let s = rope[ropeBase + ROPE + d];
        value = qsh[g * ${HEAD_DIM}u + d] * c - qsh[g * ${HEAD_DIM}u + d + HALF] * s;
      } else if (d >= HALF && d < HALF + ROPE) {
        let p = d - HALF;
        let c = rope[ropeBase + p];
        let s = rope[ropeBase + ROPE + p];
        value = qsh[g * ${HEAD_DIM}u + d] * c + qsh[g * ${HEAD_DIM}u + p] * s;
      }
      qOut[src + d] = value;
    }
    workgroupBarrier();
  }
}`;
};

/**
 * Causal GQA attention over Gemma's emulated FP8 E4M3FN KV cache.
 *
 * Bindings: q[T,QDIM], kCache, vCache, out[T,QDIM], u(basePos,T).  The
 * `WINDOW` parameter is 512 for a sliding layer and 0 for a full/global layer.
 * It uses online softmax, so global attention does not allocate workgroup
 * memory proportional to context length.  Dispatch `(T, KV_HEADS)`.
 */
export const textCausalAttention = ({
  HEADS = 8,
  KV_HEADS = 2,
  HEAD_DIM,
  MAXCTX,
  WINDOW = 0,
  ATTENTION_SCALE = 1,
}) => {
  const { GROUPS, QDIM, WORDS, CACHE_WORDS } = textAttentionShape({
    HEADS, KV_HEADS, HEAD_DIM, ROTARY_PAIRS: 0, MAXCTX,
  });
  requireInt('WINDOW', WINDOW, 0);
  if (WINDOW > MAXCTX) throw new Error(`WINDOW (${WINDOW}) cannot exceed MAXCTX (${MAXCTX})`);
  return /* wgsl */`
${FP8_E4M3FN}
struct U { basePos: u32, T: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> kCache: array<u32>;
@group(0) @binding(2) var<storage, read> vCache: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> u: U;
var<workgroup> red: array<f32, 256>;
var<workgroup> maxScore: f32;
var<workgroup> denom: f32;
var<workgroup> alpha: f32;
var<workgroup> beta: f32;
const WORDS = ${WORDS}u;
const CACHE_WORDS = ${CACHE_WORDS}u;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let t = wid.x;
  let kvh = wid.y;
  if (t >= u.T || kvh >= ${KV_HEADS}u) { return; }
  let pos = u.basePos + t;
  let start = ${WINDOW === 0
    ? '0u'
    : `select(0u, pos + 1u - ${WINDOW}u, pos + 1u > ${WINDOW}u)`};
  let qbase = t * ${QDIM}u;

  // One KV workgroup produces all Q heads sharing it.  E4B head widths are
  // 256 or 512, so each lane owns at most two output dimensions.
  for (var g = 0u; g < ${GROUPS}u; g++) {
    let qh = kvh * ${GROUPS}u + g;
    let hbase = qbase + qh * ${HEAD_DIM}u;
    var acc0 = 0.0;
    var acc1 = 0.0;
    if (lid.x == 0u) { maxScore = -1.0e30; denom = 0.0; }
    workgroupBarrier();
    for (var p = start; p <= pos; p++) {
      let dataBase = (p * ${KV_HEADS}u + kvh) * WORDS;
      let kScale = bitcast<f32>(kCache[CACHE_WORDS + p * ${KV_HEADS}u + kvh]);
      var dotp = 0.0;
      for (var d = lid.x; d < ${HEAD_DIM}u; d += 256u) {
        let kw = kCache[dataBase + (d >> 2u)];
        dotp += q[hbase + d] * gemma_fp8e4m3fn_decode_lane(kw, d & 3u) * kScale;
      }
      red[lid.x] = dotp;
      workgroupBarrier();
      var step = 128u;
      loop { if (step == 0u) { break; }
        if (lid.x < step) { red[lid.x] += red[lid.x + step]; }
        workgroupBarrier(); step = step >> 1u;
      }
      if (lid.x == 0u) {
        let score = red[0] * ${Number(ATTENTION_SCALE)};
        let nextMax = max(maxScore, score);
        alpha = exp(maxScore - nextMax);
        beta = exp(score - nextMax);
        denom = denom * alpha + beta;
        maxScore = nextMax;
      }
      workgroupBarrier();
      let vScale = bitcast<f32>(vCache[CACHE_WORDS + p * ${KV_HEADS}u + kvh]);
      if (lid.x < ${HEAD_DIM}u) {
        let d = lid.x;
        let vw = vCache[dataBase + (d >> 2u)];
        acc0 = acc0 * alpha + gemma_fp8e4m3fn_decode_lane(vw, d & 3u) * vScale * beta;
      }
      if (lid.x + 256u < ${HEAD_DIM}u) {
        let d = lid.x + 256u;
        let vw = vCache[dataBase + (d >> 2u)];
        acc1 = acc1 * alpha + gemma_fp8e4m3fn_decode_lane(vw, d & 3u) * vScale * beta;
      }
      workgroupBarrier();
    }
    if (lid.x < ${HEAD_DIM}u) { out[hbase + lid.x] = acc0 / denom; }
    if (lid.x + 256u < ${HEAD_DIM}u) { out[hbase + lid.x + 256u] = acc1 / denom; }
    workgroupBarrier();
  }
}`;
};

// ---------------------------------------------------------------------------
// Vision building blocks
//
// The E4B vision tower is a 16-layer, bidirectional 12x64-head encoder.  Its
// input processor can run one unpadded image/frame at a time, which keeps the
// attention kernel below simple and avoids giving padded patches non-zero
// attention probability.

/**
 * Per-head RMSNorm for dense encoder attention.  Bindings: x[rows,heads,dim],
 * gamma[dim], out[rows,heads,dim], u.rows.  Dispatch `(rows, HEADS)`.
 */
export const headRmsnorm = ({ HEADS, HEAD_DIM, WITH_GAMMA = 1, EPS = 1e-6 }) => {
  requireInt('HEADS', HEADS);
  requireInt('HEAD_DIM', HEAD_DIM);
  if (HEAD_DIM > 1024) throw new Error(`head RMSNorm supports HEAD_DIM <= 1024; got ${HEAD_DIM}`);
  return /* wgsl */`
struct U { rows: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> u: U;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let r = wid.x;
  let h = wid.y;
  if (r >= u.rows || h >= ${HEADS}u) { return; }
  let b = (r * ${HEADS}u + h) * ${HEAD_DIM}u;
  var ss = 0.0;
  for (var d = lid.x; d < ${HEAD_DIM}u; d += 256u) { let v = x[b + d]; ss += v * v; }
  red[lid.x] = ss;
  workgroupBarrier();
  var step = 128u;
  loop { if (step == 0u) { break; }
    if (lid.x < step) { red[lid.x] += red[lid.x + step]; }
    workgroupBarrier(); step = step >> 1u;
  }
  let ri = inverseSqrt(red[0] / ${HEAD_DIM}.0 + ${Number(EPS)});
  for (var d = lid.x; d < ${HEAD_DIM}u; d += 256u) {
    // See rmsnorm: bind-group layout must remain stable for WITH_GAMMA=0.
    out[b + d] = x[b + d] * ri * ${WITH_GAMMA ? 'gamma[d]' : '1.0 + 0.0 * gamma[d]'};
  }
}`;
};

/**
 * Apply Gemma vision's 2-D RoPE to separate Q/K arrays.
 *
 * The head is split into x and y halves.  Each half uses the same RoPE
 * frequency range and rotates its own first/second half, exactly matching
 * `apply_multidimensional_rope` in Transformers.  Bindings are qIn, kIn,
 * pixelPositions `[rows][x,y]` as i32, qOut, kOut, u.rows.  Dispatch
 * `ceil(rows * HEADS * HEAD_DIM / 256)`.
 */
export const vision2DRope = ({ HEADS = 12, HEAD_DIM = 64, THETA = 100 }) => {
  requireInt('HEADS', HEADS);
  requireInt('HEAD_DIM', HEAD_DIM);
  if (HEAD_DIM % 4) throw new Error(`vision HEAD_DIM must be divisible by 4; got ${HEAD_DIM}`);
  const axisDim = HEAD_DIM / 2;
  const axisHalf = axisDim / 2;
  return /* wgsl */`
struct U { rows: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> qIn: array<f32>;
@group(0) @binding(1) var<storage, read> kIn: array<f32>;
@group(0) @binding(2) var<storage, read> pixelPositions: array<i32>;
@group(0) @binding(3) var<storage, read_write> qOut: array<f32>;
@group(0) @binding(4) var<storage, read_write> kOut: array<f32>;
@group(0) @binding(5) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  let total = u.rows * ${HEADS * HEAD_DIM}u;
  if (i >= total) { return; }
  let row = i / ${HEADS * HEAD_DIM}u;
  let hd = i % ${HEADS * HEAD_DIM}u;
  let d = hd % ${HEAD_DIM}u;
  let axis = d / ${axisDim}u;
  let local = d % ${axisDim}u;
  let pos = pixelPositions[row * 2u + axis];
  if (pos < 0) {
    qOut[i] = qIn[i];
    kOut[i] = kIn[i];
    return;
  }
  let pair = local % ${axisHalf}u;
  let angle = f32(pos) * pow(${Number(THETA)}, -f32(pair) / ${axisDim}.0);
  let c = cos(angle);
  let s = sin(angle);
  let mateLocal = select(local + ${axisHalf}u, local - ${axisHalf}u, local >= ${axisHalf}u);
  let mate = (hd / ${HEAD_DIM}u) * ${HEAD_DIM}u + axis * ${axisDim}u + mateLocal;
  if (local < ${axisHalf}u) {
    qOut[i] = qIn[i] * c - qIn[row * ${HEADS * HEAD_DIM}u + mate] * s;
    kOut[i] = kIn[i] * c - kIn[row * ${HEADS * HEAD_DIM}u + mate] * s;
  } else {
    qOut[i] = qIn[i] * c + qIn[row * ${HEADS * HEAD_DIM}u + mate] * s;
    kOut[i] = kIn[i] * c + kIn[row * ${HEADS * HEAD_DIM}u + mate] * s;
  }
}`;
};

/**
 * Add the x/y position-table embeddings after the Q4 patch projection.
 * Position table layout is `[2][TABLE_SIZE][H]`; negative positions represent
 * a padded patch.  Bindings: x, positions i32, table, out, u.rows.  Dispatch
 * `(ceil(H / 256), rows)`.
 */
export const addVisionPositions = ({ H = 768, TABLE_SIZE = 10240 }) => {
  requireInt('H', H);
  requireInt('TABLE_SIZE', TABLE_SIZE);
  return /* wgsl */`
struct U { rows: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> positions: array<i32>;
@group(0) @binding(2) var<storage, read> table: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let row = wid.y;
  let d = wid.x * 256u + lid.x;
  if (row >= u.rows || d >= ${H}u) { return; }
  let b = row * ${H}u + d;
  let px = positions[row * 2u];
  let py = positions[row * 2u + 1u];
  var v = x[b];
  if (px >= 0 && py >= 0) {
    v += table[u32(px) * ${H}u + d] + table[${TABLE_SIZE * H}u + u32(py) * ${H}u + d];
  }
  out[b] = v;
}`;
};

/**
 * Q4-table counterpart to `addVisionPositions`.
 *
 * The current Gemma loader quantizes `vision.positionEmb`, whose flattened
 * layout is `[2 * TABLE_SIZE, H]`; bind its `{q,s}` buffers at bindings 2/3.
 * Bindings: x, positions i32, tableQ, tableS, out, u.rows.  Dispatch
 * `(ceil(H / 256), rows)`.
 */
export const addVisionPositionsQ4 = ({ H = 768, TABLE_SIZE = 10240 }) => {
  assertQ4Shape(2 * TABLE_SIZE, H);
  return /* wgsl */`
struct U { rows: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> positions: array<i32>;
@group(0) @binding(2) var<storage, read> tableQ: array<u32>;
@group(0) @binding(3) var<storage, read> tableS: array<u32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> u: U;
const WPR = ${H / 8}u;
const BPR = ${H / 32}u;
fn lookup(row: u32, d: u32) -> f32 {
  let q = f32((tableQ[row * WPR + (d >> 3u)] >> ((d & 7u) * 4u)) & 15u);
  let sm = unpack2x16float(tableS[row * BPR + (d >> 5u)]);
  return sm.x * q + sm.y;
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let row = wid.y;
  let d = wid.x * 256u + lid.x;
  if (row >= u.rows || d >= ${H}u) { return; }
  let px = positions[row * 2u];
  let py = positions[row * 2u + 1u];
  var v = x[row * ${H}u + d];
  if (px >= 0 && py >= 0) {
    v += lookup(u32(px), d) + lookup(${TABLE_SIZE}u + u32(py), d);
  }
  out[row * ${H}u + d] = v;
}`;
};

/**
 * Faithful Gemma4ClippableLinear pre/post operation.  `lo` and `hi` are
 * single-f32 buffers loaded from a tensor's `input_*` or `output_*` scalar.
 * Bindings: src, lo, hi, dst, u.elements.  It is safe to skip this only after
 * verifying the checkpoint's clip scalars are respectively -inf/+inf.
 */
export const clampByScalars = () => /* wgsl */`
struct U { elements: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read> lo: array<f32>;
@group(0) @binding(2) var<storage, read> hi: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
@group(0) @binding(4) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x < u.elements) { dst[gid.x] = clamp(src[gid.x], lo[0], hi[0]); }
}`;

/**
 * Loader-compatible Gemma4ClippableLinear clamp.  The Gemma loader packs
 * `[input_min,input_max,output_min,output_max]` into one f32 buffer, so use
 * `OFFSET=0` before the linear and `OFFSET=2` after it.  Bindings: src,
 * bounds[4], dst, u.elements.
 */
export const clampByBounds = ({ OFFSET = 0 }) => {
  requireInt('OFFSET', OFFSET, 0);
  if (OFFSET > 2) throw new Error('clampByBounds OFFSET must be 0 (input) or 2 (output)');
  return /* wgsl */`
struct U { elements: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read> bounds: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x < u.elements) { dst[gid.x] = clamp(src[gid.x], bounds[${OFFSET}u], bounds[${OFFSET + 1}u]); }
}`;
};

/** Multiply a buffer by one dynamic scalar (e.g. Gemma `layer_scalar`). */
export const scaleByScalar = () => /* wgsl */`
struct U { elements: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read> scalar: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x < u.elements) { dst[gid.x] = src[gid.x] * scalar[0]; }
}`;

/**
 * Dense f32 GQA attention for vision and other fixed encoder sequences.
 *
 * Q is `[rows][HEADS * HEAD_DIM]`; K/V are `[rows][KV_HEADS * HEAD_DIM]`.
 * The default is bidirectional vision attention.  Set `CAUSAL=1` for a
 * generic causal f32 reference path.  `WINDOW=0` means unrestricted; causal
 * windows retain only the latest WINDOW keys.  Bindings: q, k, v, out, u.rows.
 * Dispatch `(rows, KV_HEADS)`.
 */
export const denseAttention = ({
  HEADS,
  KV_HEADS = HEADS,
  HEAD_DIM,
  CAUSAL = 0,
  WINDOW = 0,
  ATTENTION_SCALE = 1,
}) => {
  requireInt('HEADS', HEADS);
  requireInt('KV_HEADS', KV_HEADS);
  requireInt('HEAD_DIM', HEAD_DIM);
  requireInt('WINDOW', WINDOW, 0);
  if (HEADS % KV_HEADS || HEAD_DIM > 512) throw new Error('denseAttention requires GQA heads and HEAD_DIM <= 512');
  const groups = HEADS / KV_HEADS;
  const qdim = HEADS * HEAD_DIM;
  const kvdim = KV_HEADS * HEAD_DIM;
  return /* wgsl */`
struct U { rows: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> u: U;
var<workgroup> red: array<f32, 256>;
var<workgroup> maxScore: f32;
var<workgroup> denom: f32;
var<workgroup> alpha: f32;
var<workgroup> beta: f32;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let t = wid.x;
  let kvh = wid.y;
  if (t >= u.rows || kvh >= ${KV_HEADS}u) { return; }
  let start = ${CAUSAL
    ? (WINDOW === 0 ? '0u' : `select(0u, t + 1u - ${WINDOW}u, t + 1u > ${WINDOW}u)`)
    : '0u'};
  let end = ${CAUSAL ? 't + 1u' : 'u.rows'};
  for (var g = 0u; g < ${groups}u; g++) {
    let qh = kvh * ${groups}u + g;
    let qb = t * ${qdim}u + qh * ${HEAD_DIM}u;
    var acc0 = 0.0;
    var acc1 = 0.0;
    if (lid.x == 0u) { maxScore = -1.0e30; denom = 0.0; }
    workgroupBarrier();
    for (var p = start; p < end; p++) {
      let kb = p * ${kvdim}u + kvh * ${HEAD_DIM}u;
      var dotp = 0.0;
      for (var d = lid.x; d < ${HEAD_DIM}u; d += 256u) { dotp += q[qb + d] * k[kb + d]; }
      red[lid.x] = dotp;
      workgroupBarrier();
      var step = 128u;
      loop { if (step == 0u) { break; }
        if (lid.x < step) { red[lid.x] += red[lid.x + step]; }
        workgroupBarrier(); step = step >> 1u;
      }
      if (lid.x == 0u) {
        let score = red[0] * ${Number(ATTENTION_SCALE)};
        let nextMax = max(maxScore, score);
        alpha = exp(maxScore - nextMax);
        beta = exp(score - nextMax);
        denom = denom * alpha + beta;
        maxScore = nextMax;
      }
      workgroupBarrier();
      let vb = p * ${kvdim}u + kvh * ${HEAD_DIM}u;
      if (lid.x < ${HEAD_DIM}u) { acc0 = acc0 * alpha + v[vb + lid.x] * beta; }
      if (lid.x + 256u < ${HEAD_DIM}u) { acc1 = acc1 * alpha + v[vb + lid.x + 256u] * beta; }
      workgroupBarrier();
    }
    if (lid.x < ${HEAD_DIM}u) { out[qb + lid.x] = acc0 / denom; }
    if (lid.x + 256u < ${HEAD_DIM}u) { out[qb + lid.x + 256u] = acc1 / denom; }
    workgroupBarrier();
  }
}`;
};

/**
 * Spatial average pool for a single unpadded image/frame sequence.
 * x is `[inH * inW][H]`; out is `[outH * outW][H]`, where out dimensions are
 * in dimensions / POOL.  `SCALE` can be set to `sqrt(768)` for Gemma's vision
 * pooler. Bindings: x, out, u(inW,inH,outW,outH). Dispatch
 * `(ceil(H/256), outW*outH)`.
 */
export const visionPool = ({ H = 768, POOL = 3, SCALE = 1 }) => {
  requireInt('H', H);
  requireInt('POOL', POOL);
  return /* wgsl */`
struct U { inW: u32, inH: u32, outW: u32, outH: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let oi = wid.y;
  let d = wid.x * 256u + lid.x;
  if (oi >= u.outW * u.outH || d >= ${H}u) { return; }
  let oy = oi / u.outW;
  let ox = oi % u.outW;
  var sum = 0.0;
  for (var ky = 0u; ky < ${POOL}u; ky++) {
    for (var kx = 0u; kx < ${POOL}u; kx++) {
      let ix = ox * ${POOL}u + kx;
      let iy = oy * ${POOL}u + ky;
      sum += x[(iy * u.inW + ix) * ${H}u + d];
    }
  }
  out[oi * ${H}u + d] = sum * ${Number(SCALE / (POOL * POOL))};
}`;
};

// ---------------------------------------------------------------------------
// Audio building blocks
//
// Audio preprocessing (PCM decoding, 16-kHz resampling, and log-mel STFT) is
// deliberately a browser/CPU concern.  These shaders begin with the
// [mel-frames, 128] feature matrix and cover the Gemma 4 audio tower itself.

/** ReLU in a contiguous f32 buffer. Bindings: src, dst, u.elements. */
export const relu = () => /* wgsl */`
struct U { elements: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x < u.elements) { dst[gid.x] = max(src[gid.x], 0.0); }
}`;

/** SiLU elementwise activation used by Gemma's audio tower. */
export const silu = () => /* wgsl */`
struct U { elements: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x < u.elements) { let x = src[gid.x]; dst[gid.x] = x / (1.0 + exp(-x)); }
}`;

/**
 * 3x3 f32 Conv2d reference for the two audio subsampling layers.
 *
 * Input/output are channels-last `[height][width][channels]`; weights are
 * `[outChannels][inChannels][ky][kx]`, matching a straightforward reshape of
 * PyTorch's `[out,in,3,3]`.  Bindings: input, weight, out,
 * u(inW,inH,inC,outW,outH,outC).  The caller uses STRIDE=2, PAD=1 for E4B.
 */
export const conv2d3x3 = ({ STRIDE = 2, PAD = 1 }) => {
  requireInt('STRIDE', STRIDE);
  requireInt('PAD', PAD, 0);
  return /* wgsl */`
struct U { inW: u32, inH: u32, inC: u32, outW: u32, outH: u32, outC: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let oi = gid.x;
  let total = u.outW * u.outH * u.outC;
  if (oi >= total) { return; }
  let oc = oi % u.outC;
  let spatial = oi / u.outC;
  let ox = spatial % u.outW;
  let oy = spatial / u.outW;
  var sum = 0.0;
  for (var ic = 0u; ic < u.inC; ic++) {
    for (var ky = 0u; ky < 3u; ky++) {
      for (var kx = 0u; kx < 3u; kx++) {
        let ix = i32(ox * ${STRIDE}u + kx) - ${PAD};
        let iy = i32(oy * ${STRIDE}u + ky) - ${PAD};
        if (ix >= 0 && iy >= 0 && ix < i32(u.inW) && iy < i32(u.inH)) {
          let src = (u32(iy) * u.inW + u32(ix)) * u.inC + ic;
          let wi = (((oc * u.inC + ic) * 3u + ky) * 3u + kx);
          sum += input[src] * weight[wi];
        }
      }
    }
  }
  out[oi] = sum;
}`;
};

/**
 * Causal depthwise Conv1d used by Gemma's audio LightConv.
 * Input/output: `[rows][C]`; weights: `[C][K]`.  Dispatch `(ceil(C/256),rows)`.
 */
export const depthwiseConv1d = ({ C = 1024, K = 5 }) => {
  requireInt('C', C);
  requireInt('K', K);
  return /* wgsl */`
struct U { rows: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let t = wid.y;
  let c = wid.x * 256u + lid.x;
  if (t >= u.rows || c >= ${C}u) { return; }
  var sum = 0.0;
  for (var k = 0u; k < ${K}u; k++) {
    let p = i32(t) + i32(k) - ${K - 1};
    if (p >= 0) { sum += input[u32(p) * ${C}u + c] * weight[c * ${K}u + k]; }
  }
  out[t * ${C}u + c] = sum;
}`;
};

/** PyTorch GLU: a * sigmoid(b) for fused `[rows,2*C]` input. */
export const glu = ({ C = 1024 }) => {
  requireInt('C', C);
  return /* wgsl */`
struct U { rows: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let r = wid.y;
  let d = wid.x * 256u + lid.x;
  if (r < u.rows && d < ${C}u) {
    let b = r * ${2 * C}u;
    out[r * ${C}u + d] = src[b + d] / (1.0 + exp(-src[b + ${C}u + d]));
  }
}`;
};

/**
 * Audio attention's q and k scale stage.  `perDimScale` has HEAD_DIM values.
 * qOut = q * Q_SCALE * softplus(perDimScale[d]); kOut = k * K_SCALE.
 * Bindings: q, k, perDimScale, qOut, kOut, u.rows.  Dispatch
 * `(ceil(HEADS*HEAD_DIM/256), rows)`.
 */
export const audioQkScale = ({
  HEADS = 8,
  HEAD_DIM = 128,
  Q_SCALE = 1 / Math.sqrt(128) / Math.log(2),
  K_SCALE = Math.log(1 + Math.E) / Math.log(2),
}) => {
  requireInt('HEADS', HEADS);
  requireInt('HEAD_DIM', HEAD_DIM);
  return /* wgsl */`
${GELU}
struct U { rows: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> perDimScale: array<f32>;
@group(0) @binding(3) var<storage, read_write> qOut: array<f32>;
@group(0) @binding(4) var<storage, read_write> kOut: array<f32>;
@group(0) @binding(5) var<uniform> u: U;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let r = wid.y;
  let d = wid.x * 256u + lid.x;
  let width = ${HEADS * HEAD_DIM}u;
  if (r < u.rows && d < width) {
    qOut[r * width + d] = q[r * width + d] * ${Number(Q_SCALE)} * gemma_softplus(perDimScale[d % ${HEAD_DIM}u]);
    kOut[r * width + d] = k[r * width + d] * ${Number(K_SCALE)};
  }
}`;
};

/**
 * Generate the sinusoidal relative-position matrix consumed by the audio
 * `relative_k_proj`.  Output layout `[REL][HIDDEN]` is
 * `[sin(freq * position), cos(freq * position)]`, with positions REL-1..0.
 * Bindings: out.  Dispatch `ceil(REL*HIDDEN/256)`.
 */
export const audioRelativePositions = ({ HIDDEN = 1024, REL = 13 }) => {
  requireInt('HIDDEN', HIDDEN);
  requireInt('REL', REL);
  if (HIDDEN % 2) throw new Error('audio relative position HIDDEN must be even');
  const half = HIDDEN / 2;
  return /* wgsl */`
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= ${REL * HIDDEN}u) { return; }
  let row = i / ${HIDDEN}u;
  let d = i % ${HIDDEN}u;
  let pair = d % ${half}u;
  let position = ${REL - 1}u - row;
  let angle = f32(position) * exp(-log(10000.0) * f32(pair) / ${Math.max(1, half - 1)}.0);
  out[i] = select(sin(angle), cos(angle), d >= ${half}u);
}`;
};

/**
 * Gemma audio's blocked local relative attention, expressed directly instead
 * of materializing the 5-D mask/relative-shift tensor.
 *
 * Inputs q/k have already passed through `audioQkScale`; v is raw projected
 * values. `relativeK` is `relative_k_proj(audioRelativePositions(...))` with
 * layout `[REL][HEADS * HEAD_DIM]`.  `valid` is one u32 (0/1) per padded row.
 * `u.rows` must be padded to a multiple of CHUNK (the processor naturally
 * produces this when batching); invalid rows have valid=0.  Candidate keys for
 * a block are `[block*CHUNK-LEFT, ..., +CHUNK+RIGHT-1]`; the relative term is
 * applied only to distances -LEFT..0, matching Transformers' rel-shift.
 * Bindings: q,k,v,relativeK,valid,out,u.rows. Dispatch
 * `(rows/CHUNK, HEADS)`.
 */
export const audioChunkAttention = ({
  HEADS = 8,
  HEAD_DIM = 128,
  CHUNK = 12,
  LEFT = 12,
  RIGHT = 0,
  SOFTCAP = 50,
}) => {
  requireInt('HEADS', HEADS);
  requireInt('HEAD_DIM', HEAD_DIM);
  requireInt('CHUNK', CHUNK);
  requireInt('LEFT', LEFT, 0);
  requireInt('RIGHT', RIGHT, 0);
  if (HEAD_DIM > 256) throw new Error('audioChunkAttention currently supports HEAD_DIM <= 256');
  const context = CHUNK + LEFT + RIGHT;
  const rel = LEFT + 1;
  const width = HEADS * HEAD_DIM;
  return /* wgsl */`
struct U { rows: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read> relativeK: array<f32>;
@group(0) @binding(4) var<storage, read> valid: array<u32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@group(0) @binding(6) var<uniform> u: U;
var<workgroup> red: array<f32, 256>;
var<workgroup> maxScore: f32;
var<workgroup> denom: f32;
var<workgroup> alpha: f32;
var<workgroup> beta: f32;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let block = wid.x;
  let head = wid.y;
  if (head >= ${HEADS}u) { return; }
  for (var qi = 0u; qi < ${CHUNK}u; qi++) {
    let qpos = block * ${CHUNK}u + qi;
    // Contract requires padded rows, so every workgroup has addressable rows.
    let qValid = valid[qpos] != 0u;
    let qb = qpos * ${width}u + head * ${HEAD_DIM}u;
    var acc = 0.0;
    if (lid.x == 0u) { maxScore = -1.0e30; denom = 0.0; }
    workgroupBarrier();
    for (var c = 0u; c < ${context}u; c++) {
      let pSigned = i32(block * ${CHUNK}u + c) - ${LEFT};
      var keyValid = false;
      if (pSigned >= 0 && pSigned < i32(u.rows)) { keyValid = valid[u32(pSigned)] != 0u; }
      let isActive = qValid && keyValid;
      var dotp = 0.0;
      if (isActive && lid.x < ${HEAD_DIM}u) {
        let p = u32(pSigned);
        let d = lid.x;
        dotp = q[qb + d] * k[p * ${width}u + head * ${HEAD_DIM}u + d];
        let relIdx = i32(c) - i32(qi);
        if (relIdx >= 0 && relIdx < ${rel}) {
          dotp += q[qb + d] * relativeK[u32(relIdx) * ${width}u + head * ${HEAD_DIM}u + d];
        }
      }
      red[lid.x] = dotp;
      workgroupBarrier();
      var step = 128u;
      loop { if (step == 0u) { break; }
        if (lid.x < step) { red[lid.x] += red[lid.x + step]; }
        workgroupBarrier(); step = step >> 1u;
      }
      if (lid.x == 0u) {
        if (isActive) {
          let raw = red[0];
          let score = ${Number(SOFTCAP)} * tanh(raw / ${Number(SOFTCAP)});
          let nextMax = max(maxScore, score);
          alpha = exp(maxScore - nextMax);
          beta = exp(score - nextMax);
          denom = denom * alpha + beta;
          maxScore = nextMax;
        } else {
          alpha = 1.0;
          beta = 0.0;
        }
      }
      workgroupBarrier();
      if (isActive && lid.x < ${HEAD_DIM}u) {
        let p = u32(pSigned);
        acc = acc * alpha + v[p * ${width}u + head * ${HEAD_DIM}u + lid.x] * beta;
      } else {
        acc *= alpha;
      }
      workgroupBarrier();
    }
    if (lid.x < ${HEAD_DIM}u) {
      out[qb + lid.x] = select(0.0, acc / denom, qValid && denom > 0.0);
    }
    workgroupBarrier();
  }
}`;
};

// A compact machine-readable hand-off for the Gemma runtime.  The detailed
// binding order is also repeated beside every generator above; keeping this
// export lets an integration test assert that a pipeline map exposes every
// architectural stage without parsing comments.
export const GEMMA_KERNEL_CONTRACTS = Object.freeze({
  q4: Object.freeze({
    gemvQ4: 'qdata, scales, x[T,K], out[T,OSTRIDE]; dispatch ceil(N/4),T',
    gemmQ4: 'qdata, scales, x[T,K], out[T,OSTRIDE], u.rows; dispatch ceil(N/64),ceil(T/4)',
    gatherQ4: 'tokens[T], shard qdata/scales, out[T,K], u.rows; clear output before all shards',
  }),
  vector: Object.freeze({
    rmsnorm: 'direct Gemma gamma; set WITH_GAMMA=0 for V/projector norms',
    geluMul: 'Gemma GELU(tanh) gated MLP',
    geluMulPair: 'Gemma PLE gate × per-layer input',
    pleGateMul: 'non-contiguous PLE slice: GELU(gate[T,256]) × ple[T,42,256] at fixed layer',
    pleCombine: 'identity and context PLE rows [tokens*42,256]',
    pleCombineMasked: 'text rows use (identity+context)/sqrt(2); media soft-token rows use context only',
    addBias: 'row-wise learned bias, used after audio output projection',
    softcap: 'final logits = 30*tanh(logits/30)',
    clampByScalars: 'Gemma4ClippableLinear input/output clipping',
    clampByBounds: 'loader-packed [inputMin,inputMax,outputMin,outputMax], OFFSET=0/2',
  }),
  text: Object.freeze({
    textKvPrep: 'QKV [Q|K|V], direct Q/K RMS, V RMS, RoPE, FP8 E4M3FN KV cache',
    textQPrep: 'Q-only direct RMS/RoPE for the declared shared-KV decoder tail',
    textCausalAttention: 'FP8 E4M3FN cached causal GQA; WINDOW=512 local or 0 global',
    e4bLocal: Object.freeze({ heads: 8, kvHeads: 2, headDim: 256, rotaryPairs: 128, window: 512 }),
    e4bGlobal: Object.freeze({ heads: 8, kvHeads: 2, headDim: 512, rotaryPairs: 64, window: 0 }),
  }),
  vision: Object.freeze({
    headRmsnorm: 'per-head direct RMS scale (or unit V norm)',
    vision2DRope: '12 heads × 64 dimensions, theta=100, x/y position pairs',
    denseAttention: 'unpadded bidirectional f32 encoder GQA attention',
    addVisionPositionsQ4: 'quantized [2,10240,768] table from Gemma loader',
    visionPool: 'per-image 3×3 spatial pool, pass SCALE=sqrt(768)',
  }),
  audio: Object.freeze({
    conv2d3x3: 'channels-last stride-2/pad-1 SSCP convolution',
    audioChunkAttention: '12-token chunk, 12 left context, relative attention, softcap=50',
    depthwiseConv1d: 'causal 1024-channel K=5 LightConv',
    silu: 'elementwise audio FFN/LightConv activation',
    audioRelativePositions: '13 × 1024 sin/cos source for relative_k_proj',
  }),
});
