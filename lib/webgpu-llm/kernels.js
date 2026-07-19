// Hand-written WGSL kernels for Qwen3.5-4B hybrid inference.
//
// Conventions:
//  - Activations f32. KV cache & nothing else packed f16 via pack2x16float.
//  - Uniform U { basePos, T } shared by all kernels; tokens in a storage buffer.
//  - GGUF converter-transformed RMSNorm parameters are consumed as stored.

const U_DEF = /*wgsl*/`
struct U { basePos: u32, T: u32, tok0: u32, pad0: u32 }
`;

const SILU = /*wgsl*/`
fn silu(x: f32) -> f32 { return x / (1.0 + exp(-x)); }
fn sigmoid_(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
`;

// int8 KV cache: each cache buffer holds [MAXCTX][4 kv-heads][64] u32 words
// (4 int8 dims per word, symmetric absmax quantization per (position, head)),
// followed by the f32 scales (one per position×head) at u32 offset MAXCTX*256.
// Halves KV memory + attention read bandwidth vs the previous f16x2 layout;
// int8 with per-token-per-head scales is near-lossless for 256-dim heads.
const KVQ = /*wgsl*/`
fn kvq_pack(v: vec4f, qinv: f32) -> u32 {
  let q = vec4i(clamp(round(v * qinv), vec4f(-127.0), vec4f(127.0)));
  return (u32(q.x) & 0xFFu) | ((u32(q.y) & 0xFFu) << 8u) | ((u32(q.z) & 0xFFu) << 16u) | ((u32(q.w) & 0xFFu) << 24u);
}
fn kvq_unpack(w: u32) -> vec4f {
  return vec4f(f32(i32(w << 24u) >> 24u), f32(i32(w << 16u) >> 24u), f32(i32(w << 8u) >> 24u), f32(i32(w) >> 24u));
}
`;

// DROW: input row = posBuf[1]-1 (chained speculation: h of the last accepted token)
export const rmsnorm = ({ K, OSTRIDE = K, OUTOFF = 0, DROW = 0 }) => /*wgsl*/`
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> outb: array<f32>;
${DROW ? '@group(0) @binding(3) var<storage, read> posBuf: array<u32>;' : ''}
var<workgroup> red: array<f32, 256>;
var<workgroup> rinv: f32;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  ${DROW ? `let base = (posBuf[1] - 1u) * ${K}u;` : `let base = wid.x * ${K}u;`}
  let obase = wid.x * ${OSTRIDE}u + ${OUTOFF}u;
  var ss = 0.0;
  var c = lid.x;
  loop { if (c >= ${K}u) { break; } let v = x[base + c]; ss += v * v; c += 256u; }
  red[lid.x] = ss;
  workgroupBarrier();
  var s = 128u;
  loop { if (s == 0u) { break; }
    if (lid.x < s) { red[lid.x] += red[lid.x + s]; }
    workgroupBarrier(); s = s >> 1u;
  }
  if (lid.x == 0u) { rinv = inverseSqrt(red[0] / ${K}.0 + 1e-6); }
  workgroupBarrier();
  c = lid.x;
  loop { if (c >= ${K}u) { break; } outb[obase + c] = x[base + c] * rinv * w[c]; c += 256u; }
}`;

// ---------------------------------------------------------------------------
// GPU-side sampler (batched decode): exact global top-K merge of the per-block
// sorted top-20 lists, then temperature / top-k / top-p sampling — writes the
// token into tokens[0] so the next iteration in the same submit can consume it
// with no CPU round-trip. ctl = [stopFlag, iterCounter].
export const sampler = ({ BLOCKS, KTOP, WIN = 512 }) => /*wgsl*/`
struct SP {
  seed: u32,
  stopCount: u32,
  topK: u32,
  temp: f32,
  topP: f32,
  pp: f32,        // presence penalty (0 = off)
  pad1: u32,
  pad2: u32,
  stopIds: vec4u,
}
@group(0) @binding(0) var<storage, read> cand: array<u32>;      // [BLOCKS][KTOP][2], sorted per block
@group(0) @binding(1) var<uniform> sp: SP;
@group(0) @binding(2) var<storage, read_write> tokens: array<u32>;
@group(0) @binding(3) var<storage, read_write> sampled: array<u32>;
@group(0) @binding(4) var<storage, read_write> ctl: array<u32>; // [flag, counter]
@group(0) @binding(5) var<storage, read_write> recent: array<u32>; // [count, ring[WIN]] generated tokens
var<workgroup> hv: array<f32, 256>;
var<workgroup> hid: array<u32, 256>;
var<workgroup> rv: array<f32, 256>;
var<workgroup> ri: array<u32, 256>;
var<workgroup> winV: array<f32, ${KTOP}>;
var<workgroup> winI: array<u32, ${KTOP}>;
var<workgroup> pen: array<f32, ${KTOP}>;
var<workgroup> ptr: array<u32, 256>;
fn pcg(x: u32) -> u32 {
  var h = x * 747796405u + 2891336453u;
  h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
  return (h >> 22u) ^ h;
}
fn isStop(tok: u32) -> bool {
  return (sp.stopCount > 0u && tok == sp.stopIds.x) ||
         (sp.stopCount > 1u && tok == sp.stopIds.y) ||
         (sp.stopCount > 2u && tok == sp.stopIds.z) ||
         (sp.stopCount > 3u && tok == sp.stopIds.w);
}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3u) {
  let fl = ctl[0];
  let it = ctl[1];
  ptr[lid.x] = 0u;
  workgroupBarrier();
  // exact top-KTOP merge: each thread walks its block's sorted list
  for (var r = 0u; r < ${KTOP}u; r++) {
    var v = -3.0e38;
    var id = 0u;
    if (lid.x < ${BLOCKS}u && ptr[lid.x] < ${KTOP}u) {
      let e = (lid.x * ${KTOP}u + ptr[lid.x]) * 2u;
      id = cand[e];
      v = bitcast<f32>(cand[e + 1u]);
    }
    hv[lid.x] = v; hid[lid.x] = id;
    rv[lid.x] = v; ri[lid.x] = lid.x;
    workgroupBarrier();
    var s = 128u;
    loop { if (s == 0u) { break; }
      if (lid.x < s && rv[lid.x + s] > rv[lid.x]) { rv[lid.x] = rv[lid.x + s]; ri[lid.x] = ri[lid.x + s]; }
      workgroupBarrier(); s = s >> 1u;
    }
    if (lid.x == ri[0]) { ptr[lid.x] += 1u; }
    if (lid.x == 0u) { winV[r] = rv[0]; winI[r] = hid[ri[0]]; }
    workgroupBarrier();
  }
  // presence penalty: one lane per candidate scans the recent-token ring
  if (lid.x < ${KTOP}u) {
    var hit = 0.0;
    if (sp.pp > 0.0) {
      let n = min(recent[0], ${WIN}u);
      let id = winI[lid.x];
      for (var i = 0u; i < n; i++) {
        if (recent[1u + i] == id) { hit = 1.0; break; }
      }
    }
    pen[lid.x] = hit * sp.pp;
  }
  workgroupBarrier();
  if (lid.x == 0u) {
    var tok: u32;
    if (fl == 1u) {
      tok = sp.stopIds.x; // stop already hit earlier in this batch: emit filler
    } else {
      var vals: array<f32, ${KTOP}>;
      var ids: array<u32, ${KTOP}>;
      for (var i = 0u; i < ${KTOP}u; i++) { vals[i] = winV[i] - pen[i]; ids[i] = winI[i]; }
      if (sp.pp > 0.0) {
        // penalties break the sort order; top-k/top-p assume descending
        for (var i = 1u; i < ${KTOP}u; i++) {
          let v = vals[i]; let d = ids[i];
          var j = i;
          loop {
            if (j == 0u) { break; }
            if (vals[j - 1u] >= v) { break; }
            vals[j] = vals[j - 1u]; ids[j] = ids[j - 1u]; j -= 1u;
          }
          vals[j] = v; ids[j] = d;
        }
      }
      if (sp.temp <= 0.0) {
        tok = ids[0];
      } else {
        let k = min(sp.topK, ${KTOP}u);
        var probs: array<f32, ${KTOP}>;
        var sum = 0.0;
        for (var i = 0u; i < k; i++) {
          probs[i] = exp((vals[i] - vals[0]) / sp.temp);
          sum += probs[i];
        }
        var cut = k;
        var cum = 0.0;
        for (var i = 0u; i < k; i++) {
          cum += probs[i] / sum;
          if (cum >= sp.topP) { cut = i + 1u; break; }
        }
        var sub = 0.0;
        for (var i = 0u; i < cut; i++) { sub += probs[i]; }
        var r = (f32(pcg(sp.seed ^ (it * 0x9E3779B9u)) & 0xFFFFFFu) / 16777216.0) * sub;
        tok = ids[cut - 1u];
        for (var i = 0u; i < cut; i++) {
          r -= probs[i];
          if (r <= 0.0) { tok = ids[i]; break; }
        }
      }
    }
    sampled[it] = tok;
    tokens[0] = tok;
    ctl[1] = it + 1u;
    if (isStop(tok)) {
      ctl[0] = 1u;
    } else if (fl == 0u) {
      // record the generated token in the presence window (stops excluded)
      let c = recent[0];
      recent[1u + (c % ${WIN}u)] = tok;
      recent[0] = c + 1u;
    }
  }
}`;

// ---------------------------------------------------------------------------
// GPU-side speculative acceptance (chained rounds): samples from each verify
// row, cascade-compares against the drafts, and advances the GPU position
// state so the next round in the same submit needs no CPU involvement.
// posBuf = [pos, aPrev, mtpExact, snapValid, round, _, _, _]
// ctl    = [stopFlag, _]
// log    = per round [a, next, d0, d1]
// Stop tokens are never fed: a match on a stop-token draft stops acceptance there.
export const acceptSpec = ({ BLOCKS, KTOP, WIN = 512 }) => /*wgsl*/`
struct SP {
  seed: u32,
  stopCount: u32,
  topK: u32,
  temp: f32,
  topP: f32,
  pp: f32,        // presence penalty (0 = off)
  pad1: u32,
  pad2: u32,
  stopIds: vec4u,
}
@group(0) @binding(0) var<storage, read> cand: array<u32>;      // [3][BLOCKS][KTOP][2]
@group(0) @binding(1) var<uniform> sp: SP;
@group(0) @binding(2) var<storage, read> draftIds: array<u32>;  // [2]
@group(0) @binding(3) var<storage, read_write> tokens: array<u32>;
@group(0) @binding(4) var<storage, read_write> posBuf: array<u32>;
@group(0) @binding(5) var<storage, read_write> ctl: array<u32>;
@group(0) @binding(6) var<storage, read_write> logb: array<u32>; // [R][4]
@group(0) @binding(7) var<storage, read_write> recent: array<u32>; // [count, ring[WIN]]
var<workgroup> hv: array<f32, 256>;
var<workgroup> hid: array<u32, 256>;
var<workgroup> rv: array<f32, 256>;
var<workgroup> ri: array<u32, 256>;
var<workgroup> winV: array<f32, ${KTOP}>;
var<workgroup> winI: array<u32, ${KTOP}>;
var<workgroup> pen: array<f32, ${KTOP}>;
var<workgroup> ptr: array<u32, 256>;
var<workgroup> pick: array<u32, 3>;
fn pcg(x: u32) -> u32 {
  var h = x * 747796405u + 2891336453u;
  h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
  return (h >> 22u) ^ h;
}
fn isStop(tok: u32) -> bool {
  return (sp.stopCount > 0u && tok == sp.stopIds.x) ||
         (sp.stopCount > 1u && tok == sp.stopIds.y) ||
         (sp.stopCount > 2u && tok == sp.stopIds.z) ||
         (sp.stopCount > 3u && tok == sp.stopIds.w);
}
const ROWSZ = ${BLOCKS * KTOP * 2}u;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3u) {
  let rnd = posBuf[4];
  for (var row = 0u; row < 3u; row++) {
    ptr[lid.x] = 0u;
    workgroupBarrier();
    for (var r = 0u; r < ${KTOP}u; r++) {
      var v = -3.0e38;
      var id = 0u;
      if (lid.x < ${BLOCKS}u && ptr[lid.x] < ${KTOP}u) {
        let e = row * ROWSZ + (lid.x * ${KTOP}u + ptr[lid.x]) * 2u;
        id = cand[e];
        v = bitcast<f32>(cand[e + 1u]);
      }
      hv[lid.x] = v; hid[lid.x] = id;
      rv[lid.x] = v; ri[lid.x] = lid.x;
      workgroupBarrier();
      var s = 128u;
      loop { if (s == 0u) { break; }
        if (lid.x < s && rv[lid.x + s] > rv[lid.x]) { rv[lid.x] = rv[lid.x + s]; ri[lid.x] = ri[lid.x + s]; }
        workgroupBarrier(); s = s >> 1u;
      }
      if (lid.x == ri[0]) { ptr[lid.x] += 1u; }
      if (lid.x == 0u) { winV[r] = rv[0]; winI[r] = hid[ri[0]]; }
      workgroupBarrier();
    }
    // presence penalty for this row's candidates (ring state as of round start)
    if (lid.x < ${KTOP}u) {
      var hit = 0.0;
      if (sp.pp > 0.0) {
        let n = min(recent[0], ${WIN}u);
        let id = winI[lid.x];
        for (var i = 0u; i < n; i++) {
          if (recent[1u + i] == id) { hit = 1.0; break; }
        }
      }
      pen[lid.x] = hit * sp.pp;
    }
    workgroupBarrier();
    if (lid.x == 0u) {
      var tok: u32;
      var vals: array<f32, ${KTOP}>;
      var ids: array<u32, ${KTOP}>;
      for (var i = 0u; i < ${KTOP}u; i++) { vals[i] = winV[i] - pen[i]; ids[i] = winI[i]; }
      if (sp.pp > 0.0) {
        for (var i = 1u; i < ${KTOP}u; i++) {
          let v = vals[i]; let d = ids[i];
          var j = i;
          loop {
            if (j == 0u) { break; }
            if (vals[j - 1u] >= v) { break; }
            vals[j] = vals[j - 1u]; ids[j] = ids[j - 1u]; j -= 1u;
          }
          vals[j] = v; ids[j] = d;
        }
      }
      if (sp.temp <= 0.0) {
        tok = ids[0];
      } else {
        let k = min(sp.topK, ${KTOP}u);
        var probs: array<f32, ${KTOP}>;
        var sum = 0.0;
        for (var i = 0u; i < k; i++) { probs[i] = exp((vals[i] - vals[0]) / sp.temp); sum += probs[i]; }
        var cut = k;
        var cum = 0.0;
        for (var i = 0u; i < k; i++) { cum += probs[i] / sum; if (cum >= sp.topP) { cut = i + 1u; break; } }
        var sub = 0.0;
        for (var i = 0u; i < cut; i++) { sub += probs[i]; }
        var rr = (f32(pcg(sp.seed ^ (rnd * 0x9E3779B9u) ^ (row * 0x85EBCA6Bu)) & 0xFFFFFFu) / 16777216.0) * sub;
        tok = ids[cut - 1u];
        for (var i = 0u; i < cut; i++) { rr -= probs[i]; if (rr <= 0.0) { tok = ids[i]; break; } }
      }
      pick[row] = tok;
    }
    workgroupBarrier();
  }
  if (lid.x == 0u) {
    let frozen = ctl[0];
    var a = 1u;
    var next = pick[0];
    if (next == draftIds[0] && !isStop(next)) {
      a = 2u;
      next = pick[1];
      if (next == draftIds[1] && !isStop(next)) {
        a = 3u;
        next = pick[2];
      }
    }
    if (frozen == 0u) {
      logb[rnd * 4u] = a;
      logb[rnd * 4u + 1u] = next;
      logb[rnd * 4u + 2u] = draftIds[0];
      logb[rnd * 4u + 3u] = draftIds[1];
      posBuf[2] = posBuf[0];       // mtpExact for the next round's catch-up
      posBuf[0] = posBuf[0] + a;
      posBuf[1] = a;
      posBuf[3] = select(0u, 1u, a < 3u); // partial accept → read snapshots next round
      // record this round's generated tokens (pick[0..a-1]) in the presence
      // window; next == pick[a-1]. Stop tokens are excluded.
      for (var i = 0u; i < a; i++) {
        let tk = pick[i];
        if (!isStop(tk)) {
          let c = recent[0];
          recent[1u + (c % ${WIN}u)] = tk;
          recent[0] = c + 1u;
        }
      }
      if (isStop(next)) { ctl[0] = 1u; }
    } else {
      logb[rnd * 4u] = 0u;         // frozen round: nothing committed
      logb[rnd * 4u + 1u] = sp.stopIds.x;
      logb[rnd * 4u + 2u] = 0u;
      logb[rnd * 4u + 3u] = 0u;
    }
    tokens[0] = next;
    posBuf[4] = rnd + 1u;
  }
}`;

// Merge per-block top-1 candidates from the topk kernel into a drafted token:
// writes tokens[SLOT] (feeds the verify chunk) and draftIds[SLOT-1] (for readback).
export const argmaxMerge = ({ SLOT, BLOCKS, KTOP }) => /*wgsl*/`
@group(0) @binding(0) var<storage, read> cand: array<u32>; // [BLOCKS][KTOP][2]
@group(0) @binding(1) var<storage, read_write> tokens: array<u32>;
@group(0) @binding(2) var<storage, read_write> draftIds: array<u32>;
var<workgroup> rv: array<f32, 256>;
var<workgroup> ri: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3u) {
  var bv = -3.0e38;
  var bi = 0u;
  var b = lid.x;
  loop {
    if (b >= ${BLOCKS}u) { break; }
    let v = bitcast<f32>(cand[b * ${KTOP * 2}u + 1u]); // block's top-1 value
    if (v > bv) { bv = v; bi = cand[b * ${KTOP * 2}u]; }
    b += 256u;
  }
  rv[lid.x] = bv; ri[lid.x] = bi;
  workgroupBarrier();
  var s = 128u;
  loop { if (s == 0u) { break; }
    if (lid.x < s && rv[lid.x + s] > rv[lid.x]) { rv[lid.x] = rv[lid.x + s]; ri[lid.x] = ri[lid.x + s]; }
    workgroupBarrier(); s = s >> 1u;
  }
  if (lid.x == 0u) {
    tokens[${SLOT}u] = ri[0];
    draftIds[${SLOT - 1}u] = ri[0];
  }
}`;


// ---------------------------------------------------------------------------
// Causal depthwise conv1d (kernel 4) + SiLU over the 8192 qkv channels.
// Input is the fused projection buffer [T][12352] (first 8192 channels).
// state holds the 3 inputs before this chunk: state[j][c], j=0 oldest.
// SHIFT=1 (decode, T==1): the same thread also advances the state — safe,
// since each channel is owned by exactly one thread.
export const conv1d = ({ SHIFT = 0, SNAP = 0, FLAG = 0, LAZY = 0, INSTR = 12352 }) => /*wgsl*/`
${SILU}
@group(0) @binding(0) var<storage, read> cw: array<f32>;      // [8192][4]
@group(0) @binding(1) var<storage, read_write> state: array<f32>;   // [3][8192]
@group(0) @binding(2) var<storage, read> inb: array<f32>;     // [T][${'' + 12352}]
@group(0) @binding(3) var<storage, read_write> outb: array<f32>;
${SNAP ? '@group(0) @binding(4) var<storage, read_write> snap: array<f32>; // [T][3][8192]' : ''}
${FLAG ? '@group(0) @binding(4) var<storage, read> ctl: array<u32>; // [stopFlag, _]: freeze state after stop' : ''}
${LAZY ? `@group(0) @binding(5) var<storage, read> posBuf: array<u32>;
@group(0) @binding(6) var<storage, read> ctl: array<u32>;` : ''}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let c = wid.x * 256u + lid.x;
  let t = wid.y;
  var acc = 0.0;
  ${LAZY ? 'let useSnap = posBuf[3] == 1u; let snapBase = (posBuf[1] - 1u) * 24576u;' : ''}
  ${SHIFT || SNAP ? 'var last: array<f32, 3>;' : ''}
  for (var j = 0u; j < 4u; j++) {
    let tau = i32(t) + i32(j) - 3;
    var v: f32;
    if (tau < 0) {
      ${LAZY
        ? 'if (useSnap) { v = snap[snapBase + u32(3 + tau) * 8192u + c]; } else { v = state[u32(3 + tau) * 8192u + c]; }'
        : 'v = state[u32(3 + tau) * 8192u + c];'}
    }
    else { v = inb[u32(tau) * ${INSTR}u + c]; }
    acc += cw[c * 4u + j] * v;
    ${SHIFT || SNAP ? 'if (j > 0u) { last[j - 1u] = v; }' : ''}
  }
  outb[t * 8192u + c] = silu(acc);
  ${SHIFT ? (FLAG
    ? 'if (ctl[0] == 0u) { for (var j = 0u; j < 3u; j++) { state[j * 8192u + c] = last[j]; } }'
    : 'for (var j = 0u; j < 3u; j++) { state[j * 8192u + c] = last[j]; }') : ''}
  ${SNAP ? `if (t < 2u ${LAZY ? '&& ctl[0] == 0u' : ''}) { for (var j = 0u; j < 3u; j++) { snap[t * 24576u + j * 8192u + c] = last[j]; } }` : ''}
}`;

// Shift conv state forward by T tokens (prefill path; reads old values per-thread).
// TBAKE=3 + FLAG: chained-speculation variant (no uniform, stop-frozen).
export const convShift = ({ INSTR = 12352, TBAKE = 0, FLAG = 0 }) => /*wgsl*/`
${TBAKE ? '' : U_DEF}
@group(0) @binding(0) var<storage, read> inb: array<f32>;     // [T][12352]
@group(0) @binding(1) var<storage, read_write> state: array<f32>;
${TBAKE
  ? (FLAG ? '@group(0) @binding(2) var<storage, read> ctl: array<u32>;' : '')
  : '@group(0) @binding(2) var<uniform> u: U;'}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let c = wid.x * 256u + lid.x;
  var nv: array<f32, 3>;
  for (var j = 0u; j < 3u; j++) {
    let tau = ${TBAKE ? `${TBAKE}` : 'i32(u.T)'} - 3 + i32(j);
    if (tau < 0) { nv[j] = state[u32(3 + tau) * 8192u + c]; }
    else { nv[j] = inb[u32(tau) * ${INSTR}u + c]; }
  }
  ${FLAG ? 'if (ctl[0] != 0u) { return; }' : ''}
  for (var j = 0u; j < 3u; j++) { state[j * 8192u + c] = nv[j]; }
}`;

// ---------------------------------------------------------------------------
// DeltaNet prep: L2-normalize q,k per 128-dim head (q also scaled by 1/sqrt(128)),
// and compute per-v-head beta = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias).
// gb layout: [T][ beta(32) | g(32) ].
export const deltaPrep = () => /*wgsl*/`
@group(0) @binding(0) var<storage, read> qkvc: array<f32>;  // [T][8192] post-conv
@group(0) @binding(1) var<storage, read> ba: array<f32>;    // [T][12352]: b at 12288, a at 12320
@group(0) @binding(2) var<storage, read> adt: array<f32>;   // [ -exp(A_log)(32) | dt_bias(32) ]
@group(0) @binding(3) var<storage, read_write> qn: array<f32>; // [T][16][128]
@group(0) @binding(4) var<storage, read_write> kn: array<f32>; // [T][16][128]
@group(0) @binding(5) var<storage, read_write> gb: array<f32>; // [T][64]
var<workgroup> red: array<f32, 128>;
var<workgroup> rinv: f32;
fn softplus_(x: f32) -> f32 { return select(log(1.0 + exp(x)), x, x > 20.0); }
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let t = wid.x;
  let s = wid.y; // 0..15 q heads, 16..31 k heads
  let isQ = s < 16u;
  let head = select(s - 16u, s, isQ);
  let src = t * 8192u + select(2048u, 0u, isQ) + head * 128u + lid.x;
  let v = qkvc[src];
  red[lid.x] = v * v;
  workgroupBarrier();
  var st = 64u;
  loop { if (st == 0u) { break; }
    if (lid.x < st) { red[lid.x] += red[lid.x + st]; }
    workgroupBarrier(); st = st >> 1u;
  }
  if (lid.x == 0u) { rinv = inverseSqrt(red[0] + 1e-6); } // FLA l2norm: sum, not mean
  workgroupBarrier();
  if (isQ) { qn[t * 2048u + head * 128u + lid.x] = v * rinv * ${1 / Math.sqrt(128)}; }
  else { kn[t * 2048u + head * 128u + lid.x] = v * rinv; }
  if (s == 0u && lid.x < 32u) { gb[t * 64u + lid.x] = 1.0 / (1.0 + exp(-ba[t * 12352u + 12288u + lid.x])); }
  if (s == 1u && lid.x < 32u) {
    gb[t * 64u + 32u + lid.x] = adt[lid.x] * softplus_(ba[t * 12352u + 12320u + lid.x] + adt[32u + lid.x]);
  }
}`;

// ---------------------------------------------------------------------------
// Gated DeltaNet recurrence. One workgroup = (head, 16-wide v slice); the state
// slice S[128,16] lives in shared memory; sequential over chunk tokens.
//   S = S * exp(g);  kv = S^T k;  delta = (v - kv) * beta;  S += k delta^T;  o = S^T q
// State layout is [head][slice(8)][k(128)][v(16)] — the slice a workgroup owns is
// contiguous (8 KB), so loads/stores are fully coalesced.
// PREP=1 (decode, T==1): computes q/k L2-norm and g/beta inline from the post-conv
// projections instead of reading the deltaPrep outputs — one dispatch fewer.
// LAZY (chained speculation): T baked to 3, no uniform; initial state is read
// from the previous round's snapshot (slot aPrev-1) when posBuf[3] says the last
// accept was partial; writes are frozen after a stop token (ctl[0]).
export const deltaRule = ({ PREP = 0, SNAP = 0, FLAG = 0, LAZY = 0 }) => /*wgsl*/`
${U_DEF}
@group(0) @binding(0) var<storage, read_write> state: array<f32>; // [32][8][128][16]
${PREP ? /*wgsl*/`
@group(0) @binding(1) var<storage, read> adt: array<f32>;  // [-exp(A_log)(32) | dt_bias(32)]
@group(0) @binding(2) var<storage, read> qkvc: array<f32>; // [T][8192]: q|k|v post-conv
@group(0) @binding(3) var<storage, read> qkvz: array<f32>; // [T][12352]: b@12288, a@12320
@group(0) @binding(4) var<storage, read_write> outb: array<f32>; // [T][32][128]
${LAZY ? '' : '@group(0) @binding(5) var<uniform> u: U;'}
` : /*wgsl*/`
@group(0) @binding(1) var<storage, read> qn: array<f32>;   // [T][16][128] (pre-scaled)
@group(0) @binding(2) var<storage, read> kn: array<f32>;   // [T][16][128]
@group(0) @binding(3) var<storage, read> qkvc: array<f32>; // [T][8192]: q|k|v post-conv
@group(0) @binding(4) var<storage, read> gb: array<f32>;   // [T][beta32|g32]
@group(0) @binding(5) var<storage, read_write> outb: array<f32>; // [T][32][128]
@group(0) @binding(6) var<uniform> u: U;
`}
${SNAP ? `@group(0) @binding(${PREP ? (LAZY ? 5 : 6) : 7}) var<storage, read_write> snap: array<f32>; // [T][32][8][128][16]` : ''}
${FLAG ? '@group(0) @binding(6) var<storage, read> ctl: array<u32>; // [stopFlag, _]: freeze state after stop' : ''}
${LAZY ? `@group(0) @binding(6) var<storage, read> posBuf: array<u32>;
@group(0) @binding(7) var<storage, read> ctl: array<u32>;` : ''}
var<workgroup> Sl: array<f32, 2048>;   // [k 128][v 16]
var<workgroup> ksh: array<f32, 128>;
var<workgroup> qsh: array<f32, 128>;
var<workgroup> stag: array<f32, 256>;
var<workgroup> dsh: array<f32, 16>;
var<workgroup> vsh: array<f32, 16>;
var<workgroup> sc: array<f32, 2>;      // [g_exp, beta]
fn softplus_(x: f32) -> f32 { return select(log(1.0 + exp(x)), x, x > 20.0); }
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let h = wid.x >> 3u;
  let sl8 = wid.x & 7u;
  let v0 = sl8 * 16u;
  let vl = lid.x & 15u;
  let kg = lid.x >> 4u; // 0..15, owns k rows kg*8..kg*8+8
  let sbase = h * 16384u + sl8 * 2048u;
${LAZY ? /*wgsl*/`
  // lazy restore: partial accept last round → resume from that snapshot
  let useSnap = posBuf[3] == 1u;
  let snapBase = (posBuf[1] - 1u) * 524288u + sbase;
  for (var j = 0u; j < 8u; j++) {
    let k = kg * 8u + j;
    if (useSnap) { Sl[k * 16u + vl] = snap[snapBase + k * 16u + vl]; }
    else { Sl[k * 16u + vl] = state[sbase + k * 16u + vl]; }
  }
` : /*wgsl*/`
  for (var j = 0u; j < 8u; j++) {
    let k = kg * 8u + j;
    Sl[k * 16u + vl] = state[sbase + k * 16u + vl];
  }
`}
  // GGUF stores V heads tiled by repeat slot: [K0..K15, K0..K15].
  // The converter applies the same permutation to V/Z/A/B/conv/out tensors,
  // so recurrent state can remain in this order and only the K-head lookup
  // changes from grouped h/2 to tiled h%16.
  let kh = h & 15u;
  for (var t = 0u; t < ${LAZY ? '3u' : 'u.T'}; t++) {
${PREP ? /*wgsl*/`
    { // inline prep: L2-norm q/k of k-head kh, g/beta of v-head h
      let isQ = lid.x < 128u;
      let d = lid.x & 127u;
      let src = t * 8192u + select(2048u, 0u, isQ) + kh * 128u + d;
      let raw = qkvc[src];
      stag[lid.x] = raw * raw;
      workgroupBarrier();
      var s = 64u;
      loop { if (s == 0u) { break; }
        if (d < s) { stag[lid.x] += stag[lid.x + s]; }
        workgroupBarrier(); s = s >> 1u;
      }
      if (isQ) { qsh[d] = raw * inverseSqrt(stag[0] + 1e-6) * ${1 / Math.sqrt(128)}; }
      else { ksh[d] = raw * inverseSqrt(stag[128] + 1e-6); }
      if (lid.x < 16u) { vsh[lid.x] = qkvc[t * 8192u + 4096u + h * 128u + v0 + lid.x]; }
      if (lid.x == 254u) { sc[0] = exp(adt[h] * softplus_(qkvz[t * 12352u + 12320u + h] + adt[32u + h])); }
      if (lid.x == 255u) { sc[1] = 1.0 / (1.0 + exp(-qkvz[t * 12352u + 12288u + h])); }
    }
` : /*wgsl*/`
    if (lid.x < 128u) {
      ksh[lid.x] = kn[t * 2048u + kh * 128u + lid.x];
      qsh[lid.x] = qn[t * 2048u + kh * 128u + lid.x];
    } else if (lid.x < 144u) {
      vsh[lid.x - 128u] = qkvc[t * 8192u + 4096u + h * 128u + v0 + (lid.x - 128u)];
    } else if (lid.x == 254u) {
      sc[0] = exp(gb[t * 64u + 32u + h]);
    } else if (lid.x == 255u) {
      sc[1] = gb[t * 64u + h];
    }
`}
    workgroupBarrier();
    var p = 0.0;
    for (var j = 0u; j < 8u; j++) {
      let k = kg * 8u + j;
      p += Sl[k * 16u + vl] * ksh[k];
    }
    stag[lid.x] = p;
    workgroupBarrier();
    if (lid.x < 16u) {
      var kv = 0.0;
      for (var g = 0u; g < 16u; g++) { kv += stag[g * 16u + lid.x]; }
      dsh[lid.x] = (vsh[lid.x] - sc[0] * kv) * sc[1];
    }
    workgroupBarrier();
    var p2 = 0.0;
    let ge = sc[0];
    let d = dsh[vl];
    for (var j = 0u; j < 8u; j++) {
      let k = kg * 8u + j;
      let idx = k * 16u + vl;
      let sn = Sl[idx] * ge + ksh[k] * d;
      Sl[idx] = sn;
      p2 += sn * qsh[k];
    }
    stag[lid.x] = p2;
    workgroupBarrier();
    if (lid.x < 16u) {
      var o = 0.0;
      for (var g = 0u; g < 16u; g++) { o += stag[g * 16u + lid.x]; }
      outb[t * 4096u + h * 128u + v0 + lid.x] = o;
    }
${SNAP ? /*wgsl*/`
    // slot 2 is never read (2 == full accept keeps the live state)
    if (t < 2u ${LAZY ? '&& ctl[0] == 0u' : ''}) {
      for (var j = 0u; j < 8u; j++) {
        let k = kg * 8u + j;
        snap[t * 524288u + sbase + k * 16u + vl] = Sl[k * 16u + vl];
      }
    }
` : ''}
    workgroupBarrier();
  }
  ${FLAG || LAZY ? 'if (ctl[0] != 0u) { return; } // stop hit earlier in this batch: keep state frozen' : ''}
  for (var j = 0u; j < 8u; j++) {
    let k = kg * 8u + j;
    state[sbase + k * 16u + vl] = Sl[k * 16u + vl];
  }
}`;

// ---------------------------------------------------------------------------
// Gated RMSNorm per 128-dim v-head: y = rms(x)*w * silu(z). (weight NOT zero-centered)
export const gatedNorm = () => /*wgsl*/`
${SILU}
@group(0) @binding(0) var<storage, read> x: array<f32>;   // [T][4096]
@group(0) @binding(1) var<storage, read> z: array<f32>;   // [T][12352], z at offset 8192
@group(0) @binding(2) var<storage, read> w: array<f32>;   // [128]
@group(0) @binding(3) var<storage, read_write> outb: array<f32>;
var<workgroup> red: array<f32, 128>;
var<workgroup> rinv: f32;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let base = wid.x * 4096u + wid.y * 128u;
  let v = x[base + lid.x];
  red[lid.x] = v * v;
  workgroupBarrier();
  var s = 64u;
  loop { if (s == 0u) { break; }
    if (lid.x < s) { red[lid.x] += red[lid.x + s]; }
    workgroupBarrier(); s = s >> 1u;
  }
  if (lid.x == 0u) { rinv = inverseSqrt(red[0] / 128.0 + 1e-6); }
  workgroupBarrier();
  let zv = z[wid.x * 12352u + 8192u + wid.y * 128u + lid.x];
  outb[base + lid.x] = v * rinv * w[lid.x] * silu(zv);
}`;

// ---------------------------------------------------------------------------
// Full-attention prep: per-head RMSNorm(q/k) [zero-centered w pre-added],
// partial RoPE (first 64 dims, rotate-half), write q to qr, pack k/v into caches.
// Slices: 0..15 q heads | 16..19 k heads | 20..23 v heads.
// SPOS modes for GPU-chained speculation:
//   'verify'  — pos = posBuf[0] + t
//   'catchup' — pos = posBuf[2] + J (baked); cache writes only when J < aPrev-1
export const attnPrep = ({ MAXCTX, SPOS = 0, J = 0 }) => /*wgsl*/`
${U_DEF}
${KVQ}
@group(0) @binding(0) var<storage, read> qg: array<f32>;     // [T][10240]: q&gate | k | v
@group(0) @binding(1) var<storage, read> qw: array<f32>;     // [256] (1+gamma)
@group(0) @binding(2) var<storage, read> kw: array<f32>;     // [256]
@group(0) @binding(3) var<storage, read> rope: array<f32>;   // [MAXCTX][cos32|sin32]
@group(0) @binding(4) var<storage, read_write> qr: array<f32>;      // [T][16][256]
@group(0) @binding(5) var<storage, read_write> kcache: array<u32>;  // [MAXCTX][4][64] i8x4 + scales
@group(0) @binding(6) var<storage, read_write> vcache: array<u32>;
${SPOS
  ? '@group(0) @binding(7) var<storage, read> posBuf: array<u32>;'
  : '@group(0) @binding(7) var<uniform> u: U;'}
var<workgroup> arr: array<f32, 256>;
var<workgroup> red: array<f32, 256>;
var<workgroup> rinv: f32;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let t = wid.x;
${SPOS ? '' : /*wgsl*/`
  // MTP prefill sets u.tok0=1 to skip row 0 (its slot would be position -1)
  if (u.tok0 == 1u && t == 0u) { return; }
`}
  let s = wid.y;
  ${{
    0: 'let pos = u.basePos + t;',
    verify: 'let pos = posBuf[0] + t;',
    catchup: `let pos = posBuf[2] + ${J}u;
  let cuValid = ${J}u + 1u < posBuf[1]; // only aPrev-1 catch-up entries are real`,
  }[SPOS || 0]}
  let d = lid.x;
  var v: f32;
  if (s < 16u) { v = qg[t * 10240u + s * 512u + d]; }
  else if (s < 20u) { v = qg[t * 10240u + 8192u + (s - 16u) * 256u + d]; }
  else { v = qg[t * 10240u + 9216u + (s - 20u) * 256u + d]; }

  if (s < 20u) {
    // RMSNorm over 256 dims
    red[d] = v * v;
    workgroupBarrier();
    var st = 128u;
    loop { if (st == 0u) { break; }
      if (d < st) { red[d] += red[d + st]; }
      workgroupBarrier(); st = st >> 1u;
    }
    if (d == 0u) { rinv = inverseSqrt(red[0] / 256.0 + 1e-6); }
    workgroupBarrier();
    v = v * rinv * select(kw[d], qw[d], s < 16u);
    arr[d] = v;
    workgroupBarrier();
    // partial RoPE on dims [0,64): pair (i, i+32), rotate-half convention
    if (d < 64u) {
      let fi = d & 31u;
      let c = rope[pos * 64u + fi];
      let sn = rope[pos * 64u + 32u + fi];
      if (d < 32u) { v = arr[d] * c - arr[d + 32u] * sn; }
      else { v = arr[d] * c + arr[d - 32u] * sn; }
    }
    workgroupBarrier();
    arr[d] = v;
    workgroupBarrier();
  } else {
    arr[d] = v;
    workgroupBarrier();
  }

  if (s < 16u) {
    qr[t * 4096u + s * 256u + d] = arr[d];
  } else {
    // int8-quantize the 256-dim k/v vector: absmax → per-(pos, head) scale
    red[d] = abs(arr[d]);
    workgroupBarrier();
    var qs = 128u;
    loop { if (qs == 0u) { break; }
      if (d < qs) { red[d] = max(red[d], red[d + qs]); }
      workgroupBarrier(); qs = qs >> 1u;
    }
    let amax = red[0];
    let qinv = select(0.0, 127.0 / amax, amax > 0.0);
    let head = s - select(20u, 16u, s < 20u);
    if (d < 64u ${SPOS === 'catchup' ? '&& cuValid' : ''}) {
      let word = kvq_pack(vec4f(arr[d * 4u], arr[d * 4u + 1u], arr[d * 4u + 2u], arr[d * 4u + 3u]), qinv);
      let base = (pos * 4u + head) * 64u + d;
      if (s < 20u) { kcache[base] = word; } else { vcache[base] = word; }
    }
    if (d == 0u ${SPOS === 'catchup' ? '&& cuValid' : ''}) {
      let si = ${MAXCTX * 256}u + pos * 4u + head;
      let sc = bitcast<u32>(amax / 127.0);
      if (s < 20u) { kcache[si] = sc; } else { vcache[si] = sc; }
    }
  }
}`;

// ---------------------------------------------------------------------------
// Streaming causal attention (decode & prefill), one workgroup per (t, kv-head),
// 4 query heads per kv head processed together; online softmax over 512-pos chunks.
// Epilogue applies the per-head sigmoid output gate from q_proj.
export const attention = ({ MAXCTX, SPOS = 0 }) => /*wgsl*/`
${U_DEF}
${SILU}
${KVQ}
@group(0) @binding(0) var<storage, read> qr: array<f32>;      // [T][16][256]
@group(0) @binding(1) var<storage, read> kcache: array<u32>;  // [MAXCTX][4][64] i8x4 + scales
@group(0) @binding(2) var<storage, read> vcache: array<u32>;
@group(0) @binding(3) var<storage, read> qg: array<f32>;      // [T][10240], gates at h*512+256
@group(0) @binding(4) var<storage, read_write> outb: array<f32>; // [T][16][256]
${SPOS
  ? '@group(0) @binding(5) var<storage, read> posBuf: array<u32>;'
  : '@group(0) @binding(5) var<uniform> u: U;'}
var<workgroup> qsh: array<f32, 1024>;     // 4 q heads x 256
var<workgroup> scores: array<f32, 2048>;  // 512 pos x 4 heads
var<workgroup> red4: array<vec4f, 256>;
var<workgroup> mnew: vec4f;
var<workgroup> lchunk: vec4f;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let t = wid.x;
  let kvh = wid.y;
  ${SPOS ? 'let npos = posBuf[0] + t + 1u;' : 'let npos = u.basePos + t + 1u;'}
  for (var e = 0u; e < 4u; e++) {
    let idx = e * 256u + lid.x;
    qsh[idx] = qr[t * 4096u + kvh * 1024u + idx];
  }
  var m = vec4f(-3.0e38);
  var l = vec4f(0.0);
  var acc = vec4f(0.0);
  let dword = lid.x >> 2u;                    // this thread's dim → packed word
  let dshift = (3u - (lid.x & 3u)) * 8u;      // …and sign-extension shift
  var p0 = 0u;
  loop {
    if (p0 >= npos) { break; }
    workgroupBarrier();
    // scores for up to 512 positions x 4 q heads; each thread does 2 positions
    for (var e = 0u; e < 2u; e++) {
      let pl = e * 256u + lid.x;
      let p = p0 + pl;
      var sv = vec4f(-3.0e38);
      if (p < npos) {
        sv = vec4f(0.0);
        let kbase = (p * 4u + kvh) * 64u;
        for (var w = 0u; w < 64u; w++) {
          let kk = kvq_unpack(kcache[kbase + w]);
          let d0 = w * 4u;
          sv += kk.x * vec4f(qsh[d0], qsh[256u + d0], qsh[512u + d0], qsh[768u + d0]);
          sv += kk.y * vec4f(qsh[d0 + 1u], qsh[256u + d0 + 1u], qsh[512u + d0 + 1u], qsh[768u + d0 + 1u]);
          sv += kk.z * vec4f(qsh[d0 + 2u], qsh[256u + d0 + 2u], qsh[512u + d0 + 2u], qsh[768u + d0 + 2u]);
          sv += kk.w * vec4f(qsh[d0 + 3u], qsh[256u + d0 + 3u], qsh[512u + d0 + 3u], qsh[768u + d0 + 3u]);
        }
        // dequant scale folded with 1/sqrt(256)
        sv *= bitcast<f32>(kcache[${MAXCTX * 256}u + p * 4u + kvh]) * 0.0625;
      }
      scores[pl * 4u] = sv.x; scores[pl * 4u + 1u] = sv.y; scores[pl * 4u + 2u] = sv.z; scores[pl * 4u + 3u] = sv.w;
    }
    workgroupBarrier();
    // chunk max (vec4 over q heads)
    {
      let a = vec4f(scores[(lid.x * 2u) * 4u], scores[(lid.x * 2u) * 4u + 1u], scores[(lid.x * 2u) * 4u + 2u], scores[(lid.x * 2u) * 4u + 3u]);
      let b = vec4f(scores[(lid.x * 2u + 1u) * 4u], scores[(lid.x * 2u + 1u) * 4u + 1u], scores[(lid.x * 2u + 1u) * 4u + 2u], scores[(lid.x * 2u + 1u) * 4u + 3u]);
      red4[lid.x] = max(a, b);
    }
    workgroupBarrier();
    var st = 128u;
    loop { if (st == 0u) { break; }
      if (lid.x < st) { red4[lid.x] = max(red4[lid.x], red4[lid.x + st]); }
      workgroupBarrier(); st = st >> 1u;
    }
    if (lid.x == 0u) { mnew = max(m, red4[0]); }
    workgroupBarrier();
    let mn = mnew;
    // exponentiate in place + chunk sum
    var psum = vec4f(0.0);
    for (var e = 0u; e < 2u; e++) {
      let pl = lid.x * 2u + e;
      let p = p0 + pl;
      var ev = vec4f(0.0);
      if (p < npos) {
        ev = exp(vec4f(scores[pl * 4u], scores[pl * 4u + 1u], scores[pl * 4u + 2u], scores[pl * 4u + 3u]) - mn);
      }
      scores[pl * 4u] = ev.x; scores[pl * 4u + 1u] = ev.y; scores[pl * 4u + 2u] = ev.z; scores[pl * 4u + 3u] = ev.w;
      psum += ev;
    }
    red4[lid.x] = psum;
    workgroupBarrier();
    st = 128u;
    loop { if (st == 0u) { break; }
      if (lid.x < st) { red4[lid.x] += red4[lid.x + st]; }
      workgroupBarrier(); st = st >> 1u;
    }
    if (lid.x == 0u) { lchunk = red4[0]; }
    workgroupBarrier();
    let rescale = exp(m - mn);
    l = l * rescale + lchunk;
    acc = acc * rescale;
    m = mn;
    // accumulate V: this thread owns output dim d = lid.x for all 4 q heads
    let cnt = min(512u, npos - p0);
    for (var j = 0u; j < cnt; j++) {
      let vrow = (p0 + j) * 4u + kvh;
      let vq = i32(vcache[vrow * 64u + dword] << dshift) >> 24u;
      let vd = f32(vq) * bitcast<f32>(vcache[${MAXCTX * 256}u + vrow]);
      acc += vd * vec4f(scores[j * 4u], scores[j * 4u + 1u], scores[j * 4u + 2u], scores[j * 4u + 3u]);
    }
    p0 += 512u;
  }
  for (var qi = 0u; qi < 4u; qi++) {
    let h = kvh * 4u + qi;
    let gate = qg[t * 10240u + h * 512u + 256u + lid.x];
    outb[t * 4096u + h * 256u + lid.x] = (acc[qi] / l[qi]) * sigmoid_(gate);
  }
}`;

// Decode-only q/k/v preparation, one workgroup per KV head. Unlike attnPart,
// this work is performed once per token rather than once per 512-pos partition.
// The prepared q buffer is consumed by attnPartPrepared below.
export const attnDecodePrep = ({ MAXCTX, SPOS = 0, OFFS = 0 }) => /*wgsl*/`
${U_DEF}
${KVQ}
@group(0) @binding(0) var<storage, read> qg: array<f32>;
@group(0) @binding(1) var<storage, read> qw: array<f32>;
@group(0) @binding(2) var<storage, read> kw: array<f32>;
@group(0) @binding(3) var<storage, read> rope: array<f32>;
@group(0) @binding(4) var<storage, read_write> qr: array<f32>;
@group(0) @binding(5) var<storage, read_write> kcache: array<u32>;
@group(0) @binding(6) var<storage, read_write> vcache: array<u32>;
${SPOS
  ? '@group(0) @binding(7) var<storage, read> posBuf: array<u32>;'
  : '@group(0) @binding(7) var<uniform> u: U;'}
var<workgroup> qsh: array<f32, 1024>;
var<workgroup> arr: array<f32, 256>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let kvh = wid.x;
  ${SPOS ? `let pos = u32(i32(posBuf[0]) + (${OFFS}));` : 'let pos = u.basePos;'}
  // Four query heads share this KV head; 64 lanes prepare each query head.
  let hq = lid.x >> 6u;
  let ln = lid.x & 63u;
  let h = kvh * 4u + hq;
  var ss = 0.0;
  for (var e = 0u; e < 4u; e++) {
    let v = qg[h * 512u + ln + e * 64u];
    ss += v * v;
  }
  red[lid.x] = ss;
  workgroupBarrier();
  var s = 32u;
  loop { if (s == 0u) { break; }
    if (ln < s) { red[lid.x] += red[lid.x + s]; }
    workgroupBarrier(); s = s >> 1u;
  }
  let ri = inverseSqrt(red[hq * 64u] / 256.0 + 1e-6);
  for (var e = 0u; e < 4u; e++) {
    let d = ln + e * 64u;
    qsh[hq * 256u + d] = qg[h * 512u + d] * ri * qw[d];
  }
  workgroupBarrier();
  let fi = ln & 31u;
  let c = rope[pos * 64u + fi];
  let sn = rope[pos * 64u + 32u + fi];
  var qrot: f32;
  if (ln < 32u) { qrot = qsh[hq * 256u + ln] * c - qsh[hq * 256u + ln + 32u] * sn; }
  else { qrot = qsh[hq * 256u + ln] * c + qsh[hq * 256u + ln - 32u] * sn; }
  workgroupBarrier();
  qsh[hq * 256u + ln] = qrot;
  workgroupBarrier();
  for (var e = 0u; e < 4u; e++) {
    let d = ln + e * 64u;
    qr[kvh * 1024u + hq * 256u + d] = qsh[hq * 256u + d];
  }

  // This workgroup also appends its single K/V head to the quantized cache.
  let d = lid.x;
  let kraw = qg[8192u + kvh * 256u + d];
  red[d] = kraw * kraw;
  workgroupBarrier();
  s = 128u;
  loop { if (s == 0u) { break; }
    if (d < s) { red[d] += red[d + s]; }
    workgroupBarrier(); s = s >> 1u;
  }
  let kri = inverseSqrt(red[0] / 256.0 + 1e-6);
  arr[d] = kraw * kri * kw[d];
  workgroupBarrier();
  var krot = arr[d];
  if (d < 64u) {
    let kfi = d & 31u;
    let kc = rope[pos * 64u + kfi];
    let ks = rope[pos * 64u + 32u + kfi];
    if (d < 32u) { krot = arr[d] * kc - arr[d + 32u] * ks; }
    else { krot = arr[d] * kc + arr[d - 32u] * ks; }
  }
  workgroupBarrier();
  arr[d] = krot;
  workgroupBarrier();
  red[d] = abs(arr[d]);
  workgroupBarrier();
  s = 128u;
  loop { if (s == 0u) { break; }
    if (d < s) { red[d] = max(red[d], red[d + s]); }
    workgroupBarrier(); s = s >> 1u;
  }
  let kamax = red[0];
  let kqinv = select(0.0, 127.0 / kamax, kamax > 0.0);
  if (d < 64u) {
    kcache[(pos * 4u + kvh) * 64u + d] =
      kvq_pack(vec4f(arr[d * 4u], arr[d * 4u + 1u], arr[d * 4u + 2u], arr[d * 4u + 3u]), kqinv);
  }
  if (d == 0u) { kcache[${MAXCTX * 256}u + pos * 4u + kvh] = bitcast<u32>(kamax / 127.0); }
  workgroupBarrier();
  arr[d] = qg[9216u + kvh * 256u + d];
  red[d] = abs(arr[d]);
  workgroupBarrier();
  s = 128u;
  loop { if (s == 0u) { break; }
    if (d < s) { red[d] = max(red[d], red[d + s]); }
    workgroupBarrier(); s = s >> 1u;
  }
  let vamax = red[0];
  let vqinv = select(0.0, 127.0 / vamax, vamax > 0.0);
  if (d < 64u) {
    vcache[(pos * 4u + kvh) * 64u + d] =
      kvq_pack(vec4f(arr[d * 4u], arr[d * 4u + 1u], arr[d * 4u + 2u], arr[d * 4u + 3u]), vqinv);
  }
  if (d == 0u) { vcache[${MAXCTX * 256}u + pos * 4u + kvh] = bitcast<u32>(vamax / 127.0); }
}`;

// Flash-decoding partitions over q/k/v prepared once by attnDecodePrep.
export const attnPartPrepared = ({ PMAX, MAXCTX, SPOS = 0, OFFS = 0, SUBGROUPS = 0 }) => /*wgsl*/`
${SUBGROUPS ? 'enable subgroups;' : ''}
${U_DEF}
${KVQ}
@group(0) @binding(0) var<storage, read> qr: array<f32>;
@group(0) @binding(1) var<storage, read> kcache: array<u32>;
@group(0) @binding(2) var<storage, read> vcache: array<u32>;
@group(0) @binding(3) var<storage, read_write> parts: array<f32>;
${SPOS
  ? '@group(0) @binding(4) var<storage, read> posBuf: array<u32>;'
  : '@group(0) @binding(4) var<uniform> u: U;'}
var<workgroup> qsh: array<f32, 1024>;
var<workgroup> scores: array<vec4f, 512>;
var<workgroup> red4: array<vec4f, 256>;
var<workgroup> mch: vec4f;
var<workgroup> lch: vec4f;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u
        ${SUBGROUPS ? ', @builtin(subgroup_invocation_id) sgi: u32, @builtin(subgroup_size) sgs: u32' : ''}) {
  let p = wid.x;
  let kvh = wid.y;
  ${SPOS ? `let pos = u32(i32(posBuf[0]) + (${OFFS}));` : 'let pos = u.basePos;'}
  let npos = pos + 1u;
  for (var e = 0u; e < 4u; e++) {
    let i = e * 256u + lid.x;
    qsh[i] = qr[kvh * 1024u + i];
  }
  workgroupBarrier();
  let p0 = p * 512u;
  let pend = min(p0 + 512u, npos);
  for (var e = 0u; e < 2u; e++) {
    let pl = e * 256u + lid.x;
    let pp = p0 + pl;
    var sv = vec4f(-3.0e38);
    if (pp < pend) {
      sv = vec4f(0.0);
      let kbase = (pp * 4u + kvh) * 64u;
      for (var w = 0u; w < 64u; w++) {
        let kk = kvq_unpack(kcache[kbase + w]);
        let d0 = w * 4u;
        sv += kk.x * vec4f(qsh[d0], qsh[256u + d0], qsh[512u + d0], qsh[768u + d0]);
        sv += kk.y * vec4f(qsh[d0 + 1u], qsh[256u + d0 + 1u], qsh[512u + d0 + 1u], qsh[768u + d0 + 1u]);
        sv += kk.z * vec4f(qsh[d0 + 2u], qsh[256u + d0 + 2u], qsh[512u + d0 + 2u], qsh[768u + d0 + 2u]);
        sv += kk.w * vec4f(qsh[d0 + 3u], qsh[256u + d0 + 3u], qsh[512u + d0 + 3u], qsh[768u + d0 + 3u]);
      }
      sv *= bitcast<f32>(kcache[${MAXCTX * 256}u + pp * 4u + kvh]) * 0.0625;
    }
    scores[pl] = sv;
  }
  workgroupBarrier();
  let localMax = max(scores[lid.x * 2u], scores[lid.x * 2u + 1u]);
${SUBGROUPS ? /*wgsl*/`
  let sgMax = subgroupMax(localMax);
  if (sgi == 0u) { red4[lid.x / sgs] = sgMax; }
  workgroupBarrier();
  if (lid.x == 0u) {
    var v = red4[0];
    for (var i = 1u; i < 256u / sgs; i++) { v = max(v, red4[i]); }
    mch = v;
  }
` : /*wgsl*/`
  red4[lid.x] = localMax;
  workgroupBarrier();
  var st = 128u;
  loop { if (st == 0u) { break; }
    if (lid.x < st) { red4[lid.x] = max(red4[lid.x], red4[lid.x + st]); }
    workgroupBarrier(); st = st >> 1u;
  }
  if (lid.x == 0u) { mch = red4[0]; }
`}
  workgroupBarrier();
  let mn = mch;
  var psum = vec4f(0.0);
  for (var e = 0u; e < 2u; e++) {
    let pl = lid.x * 2u + e;
    let pp = p0 + pl;
    var ev = vec4f(0.0);
    if (pp < pend) { ev = exp(scores[pl] - mn); }
    scores[pl] = ev;
    psum += ev;
  }
${SUBGROUPS ? /*wgsl*/`
  let sgSum = subgroupAdd(psum);
  if (sgi == 0u) { red4[lid.x / sgs] = sgSum; }
  workgroupBarrier();
  if (lid.x == 0u) {
    var v = red4[0];
    for (var i = 1u; i < 256u / sgs; i++) { v += red4[i]; }
    lch = v;
  }
` : /*wgsl*/`
  red4[lid.x] = psum;
  workgroupBarrier();
  st = 128u;
  loop { if (st == 0u) { break; }
    if (lid.x < st) { red4[lid.x] += red4[lid.x + st]; }
    workgroupBarrier(); st = st >> 1u;
  }
  if (lid.x == 0u) { lch = red4[0]; }
`}
  workgroupBarrier();
  var acc = vec4f(0.0);
  let dword = lid.x >> 2u;
  let dshift = (3u - (lid.x & 3u)) * 8u;
  let cnt = pend - min(p0, pend);
  for (var j = 0u; j < cnt; j++) {
    let vrow = (p0 + j) * 4u + kvh;
    let vq = i32(vcache[vrow * 64u + dword] << dshift) >> 24u;
    let vd = f32(vq) * bitcast<f32>(vcache[${MAXCTX * 256}u + vrow]);
    acc += vd * scores[j];
  }
  for (var qi = 0u; qi < 4u; qi++) {
    let base = ((kvh * ${PMAX}u + p) * 4u + qi) * 258u;
    parts[base + lid.x] = acc[qi];
    if (lid.x == 0u) { parts[base + 256u] = mn[qi]; parts[base + 257u] = lch[qi]; }
  }
}`;

// ---------------------------------------------------------------------------
// Decode attention, flash-decoding style: grid (P partitions, 4 kv heads).
// Each workgroup handles one 512-position partition for its kv head's 4 query
// heads and emits (acc[256], m, l) partials; q norm+RoPE is recomputed inline
// (cheap) so no qr buffer or prep dispatch is needed. The LAST partition also
// appends this token's k/v to the caches before scoring (it is the only
// partition that reads the new position — no cross-workgroup hazard).
export const attnPart = ({ PMAX, MAXCTX, SPOS = 0, OFFS = 0 }) => /*wgsl*/`
${U_DEF}
${KVQ}
@group(0) @binding(0) var<storage, read> qg: array<f32>;     // [10240]: q&gate | k | v
@group(0) @binding(1) var<storage, read> qw: array<f32>;     // [256]
@group(0) @binding(2) var<storage, read> kw: array<f32>;     // [256]
@group(0) @binding(3) var<storage, read> rope: array<f32>;
@group(0) @binding(4) var<storage, read_write> kcache: array<u32>; // [MAXCTX][4][64] i8x4 + scales
@group(0) @binding(5) var<storage, read_write> vcache: array<u32>;
@group(0) @binding(6) var<storage, read_write> parts: array<f32>; // [4][PMAX][4][258]
${SPOS
  ? '@group(0) @binding(7) var<storage, read> posBuf: array<u32>; // GPU-chained position'
  : '@group(0) @binding(7) var<uniform> u: U;'}
var<workgroup> qsh: array<f32, 1024>;
var<workgroup> arr: array<f32, 256>;
var<workgroup> red: array<f32, 256>;
var<workgroup> scores: array<f32, 2048>;
var<workgroup> red4: array<vec4f, 256>;
var<workgroup> mch: vec4f;
var<workgroup> lch: vec4f;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let p = wid.x;
  let kvh = wid.y;
  ${SPOS
    ? `let pos = u32(i32(posBuf[0]) + (${OFFS}));`
    : 'let pos = u.basePos;'}
  let npos = pos + 1u;
  let lastP = pos / 512u;
  // ---- q prep: 4 heads, 64 lanes each ----
  {
    let hq = lid.x >> 6u;        // 0..3
    let ln = lid.x & 63u;
    let h = kvh * 4u + hq;
    var ss = 0.0;
    for (var e = 0u; e < 4u; e++) {
      let v = qg[h * 512u + ln + e * 64u];
      ss += v * v;
    }
    red[lid.x] = ss;
    workgroupBarrier();
    var s = 32u;
    loop { if (s == 0u) { break; }
      if (ln < s) { red[lid.x] += red[lid.x + s]; }
      workgroupBarrier(); s = s >> 1u;
    }
    let ri = inverseSqrt(red[hq * 64u] / 256.0 + 1e-6);
    workgroupBarrier();
    for (var e = 0u; e < 4u; e++) {
      let d = ln + e * 64u;
      qsh[hq * 256u + d] = qg[h * 512u + d] * ri * qw[d];
    }
    workgroupBarrier();
    // partial RoPE per head: this head's 64 lanes are exactly its 64 rotary dims
    let fi = ln & 31u;
    let c = rope[pos * 64u + fi];
    let sn = rope[pos * 64u + 32u + fi];
    var nv: f32;
    if (ln < 32u) { nv = qsh[hq * 256u + ln] * c - qsh[hq * 256u + ln + 32u] * sn; }
    else { nv = qsh[hq * 256u + ln] * c + qsh[hq * 256u + ln - 32u] * sn; }
    workgroupBarrier();
    qsh[hq * 256u + ln] = nv;
    workgroupBarrier();
  }
  // ---- last partition appends k/v for this kv head ----
  if (p == lastP) {
    let d = lid.x;
    let kraw = qg[8192u + kvh * 256u + d];
    red[d] = kraw * kraw;
    workgroupBarrier();
    var s = 128u;
    loop { if (s == 0u) { break; }
      if (d < s) { red[d] += red[d + s]; }
      workgroupBarrier(); s = s >> 1u;
    }
    let ri = inverseSqrt(red[0] / 256.0 + 1e-6);
    workgroupBarrier();
    arr[d] = kraw * ri * kw[d];
    workgroupBarrier();
    var nv = arr[d];
    if (d < 64u) {
      let fi = d & 31u;
      let c = rope[pos * 64u + fi];
      let sn = rope[pos * 64u + 32u + fi];
      if (d < 32u) { nv = arr[d] * c - arr[d + 32u] * sn; }
      else { nv = arr[d] * c + arr[d - 32u] * sn; }
    }
    workgroupBarrier();
    arr[d] = nv;
    workgroupBarrier();
    // int8-quantize k (in arr), then stage v through arr and do the same
    red[d] = abs(arr[d]);
    workgroupBarrier();
    s = 128u;
    loop { if (s == 0u) { break; }
      if (d < s) { red[d] = max(red[d], red[d + s]); }
      workgroupBarrier(); s = s >> 1u;
    }
    let kamax = red[0];
    let kqinv = select(0.0, 127.0 / kamax, kamax > 0.0);
    if (d < 64u) {
      kcache[(pos * 4u + kvh) * 64u + d] =
        kvq_pack(vec4f(arr[d * 4u], arr[d * 4u + 1u], arr[d * 4u + 2u], arr[d * 4u + 3u]), kqinv);
    }
    if (d == 0u) { kcache[${MAXCTX * 256}u + pos * 4u + kvh] = bitcast<u32>(kamax / 127.0); }
    workgroupBarrier(); // k packers are done reading arr/red before v reuses them
    arr[d] = qg[9216u + kvh * 256u + d];
    red[d] = abs(arr[d]);
    workgroupBarrier();
    s = 128u;
    loop { if (s == 0u) { break; }
      if (d < s) { red[d] = max(red[d], red[d + s]); }
      workgroupBarrier(); s = s >> 1u;
    }
    let vamax = red[0];
    let vqinv = select(0.0, 127.0 / vamax, vamax > 0.0);
    if (d < 64u) {
      vcache[(pos * 4u + kvh) * 64u + d] =
        kvq_pack(vec4f(arr[d * 4u], arr[d * 4u + 1u], arr[d * 4u + 2u], arr[d * 4u + 3u]), vqinv);
    }
    if (d == 0u) { vcache[${MAXCTX * 256}u + pos * 4u + kvh] = bitcast<u32>(vamax / 127.0); }
    storageBarrier();
    workgroupBarrier();
  }
  // ---- scores for this partition ----
  let p0 = p * 512u;
  let pend = min(p0 + 512u, npos);
  for (var e = 0u; e < 2u; e++) {
    let pl = e * 256u + lid.x;
    let pp = p0 + pl;
    var sv = vec4f(-3.0e38);
    if (pp < pend) {
      sv = vec4f(0.0);
      let kbase = (pp * 4u + kvh) * 64u;
      for (var w = 0u; w < 64u; w++) {
        let kk = kvq_unpack(kcache[kbase + w]);
        let d0 = w * 4u;
        sv += kk.x * vec4f(qsh[d0], qsh[256u + d0], qsh[512u + d0], qsh[768u + d0]);
        sv += kk.y * vec4f(qsh[d0 + 1u], qsh[256u + d0 + 1u], qsh[512u + d0 + 1u], qsh[768u + d0 + 1u]);
        sv += kk.z * vec4f(qsh[d0 + 2u], qsh[256u + d0 + 2u], qsh[512u + d0 + 2u], qsh[768u + d0 + 2u]);
        sv += kk.w * vec4f(qsh[d0 + 3u], qsh[256u + d0 + 3u], qsh[512u + d0 + 3u], qsh[768u + d0 + 3u]);
      }
      // dequant scale folded with 1/sqrt(256)
      sv *= bitcast<f32>(kcache[${MAXCTX * 256}u + pp * 4u + kvh]) * 0.0625;
    }
    scores[pl * 4u] = sv.x; scores[pl * 4u + 1u] = sv.y; scores[pl * 4u + 2u] = sv.z; scores[pl * 4u + 3u] = sv.w;
  }
  workgroupBarrier();
  let a = vec4f(scores[(lid.x * 2u) * 4u], scores[(lid.x * 2u) * 4u + 1u], scores[(lid.x * 2u) * 4u + 2u], scores[(lid.x * 2u) * 4u + 3u]);
  let b = vec4f(scores[(lid.x * 2u + 1u) * 4u], scores[(lid.x * 2u + 1u) * 4u + 1u], scores[(lid.x * 2u + 1u) * 4u + 2u], scores[(lid.x * 2u + 1u) * 4u + 3u]);
  red4[lid.x] = max(a, b);
  workgroupBarrier();
  var st = 128u;
  loop { if (st == 0u) { break; }
    if (lid.x < st) { red4[lid.x] = max(red4[lid.x], red4[lid.x + st]); }
    workgroupBarrier(); st = st >> 1u;
  }
  if (lid.x == 0u) { mch = red4[0]; }
  workgroupBarrier();
  let mn = mch;
  var psum = vec4f(0.0);
  for (var e = 0u; e < 2u; e++) {
    let pl = lid.x * 2u + e;
    let pp = p0 + pl;
    var ev = vec4f(0.0);
    if (pp < pend) {
      ev = exp(vec4f(scores[pl * 4u], scores[pl * 4u + 1u], scores[pl * 4u + 2u], scores[pl * 4u + 3u]) - mn);
    }
    scores[pl * 4u] = ev.x; scores[pl * 4u + 1u] = ev.y; scores[pl * 4u + 2u] = ev.z; scores[pl * 4u + 3u] = ev.w;
    psum += ev;
  }
  red4[lid.x] = psum;
  workgroupBarrier();
  st = 128u;
  loop { if (st == 0u) { break; }
    if (lid.x < st) { red4[lid.x] += red4[lid.x + st]; }
    workgroupBarrier(); st = st >> 1u;
  }
  if (lid.x == 0u) { lch = red4[0]; }
  workgroupBarrier();
  var acc = vec4f(0.0);
  let dword = lid.x >> 2u;                    // this thread's dim → packed word
  let dshift = (3u - (lid.x & 3u)) * 8u;      // …and sign-extension shift
  let cnt = pend - min(p0, pend); // 0 for over-dispatched partitions (p0 >= npos)
  for (var j = 0u; j < cnt; j++) {
    let vrow = (p0 + j) * 4u + kvh;
    let vq = i32(vcache[vrow * 64u + dword] << dshift) >> 24u;
    let vd = f32(vq) * bitcast<f32>(vcache[${MAXCTX * 256}u + vrow]);
    acc += vd * vec4f(scores[j * 4u], scores[j * 4u + 1u], scores[j * 4u + 2u], scores[j * 4u + 3u]);
  }
  for (var qi = 0u; qi < 4u; qi++) {
    let base = ((kvh * ${PMAX}u + p) * 4u + qi) * 258u;
    parts[base + lid.x] = acc[qi];
    if (lid.x == 0u) { parts[base + 256u] = mn[qi]; parts[base + 257u] = lch[qi]; }
  }
}`;

// Combine attention partials across partitions; applies the sigmoid output gate.
export const attnReduce = ({ PMAX, SPOS = 0, OFFS = 0 }) => /*wgsl*/`
${U_DEF}
${SILU}
@group(0) @binding(0) var<storage, read> parts: array<f32>; // [4][PMAX][4][258]
@group(0) @binding(1) var<storage, read> qg: array<f32>;    // [10240]
@group(0) @binding(2) var<storage, read_write> outb: array<f32>; // [16][256]
${SPOS
  ? '@group(0) @binding(3) var<storage, read> posBuf: array<u32>;'
  : '@group(0) @binding(3) var<uniform> u: U;'}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let kvh = wid.x;
  let qi = wid.y;
  ${SPOS
    ? `let P = u32(i32(posBuf[0]) + (${OFFS})) / 512u + 1u;`
    : 'let P = u.basePos / 512u + 1u;'}
  var M = -3.0e38;
  for (var p = 0u; p < P; p++) {
    M = max(M, parts[((kvh * ${PMAX}u + p) * 4u + qi) * 258u + 256u]);
  }
  var L = 0.0;
  var acc = 0.0;
  for (var p = 0u; p < P; p++) {
    let base = ((kvh * ${PMAX}u + p) * 4u + qi) * 258u;
    let w = exp(parts[base + 256u] - M);
    L += parts[base + 257u] * w;
    acc += parts[base + lid.x] * w;
  }
  let h = kvh * 4u + qi;
  let gate = qg[h * 512u + 256u + lid.x];
  outb[h * 256u + lid.x] = (acc / L) * sigmoid_(gate);
}`;

// ---------------------------------------------------------------------------
// SwiGLU elementwise: act = silu(gu[:9216]) * gu[9216:]
export const siluMul = () => /*wgsl*/`
${SILU}
@group(0) @binding(0) var<storage, read> gu: array<f32>;   // [T][18432]
@group(0) @binding(1) var<storage, read_write> outb: array<f32>; // [T][9216]
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let i = wid.x * 256u + lid.x;
  let t = wid.y;
  outb[t * 9216u + i] = silu(gu[t * 18432u + i]) * gu[t * 18432u + 9216u + i];
}`;

// ---------------------------------------------------------------------------
// Per-block top-K over logits: each workgroup scans a 1024 slice and emits its
// local top-20 (id, value) pairs. CPU merges 243 blocks and samples.
export const topk = ({ VOCAB, KTOP, ALLOWED = false, DENIED = false, SOFTCAP = 0 }) => {
  if (ALLOWED && DENIED) throw new Error('topk cannot use allow and deny filters together');
  if (!Number.isFinite(SOFTCAP) || SOFTCAP < 0) throw new Error('topk SOFTCAP must be finite and non-negative');
  const FILTERED = ALLOWED || DENIED;
  const logit = SOFTCAP > 0
    ? `${Number(SOFTCAP).toFixed(8)} * tanh(logits[rowOff + g] / ${Number(SOFTCAP).toFixed(8)})`
    : 'logits[rowOff + g]';
  return /*wgsl*/`
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> cand: array<u32>; // [row][blocks][KTOP][2]
${FILTERED ? `@group(0) @binding(2) var<storage, read> filterIds: array<u32>;
@group(0) @binding(3) var<uniform> allowedCount: u32;
fn isListed(id: u32) -> bool {
  var lo = 0u;
  var hi = allowedCount;
  loop {
    if (lo >= hi) { break; }
    let mid = lo + (hi - lo) / 2u;
    let value = filterIds[mid];
    if (value == id) { return true; }
    if (value < id) { lo = mid + 1u; } else { hi = mid; }
  }
  return false;
}` : ''}
var<workgroup> vals: array<f32, 1024>;
var<workgroup> rv: array<f32, 256>;
var<workgroup> ri: array<u32, 256>;
const NBLK = ${Math.ceil(VOCAB / 1024)}u;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let rowOff = wid.y * ${VOCAB}u;         // logits row (verify processes several)
  let candOff = wid.y * NBLK * ${KTOP * 2}u;
  let base = wid.x * 1024u;
  for (var e = 0u; e < 4u; e++) {
    let i = lid.x * 4u + e;
    let g = base + i;
    vals[i] = select(-3.0e38, ${logit}, g < ${VOCAB}u${ALLOWED ? ' && isListed(g)' : DENIED ? ' && !isListed(g)' : ''});
  }
  workgroupBarrier();
  for (var it = 0u; it < ${KTOP}u; it++) {
    var bv = -3.0e38;
    var bi = 0u;
    for (var e = 0u; e < 4u; e++) {
      let i = lid.x * 4u + e;
      if (vals[i] > bv) { bv = vals[i]; bi = i; }
    }
    rv[lid.x] = bv; ri[lid.x] = bi;
    workgroupBarrier();
    var s = 128u;
    loop { if (s == 0u) { break; }
      if (lid.x < s && rv[lid.x + s] > rv[lid.x]) { rv[lid.x] = rv[lid.x + s]; ri[lid.x] = ri[lid.x + s]; }
      workgroupBarrier(); s = s >> 1u;
    }
    if (lid.x == 0u) {
      cand[candOff + (wid.x * ${KTOP}u + it) * 2u] = base + ri[0];
      cand[candOff + (wid.x * ${KTOP}u + it) * 2u + 1u] = bitcast<u32>(rv[0]);
      vals[ri[0]] = -3.0e38;
    }
    workgroupBarrier();
  }
}`;
};
