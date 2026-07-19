import { GGML_LAYOUT, GGML_TYPE } from '../gguf/parser.js';

// Decode tiling knobs for fixed 32-lane subgroups. A subgroup shares each f32
// activation block across these many packed output rows; weights stay native.
const Q4_0_ROWS_PER_SUBGROUP = 8;
const Q4_0_WORKGROUP_SIZE = 256;
const Q8_0_ROWS_PER_SUBGROUP = 8;

const requireInt = (name, value, min = 1) => {
  if (!Number.isInteger(value) || value < min) throw new Error(`${name} must be an integer >= ${min}; got ${value}`);
};

export function ggmlLoadWGSL(type, K) {
  const layout = GGML_LAYOUT[type];
  if (!layout) throw new Error(`Unsupported GGML type ${type}`);
  if (K % layout.blockSize) throw new Error(`K=${K} is not divisible by GGML block size ${layout.blockSize}`);
  const rowBytes = K / layout.blockSize * layout.typeSize;
  const common = /* wgsl */`
fn rawByte(offset: u32) -> u32 {
  return (weight[offset >> 2u] >> ((offset & 3u) * 8u)) & 255u;
}
fn rawU16(offset: u32) -> u32 { return rawByte(offset) | (rawByte(offset + 1u) << 8u); }
fn rawI8(offset: u32) -> i32 { return i32(rawByte(offset) << 24u) >> 24; }
fn halfAt(offset: u32) -> f32 { return unpack2x16float(rawU16(offset)).x; }
`;
  let load;
  switch (type) {
    case GGML_TYPE.F32:
      load = `return bitcast<f32>(weight[(row * ${rowBytes}u + col * 4u) >> 2u]);`;
      break;
    case GGML_TYPE.BF16:
      load = `return bitcast<f32>(rawU16(row * ${rowBytes}u + col * 2u) << 16u);`;
      break;
    case GGML_TYPE.Q4_0:
      load = /* wgsl */`
  let block = col >> 5u; let i = col & 31u; let base = row * ${rowBytes}u + block * 18u;
  let q = rawByte(base + 2u + (i & 15u));
  return halfAt(base) * (f32(select(q & 15u, q >> 4u, i >= 16u)) - 8.0);`;
      break;
    case GGML_TYPE.Q4_1:
      load = /* wgsl */`
  let block = col >> 5u; let i = col & 31u; let base = row * ${rowBytes}u + block * 20u;
  let q = rawByte(base + 4u + (i & 15u));
  return halfAt(base) * f32(select(q & 15u, q >> 4u, i >= 16u)) + halfAt(base + 2u);`;
      break;
    case GGML_TYPE.Q8_0:
      load = /* wgsl */`
  let block = col >> 5u; let i = col & 31u; let base = row * ${rowBytes}u + block * 34u;
  return halfAt(base) * f32(rawI8(base + 2u + i));`;
      break;
    case GGML_TYPE.Q5_K:
      load = /* wgsl */`
  let block = col >> 8u; let i = col & 255u; let base = row * ${rowBytes}u + block * 176u;
  let group = i >> 5u; let l = i & 31u; let pair = group >> 1u;
  var sc: u32; var mn: u32;
  if (group < 4u) {
    sc = rawByte(base + 4u + group) & 63u;
    mn = rawByte(base + 8u + group) & 63u;
  } else {
    sc = (rawByte(base + 8u + group) & 15u) | ((rawByte(base + group) >> 6u) << 4u);
    mn = (rawByte(base + 8u + group) >> 4u) | ((rawByte(base + 4u + group) >> 6u) << 4u);
  }
  let packed = rawByte(base + 48u + pair * 32u + l);
  let low = (group & 1u) == 0u;
  let qhMask = select(2u, 1u, low) << (pair * 2u);
  let q = select(packed >> 4u, packed & 15u, low) + select(0u, 16u, (rawByte(base + 16u + l) & qhMask) != 0u);
  return halfAt(base) * f32(sc * q) - halfAt(base + 2u) * f32(mn);`;
      break;
    case GGML_TYPE.Q6_K:
      load = /* wgsl */`
  let block = col >> 8u; let i = col & 255u; let base = row * ${rowBytes}u + block * 210u;
  let half = i >> 7u; let p = i & 127u; let quarter = p >> 5u; let l = p & 31u;
  let qlOffset = select(0u, 32u, (quarter & 1u) != 0u);
  let packed = rawByte(base + half * 64u + qlOffset + l);
  let nibble = select(packed & 15u, packed >> 4u, quarter >= 2u);
  let high = (rawByte(base + 128u + half * 32u + l) >> (quarter * 2u)) & 3u;
  let scaleIndex = half * 8u + (l >> 4u) + quarter * 2u;
  return halfAt(base + 208u) * f32(rawI8(base + 192u + scaleIndex)) * (f32(nibble | (high << 4u)) - 32.0);`;
      break;
  }
  return `${common}\nfn loadWeight(row: u32, col: u32) -> f32 { ${load}\n}`;
}

export const gemvGGML = ({ N, K, TYPE, RESIDUAL = 0, OSTRIDE = N, OUTOFF = 0, WEIGHT_ROW_OFFSET = 0, ACTIVATION = 'plain', SUBGROUPS = 0 }) => {
  requireInt('N', N); requireInt('K', K); requireInt('OSTRIDE', OSTRIDE); requireInt('OUTOFF', OUTOFF, 0);
  if (!['plain', 'silu', 'gelu'].includes(ACTIVATION)) throw new Error(`Unsupported activation ${ACTIVATION}`);
  return /* wgsl */`
${SUBGROUPS ? 'enable subgroups;' : ''}
@group(0) @binding(0) var<storage, read> weight: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
var<workgroup> red: array<f32, 256>;
${ggmlLoadWGSL(TYPE, K)}
fn gelu(v: vec4f) -> vec4f { return 0.5 * v * (vec4f(1.0) + tanh(vec4f(0.7978845608028654) * (v + vec4f(0.044715) * v * v * v))); }
fn loadX(base: u32, i: u32) -> vec4f {
  let v = x[base + i];
  ${ACTIVATION === 'silu' ? `return (v / (vec4f(1.0) + exp(-v))) * x[base + ${K / 4}u + i];`
    : ACTIVATION === 'gelu' ? `return gelu(v) * x[base + ${K / 4}u + i];` : 'return v;'}
}
@compute @workgroup_size(${Q4_0_WORKGROUP_SIZE})
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u
        ${SUBGROUPS ? ', @builtin(subgroup_invocation_id) sgi: u32, @builtin(subgroup_size) sgs: u32' : ''}) {
  let inputRow = wid.y; let localRow = lid.x >> 6u; let lane = lid.x & 63u;
  let row = wid.x * 4u + localRow; let xbase = inputRow * ${ACTIVATION === 'plain' ? K / 4 : K / 2}u;
  var sum = 0.0;
  if (row < ${N}u) {
    for (var col = lane * 4u; col < ${K}u; col += 256u) {
      let xv = loadX(xbase, col >> 2u); let wr = row + ${WEIGHT_ROW_OFFSET}u;
      sum += dot(vec4f(loadWeight(wr, col), loadWeight(wr, col + 1u), loadWeight(wr, col + 2u), loadWeight(wr, col + 3u)), xv);
    }
  }
${SUBGROUPS ? `
  sum = subgroupAdd(sum); let groups = 64u / sgs;
  if (sgi == 0u) { red[lid.x / sgs] = sum; } workgroupBarrier();
  if (lane == 0u && row < ${N}u) { var total = 0.0; for (var i=0u;i<groups;i++){ total += red[localRow*groups+i]; }
    let dst=inputRow*${OSTRIDE}u+${OUTOFF}u+row; ${RESIDUAL ? 'out[dst] += total;' : 'out[dst] = total;'} }
` : `
  red[lid.x]=sum; workgroupBarrier(); var step=32u;
  loop { if(step==0u){break;} if(lane<step){red[lid.x]+=red[lid.x+step];} workgroupBarrier(); step>>=1u; }
  if(lane==0u && row<${N}u){let dst=inputRow*${OSTRIDE}u+${OUTOFF}u+row; ${RESIDUAL ? 'out[dst] += red[lid.x];' : 'out[dst] = red[lid.x];'}}
`}
}`;
};

// Fast native Q4_0 decode path for fixed 32-lane subgroups. Each subgroup
// owns one output row and each lane consumes whole 32-value GGML blocks. This
// mirrors llama.cpp's warp-level MMV layout: all lanes follow one contiguous
// packed row, and no intermediate floating-point weight buffer is produced.
export const gemvQ40Subgroup = ({ N, K, RESIDUAL = 0, OSTRIDE = N, OUTOFF = 0, WEIGHT_ROW_OFFSET = 0 }) => {
  requireInt('N', N); requireInt('K', K); requireInt('OSTRIDE', OSTRIDE); requireInt('OUTOFF', OUTOFF, 0);
  if (K % 32) throw new Error(`Q4_0 GEMV K=${K} is not divisible by 32`);
  const rowBytes = K / 32 * 18;
  const rowsPerSubgroup = Q4_0_ROWS_PER_SUBGROUP;
  const accumulators = Array.from({ length: rowsPerSubgroup }, (_, i) => `var sum${i}=0.0;`).join(' ');
  const dotRows = Array.from({ length: rowsPerSubgroup }, (_, i) =>
    `if(row+${i}u<${N}u){sum${i}+=q4BlockDot((row+${WEIGHT_ROW_OFFSET + i}u)*${rowBytes}u+block*18u,x0,x1,x2,x3,x4,x5,x6,x7);}`
  ).join('\n');
  const reductions = Array.from({ length: rowsPerSubgroup }, (_, i) => `sum${i}=subgroupAdd(sum${i});`).join(' ');
  const stores = Array.from({ length: rowsPerSubgroup }, (_, i) =>
    `if(row+${i}u<${N}u){${RESIDUAL ? `out[dst+${i}u]+=sum${i};` : `out[dst+${i}u]=sum${i};`}}`
  ).join('\n');
  return /* wgsl */`
enable subgroups;
@group(0) @binding(0) var<storage, read> weight: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
fn low4(q: u32) -> vec4f {
  return vec4f(f32(q & 15u), f32((q >> 8u) & 15u), f32((q >> 16u) & 15u), f32((q >> 24u) & 15u)) - vec4f(8.0);
}
fn high4(q: u32) -> vec4f {
  return vec4f(f32((q >> 4u) & 15u), f32((q >> 12u) & 15u), f32((q >> 20u) & 15u), f32((q >> 28u) & 15u)) - vec4f(8.0);
}
fn q4BlockDot(base: u32, x0: vec4f, x1: vec4f, x2: vec4f, x3: vec4f,
              x4: vec4f, x5: vec4f, x6: vec4f, x7: vec4f) -> f32 {
  // A Q4_0 block is 18 bytes and alternates between 0- and 2-byte u32
  // alignment. Load its five containing words once instead of expressing
  // five overlapping unaligned reads and relying on driver CSE.
  let wi = base >> 2u; let shifted = (base & 2u) != 0u;
  let w0 = weight[wi]; let w1 = weight[wi + 1u]; let w2 = weight[wi + 2u];
  let w3 = weight[wi + 3u]; let w4 = weight[wi + 4u];
  let scaleBits = select(w0 & 65535u, w0 >> 16u, shifted);
  let q0 = select((w0 >> 16u) | (w1 << 16u), w1, shifted);
  let q1 = select((w1 >> 16u) | (w2 << 16u), w2, shifted);
  let q2 = select((w2 >> 16u) | (w3 << 16u), w3, shifted);
  let q3 = select((w3 >> 16u) | (w4 << 16u), w4, shifted);
  let qdot = dot(low4(q0), x0) + dot(low4(q1), x1)
    + dot(low4(q2), x2) + dot(low4(q3), x3)
    + dot(high4(q0), x4) + dot(high4(q1), x5)
    + dot(high4(q2), x6) + dot(high4(q3), x7);
  return unpack2x16float(scaleBits).x * qdot;
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u,
        @builtin(subgroup_invocation_id) lane: u32) {
  let inputRow=wid.y;let row=wid.x*${rowsPerSubgroup * (Q4_0_WORKGROUP_SIZE / 32)}u+(lid.x>>5u)*${rowsPerSubgroup}u;
  let xbase = inputRow * ${K / 4}u;
  ${accumulators}
  for (var block = lane; block < ${K / 32}u; block += 32u) {
    let xb = xbase + block * 8u;
    let x0=x[xb]; let x1=x[xb+1u]; let x2=x[xb+2u]; let x3=x[xb+3u];
    let x4=x[xb+4u]; let x5=x[xb+5u]; let x6=x[xb+6u]; let x7=x[xb+7u];
    ${dotRows}
  }
  ${reductions}
  if (lane == 0u) {
    let dst = inputRow * ${OSTRIDE}u + ${OUTOFF}u + row;
    ${stores}
  }
}`;
};

const nativeSubgroupPreamble = (bindings = '') => /* wgsl */`
enable subgroups;
@group(0) @binding(0) var<storage, read> weight: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
${bindings}
fn rawU32(offset:u32)->u32{let w=offset>>2u;let s=(offset&3u)*8u;if(s==0u){return weight[w];}return (weight[w]>>s)|(weight[w+1u]<<(32u-s));}
fn rawByte(offset:u32)->u32{return (weight[offset>>2u]>>((offset&3u)*8u))&255u;}
fn rawI8(offset:u32)->i32{return i32(rawByte(offset)<<24u)>>24;}
fn halfAt(offset:u32)->f32{return unpack2x16float(rawU32(offset)&65535u).x;}
fn bytes4(q:u32)->vec4f{return vec4f(f32(q&255u),f32((q>>8u)&255u),f32((q>>16u)&255u),f32((q>>24u)&255u));}
fn signed4(q:u32)->vec4f{return vec4f(f32(i32(q<<24u)>>24),f32(i32(q<<16u)>>24),f32(i32(q<<8u)>>24),f32(i32(q)>>24));}
`;

const validateSubgroupGemv = (type, N, K, blockSize, OSTRIDE, OUTOFF) => {
  requireInt('N', N); requireInt('K', K); requireInt('OSTRIDE', OSTRIDE); requireInt('OUTOFF', OUTOFF, 0);
  if (K % blockSize) throw new Error(`${type} GEMV K=${K} is not divisible by ${blockSize}`);
};

/** Direct f32-activation Q4_1 GEMV for fixed 32-lane subgroups. */
export const gemvQ41Subgroup = ({ N, K, RESIDUAL = 0, OSTRIDE = N, OUTOFF = 0, WEIGHT_ROW_OFFSET = 0 }) => {
  validateSubgroupGemv('Q4_1', N, K, 32, OSTRIDE, OUTOFF);
  const rowBytes = K / 32 * 20;
  return /* wgsl */`
${nativeSubgroupPreamble()}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_id) lid:vec3u,@builtin(subgroup_invocation_id) lane:u32){
 let halfLane=lane&15u;let inputRow=wid.y;let row=wid.x*16u+(lid.x>>4u);var acc=0.0;
 if(row<${N}u){let wr=row+${WEIGHT_ROW_OFFSET}u;let xb=inputRow*${K / 4}u;
  for(var block=halfLane;block<${K / 32}u;block+=16u){let base=wr*${rowBytes}u+block*20u;var qdot=0.0;var xsum=0.0;
   for(var j=0u;j<4u;j++){let q=rawU32(base+4u+j*4u);let xv0=x[xb+block*8u+j];let xv1=x[xb+block*8u+4u+j];
    qdot+=dot(bytes4(q&0x0f0f0f0fu),xv0)+dot(bytes4((q>>4u)&0x0f0f0f0fu),xv1);xsum+=dot(vec4f(1.0),xv0+xv1);}
   acc+=halfAt(base)*qdot+halfAt(base+2u)*xsum;
  }
 }
 acc+=subgroupShuffleXor(acc,8u);acc+=subgroupShuffleXor(acc,4u);acc+=subgroupShuffleXor(acc,2u);acc+=subgroupShuffleXor(acc,1u);if(halfLane==0u&&row<${N}u){let dst=inputRow*${OSTRIDE}u+${OUTOFF}u+row;${RESIDUAL ? 'out[dst]+=acc;' : 'out[dst]=acc;'}}
}`;
};

/** Direct f32-activation Q8_0 GEMV for fixed 32-lane subgroups. */
export const gemvQ80Subgroup = ({ N, K, RESIDUAL = 0, OSTRIDE = N, OUTOFF = 0, WEIGHT_ROW_OFFSET = 0 }) => {
  validateSubgroupGemv('Q8_0', N, K, 32, OSTRIDE, OUTOFF);
  const rowBytes = K / 32 * 34;
  const rowsPerSubgroup = Q8_0_ROWS_PER_SUBGROUP;
  const accumulators = Array.from({ length: rowsPerSubgroup }, (_, i) => `var sum${i}=0.0;`).join(' ');
  const dotRows = Array.from({ length: rowsPerSubgroup }, (_, i) =>
    `if(row+${i}u<${N}u){sum${i}+=q8BlockDot((row+${WEIGHT_ROW_OFFSET + i}u)*${rowBytes}u+block*34u,x0,x1,x2,x3,x4,x5,x6,x7);}`
  ).join('\n');
  const reductions = Array.from({ length: rowsPerSubgroup }, (_, i) => `sum${i}=subgroupAdd(sum${i});`).join(' ');
  const stores = Array.from({ length: rowsPerSubgroup }, (_, i) =>
    `if(row+${i}u<${N}u){${RESIDUAL ? `out[dst+${i}u]+=sum${i};` : `out[dst+${i}u]=sum${i};`}}`
  ).join('\n');
  return /* wgsl */`
${nativeSubgroupPreamble()}
fn q8BlockDot(base:u32,x0:vec4f,x1:vec4f,x2:vec4f,x3:vec4f,x4:vec4f,x5:vec4f,x6:vec4f,x7:vec4f)->f32{
 let wi=base>>2u;let shifted=(base&2u)!=0u;
 let w0=weight[wi];let w1=weight[wi+1u];let w2=weight[wi+2u];let w3=weight[wi+3u];let w4=weight[wi+4u];
 let w5=weight[wi+5u];let w6=weight[wi+6u];let w7=weight[wi+7u];let w8=weight[wi+8u];
 let scaleBits=select(w0&65535u,w0>>16u,shifted);
 let q0=select((w0>>16u)|(w1<<16u),w1,shifted);let q1=select((w1>>16u)|(w2<<16u),w2,shifted);
 let q2=select((w2>>16u)|(w3<<16u),w3,shifted);let q3=select((w3>>16u)|(w4<<16u),w4,shifted);
 let q4=select((w4>>16u)|(w5<<16u),w5,shifted);let q5=select((w5>>16u)|(w6<<16u),w6,shifted);
 let q6=select((w6>>16u)|(w7<<16u),w7,shifted);let q7=select((w7>>16u)|(w8<<16u),w8,shifted);
 return unpack2x16float(scaleBits).x*(dot(signed4(q0),x0)+dot(signed4(q1),x1)+dot(signed4(q2),x2)+dot(signed4(q3),x3)+dot(signed4(q4),x4)+dot(signed4(q5),x5)+dot(signed4(q6),x6)+dot(signed4(q7),x7));
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_id) lid:vec3u,@builtin(subgroup_invocation_id) lane:u32){
 let inputRow=wid.y;let row=wid.x*${rowsPerSubgroup * 8}u+(lid.x>>5u)*${rowsPerSubgroup}u;let xb0=inputRow*${K / 4}u;
 ${accumulators}
 for(var block=lane;block<${K / 32}u;block+=32u){let xb=xb0+block*8u;
  let x0=x[xb];let x1=x[xb+1u];let x2=x[xb+2u];let x3=x[xb+3u];let x4=x[xb+4u];let x5=x[xb+5u];let x6=x[xb+6u];let x7=x[xb+7u];
  ${dotRows}
 }
 ${reductions}
 if(lane==0u){let dst=inputRow*${OSTRIDE}u+${OUTOFF}u+row;${stores}}
}`;
};

/** Direct f32-activation Q5_K GEMV for fixed 32-lane subgroups. */
export const gemvQ5KSubgroup = ({ N, K, RESIDUAL = 0, OSTRIDE = N, OUTOFF = 0, WEIGHT_ROW_OFFSET = 0 }) => {
  validateSubgroupGemv('Q5_K', N, K, 256, OSTRIDE, OUTOFF);
  const rowBytes = K / 256 * 176;
  return /* wgsl */`
${nativeSubgroupPreamble()}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_id) lid:vec3u,@builtin(subgroup_invocation_id) lane:u32){
 let halfLane=lane&15u;let inputRow=wid.y;let row=wid.x*16u+(lid.x>>4u);var acc=0.0;
 if(row<${N}u){let wr=row+${WEIGHT_ROW_OFFSET}u;let xb=inputRow*${K / 4}u;
  for(var gi=halfLane;gi<${K / 32}u;gi+=16u){let block=gi>>3u;let group=gi&7u;let base=wr*${rowBytes}u+block*176u;let pair=group>>1u;let low=(group&1u)==0u;
   var sc:u32;var mn:u32;if(group<4u){sc=rawByte(base+4u+group)&63u;mn=rawByte(base+8u+group)&63u;}else{sc=(rawByte(base+8u+group)&15u)|((rawByte(base+group)>>6u)<<4u);mn=(rawByte(base+8u+group)>>4u)|((rawByte(base+4u+group)>>6u)<<4u);}
   let bitPos=pair*2u+select(1u,0u,low);var qdot=0.0;var xsum=0.0;
   for(var j=0u;j<8u;j++){let packed=rawU32(base+48u+pair*32u+j*4u);let high=((rawU32(base+16u+j*4u)>>bitPos)&0x01010101u)<<4u;let q=select((packed>>4u)&0x0f0f0f0fu,packed&0x0f0f0f0fu,low)|high;let xv=x[xb+gi*8u+j];qdot+=dot(bytes4(q),xv);xsum+=dot(vec4f(1.0),xv);}
   acc+=halfAt(base)*f32(sc)*qdot-halfAt(base+2u)*f32(mn)*xsum;
  }
 }
 acc+=subgroupShuffleXor(acc,8u);acc+=subgroupShuffleXor(acc,4u);acc+=subgroupShuffleXor(acc,2u);acc+=subgroupShuffleXor(acc,1u);if(halfLane==0u&&row<${N}u){let dst=inputRow*${OSTRIDE}u+${OUTOFF}u+row;${RESIDUAL ? 'out[dst]+=acc;' : 'out[dst]=acc;'}}
}`;
};

/** Direct f32-activation Q6_K GEMV for fixed 32-lane subgroups. */
export const gemvQ6KSubgroup = ({ N, K, RESIDUAL = 0, OSTRIDE = N, OUTOFF = 0, WEIGHT_ROW_OFFSET = 0 }) => {
  validateSubgroupGemv('Q6_K', N, K, 256, OSTRIDE, OUTOFF);
  const rowBytes = K / 256 * 210;
  return /* wgsl */`
${nativeSubgroupPreamble()}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_id) lid:vec3u,@builtin(subgroup_invocation_id) lane:u32){
 let halfLane=lane&15u;let inputRow=wid.y;let row=wid.x*16u+(lid.x>>4u);var acc=0.0;
 if(row<${N}u){let wr=row+${WEIGHT_ROW_OFFSET}u;let xb=inputRow*${K / 4}u;
  for(var bi=halfLane;bi<${K / 16}u;bi+=16u){let block=bi>>4u;let local=bi&15u;let half=local>>3u;let quarter=(local>>1u)&3u;let lbase=(local&1u)*16u;let base=wr*${rowBytes}u+block*210u;let qlo=select(0u,32u,(quarter&1u)!=0u);let shift=quarter*2u;var qdot=0.0;
   for(var j=0u;j<4u;j++){let packed=rawU32(base+half*64u+qlo+lbase+j*4u);let nib=select(packed&0x0f0f0f0fu,(packed>>4u)&0x0f0f0f0fu,quarter>=2u);let qh=((rawU32(base+128u+half*32u+lbase+j*4u)>>shift)&0x03030303u)<<4u;qdot+=dot(bytes4(nib|qh)-vec4f(32.0),x[xb+bi*4u+j]);}
   let scaleIndex=half*8u+(lbase>>4u)+quarter*2u;acc+=halfAt(base+208u)*f32(rawI8(base+192u+scaleIndex))*qdot;
  }
 }
 acc+=subgroupShuffleXor(acc,8u);acc+=subgroupShuffleXor(acc,4u);acc+=subgroupShuffleXor(acc,2u);acc+=subgroupShuffleXor(acc,1u);if(halfLane==0u&&row<${N}u){let dst=inputRow*${OSTRIDE}u+${OUTOFF}u+row;${RESIDUAL ? 'out[dst]+=acc;' : 'out[dst]=acc;'}}
}`;
};

const NATIVE_SUBGROUP_GEMV = new Map([
  [GGML_TYPE.Q4_0, gemvQ40Subgroup],
  [GGML_TYPE.Q4_1, gemvQ41Subgroup],
  [GGML_TYPE.Q5_K, gemvQ5KSubgroup],
  [GGML_TYPE.Q6_K, gemvQ6KSubgroup],
  [GGML_TYPE.Q8_0, gemvQ80Subgroup],
]);

export const hasNativeSubgroupGemv = (type) => NATIVE_SUBGROUP_GEMV.has(type);
export const nativeSubgroupRowsPerWorkgroup = (type) => type === GGML_TYPE.Q4_0
  ? Q4_0_ROWS_PER_SUBGROUP * (Q4_0_WORKGROUP_SIZE / 32)
  : type === GGML_TYPE.Q8_0
    ? Q8_0_ROWS_PER_SUBGROUP * 8
    : hasNativeSubgroupGemv(type) ? 16 : 4;

/** Build a packed-GGML GEMV without materializing a floating-point weight tensor. */
export function gemvSubgroupGGML({ TYPE, ...options }) {
  const factory = NATIVE_SUBGROUP_GEMV.get(TYPE);
  if (!factory) throw new Error(`GGML type ${TYPE} has no native subgroup GEMV`);
  return factory(options);
}

export const gemmGGML = ({ N, K, TYPE, RESIDUAL = 0, OSTRIDE = N, OUTOFF = 0 }) => /* wgsl */`
struct U { rows: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> weight: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> u: U;
var<workgroup> xt: array<f32, 544>; var<workgroup> wt: array<f32, 2176>;
${ggmlLoadWGSL(TYPE, K)}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_id) lid:vec3u){
 let n0=wid.x*128u; let t0=wid.y*32u; let nx=lid.x&31u; let ty=lid.x>>5u; var acc:array<f32,16>;
 for(var i=0u;i<16u;i++){acc[i]=0.0;}
 for(var k0=0u;k0<${K}u;k0+=16u){
  for(var e=0u;e<2u;e++){let z=lid.x*2u+e;let tr=z>>4u;let k=z&15u;let t=t0+tr;xt[tr*17u+k]=select(0.0,x[t*${K}u+k0+k],t<u.rows);}
  let wr=lid.x>>1u;let half=lid.x&1u;let row=n0+wr;
  for(var j=0u;j<8u;j++){wt[wr*17u+half*8u+j]=select(0.0,loadWeight(row,k0+half*8u+j),row<${N}u);}
  workgroupBarrier();
  for(var kk=0u;kk<16u;kk++){let a0=xt[ty*17u+kk];let a1=xt[(ty+8u)*17u+kk];let a2=xt[(ty+16u)*17u+kk];let a3=xt[(ty+24u)*17u+kk];
   for(var i=0u;i<4u;i++){let b=wt[(nx+i*32u)*17u+kk];acc[i*4u]+=b*a0;acc[i*4u+1u]+=b*a1;acc[i*4u+2u]+=b*a2;acc[i*4u+3u]+=b*a3;}}
  workgroupBarrier();
 }
 for(var i=0u;i<4u;i++){let n=n0+nx+i*32u;if(n>=${N}u){continue;}for(var j=0u;j<4u;j++){let t=t0+ty+j*8u;if(t<u.rows){let dst=t*${OSTRIDE}u+${OUTOFF}u+n;${RESIDUAL ? 'out[dst]+=acc[i*4u+j];' : 'out[dst]=acc[i*4u+j];'}}}}
}`;

export const gatherGGML = ({ START = 0, NUM, K, TYPE, SCALE = 1, UNIFORM = 1 }) => /* wgsl */`
${UNIFORM ? 'struct U { rows:u32,pad0:u32,pad1:u32,pad2:u32 }' : ''}
@group(0) @binding(0) var<storage,read> tokens:array<u32>;
@group(0) @binding(1) var<storage,read> weight:array<u32>;
@group(0) @binding(2) var<storage,read_write> out:array<f32>;
${UNIFORM ? '@group(0) @binding(3) var<uniform> u:U;' : ''}
${ggmlLoadWGSL(TYPE, K)}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_id) lid:vec3u){let t=wid.y;let d=wid.x*256u+lid.x;if(${UNIFORM ? 't>=u.rows||' : ''}d>=${K}u){return;}let id=tokens[t];if(id<${START}u||id>=${START + NUM}u){return;}out[t*${K}u+d]=loadWeight(id-${START}u,d)*${Number(SCALE)};}`;
