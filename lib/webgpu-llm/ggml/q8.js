import { GGML_TYPE } from '../gguf/parser.js';

// llama.cpp's transient mat-vec operand. Each 32-value block occupies nine
// u32 words: packed f16 (d, d*sum(q)) followed by eight packed i8x4 words.
export const Q8_1_BLOCK_SIZE = 32;
export const Q8_1_WORDS = 9;
export const Q8_1_BYTES = Q8_1_WORDS * Uint32Array.BYTES_PER_ELEMENT;
// Eight fixed 32-lane subgroups each evaluate a small tile of output rows. The Q8_1
// representation and dot products match llama.cpp; this scheduling avoids the
// cross-subgroup scratch reduction used by its portable WebGPU shader.
export const q81OutputsPerSubgroup = () => 3;
export const q81RowsPerWorkgroup = (type) => q81OutputsPerSubgroup(type) * 8;

export const supportsQ81Gemv = (type) => type === GGML_TYPE.Q4_0
  || type === GGML_TYPE.Q4_1
  || type === GGML_TYPE.Q5_K
  || type === GGML_TYPE.Q6_K
  || type === GGML_TYPE.Q8_0;

export const q81ByteLength = (rows, K) => {
  if (!Number.isInteger(rows) || rows < 1 || !Number.isInteger(K) || K < 32 || K % 32) {
    throw new Error(`Q8_1 requires positive rows and K divisible by 32; got rows=${rows}, K=${K}`);
  }
  return rows * K / Q8_1_BLOCK_SIZE * Q8_1_BYTES;
};

/** Quantize f32 activation rows to llama.cpp-compatible Q8_1 blocks. */
export const quantizeQ81 = ({ K }) => {
  if (!Number.isInteger(K) || K < 32 || K % 32) throw new Error(`Q8_1 K must be divisible by 32; got ${K}`);
  const blocks = K / 32;
  return /* wgsl */`
enable subgroups;
requires packed_4x8_integer_dot_product;
@group(0) @binding(0) var<storage,read> x:array<vec4f>;
@group(0) @binding(1) var<storage,read_write> q8:array<u32>;
fn clusterMax8(v:f32)->f32{var r=v;r=max(r,subgroupShuffleXor(r,1u));r=max(r,subgroupShuffleXor(r,2u));r=max(r,subgroupShuffleXor(r,4u));return r;}
fn clusterAdd8(v:i32)->i32{var r=v;r+=subgroupShuffleXor(r,1u);r+=subgroupShuffleXor(r,2u);r+=subgroupShuffleXor(r,4u);return r;}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_id) lid:vec3u){
 let lane=lid.x&7u;let block=wid.x*32u+(lid.x>>3u);let valid=block<${blocks}u;var values=vec4f(0.0);
 if(valid){values=x[wid.y*${K / 4}u+block*8u+lane];}
 let av=abs(values);let amax=clusterMax8(max(max(av.x,av.y),max(av.z,av.w)));let d=amax/127.0;let inv=select(0.0,1.0/d,d>0.0);
 let packed=pack4xI8(vec4i(round(values*inv)));let sum=clusterAdd8(dot4I8Packed(packed,0x01010101u));
 if(valid){let base=(wid.y*${blocks}u+block)*9u;q8[base+1u+lane]=packed;if(lane==0u){q8[base]=pack2x16float(vec2f(d,d*f32(sum)));}}
}`;
};

/** Fuse direct-scale RMSNorm with llama.cpp-compatible Q8_1 activation
 * quantization. This is an activation transform only; GGUF weights remain in
 * their native packed format. One workgroup handles one activation row. */
export const rmsNormQuantizeQ81 = ({ K, EPS = 1e-6 }) => {
  if (!Number.isInteger(K) || K < 32 || K % 32) throw new Error(`RMSNorm Q8_1 K must be divisible by 32; got ${K}`);
  const blocks = K / 32;
  return /* wgsl */`
enable subgroups;
requires packed_4x8_integer_dot_product;
struct U { rows:u32, pad0:u32, pad1:u32, pad2:u32 }
@group(0) @binding(0) var<storage,read> x:array<vec4f>;
@group(0) @binding(1) var<storage,read> gamma:array<vec4f>;
@group(0) @binding(2) var<storage,read_write> q8:array<u32>;
@group(0) @binding(3) var<uniform> u:U;
var<workgroup> partial:array<f32,8>;
var<workgroup> rmsInv:f32;
fn clusterMax8(v:f32)->f32{var r=v;r=max(r,subgroupShuffleXor(r,1u));r=max(r,subgroupShuffleXor(r,2u));r=max(r,subgroupShuffleXor(r,4u));return r;}
fn clusterAdd8(v:i32)->i32{var r=v;r+=subgroupShuffleXor(r,1u);r+=subgroupShuffleXor(r,2u);r+=subgroupShuffleXor(r,4u);return r;}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_id) lid:vec3u,@builtin(subgroup_invocation_id) lane:u32){
 let row=wid.x;if(row>=u.rows){return;}let vecBase=row*${K / 4}u;var ss=0.0;
 for(var i=lid.x;i<${K / 4}u;i+=256u){let v=x[vecBase+i];ss+=dot(v,v);}
 let subgroupSum=subgroupAdd(ss);if(lane==0u){partial[lid.x>>5u]=subgroupSum;}workgroupBarrier();
 let pv=select(0.0,partial[lane],(lid.x>>5u)==0u&&lane<8u);let total=subgroupAdd(pv);
 if(lid.x==0u){rmsInv=inverseSqrt(total/${K}.0+${Number(EPS)});}workgroupBarrier();
 let cluster=lid.x>>3u;let clusterLane=lid.x&7u;
 for(var iter=0u;iter<${Math.ceil(blocks / 32)}u;iter++){let block=iter*32u+cluster;let valid=block<${blocks}u;var values=vec4f(0.0);
  if(valid){let vi=vecBase+block*8u+clusterLane;values=x[vi]*gamma[block*8u+clusterLane]*rmsInv;}
  let av=abs(values);let amax=clusterMax8(max(max(av.x,av.y),max(av.z,av.w)));let d=amax/127.0;let inv=select(0.0,1.0/d,d>0.0);
  let packed=pack4xI8(vec4i(round(values*inv)));let sum=clusterAdd8(dot4I8Packed(packed,0x01010101u));
  if(valid){let base=(row*${blocks}u+block)*9u;q8[base+1u+clusterLane]=packed;if(clusterLane==0u){q8[base]=pack2x16float(vec2f(d,d*f32(sum)));}}
 }
}`;
};

/** Fuse Gemma's RMSNorm residual update with Q8_1 production for the next
 * projection. The f32 residual result is preserved for the following layer. */
export const rmsNormAddQuantizeQ81 = ({ K, EPS = 1e-6 }) => {
  if (!Number.isInteger(K) || K < 32 || K % 32) throw new Error(`RMSNorm-add Q8_1 K must be divisible by 32; got ${K}`);
  const blocks = K / 32;
  return /* wgsl */`
enable subgroups;
requires packed_4x8_integer_dot_product;
struct U { rows:u32, pad0:u32, pad1:u32, pad2:u32 }
@group(0) @binding(0) var<storage,read> x:array<vec4f>;
@group(0) @binding(1) var<storage,read> gamma:array<vec4f>;
@group(0) @binding(2) var<storage,read> residual:array<vec4f>;
@group(0) @binding(3) var<storage,read_write> out:array<vec4f>;
@group(0) @binding(4) var<storage,read_write> q8:array<u32>;
@group(0) @binding(5) var<uniform> u:U;
var<workgroup> red:array<f32,256>;
fn clusterMax8(v:f32)->f32{var r=v;r=max(r,subgroupShuffleXor(r,1u));r=max(r,subgroupShuffleXor(r,2u));r=max(r,subgroupShuffleXor(r,4u));return r;}
fn clusterAdd8(v:i32)->i32{var r=v;r+=subgroupShuffleXor(r,1u);r+=subgroupShuffleXor(r,2u);r+=subgroupShuffleXor(r,4u);return r;}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_id) lid:vec3u){
 let row=wid.x;if(row>=u.rows){return;}let vecBase=row*${K / 4}u;var ss=0.0;
 for(var i=lid.x;i<${K / 4}u;i+=256u){let v=x[vecBase+i];ss+=dot(v,v);}red[lid.x]=ss;workgroupBarrier();
 var step=128u;loop{if(step==0u){break;}if(lid.x<step){red[lid.x]+=red[lid.x+step];}workgroupBarrier();step>>=1u;}
 let rmsInv=inverseSqrt(red[0]/${K}.0+${Number(EPS)});let cluster=lid.x>>3u;let lane=lid.x&7u;
 for(var iter=0u;iter<${Math.ceil(blocks / 32)}u;iter++){let block=iter*32u+cluster;let valid=block<${blocks}u;var values=vec4f(0.0);var vi=0u;
  if(valid){vi=vecBase+block*8u+lane;values=residual[vi]+x[vi]*gamma[block*8u+lane]*rmsInv;out[vi]=values;}
  let av=abs(values);let amax=clusterMax8(max(max(av.x,av.y),max(av.z,av.w)));let d=amax/127.0;let inv=select(0.0,1.0/d,d>0.0);let packed=pack4xI8(vec4i(round(values*inv)));let sum=clusterAdd8(dot4I8Packed(packed,0x01010101u));
  if(valid){let base=(row*${blocks}u+block)*9u;q8[base+1u+lane]=packed;if(lane==0u){q8[base]=pack2x16float(vec2f(d,d*f32(sum)));}}
 }
}`;
};

/** Fuse a normalized residual update with the following direct-scale RMSNorm
 * and Q8_1 production. Both f32 residual semantics and packed Q8_1 semantics
 * match the standalone Gemma kernels. */
export const rmsNormAddNormQuantizeQ81 = ({ K, EPS = 1e-6 }) => {
  if (!Number.isInteger(K) || K < 32 || K % 32) throw new Error(`RMSNorm-add-norm Q8_1 K must be divisible by 32; got ${K}`);
  const blocks = K / 32;
  return /* wgsl */`
enable subgroups;
requires packed_4x8_integer_dot_product;
struct U { rows:u32, pad0:u32, pad1:u32, pad2:u32 }
@group(0) @binding(0) var<storage,read> x:array<vec4f>;
@group(0) @binding(1) var<storage,read> addGamma:array<vec4f>;
@group(0) @binding(2) var<storage,read> residual:array<vec4f>;
@group(0) @binding(3) var<storage,read_write> out:array<vec4f>;
@group(0) @binding(4) var<storage,read> normGamma:array<vec4f>;
@group(0) @binding(5) var<storage,read_write> q8:array<u32>;
@group(0) @binding(6) var<uniform> u:U;
var<workgroup> red:array<f32,256>;
fn clusterMax8(v:f32)->f32{var r=v;r=max(r,subgroupShuffleXor(r,1u));r=max(r,subgroupShuffleXor(r,2u));r=max(r,subgroupShuffleXor(r,4u));return r;}
fn clusterAdd8(v:i32)->i32{var r=v;r+=subgroupShuffleXor(r,1u);r+=subgroupShuffleXor(r,2u);r+=subgroupShuffleXor(r,4u);return r;}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_id) lid:vec3u){
 let row=wid.x;if(row>=u.rows){return;}let vecBase=row*${K / 4}u;var ss=0.0;
 for(var i=lid.x;i<${K / 4}u;i+=256u){let v=x[vecBase+i];ss+=dot(v,v);}red[lid.x]=ss;workgroupBarrier();
 var step=128u;loop{if(step==0u){break;}if(lid.x<step){red[lid.x]+=red[lid.x+step];}workgroupBarrier();step=step>>1u;}
 let addInv=inverseSqrt(red[0]/${K}.0+${Number(EPS)});var outSs=0.0;
 for(var i=lid.x;i<${K / 4}u;i+=256u){let vi=vecBase+i;let v=residual[vi]+x[vi]*addGamma[i]*addInv;out[vi]=v;outSs+=dot(v,v);}
 red[lid.x]=outSs;workgroupBarrier();step=128u;loop{if(step==0u){break;}if(lid.x<step){red[lid.x]+=red[lid.x+step];}workgroupBarrier();step=step>>1u;}
 let normInv=inverseSqrt(red[0]/${K}.0+${Number(EPS)});let cluster=lid.x>>3u;let lane=lid.x&7u;
 for(var iter=0u;iter<${Math.ceil(blocks / 32)}u;iter++){let block=iter*32u+cluster;let valid=block<${blocks}u;var values=vec4f(0.0);
  if(valid){let vi=vecBase+block*8u+lane;values=out[vi]*normGamma[block*8u+lane]*normInv;}
  let av=abs(values);let amax=clusterMax8(max(max(av.x,av.y),max(av.z,av.w)));let d=amax/127.0;let inv=select(0.0,1.0/d,d>0.0);let packed=pack4xI8(vec4i(round(values*inv)));let sum=clusterAdd8(dot4I8Packed(packed,0x01010101u));
  if(valid){let base=(row*${blocks}u+block)*9u;q8[base+1u+lane]=packed;if(lane==0u){q8[base]=pack2x16float(vec2f(d,d*f32(sum)));}}
 }
}`;
};

const formatHelpers = /* wgsl */`
fn rawU32(offset:u32)->u32{let w=offset>>2u;let s=(offset&3u)*8u;if(s==0u){return weight[w];}return (weight[w]>>s)|(weight[w+1u]<<(32u-s));}
fn rawByte(offset:u32)->u32{return (weight[offset>>2u]>>((offset&3u)*8u))&255u;}
fn rawI8(offset:u32)->i32{return i32(rawByte(offset)<<24u)>>24;}
fn halfAt(offset:u32)->f32{return unpack2x16float(rawU32(offset)&65535u).x;}
`;

const q81Accumulators = (outputs) => Array.from(
  { length: outputs },
  (_, row) => `var acc${row}=0.0;`,
).join('');

const q81Rows = (outputs, body) => Array.from(
  { length: outputs },
  (_, row) => `if(row0+${row}u<OUTPUT_ROWS){let outputRow=row0+${row}u;${body(row)}}`,
).join('\n');

const q81Store = ({ outputs, N, OSTRIDE, OUTOFF, RESIDUAL }) => Array.from(
  { length: outputs },
  (_, row) => `acc${row}=subgroupAdd(acc${row});if(lane==0u&&row0+${row}u<${N}u){let dst=inputRow*${OSTRIDE}u+${OUTOFF}u+row0+${row}u;${RESIDUAL ? `out[dst]+=acc${row};` : `out[dst]=acc${row};`}}`,
).join('\n');

const gemvQ5KQ81 = ({ N, K, RESIDUAL, OSTRIDE, OUTOFF, WEIGHT_ROW_OFFSET }) => {
  const outputs = q81OutputsPerSubgroup(GGML_TYPE.Q5_K);
  const rowsPerWorkgroup = outputs * 8;
  const rowBytes = K / 256 * 176;
  const rows = q81Rows(outputs, (row) => `let base=(outputRow+${WEIGHT_ROW_OFFSET}u)*${rowBytes}u+block*176u;
   var sc:u32;var mn:u32;if(group<4u){sc=rawByte(base+4u+group)&63u;mn=rawByte(base+8u+group)&63u;}else{sc=(rawByte(base+8u+group)&15u)|((rawByte(base+group)>>6u)<<4u);mn=(rawByte(base+8u+group)>>4u)|((rawByte(base+4u+group)>>6u)<<4u);}
   let bitPos=pair*2u+select(1u,0u,low);var dot=0i;
   for(var j=0u;j<8u;j++){let packed=rawU32(base+48u+pair*32u+j*4u);let high=((rawU32(base+16u+j*4u)>>bitPos)&0x01010101u)<<4u;let qw=select((packed>>4u)&0x0f0f0f0fu,packed&0x0f0f0f0fu,low)|high;dot+=dot4I8Packed(qw,q8[qbase+1u+j]);}
   let ds=unpack2x16float(q8[qbase]);acc${row}+=halfAt(base)*f32(sc)*ds.x*f32(dot)-halfAt(base+2u)*f32(mn)*ds.y;`);
  return /* wgsl */`
enable subgroups;
requires packed_4x8_integer_dot_product;
@group(0) @binding(0) var<storage,read> weight:array<u32>;
@group(0) @binding(1) var<storage,read> q8:array<u32>;
@group(0) @binding(2) var<storage,read_write> out:array<f32>;
const OUTPUT_ROWS=${N}u;
${formatHelpers}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_id) lid:vec3u,@builtin(subgroup_invocation_id) lane:u32){
 let inputRow=wid.y;let row0=wid.x*${rowsPerWorkgroup}u+(lid.x>>5u)*${outputs}u;${q81Accumulators(outputs)}
 for(var gi=lane;gi<${K / 32}u;gi+=32u){let block=gi>>3u;let group=gi&7u;let pair=group>>1u;let low=(group&1u)==0u;let qbase=(inputRow*${K / 32}u+gi)*9u;
  ${rows}
 }
 ${q81Store({ outputs, N, OSTRIDE, OUTOFF, RESIDUAL })}
}`;
};

const gemvQ6KQ81 = ({ N, K, RESIDUAL, OSTRIDE, OUTOFF, WEIGHT_ROW_OFFSET }) => {
  const outputs = q81OutputsPerSubgroup(GGML_TYPE.Q6_K);
  const rowsPerWorkgroup = outputs * 8;
  const rowBytes = K / 256 * 210;
  const rows = q81Rows(outputs, (row) => `let base=(outputRow+${WEIGHT_ROW_OFFSET}u)*${rowBytes}u+block*210u;let qlo=select(0u,32u,(quarter&1u)!=0u);let shift=quarter*2u;var dot=0i;
   for(var j=0u;j<4u;j++){let packed=rawU32(base+half*64u+qlo+lbase+j*4u);let nib=select(packed&0x0f0f0f0fu,(packed>>4u)&0x0f0f0f0fu,quarter>=2u);let raw=nib|(((rawU32(base+128u+half*32u+lbase+j*4u)>>shift)&0x03030303u)<<4u);let sign=(~raw)&0x20202020u;let centered=(raw^0x20202020u)|(sign<<1u)|(sign<<2u);dot+=dot4I8Packed(centered,q8[qbase+qoff+j]);}
   let scaleIndex=half*8u+(lbase>>4u)+quarter*2u;let d8=unpack2x16float(q8[qbase]).x;acc${row}+=halfAt(base+208u)*f32(rawI8(base+192u+scaleIndex))*d8*f32(dot);`);
  return /* wgsl */`
enable subgroups;
requires packed_4x8_integer_dot_product;
@group(0) @binding(0) var<storage,read> weight:array<u32>;
@group(0) @binding(1) var<storage,read> q8:array<u32>;
@group(0) @binding(2) var<storage,read_write> out:array<f32>;
const OUTPUT_ROWS=${N}u;
${formatHelpers}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_id) lid:vec3u,@builtin(subgroup_invocation_id) lane:u32){
 let inputRow=wid.y;let row0=wid.x*${rowsPerWorkgroup}u+(lid.x>>5u)*${outputs}u;${q81Accumulators(outputs)}
 for(var bi=lane;bi<${K / 16}u;bi+=32u){let block=bi>>4u;let local=bi&15u;let half=local>>3u;let quarter=(local>>1u)&3u;let lbase=(local&1u)*16u;let qbase=(inputRow*${K / 32}u+(bi>>1u))*9u;let qoff=1u+(bi&1u)*4u;
  ${rows}
 }
 ${q81Store({ outputs, N, OSTRIDE, OUTOFF, RESIDUAL })}
}`;
};

/** llama.cpp-style Q4_0/Q4_1/Q8_0 x Q8_1 packed integer GEMV. */
export const gemvQ81GGML = ({ TYPE, N, K, RESIDUAL = 0, OSTRIDE = N, OUTOFF = 0, WEIGHT_ROW_OFFSET = 0 }) => {
  if (!supportsQ81Gemv(TYPE)) throw new Error(`GGML type ${TYPE} has no Q8_1 GEMV`);
  if (![N, K, OSTRIDE].every((value) => Number.isInteger(value) && value > 0) || !Number.isInteger(OUTOFF) || OUTOFF < 0) {
    throw new Error('Invalid Q8_1 GEMV dimensions');
  }
  if (K % 32) throw new Error(`Q8_1 GEMV K=${K} is not divisible by 32`);
  if (TYPE === GGML_TYPE.Q5_K) return gemvQ5KQ81({ N, K, RESIDUAL, OSTRIDE, OUTOFF, WEIGHT_ROW_OFFSET });
  if (TYPE === GGML_TYPE.Q6_K) return gemvQ6KQ81({ N, K, RESIDUAL, OSTRIDE, OUTOFF, WEIGHT_ROW_OFFSET });
  const typeBytes = TYPE === GGML_TYPE.Q4_0 ? 18 : TYPE === GGML_TYPE.Q4_1 ? 20 : 34;
  const rowBytes = K / 32 * typeBytes;
  const outputs = q81OutputsPerSubgroup(TYPE);
  const rowsPerWorkgroup = outputs * 8;
  const load = TYPE === GGML_TYPE.Q4_0 ? (row) => /* wgsl */`
   let base=(outputRow+${WEIGHT_ROW_OFFSET}u)*${rowBytes}u+block*18u;var dot=0i;
   for(var j=0u;j<4u;j++){let packed=rawU32(base+2u+j*4u);dot+=dot4I8Packed(packed&0x0f0f0f0fu,q8[qbase+1u+j])+dot4I8Packed((packed>>4u)&0x0f0f0f0fu,q8[qbase+5u+j]);}
   let ds=unpack2x16float(q8[qbase]);let d=halfAt(base);acc${row}+=f32(dot)*d*ds.x-8.0*d*ds.y;`
    : TYPE === GGML_TYPE.Q4_1 ? (row) => /* wgsl */`
   let base=(outputRow+${WEIGHT_ROW_OFFSET}u)*${rowBytes}u+block*20u;var dot=0i;
   for(var j=0u;j<4u;j++){let packed=rawU32(base+4u+j*4u);dot+=dot4I8Packed(packed&0x0f0f0f0fu,q8[qbase+1u+j])+dot4I8Packed((packed>>4u)&0x0f0f0f0fu,q8[qbase+5u+j]);}
   let ds=unpack2x16float(q8[qbase]);acc${row}+=f32(dot)*halfAt(base)*ds.x+halfAt(base+2u)*ds.y;`
      : (row) => /* wgsl */`
   let base=(outputRow+${WEIGHT_ROW_OFFSET}u)*${rowBytes}u+block*34u;var dot=0i;
   for(var j=0u;j<4u;j++){dot+=dot4I8Packed(rawU32(base+2u+j*8u),q8[qbase+1u+j*2u])+dot4I8Packed(rawU32(base+6u+j*8u),q8[qbase+2u+j*2u]);}
   acc${row}+=f32(dot)*halfAt(base)*unpack2x16float(q8[qbase]).x;`;
  const rows = q81Rows(outputs, load);
  return /* wgsl */`
enable subgroups;
requires packed_4x8_integer_dot_product;
@group(0) @binding(0) var<storage,read> weight:array<u32>;
@group(0) @binding(1) var<storage,read> q8:array<u32>;
@group(0) @binding(2) var<storage,read_write> out:array<f32>;
const OUTPUT_ROWS=${N}u;
${formatHelpers}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_id) lid:vec3u,@builtin(subgroup_invocation_id) lane:u32){
 let inputRow=wid.y;let row0=wid.x*${rowsPerWorkgroup}u+(lid.x>>5u)*${outputs}u;${q81Accumulators(outputs)}
 for(var block=lane;block<${K / 32}u;block+=32u){let qbase=(inputRow*${K / 32}u+block)*9u;
  ${rows}
 }
 ${q81Store({ outputs, N, OSTRIDE, OUTOFF, RESIDUAL })}
}`;
};

/** Gemma Q4_0 gate/up projection with an in-kernel GELU multiply. Both GGUF
 * matrices stay packed and share one transient Q8_1 activation. */
export const gemvGateUpQ81GGML = ({ N, K }) => {
  if (!Number.isInteger(N) || N < 1 || !Number.isInteger(K) || K < 32 || K % 32) {
    throw new Error(`Invalid fused gate/up dimensions N=${N}, K=${K}`);
  }
  const outputs = q81OutputsPerSubgroup(GGML_TYPE.Q4_0);
  const rowsPerWorkgroup = outputs * 8;
  const rowBytes = K / 32 * 18;
  const accumulators = Array.from({ length: outputs }, (_, row) => `var gate${row}=0.0;var up${row}=0.0;`).join('');
  const rows = Array.from({ length: outputs }, (_, row) => `
  if(row0+${row}u<${N}u){let outputRow=row0+${row}u;let base=outputRow*${rowBytes}u+block*18u;var gd=0i;var ud=0i;
   for(var j=0u;j<4u;j++){let gp=gateRaw(base+2u+j*4u);let up=upRaw(base+2u+j*4u);let q0=q8[qbase+1u+j];let q1=q8[qbase+5u+j];
    gd+=dot4I8Packed(gp&0x0f0f0f0fu,q0)+dot4I8Packed((gp>>4u)&0x0f0f0f0fu,q1);ud+=dot4I8Packed(up&0x0f0f0f0fu,q0)+dot4I8Packed((up>>4u)&0x0f0f0f0fu,q1);}
   let ds=unpack2x16float(q8[qbase]);let gscale=gateHalf(base);let uscale=upHalf(base);gate${row}+=f32(gd)*gscale*ds.x-8.0*gscale*ds.y;up${row}+=f32(ud)*uscale*ds.x-8.0*uscale*ds.y;}`
  ).join('');
  const stores = Array.from({ length: outputs }, (_, row) => `
 gate${row}=subgroupAdd(gate${row});up${row}=subgroupAdd(up${row});if(lane==0u&&row0+${row}u<${N}u){out[inputRow*${N}u+row0+${row}u]=gelu(gate${row})*up${row};}`
  ).join('');
  return /* wgsl */`
enable subgroups;
requires packed_4x8_integer_dot_product;
@group(0) @binding(0) var<storage,read> gateWeight:array<u32>;
@group(0) @binding(1) var<storage,read> upWeight:array<u32>;
@group(0) @binding(2) var<storage,read> q8:array<u32>;
@group(0) @binding(3) var<storage,read_write> out:array<f32>;
fn gateRaw(offset:u32)->u32{let w=offset>>2u;let s=(offset&3u)*8u;if(s==0u){return gateWeight[w];}return (gateWeight[w]>>s)|(gateWeight[w+1u]<<(32u-s));}
fn upRaw(offset:u32)->u32{let w=offset>>2u;let s=(offset&3u)*8u;if(s==0u){return upWeight[w];}return (upWeight[w]>>s)|(upWeight[w+1u]<<(32u-s));}
fn gateHalf(offset:u32)->f32{return unpack2x16float(gateRaw(offset)&65535u).x;}
fn upHalf(offset:u32)->f32{return unpack2x16float(upRaw(offset)&65535u).x;}
fn gelu(v:f32)->f32{return 0.5*v*(1.0+tanh(0.7978845608028654*(v+0.044715*v*v*v)));}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_id) lid:vec3u,@builtin(subgroup_invocation_id) lane:u32){
 let inputRow=wid.y;let row0=wid.x*${rowsPerWorkgroup}u+(lid.x>>5u)*${outputs}u;${accumulators}
 for(var block=lane;block<${K / 32}u;block+=32u){let qbase=(inputRow*${K / 32}u+block)*9u;${rows}}
 ${stores}
}`;
};

/** Gemma PLE gate projection with its GELU and per-layer PLE multiply folded
 * into the Q4_0 x Q8_1 GEMV. The 256-wide result is emitted directly as Q8_1
 * for the following PLE projection. */
export const gemvPleGateQ81GGML = ({ N, K, LAYERS, LAYER }) => {
  if (![N, K, LAYERS].every((value) => Number.isInteger(value) && value > 0)
      || K % 32 || !Number.isInteger(LAYER) || LAYER < 0 || LAYER >= LAYERS) {
    throw new Error(`Invalid fused PLE gate dimensions N=${N}, K=${K}, layer=${LAYER}/${LAYERS}`);
  }
  if (N % 32) throw new Error(`Fused PLE gate N=${N} must be divisible by 32`);
  const outputs = 4;
  const rowsPerWorkgroup = outputs * 8;
  const rowBytes = K / 32 * 18;
  const rows = q81Rows(outputs, (row) => `let base=outputRow*${rowBytes}u+block*18u;var dot=0i;
   for(var j=0u;j<4u;j++){let packed=rawU32(base+2u+j*4u);dot+=dot4I8Packed(packed&0x0f0f0f0fu,q8[qbase+1u+j])+dot4I8Packed((packed>>4u)&0x0f0f0f0fu,q8[qbase+5u+j]);}
   let ds=unpack2x16float(q8[qbase]);let d=halfAt(base);acc${row}+=f32(dot)*d*ds.x-8.0*d*ds.y;`);
  const stores = Array.from({ length: outputs }, (_, row) => `acc${row}=subgroupAdd(acc${row});if(lane==0u){let r=row0+${row}u;act[(lid.x>>5u)*4u+${row}u]=gelu(acc${row})*ple[(inputRow*${LAYERS}u+${LAYER}u)*${N}u+r];}`).join('');
  return /* wgsl */`
enable subgroups;
requires packed_4x8_integer_dot_product;
@group(0) @binding(0) var<storage,read> weight:array<u32>;
@group(0) @binding(1) var<storage,read> q8:array<u32>;
@group(0) @binding(2) var<storage,read> ple:array<f32>;
@group(0) @binding(3) var<storage,read_write> outQ8:array<u32>;
var<workgroup> act:array<f32,32>;
const OUTPUT_ROWS=${N}u;
${formatHelpers}
fn gelu(v:f32)->f32{return 0.5*v*(1.0+tanh(0.7978845608028654*(v+0.044715*v*v*v)));}
fn clusterMax8(v:f32)->f32{var r=v;r=max(r,subgroupShuffleXor(r,1u));r=max(r,subgroupShuffleXor(r,2u));r=max(r,subgroupShuffleXor(r,4u));return r;}
fn clusterAdd8(v:i32)->i32{var r=v;r+=subgroupShuffleXor(r,1u);r+=subgroupShuffleXor(r,2u);r+=subgroupShuffleXor(r,4u);return r;}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_id) lid:vec3u,@builtin(subgroup_invocation_id) lane:u32){
 let inputRow=wid.y;let row0=wid.x*${rowsPerWorkgroup}u+(lid.x>>5u)*${outputs}u;${q81Accumulators(outputs)}
 for(var block=lane;block<${K / 32}u;block+=32u){let qbase=(inputRow*${K / 32}u+block)*9u;${rows}}
 ${stores}
 workgroupBarrier();let cl=lid.x&7u;var values=vec4f(0.0);if(lid.x<8u){values=vec4f(act[cl*4u],act[cl*4u+1u],act[cl*4u+2u],act[cl*4u+3u]);}
 let av=abs(values);let amax=clusterMax8(max(max(av.x,av.y),max(av.z,av.w)));let d=amax/127.0;let inv=select(0.0,1.0/d,d>0.0);let packed=pack4xI8(vec4i(round(values*inv)));let sum=clusterAdd8(dot4I8Packed(packed,0x01010101u));
 if(lid.x<8u){let base=(inputRow*${N / 32}u+wid.x)*9u;outQ8[base+1u+cl]=packed;if(cl==0u){outQ8[base]=pack2x16float(vec2f(d,d*f32(sum)));}}
}`;
};
