import { GGML_LAYOUT, GGML_TYPE, tensorMatrixShape } from './parser.js';
import { CFG, isFullAttn } from '../config.js';
import {
  GEMMA_E2B_CFG,
  GEMMA_E4B_CFG,
  gemmaLayerTypes,
  isGemmaDeclaredKvSharedLayer,
} from '../gemma-config.js';

const hf = (repo, revision, filename) =>
  `https://huggingface.co/${repo}/resolve/${revision}/${encodeURIComponent(filename)}`;

const source = (value) => Object.freeze({ ...value, url: hf(value.repo, value.revision, value.filename) });

export const QWEN_GGUF_SOURCE = source({
  role: 'model',
  repo: 'unsloth/Qwen3.5-4B-MTP-GGUF',
  revision: '86835bf9949e4d14d6860f7910b1340ad4f271a9',
  filename: 'Qwen3.5-4B-Q4_0.gguf',
  byteLength: 2_669_209_920,
  sha256: '14e6ef39302330c63c2c1a1ab548c7f6f1b7e36b3150ca8b42cab7193b0c3669',
  architecture: 'qwen35',
  tensorCount: 441,
});

const GEMMA_E2B_BASE = source({
  role: 'model',
  repo: 'ggml-org/gemma-4-E2B-it-GGUF',
  revision: '858dcdf955fb1b5a43ed2301aea00362fc443a5c',
  filename: 'gemma-4-E2B-it-Q4_0.gguf',
  byteLength: 2_841_481_184,
  sha256: '8e30dff3ac4c8434c49a7036fa15564bdbb6044e42bf04550bf1a096ad7e6a52',
  architecture: 'gemma4',
  tensorCount: 541,
});

const GEMMA_E2B_MMPROJ = source({
  role: 'mmproj',
  repo: 'ggml-org/gemma-4-E2B-it-GGUF',
  revision: '858dcdf955fb1b5a43ed2301aea00362fc443a5c',
  filename: 'mmproj-gemma-4-E2B-it-BF16.gguf',
  byteLength: 986_833_664,
  sha256: '711e1e8f43fa0664adbac493129be1e6c25b81af4b4cdea97c7d798b25c0a3a4',
  architecture: 'clip',
  tensorCount: 1411,
});

const GEMMA_E4B_BASE = source({
  role: 'model',
  repo: 'ggml-org/gemma-4-E4B-it-GGUF',
  revision: '06f24bb269339b2a19a5167199b81e89ef813c10',
  filename: 'gemma-4-E4B-it-Q4_0.gguf',
  byteLength: 4_590_807_392,
  sha256: 'a555b900214b477d8880e7832e0b8925e139b0159640036b09fe472b6f2097f2',
  architecture: 'gemma4',
  tensorCount: 666,
});

const GEMMA_E4B_MMPROJ = source({
  role: 'mmproj',
  repo: 'ggml-org/gemma-4-E4B-it-GGUF',
  revision: '06f24bb269339b2a19a5167199b81e89ef813c10',
  filename: 'mmproj-gemma-4-E4B-it-BF16.gguf',
  byteLength: 991_552_256,
  sha256: 'f77995e4b6a569ab8f0d1bfdb7e8da4a0fa5b9e6f309b9bf3bdb76164d75e29f',
  architecture: 'clip',
  tensorCount: 1411,
});

export const GGUF_MODEL_SOURCES = Object.freeze({
  'Qwen/Qwen3.5-4B': Object.freeze({
    cacheNamespace: 'qwen35-4b-gguf-v1',
    legacyNamespaces: Object.freeze(['qwen35-4b']),
    config: null,
    files: Object.freeze([QWEN_GGUF_SOURCE]),
  }),
  'google/gemma-4-E2B': Object.freeze({
    cacheNamespace: 'gemma4-e2b-it-gguf-v1',
    legacyNamespaces: Object.freeze(['gemma4-e2b-it-q4-v1']),
    config: GEMMA_E2B_CFG,
    files: Object.freeze([GEMMA_E2B_BASE, GEMMA_E2B_MMPROJ]),
  }),
  'google/gemma-4-E4B': Object.freeze({
    cacheNamespace: 'gemma4-e4b-it-gguf-v1',
    legacyNamespaces: Object.freeze(['gemma4-e4b-it-q4-v1']),
    config: GEMMA_E4B_CFG,
    files: Object.freeze([GEMMA_E4B_BASE, GEMMA_E4B_MMPROJ]),
  }),
});

function needTensor(gguf, name, dimensions = null, types = null) {
  const tensor = gguf.tensorsByName.get(name);
  if (!tensor) throw new Error(`GGUF is missing required tensor ${name}`);
  if (dimensions && (tensor.dimensions.length !== dimensions.length || tensor.dimensions.some((v, i) => v !== dimensions[i]))) {
    throw new Error(`${name}: expected dimensions [${dimensions}], got [${tensor.dimensions}]`);
  }
  if (types && !types.includes(tensor.type)) {
    throw new Error(`${name}: expected ${types.join(' or ')}, got ${tensor.typeName}`);
  }
  return tensor;
}

function validateCommon(source, gguf) {
  if (gguf.version !== 3) throw new Error(`${source.filename}: GGUF v3 is required`);
  if (gguf.tensorCount !== source.tensorCount) {
    throw new Error(`${source.filename}: expected ${source.tensorCount} tensors, got ${gguf.tensorCount}`);
  }
  if (gguf.metadata['general.architecture'] !== source.architecture) {
    throw new Error(`${source.filename}: expected ${source.architecture} architecture, got ${gguf.metadata['general.architecture']}`);
  }
  if (gguf.metadata['general.type'] !== source.role) {
    throw new Error(`${source.filename}: expected general.type=${source.role}, got ${gguf.metadata['general.type']}`);
  }
  for (const tensor of gguf.tensors) {
    if (!GGML_LAYOUT[tensor.type]) throw new Error(`${tensor.name}: unsupported GGML type ${tensor.type}`);
  }
}

function validateQwen(gguf) {
  if (Number(gguf.metadata['qwen35.block_count']) !== CFG.layers + 1) throw new Error('Qwen GGUF must contain 32 decoder blocks plus integrated MTP block 32');
  if (Number(gguf.metadata['qwen35.embedding_length']) !== CFG.hidden) throw new Error('Qwen GGUF hidden size mismatch');
  needTensor(gguf, 'token_embd.weight', [CFG.hidden, CFG.vocab], [GGML_TYPE.Q6_K]);
  needTensor(gguf, 'output_norm.weight', [CFG.hidden], [GGML_TYPE.F32]);
  for (let i = 0; i < CFG.layers; i++) {
    needTensor(gguf, `blk.${i}.attn_norm.weight`, [CFG.hidden], [GGML_TYPE.F32]);
    needTensor(gguf, `blk.${i}.post_attention_norm.weight`, [CFG.hidden], [GGML_TYPE.F32]);
    needTensor(gguf, `blk.${i}.ffn_gate.weight`, [CFG.hidden, CFG.interm]);
    needTensor(gguf, `blk.${i}.ffn_up.weight`, [CFG.hidden, CFG.interm]);
    needTensor(gguf, `blk.${i}.ffn_down.weight`, [CFG.interm, CFG.hidden]);
    if (isFullAttn(i)) {
      needTensor(gguf, `blk.${i}.attn_q.weight`);
      needTensor(gguf, `blk.${i}.attn_k.weight`);
      needTensor(gguf, `blk.${i}.attn_v.weight`);
      needTensor(gguf, `blk.${i}.attn_output.weight`);
      needTensor(gguf, `blk.${i}.attn_q_norm.weight`);
      needTensor(gguf, `blk.${i}.attn_k_norm.weight`);
    } else {
      needTensor(gguf, `blk.${i}.attn_qkv.weight`, [CFG.hidden, CFG.qkvDim]);
      needTensor(gguf, `blk.${i}.attn_gate.weight`, [CFG.hidden, CFG.zDim]);
      needTensor(gguf, `blk.${i}.ssm_alpha.weight`, [CFG.hidden, CFG.vHeads]);
      needTensor(gguf, `blk.${i}.ssm_beta.weight`, [CFG.hidden, CFG.vHeads]);
      needTensor(gguf, `blk.${i}.ssm_out.weight`, [CFG.vHeads * CFG.vDim, CFG.hidden]);
      needTensor(gguf, `blk.${i}.ssm_conv1d.weight`, [CFG.convK, CFG.convDim], [GGML_TYPE.F32]);
      needTensor(gguf, `blk.${i}.ssm_a`, [CFG.vHeads], [GGML_TYPE.F32]);
      needTensor(gguf, `blk.${i}.ssm_dt.bias`, [CFG.vHeads], [GGML_TYPE.F32]);
    }
  }
  for (const name of ['nextn.eh_proj.weight', 'nextn.enorm.weight', 'nextn.hnorm.weight', 'nextn.shared_head_norm.weight']) {
    needTensor(gguf, `blk.32.${name}`);
  }
  for (const name of [
    'attn_q.weight', 'attn_k.weight', 'attn_v.weight', 'attn_output.weight',
    'attn_q_norm.weight', 'attn_k_norm.weight', 'attn_norm.weight',
    'post_attention_norm.weight', 'ffn_gate.weight', 'ffn_up.weight',
    'ffn_down.weight',
  ]) needTensor(gguf, `blk.32.${name}`);
}

function validateGemmaBase(gguf, config) {
  const T = config.text;
  if (Number(gguf.metadata['gemma4.block_count']) !== T.layers) throw new Error('Gemma GGUF layer count mismatch');
  if (Number(gguf.metadata['gemma4.embedding_length']) !== T.hidden) throw new Error('Gemma GGUF hidden size mismatch');
  needTensor(gguf, 'token_embd.weight', [T.hidden, T.vocab]);
  needTensor(gguf, 'per_layer_token_embd.weight', [T.plePackedDim, T.vocab]);
  needTensor(gguf, 'per_layer_model_proj.weight', [T.hidden, T.plePackedDim], [GGML_TYPE.BF16]);
  needTensor(gguf, 'output_norm.weight', [T.hidden], [GGML_TYPE.F32]);
  needTensor(gguf, 'per_layer_proj_norm.weight');
  for (let i = 0; i < T.layers; i++) {
    const full = gemmaLayerTypes(config)[i] === 'full_attention';
    const headDim = full ? T.globalHeadDim : T.slidingHeadDim;
    const qRows = T.heads * headDim;
    needTensor(gguf, `blk.${i}.attn_q.weight`, [T.hidden, qRows]);
    if (!isGemmaDeclaredKvSharedLayer(config, i)) {
      needTensor(gguf, `blk.${i}.attn_k.weight`, [T.hidden, T.kvHeads * headDim]);
      needTensor(gguf, `blk.${i}.attn_v.weight`, [T.hidden, T.kvHeads * headDim]);
    }
    needTensor(gguf, `blk.${i}.attn_output.weight`, [qRows, T.hidden]);
    for (const suffix of ['ffn_gate.weight', 'ffn_up.weight', 'ffn_down.weight', 'inp_gate.weight', 'proj.weight']) {
      needTensor(gguf, `blk.${i}.${suffix}`);
    }
    for (const suffix of [
      'attn_norm.weight', 'post_attention_norm.weight', 'ffn_norm.weight',
      'post_ffw_norm.weight', 'post_norm.weight', 'attn_q_norm.weight',
      'layer_output_scale.weight',
    ]) needTensor(gguf, `blk.${i}.${suffix}`);
    if (!isGemmaDeclaredKvSharedLayer(config, i)) needTensor(gguf, `blk.${i}.attn_k_norm.weight`);
  }
}

function validateGemmaMmproj(gguf, config) {
  if (gguf.metadata['clip.vision.projector_type'] !== 'gemma4v' || gguf.metadata['clip.audio.projector_type'] !== 'gemma4a') {
    throw new Error('Gemma mmproj does not contain the required vision and audio projectors');
  }
  if (Number(gguf.metadata['clip.vision.block_count']) !== config.vision.layers
      || Number(gguf.metadata['clip.audio.block_count']) !== config.audio.layers) {
    throw new Error('Gemma mmproj encoder layer count mismatch');
  }
  for (const name of ['v.patch_embd.weight', 'v.position_embd.weight', 'mm.input_projection.weight', 'a.input_projection.weight', 'a.pre_encode.out.weight', 'a.pre_encode.out.bias', 'mm.a.input_projection.weight']) {
    needTensor(gguf, name);
  }
  const needClip = (prefix) => {
    for (const suffix of ['input_min', 'input_max', 'output_min', 'output_max']) needTensor(gguf, `${prefix}.${suffix}`, null, [GGML_TYPE.F32]);
  };
  for (let i = 0; i < config.vision.layers; i++) {
    for (const suffix of ['attn_q.weight', 'attn_k.weight', 'attn_v.weight', 'attn_out.weight', 'ffn_gate.weight', 'ffn_up.weight', 'ffn_down.weight']) {
      needTensor(gguf, `v.blk.${i}.${suffix}`, null, [GGML_TYPE.BF16]);
    }
    for (const source of ['attn_q', 'attn_k', 'attn_v', 'attn_out', 'ffn_gate', 'ffn_up', 'ffn_down']) needClip(`v.blk.${i}.${source}`);
    for (const suffix of ['ln1.weight', 'attn_post_norm.weight', 'ln2.weight', 'ffn_post_norm.weight', 'attn_q_norm.weight', 'attn_k_norm.weight']) {
      needTensor(gguf, `v.blk.${i}.${suffix}`, null, [GGML_TYPE.F32]);
    }
  }
  for (let i = 0; i < config.audio.layers; i++) {
    for (const suffix of ['attn_q.weight', 'attn_k.weight', 'attn_v.weight', 'attn_out.weight', 'attn_k_rel.weight', 'conv_pw1.weight', 'conv_pw2.weight', 'ffn_up.weight', 'ffn_down.weight', 'ffn_up_1.weight', 'ffn_down_1.weight']) {
      needTensor(gguf, `a.blk.${i}.${suffix}`, null, [GGML_TYPE.BF16]);
    }
    for (const source of ['attn_q', 'attn_k', 'attn_v', 'attn_out', 'conv_pw1', 'conv_pw2', 'ffn_up', 'ffn_down', 'ffn_up_1', 'ffn_down_1']) needClip(`a.blk.${i}.${source}`);
    for (const suffix of [
      'per_dim_scale.weight', 'attn_pre_norm.weight', 'attn_post_norm.weight',
      'ln2.weight', 'conv_dw.weight', 'norm_conv.weight', 'conv_norm.weight',
      'ffn_norm.weight', 'ffn_post_norm.weight', 'ffn_norm_1.weight',
      'ffn_post_norm_1.weight',
    ]) needTensor(gguf, `a.blk.${i}.${suffix}`, null, [GGML_TYPE.F32]);
  }
  for (const name of ['a.conv1d.0.weight', 'a.conv1d.0.norm.weight', 'a.conv1d.1.weight', 'a.conv1d.1.norm.weight']) {
    needTensor(gguf, name, null, [GGML_TYPE.F32]);
  }
}

export function validateGGUFSource(source, gguf, config = null) {
  validateCommon(source, gguf);
  if (source.architecture === 'qwen35') validateQwen(gguf);
  else if (source.role === 'model') validateGemmaBase(gguf, config);
  else validateGemmaMmproj(gguf, config);
  return gguf;
}

export function sourceForModel(modelId) {
  const result = GGUF_MODEL_SOURCES[modelId];
  if (!result) throw new Error(`No pinned GGUF source for ${modelId}`);
  return result;
}

export function describeMatrix(gguf, name) { return tensorMatrixShape(needTensor(gguf, name)); }
