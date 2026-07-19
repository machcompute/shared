// Model-specific tensor vocabulary for Gemma 4.  All GGUF mechanics live in
// GGUFCache; this adapter only defines how official llama.cpp tensor names map
// onto the stable logical names consumed by GemmaModel and its media towers.
import { GGUFCache } from './gguf/cache.js';
import { sourceForModel } from './gguf/sources.js';
import {
  GEMMA_E4B_CFG as DEFAULT_CFG,
  gemmaLayerTypes,
  isGemmaDeclaredKvSharedLayer,
} from './gemma-config.js';

const raw = (logical, role, tensors) => ({ logical, role, tensors: (Array.isArray(tensors) ? tensors : [tensors]).map((tensor) => ({ tensor })), raw: true });
const matrix = (logical, role, tensors, N, K, extra = {}) => ({
  logical, role, N, K, ...extra,
  tensors: tensors.map((part) => typeof part === 'string' ? { tensor: part, outOffset: 0 } : part),
});
const clip = (logical, role, prefix) => raw(logical, role, [
  `${prefix}.input_min`, `${prefix}.input_max`, `${prefix}.output_min`, `${prefix}.output_max`,
]);

export function buildGemmaGGUFMap(CFG = DEFAULT_CFG) {
  const specs = [];
  const T = CFG.text, V = CFG.vision, A = CFG.audio;
  for (let i = 0; i < T.layers; i++) {
    const B = `blk.${i}`;
    const full = gemmaLayerTypes(CFG)[i] === 'full_attention';
    const headDim = full ? T.globalHeadDim : T.slidingHeadDim;
    const qRows = T.heads * headDim;
    const kvRows = T.kvHeads * headDim;
    const shared = isGemmaDeclaredKvSharedLayer(CFG, i);
    const parts = [{ tensor: `${B}.attn_q.weight`, outOffset: 0 }];
    if (!shared) parts.push(
      { tensor: `${B}.attn_k.weight`, outOffset: qRows },
      { tensor: `${B}.attn_v.weight`, outOffset: qRows + kvRows },
    );
    specs.push(matrix(`text.L${i}.qkv`, 'model', parts, shared ? qRows : qRows + 2 * kvRows, T.hidden, {
      layout: { qRows, kRows: shared ? 0 : kvRows, vRows: shared ? 0 : kvRows, headDim, shared },
    }));
    specs.push(matrix(`text.L${i}.o`, 'model', [`${B}.attn_output.weight`], T.hidden, qRows));
    const interm = i >= (T.doubleWideMlpFrom ?? Infinity) ? T.interm * 2 : T.interm;
    specs.push(matrix(`text.L${i}.gateup`, 'model', [
      { tensor: `${B}.ffn_gate.weight`, outOffset: 0 },
      { tensor: `${B}.ffn_up.weight`, outOffset: interm },
    ], 2 * interm, T.hidden));
    specs.push(matrix(`text.L${i}.down`, 'model', [`${B}.ffn_down.weight`], T.hidden, interm));
    specs.push(matrix(`text.L${i}.pleGate`, 'model', [`${B}.inp_gate.weight`], T.pleDim, T.hidden));
    specs.push(matrix(`text.L${i}.pleProj`, 'model', [`${B}.proj.weight`], T.hidden, T.pleDim));
    specs.push(raw(`text.L${i}.ln1`, 'model', `${B}.attn_norm.weight`));
    specs.push(raw(`text.L${i}.postAttnNorm`, 'model', `${B}.post_attention_norm.weight`));
    specs.push(raw(`text.L${i}.preFfnNorm`, 'model', `${B}.ffn_norm.weight`));
    specs.push(raw(`text.L${i}.postFfnNorm`, 'model', `${B}.post_ffw_norm.weight`));
    specs.push(raw(`text.L${i}.pleNorm`, 'model', `${B}.post_norm.weight`));
    specs.push(raw(`text.L${i}.qNorm`, 'model', `${B}.attn_q_norm.weight`));
    // Shared-KV tail intentionally omits K norm along with K/V matrices.
    if (!shared) specs.push(raw(`text.L${i}.kNorm`, 'model', `${B}.attn_k_norm.weight`));
    specs.push(raw(`text.L${i}.layerScalar`, 'model', `${B}.layer_output_scale.weight`));
  }
  specs.push(matrix('text.emb', 'model', ['token_embd.weight'], T.vocab, T.hidden));
  specs.push(matrix('text.pleEmb', 'model', ['per_layer_token_embd.weight'], T.vocab, T.plePackedDim));
  specs.push(matrix('text.pleModelProj', 'model', ['per_layer_model_proj.weight'], T.plePackedDim, T.hidden));
  specs.push(raw('text.norm', 'model', 'output_norm.weight'));
  specs.push(raw('text.pleProjectionNorm', 'model', 'per_layer_proj_norm.weight'));

  for (let i = 0; i < V.layers; i++) {
    const B = `v.blk.${i}`;
    specs.push(matrix(`vision.L${i}.qkv`, 'mmproj', [
      { tensor: `${B}.attn_q.weight`, outOffset: 0 },
      { tensor: `${B}.attn_k.weight`, outOffset: V.hidden },
      { tensor: `${B}.attn_v.weight`, outOffset: 2 * V.hidden },
    ], 3 * V.hidden, V.hidden));
    specs.push(matrix(`vision.L${i}.o`, 'mmproj', [`${B}.attn_out.weight`], V.hidden, V.hidden));
    specs.push(matrix(`vision.L${i}.gateup`, 'mmproj', [
      { tensor: `${B}.ffn_gate.weight`, outOffset: 0 },
      { tensor: `${B}.ffn_up.weight`, outOffset: V.interm },
    ], 2 * V.interm, V.hidden));
    specs.push(matrix(`vision.L${i}.down`, 'mmproj', [`${B}.ffn_down.weight`], V.hidden, V.interm));
    for (const [short, source] of [['q', 'attn_q'], ['k', 'attn_k'], ['v', 'attn_v'], ['o', 'attn_out']]) {
      specs.push(clip(`vision.L${i}.${short}.clip`, 'mmproj', `${B}.${source}`));
    }
    for (const source of ['gate', 'up', 'down']) specs.push(clip(`vision.L${i}.${source}.clip`, 'mmproj', `${B}.ffn_${source}`));
    specs.push(raw(`vision.L${i}.ln1`, 'mmproj', `${B}.ln1.weight`));
    specs.push(raw(`vision.L${i}.postAttnNorm`, 'mmproj', `${B}.attn_post_norm.weight`));
    specs.push(raw(`vision.L${i}.preFfnNorm`, 'mmproj', `${B}.ln2.weight`));
    specs.push(raw(`vision.L${i}.postFfnNorm`, 'mmproj', `${B}.ffn_post_norm.weight`));
    specs.push(raw(`vision.L${i}.qNorm`, 'mmproj', `${B}.attn_q_norm.weight`));
    specs.push(raw(`vision.L${i}.kNorm`, 'mmproj', `${B}.attn_k_norm.weight`));
  }
  specs.push(matrix('vision.patchProj', 'mmproj', ['v.patch_embd.weight'], V.hidden, V.hidden, { reshape: [V.hidden, V.hidden] }));
  specs.push(matrix('vision.positionEmb', 'mmproj', ['v.position_embd.weight'], 2 * V.positionEmbeddingSize, V.hidden));
  specs.push(matrix('vision.proj', 'mmproj', ['mm.input_projection.weight'], T.hidden, V.hidden));

  for (let i = 0; i < A.layers; i++) {
    const B = `a.blk.${i}`;
    const addFfw = (logical, suffix) => {
      specs.push(matrix(`audio.L${i}.${logical}.fc1`, 'mmproj', [`${B}.ffn_up${suffix}.weight`], A.interm, A.hidden));
      specs.push(matrix(`audio.L${i}.${logical}.fc2`, 'mmproj', [`${B}.ffn_down${suffix}.weight`], A.hidden, A.interm));
      specs.push(clip(`audio.L${i}.${logical}.fc1.clip`, 'mmproj', `${B}.ffn_up${suffix}`));
      specs.push(clip(`audio.L${i}.${logical}.fc2.clip`, 'mmproj', `${B}.ffn_down${suffix}`));
      specs.push(raw(`audio.L${i}.${logical}.preNorm`, 'mmproj', `${B}.ffn_norm${suffix}.weight`));
      specs.push(raw(`audio.L${i}.${logical}.postNorm`, 'mmproj', `${B}.ffn_post_norm${suffix}.weight`));
    };
    addFfw('ff1', '');
    specs.push(matrix(`audio.L${i}.attn.qkv`, 'mmproj', [
      { tensor: `${B}.attn_q.weight`, outOffset: 0 },
      { tensor: `${B}.attn_k.weight`, outOffset: A.hidden },
      { tensor: `${B}.attn_v.weight`, outOffset: 2 * A.hidden },
    ], 3 * A.hidden, A.hidden));
    specs.push(matrix(`audio.L${i}.attn.o`, 'mmproj', [`${B}.attn_out.weight`], A.hidden, A.hidden));
    specs.push(matrix(`audio.L${i}.attn.relativeK`, 'mmproj', [`${B}.attn_k_rel.weight`], A.hidden, A.hidden));
    for (const [short, source] of [['q', 'attn_q'], ['k', 'attn_k'], ['v', 'attn_v'], ['o', 'attn_out']]) {
      specs.push(clip(`audio.L${i}.attn.${short}.clip`, 'mmproj', `${B}.${source}`));
    }
    specs.push(raw(`audio.L${i}.attn.perDimScale`, 'mmproj', `${B}.per_dim_scale.weight`));
    specs.push(raw(`audio.L${i}.normPreAttn`, 'mmproj', `${B}.attn_pre_norm.weight`));
    specs.push(raw(`audio.L${i}.normPostAttn`, 'mmproj', `${B}.attn_post_norm.weight`));
    specs.push(raw(`audio.L${i}.normOut`, 'mmproj', `${B}.ln2.weight`));
    specs.push(matrix(`audio.L${i}.lconv.start`, 'mmproj', [`${B}.conv_pw1.weight`], 2 * A.hidden, A.hidden));
    specs.push(matrix(`audio.L${i}.lconv.end`, 'mmproj', [`${B}.conv_pw2.weight`], A.hidden, A.hidden));
    specs.push(clip(`audio.L${i}.lconv.start.clip`, 'mmproj', `${B}.conv_pw1`));
    specs.push(clip(`audio.L${i}.lconv.end.clip`, 'mmproj', `${B}.conv_pw2`));
    specs.push(raw(`audio.L${i}.lconv.depthwise`, 'mmproj', `${B}.conv_dw.weight`));
    specs.push(raw(`audio.L${i}.lconv.preNorm`, 'mmproj', `${B}.norm_conv.weight`));
    specs.push(raw(`audio.L${i}.lconv.convNorm`, 'mmproj', `${B}.conv_norm.weight`));
    addFfw('ff2', '_1');
  }
  specs.push(matrix('audio.subsample.inputProj', 'mmproj', ['a.input_projection.weight'], A.hidden, A.hidden));
  specs.push(raw('audio.subsample.conv0', 'mmproj', 'a.conv1d.0.weight'));
  specs.push(raw('audio.subsample.norm0', 'mmproj', 'a.conv1d.0.norm.weight'));
  specs.push(raw('audio.subsample.conv1', 'mmproj', 'a.conv1d.1.weight'));
  specs.push(raw('audio.subsample.norm1', 'mmproj', 'a.conv1d.1.norm.weight'));
  specs.push(matrix('audio.outputProj', 'mmproj', ['a.pre_encode.out.weight'], A.outputDim, A.hidden));
  specs.push(raw('audio.outputBias', 'mmproj', 'a.pre_encode.out.bias'));
  specs.push(matrix('audio.proj', 'mmproj', ['mm.a.input_projection.weight'], T.hidden, A.outputDim));
  return Object.freeze(specs);
}

// Kept as a diagnostic alias, now returning the native logical manifest.
export const buildGemmaSpecs = buildGemmaGGUFMap;
export const buildGemmaE4BSpecs = () => buildGemmaGGUFMap(DEFAULT_CFG);

export class GemmaLoader {
  constructor(gpu, status = () => {}, config = DEFAULT_CFG) {
    this.gpu = gpu;
    this.status = status;
    this.cfg = config;
    this.source = sourceForModel(config.id);
    this.cache = new GGUFCache(this.source, status);
    this.specs = buildGemmaGGUFMap(config);
  }

  cacheValid() { return this.cache.cacheValid(); }
  clearCache() { return this.cache.clear(); }

  async #rawConcat(filename, tensors, label) {
    const gguf = await this.cache.header(filename);
    const infos = tensors.map(({ tensor }) => {
      const info = gguf.tensorsByName.get(tensor);
      if (!info) throw new Error(`${filename}: missing ${tensor}`);
      return info;
    });
    const bytes = infos.reduce((n, info) => n + info.byteLength, 0);
    const buffer = this.gpu.storage(bytes, label);
    const file = await this.cache.file(filename);
    let offset = 0;
    for (const info of infos) {
      const data = new Uint8Array(await file.slice(info.offset, info.end).arrayBuffer());
      this.gpu.upload(buffer, data, offset);
      offset += data.byteLength;
    }
    return buffer;
  }

  async load() {
    if (!this.gpu?.device) throw new Error(`A ready WebGPU device is required before loading ${this.cfg.label} GGUF weights.`);
    await this.cache.ensure();
    const byRole = Object.fromEntries(this.source.files.map((file) => [file.role, file]));
    const weights = {};
    const dev = this.gpu.device;
    dev.pushErrorScope('out-of-memory');
    dev.pushErrorScope('validation');
    let validationError;
    let memoryError;
    try {
      let done = 0;
      for (const spec of this.specs) {
        const file = byRole[spec.role];
        if (spec.raw) {
          weights[spec.logical] = await this.#rawConcat(file.filename, spec.tensors, spec.logical);
        } else {
          const parts = [];
          for (const part of spec.tensors) {
            let weight = await this.cache.uploadTensor(this.gpu, file.filename, part.tensor, { label: `${spec.logical}.${parts.length}` });
            if (spec.reshape && weight.dimensions.length !== 2) {
              weight = Object.freeze({ ...weight, dimensions: spec.reshape, N: spec.N, K: spec.K, rowStride: weight.byteLength / spec.N });
            }
            parts.push(Object.freeze({ ...weight, outOffset: part.outOffset ?? 0 }));
          }
          weights[spec.logical] = parts.length === 1 && parts[0].outOffset === 0
            ? parts[0]
            : Object.freeze({ segments: Object.freeze(parts), N: spec.N, K: spec.K, byteLength: parts.reduce((n, p) => n + p.byteLength, 0), layout: spec.layout });
        }
        done++;
        this.status(`Uploading native ${this.cfg.label} GGUF tensors… ${done} / ${this.specs.length}`, 'cache', done / this.specs.length);
      }
      weights.__nativeBytes = Object.values(weights).reduce((sum, value) => {
        if (!value || typeof value === 'number') return sum;
        return sum + (value.byteLength ?? value.size ?? 0);
      }, 0);
      await dev.queue.onSubmittedWorkDone();
    } finally {
      validationError = await dev.popErrorScope().catch(() => null);
      memoryError = await dev.popErrorScope().catch(() => null);
    }
    if (memoryError) throw new Error(`Out of GPU memory while uploading ${this.cfg.label} native GGUF tensors.`);
    if (validationError) throw new Error(`GPU error while uploading ${this.cfg.label} GGUF weights: ${validationError.message}`);
    this.status(`${this.cfg.label} GGUF weights ready.`, 'done', 1);
    return weights;
  }
}
