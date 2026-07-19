// Native GGUF loader for Qwen3.5-4B.  This module only maps the pinned GGUF
// tensor vocabulary into architecture-level logical weights; parsing,
// validation, download verification, OPFS transactions, sharding, and GPU
// upload are shared with Gemma through GGUFCache.
import { CFG, isFullAttn } from './config.js';
import { GGUFCache } from './gguf/cache.js';
import { sourceForModel } from './gguf/sources.js';

const MODEL_ID = 'Qwen/Qwen3.5-4B';

const segment = (weight, outOffset = 0) => Object.freeze({ ...weight, outOffset });
const segmented = (parts, N, K) => Object.freeze({
  segments: Object.freeze(parts), N, K,
  byteLength: parts.reduce((sum, part) => sum + part.byteLength, 0),
});

export function buildQwenGGUFMap() {
  const specs = [];
  const one = (logical, tensor) => specs.push({ logical, tensors: [{ tensor, outOffset: 0 }] });
  const many = (logical, tensors, N, K) => specs.push({ logical, tensors, N, K });
  for (let i = 0; i < CFG.layers; i++) {
    const B = `blk.${i}`;
    one(`L${i}.ln1`, `${B}.attn_norm.weight`);
    one(`L${i}.ln2`, `${B}.post_attention_norm.weight`);
    if (isFullAttn(i)) {
      many(`L${i}.in`, [
        { tensor: `${B}.attn_q.weight`, outOffset: 0 },
        { tensor: `${B}.attn_k.weight`, outOffset: CFG.qgDim },
        { tensor: `${B}.attn_v.weight`, outOffset: CFG.qgDim + CFG.kvDim / 2 },
      ], CFG.inF, CFG.hidden);
      one(`L${i}.o`, `${B}.attn_output.weight`);
      one(`L${i}.qnw`, `${B}.attn_q_norm.weight`);
      one(`L${i}.knw`, `${B}.attn_k_norm.weight`);
    } else {
      many(`L${i}.in`, [
        { tensor: `${B}.attn_qkv.weight`, outOffset: 0 },
        { tensor: `${B}.attn_gate.weight`, outOffset: CFG.qkvDim },
        { tensor: `${B}.ssm_beta.weight`, outOffset: CFG.qkvDim + CFG.zDim },
        { tensor: `${B}.ssm_alpha.weight`, outOffset: CFG.qkvDim + CFG.zDim + CFG.vHeads },
      ], CFG.inL, CFG.hidden);
      one(`L${i}.out`, `${B}.ssm_out.weight`);
      one(`L${i}.convw`, `${B}.ssm_conv1d.weight`);
      many(`L${i}.adt`, [
        { tensor: `${B}.ssm_a`, outOffset: 0 },
        { tensor: `${B}.ssm_dt.bias`, outOffset: CFG.vHeads },
      ], 1, CFG.vHeads * 2);
      one(`L${i}.gnw`, `${B}.ssm_norm.weight`);
    }
    many(`L${i}.gateup`, [
      { tensor: `${B}.ffn_gate.weight`, outOffset: 0 },
      { tensor: `${B}.ffn_up.weight`, outOffset: CFG.interm },
    ], CFG.interm * 2, CFG.hidden);
    one(`L${i}.down`, `${B}.ffn_down.weight`);
  }
  one('norm', 'output_norm.weight');
  one('emb', 'token_embd.weight');

  const B = 'blk.32';
  many('mtp.in', [
    { tensor: `${B}.attn_q.weight`, outOffset: 0 },
    { tensor: `${B}.attn_k.weight`, outOffset: CFG.qgDim },
    { tensor: `${B}.attn_v.weight`, outOffset: CFG.qgDim + CFG.kvDim / 2 },
  ], CFG.inF, CFG.hidden);
  one('mtp.o', `${B}.attn_output.weight`);
  many('mtp.gateup', [
    { tensor: `${B}.ffn_gate.weight`, outOffset: 0 },
    { tensor: `${B}.ffn_up.weight`, outOffset: CFG.interm },
  ], CFG.interm * 2, CFG.hidden);
  one('mtp.down', `${B}.ffn_down.weight`);
  one('mtp.fc', `${B}.nextn.eh_proj.weight`);
  one('mtp.ln1', `${B}.attn_norm.weight`);
  one('mtp.ln2', `${B}.post_attention_norm.weight`);
  one('mtp.qnw', `${B}.attn_q_norm.weight`);
  one('mtp.knw', `${B}.attn_k_norm.weight`);
  one('mtp.norm', `${B}.nextn.shared_head_norm.weight`);
  one('mtp.preE', `${B}.nextn.enorm.weight`);
  one('mtp.preH', `${B}.nextn.hnorm.weight`);
  return Object.freeze(specs);
}

export class Loader {
  constructor(gpu, status = () => {}) {
    this.gpu = gpu;
    this.status = status;
    this.source = sourceForModel(MODEL_ID);
    this.cache = new GGUFCache(this.source, status);
    this.specs = buildQwenGGUFMap();
  }

  cacheValid() { return this.cache.cacheValid(); }
  clearCache() { return this.cache.clear(); }

  async #rawConcat(filename, tensors, label) {
    const gguf = await this.cache.header(filename);
    const file = await this.cache.file(filename);
    const infos = tensors.map(({ tensor }) => gguf.tensorsByName.get(tensor));
    if (infos.some((info) => !info)) throw new Error(`${label}: missing raw GGUF tensor`);
    const bytes = infos.reduce((n, info) => n + info.byteLength, 0);
    const buffer = this.gpu.storage(bytes, label);
    let offset = 0;
    for (const info of infos) {
      const data = new Uint8Array(await file.slice(info.offset, info.end).arrayBuffer());
      this.gpu.upload(buffer, data, offset);
      offset += data.byteLength;
    }
    return buffer;
  }

  async load() {
    if (!this.gpu?.device) throw new Error('A ready WebGPU device is required before loading Qwen GGUF weights.');
    await this.cache.ensure();
    const source = this.source.files[0];
    const weights = {};
    const dev = this.gpu.device;
    dev.pushErrorScope('out-of-memory');
    dev.pushErrorScope('validation');
    let validationError;
    let memoryError;
    try {
      let done = 0;
      const total = this.specs.reduce((n, spec) => n + spec.tensors.length, 0);
      for (const spec of this.specs) {
        const rawWeight = /(?:\.ln[12]|\.qnw|\.knw|\.convw|\.adt|\.gnw|^norm$|^mtp\.(?:ln[12]|qnw|knw|norm|preE|preH))$/.test(spec.logical);
        if (rawWeight) {
          weights[spec.logical] = await this.#rawConcat(source.filename, spec.tensors, spec.logical);
          done += spec.tensors.length;
          this.status(`Uploading native Qwen GGUF tensors… ${done} / ${total}`, 'cache', done / total);
          continue;
        }
        const parts = [];
        for (const part of spec.tensors) {
          const weight = await this.cache.uploadTensor(this.gpu, source.filename, part.tensor, {
            label: `${spec.logical}.${parts.length}`,
          });
          parts.push(segment(weight, part.outOffset));
          done++;
          this.status(`Uploading native Qwen GGUF tensors… ${done} / ${total}`, 'cache', done / total);
        }
        weights[spec.logical] = parts.length === 1 && parts[0].outOffset === 0
          ? parts[0]
          : segmented(parts, spec.N, spec.K);
      }
      weights.embShards = weights.emb.shards
        ?? [segment({ ...weights.emb, start: 0, rows: weights.emb.N }, 0)];
      weights.__nativeBytes = Object.values(weights).reduce((sum, value) => {
        if (!value || Array.isArray(value)) return sum;
        return sum + (value.byteLength ?? value.size ?? 0);
      }, 0);
      await dev.queue.onSubmittedWorkDone();
    } finally {
      validationError = await dev.popErrorScope().catch(() => null);
      memoryError = await dev.popErrorScope().catch(() => null);
    }
    if (memoryError) throw new Error('Out of GPU memory while uploading the native Qwen GGUF.');
    if (validationError) throw new Error(`GPU error while uploading Qwen GGUF weights: ${validationError.message}`);
    this.status('Qwen GGUF weights ready.', 'done', 1);
    return weights;
  }
}
