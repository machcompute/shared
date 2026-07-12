// Complete Gemma 4 E4B safetensors loader.
//
// This intentionally has no import from model.js. It validates the entire
// one-shard checkpoint header and loads text, image, and audio tensors into a
// Q4/FP32 GPU-weight map.
import {
  GEMMA_E4B_CFG as DEFAULT_CFG,
  gemmaLayerTypes,
  isGemmaFullAttention,
} from './gemma-config.js';

const checkpointUrl = (config) => `https://huggingface.co/${config.repo}/resolve/${config.revision}/${config.checkpoint}`;
const STALL_TIMEOUT_MS = 20000;
const QUANT_INPUT_CHUNK_BYTES = 4 * 2 ** 20;
const CACHE_UPLOAD_CHUNK_BYTES = 16 * 2 ** 20;
const MAX_PENDING_GPU_UPLOAD_BYTES = 128 * 2 ** 20;

const shapeSize = (shape) => shape.length ? shape.reduce((n, d) => n * d, 1) : 1;
const matrixRows = (shape) => shape.length > 1 ? shapeSize(shape.slice(0, -1)) : 1;
const sameShape = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);
const p = (name, shape) => ({ name, shape });
const scalar = (name) => p(name, []);

const q4Bytes = ({ N, K }) => ({ q: N * K / 2, s: N * K / 8 });

function addQ4(out, name, parts, { N, K, shape = [N, K], layout, modality }) {
  if (!Number.isInteger(N) || !Number.isInteger(K) || K % 32) {
    throw new Error(`Invalid Q4 layout for ${name}: ${N} x ${K}`);
  }
  if (parts.some((part) => part.shape[part.shape.length - 1] !== K)
      || parts.reduce((n, part) => n + matrixRows(part.shape), 0) !== N) {
    throw new Error(`Q4 source shape does not match ${name}`);
  }
  out.push({ name, parts, N, K, shape, layout, modality, transform: 'bf16-q4' });
}

function addF32(out, name, parts, { shape, modality }) {
  const elements = parts.reduce((n, part) => n + shapeSize(part.shape), 0);
  const outputShape = shape ?? (parts.length === 1 ? parts[0].shape : [elements]);
  if (shapeSize(outputShape) !== elements) throw new Error(`FP32 source shape does not match ${name}`);
  out.push({ name, parts, shape: outputShape, elements, modality, transform: 'bf16-f32' });
}

function addClip(out, name, prefix, modality) {
  // Gemma's image/audio linears clamp both their input and output.  Retaining
  // all four BF16 scalar bounds is required for numerically faithful media
  // encoders (they are not optional calibration metadata).
  addF32(out, name, [
    scalar(`${prefix}.input_min`),
    scalar(`${prefix}.input_max`),
    scalar(`${prefix}.output_min`),
    scalar(`${prefix}.output_max`),
  ], { shape: [4], modality });
}

/**
 * Returns the full, header-complete GPU tensor schema.  Q4 entries are dense
 * row-major matrices/embeddings (concatenated in `parts` order); FP32 entries
 * are norms, clip bounds, biases, layer scalars, and non-row-aligned convs.
 */
export function buildGemmaSpecs(CFG = DEFAULT_CFG) {
  const q4 = [];
  const f32 = [];
  const T = CFG.text;
  const V = CFG.vision;
  const A = CFG.audio;

  // ---- Text decoder -------------------------------------------------------
  // The checkpoint currently stores K/V and K norm for every layer.  This is
  // intentional even though config.json declares a shared-KV tail: omitting
  // them would make the loader disagree with the authoritative header.
  for (let i = 0; i < T.layers; i++) {
    const L = (name) => `model.language_model.layers.${i}.${name}`;
    const full = isGemmaFullAttention(CFG, i);
    const headDim = full ? T.globalHeadDim : T.slidingHeadDim;
    const qRows = T.heads * headDim;
    const kvRows = T.kvHeads * headDim;
    const kind = gemmaLayerTypes(CFG)[i];
    const interm = i >= (T.doubleWideMlpFrom ?? Infinity) ? T.interm * 2 : T.interm;

    addQ4(q4, `text.L${i}.qkv`, [
      p(L('self_attn.q_proj.weight'), [qRows, T.hidden]),
      p(L('self_attn.k_proj.weight'), [kvRows, T.hidden]),
      p(L('self_attn.v_proj.weight'), [kvRows, T.hidden]),
    ], {
      N: qRows + 2 * kvRows,
      K: T.hidden,
      layout: { attention: kind, qRows, kRows: kvRows, vRows: kvRows, headDim },
      modality: 'text',
    });
    addQ4(q4, `text.L${i}.o`, [p(L('self_attn.o_proj.weight'), [T.hidden, qRows])], {
      N: T.hidden, K: qRows, modality: 'text',
    });
    addQ4(q4, `text.L${i}.gateup`, [
      p(L('mlp.gate_proj.weight'), [interm, T.hidden]),
      p(L('mlp.up_proj.weight'), [interm, T.hidden]),
    ], { N: 2 * interm, K: T.hidden, modality: 'text' });
    addQ4(q4, `text.L${i}.down`, [p(L('mlp.down_proj.weight'), [T.hidden, interm])], {
      N: T.hidden, K: interm, modality: 'text',
    });
    addQ4(q4, `text.L${i}.pleGate`, [p(L('per_layer_input_gate.weight'), [T.pleDim, T.hidden])], {
      N: T.pleDim, K: T.hidden, modality: 'text',
    });
    addQ4(q4, `text.L${i}.pleProj`, [p(L('per_layer_projection.weight'), [T.hidden, T.pleDim])], {
      N: T.hidden, K: T.pleDim, modality: 'text',
    });

    // Gemma 4 RMSNorm uses checkpoint scales directly (unlike some earlier
    // Gemma-family checkpoints there is no loader-side `+ 1` transform).
    addF32(f32, `text.L${i}.ln1`, [p(L('input_layernorm.weight'), [T.hidden])], { modality: 'text' });
    addF32(f32, `text.L${i}.postAttnNorm`, [p(L('post_attention_layernorm.weight'), [T.hidden])], { modality: 'text' });
    addF32(f32, `text.L${i}.preFfnNorm`, [p(L('pre_feedforward_layernorm.weight'), [T.hidden])], { modality: 'text' });
    addF32(f32, `text.L${i}.postFfnNorm`, [p(L('post_feedforward_layernorm.weight'), [T.hidden])], { modality: 'text' });
    addF32(f32, `text.L${i}.pleNorm`, [p(L('post_per_layer_input_norm.weight'), [T.hidden])], { modality: 'text' });
    addF32(f32, `text.L${i}.qNorm`, [p(L('self_attn.q_norm.weight'), [headDim])], { modality: 'text' });
    addF32(f32, `text.L${i}.kNorm`, [p(L('self_attn.k_norm.weight'), [headDim])], { modality: 'text' });
    addF32(f32, `text.L${i}.layerScalar`, [p(L('layer_scalar'), [1])], { modality: 'text' });
  }
  addQ4(q4, 'text.emb', [p('model.language_model.embed_tokens.weight', [T.vocab, T.hidden])], {
    N: T.vocab, K: T.hidden, modality: 'text',
  });
  // This 1.4 GB Q4 payload is row-sharded automatically when a browser's
  // storage-binding limit requires it.
  addQ4(q4, 'text.pleEmb', [p('model.language_model.embed_tokens_per_layer.weight', [T.vocab, T.plePackedDim])], {
    N: T.vocab, K: T.plePackedDim,
    shape: [T.vocab, T.layers, T.pleDim], modality: 'text',
  });
  addQ4(q4, 'text.pleModelProj', [p('model.language_model.per_layer_model_projection.weight', [T.plePackedDim, T.hidden])], {
    N: T.plePackedDim, K: T.hidden, modality: 'text',
  });
  addF32(f32, 'text.norm', [p('model.language_model.norm.weight', [T.hidden])], { modality: 'text' });
  addF32(f32, 'text.pleProjectionNorm', [p('model.language_model.per_layer_projection_norm.weight', [T.pleDim])], { modality: 'text' });

  // ---- Vision tower -------------------------------------------------------
  for (let i = 0; i < V.layers; i++) {
    const L = (name) => `model.vision_tower.encoder.layers.${i}.${name}`;
    const linear = (name) => `${name}.linear.weight`;
    addQ4(q4, `vision.L${i}.qkv`, [
      p(L(linear('self_attn.q_proj')), [V.hidden, V.hidden]),
      p(L(linear('self_attn.k_proj')), [V.hidden, V.hidden]),
      p(L(linear('self_attn.v_proj')), [V.hidden, V.hidden]),
    ], {
      N: 3 * V.hidden, K: V.hidden,
      layout: { qRows: V.hidden, kRows: V.hidden, vRows: V.hidden, headDim: V.headDim },
      modality: 'vision',
    });
    addQ4(q4, `vision.L${i}.o`, [p(L(linear('self_attn.o_proj')), [V.hidden, V.hidden])], {
      N: V.hidden, K: V.hidden, modality: 'vision',
    });
    addQ4(q4, `vision.L${i}.gateup`, [
      p(L(linear('mlp.gate_proj')), [V.interm, V.hidden]),
      p(L(linear('mlp.up_proj')), [V.interm, V.hidden]),
    ], { N: 2 * V.interm, K: V.hidden, modality: 'vision' });
    addQ4(q4, `vision.L${i}.down`, [p(L(linear('mlp.down_proj')), [V.hidden, V.interm])], {
      N: V.hidden, K: V.interm, modality: 'vision',
    });

    for (const projection of ['q_proj', 'k_proj', 'v_proj', 'o_proj']) {
      addClip(f32, `vision.L${i}.${projection[0]}.clip`, L(`self_attn.${projection}`), 'vision');
    }
    for (const projection of ['gate_proj', 'up_proj', 'down_proj']) {
      const short = projection === 'gate_proj' ? 'gate' : projection === 'up_proj' ? 'up' : 'down';
      addClip(f32, `vision.L${i}.${short}.clip`, L(`mlp.${projection}`), 'vision');
    }
    addF32(f32, `vision.L${i}.ln1`, [p(L('input_layernorm.weight'), [V.hidden])], { modality: 'vision' });
    addF32(f32, `vision.L${i}.postAttnNorm`, [p(L('post_attention_layernorm.weight'), [V.hidden])], { modality: 'vision' });
    addF32(f32, `vision.L${i}.preFfnNorm`, [p(L('pre_feedforward_layernorm.weight'), [V.hidden])], { modality: 'vision' });
    addF32(f32, `vision.L${i}.postFfnNorm`, [p(L('post_feedforward_layernorm.weight'), [V.hidden])], { modality: 'vision' });
    addF32(f32, `vision.L${i}.qNorm`, [p(L('self_attn.q_norm.weight'), [V.headDim])], { modality: 'vision' });
    addF32(f32, `vision.L${i}.kNorm`, [p(L('self_attn.k_norm.weight'), [V.headDim])], { modality: 'vision' });
  }
  addQ4(q4, 'vision.patchProj', [p('model.vision_tower.patch_embedder.input_proj.weight', [V.hidden, V.hidden])], {
    N: V.hidden, K: V.hidden, modality: 'vision',
  });
  addQ4(q4, 'vision.positionEmb', [p('model.vision_tower.patch_embedder.position_embedding_table', [2, V.positionEmbeddingSize, V.hidden])], {
    N: 2 * V.positionEmbeddingSize, K: V.hidden,
    shape: [2, V.positionEmbeddingSize, V.hidden], modality: 'vision',
  });
  addQ4(q4, 'vision.proj', [p('model.embed_vision.embedding_projection.weight', [T.hidden, V.hidden])], {
    N: T.hidden, K: V.hidden, modality: 'vision',
  });

  // ---- Audio tower --------------------------------------------------------
  for (let i = 0; i < A.layers; i++) {
    const L = (name) => `model.audio_tower.layers.${i}.${name}`;
    const linear = (name) => `${name}.linear.weight`;
    const addFfw = (short, checkpoint) => {
      addQ4(q4, `audio.L${i}.${short}.fc1`, [p(L(linear(`${checkpoint}.ffw_layer_1`)), [A.interm, A.hidden])], {
        N: A.interm, K: A.hidden, modality: 'audio',
      });
      addQ4(q4, `audio.L${i}.${short}.fc2`, [p(L(linear(`${checkpoint}.ffw_layer_2`)), [A.hidden, A.interm])], {
        N: A.hidden, K: A.interm, modality: 'audio',
      });
      addClip(f32, `audio.L${i}.${short}.fc1.clip`, L(`${checkpoint}.ffw_layer_1`), 'audio');
      addClip(f32, `audio.L${i}.${short}.fc2.clip`, L(`${checkpoint}.ffw_layer_2`), 'audio');
      addF32(f32, `audio.L${i}.${short}.preNorm`, [p(L(`${checkpoint}.pre_layer_norm.weight`), [A.hidden])], { modality: 'audio' });
      addF32(f32, `audio.L${i}.${short}.postNorm`, [p(L(`${checkpoint}.post_layer_norm.weight`), [A.hidden])], { modality: 'audio' });
    };

    addFfw('ff1', 'feed_forward1');
    addQ4(q4, `audio.L${i}.attn.qkv`, [
      p(L(linear('self_attn.q_proj')), [A.hidden, A.hidden]),
      p(L(linear('self_attn.k_proj')), [A.hidden, A.hidden]),
      p(L(linear('self_attn.v_proj')), [A.hidden, A.hidden]),
    ], {
      N: 3 * A.hidden, K: A.hidden,
      layout: { qRows: A.hidden, kRows: A.hidden, vRows: A.hidden, headDim: A.headDim },
      modality: 'audio',
    });
    addQ4(q4, `audio.L${i}.attn.o`, [p(L(linear('self_attn.post')), [A.hidden, A.hidden])], {
      N: A.hidden, K: A.hidden, modality: 'audio',
    });
    addQ4(q4, `audio.L${i}.attn.relativeK`, [p(L('self_attn.relative_k_proj.weight'), [A.hidden, A.hidden])], {
      N: A.hidden, K: A.hidden, modality: 'audio',
    });
    for (const projection of ['q_proj', 'k_proj', 'v_proj']) {
      addClip(f32, `audio.L${i}.attn.${projection[0]}.clip`, L(`self_attn.${projection}`), 'audio');
    }
    addClip(f32, `audio.L${i}.attn.o.clip`, L('self_attn.post'), 'audio');
    addF32(f32, `audio.L${i}.attn.perDimScale`, [p(L('self_attn.per_dim_scale'), [A.headDim])], { modality: 'audio' });
    addF32(f32, `audio.L${i}.normPreAttn`, [p(L('norm_pre_attn.weight'), [A.hidden])], { modality: 'audio' });
    addF32(f32, `audio.L${i}.normPostAttn`, [p(L('norm_post_attn.weight'), [A.hidden])], { modality: 'audio' });
    addF32(f32, `audio.L${i}.normOut`, [p(L('norm_out.weight'), [A.hidden])], { modality: 'audio' });

    addQ4(q4, `audio.L${i}.lconv.start`, [p(L(linear('lconv1d.linear_start')), [2 * A.hidden, A.hidden])], {
      N: 2 * A.hidden, K: A.hidden, modality: 'audio',
    });
    addQ4(q4, `audio.L${i}.lconv.end`, [p(L(linear('lconv1d.linear_end')), [A.hidden, A.hidden])], {
      N: A.hidden, K: A.hidden, modality: 'audio',
    });
    addClip(f32, `audio.L${i}.lconv.start.clip`, L('lconv1d.linear_start'), 'audio');
    addClip(f32, `audio.L${i}.lconv.end.clip`, L('lconv1d.linear_end'), 'audio');
    addF32(f32, `audio.L${i}.lconv.depthwise`, [p(L('lconv1d.depthwise_conv1d.weight'), [A.hidden, 1, A.convKernel])], { modality: 'audio' });
    addF32(f32, `audio.L${i}.lconv.preNorm`, [p(L('lconv1d.pre_layer_norm.weight'), [A.hidden])], { modality: 'audio' });
    addF32(f32, `audio.L${i}.lconv.convNorm`, [p(L('lconv1d.conv_norm.weight'), [A.hidden])], { modality: 'audio' });
    addFfw('ff2', 'feed_forward2');
  }
  addQ4(q4, 'audio.subsample.inputProj', [p('model.audio_tower.subsample_conv_projection.input_proj_linear.weight', [A.hidden, A.hidden])], {
    N: A.hidden, K: A.hidden, modality: 'audio',
  });
  addF32(f32, 'audio.subsample.conv0', [p('model.audio_tower.subsample_conv_projection.layer0.conv.weight', [A.subsampleChannels[0], 1, 3, 3])], { modality: 'audio' });
  addF32(f32, 'audio.subsample.norm0', [p('model.audio_tower.subsample_conv_projection.layer0.norm.weight', [A.subsampleChannels[0]])], { modality: 'audio' });
  addF32(f32, 'audio.subsample.conv1', [p('model.audio_tower.subsample_conv_projection.layer1.conv.weight', [A.subsampleChannels[1], A.subsampleChannels[0], 3, 3])], { modality: 'audio' });
  addF32(f32, 'audio.subsample.norm1', [p('model.audio_tower.subsample_conv_projection.layer1.norm.weight', [A.subsampleChannels[1]])], { modality: 'audio' });
  addQ4(q4, 'audio.outputProj', [p('model.audio_tower.output_proj.weight', [A.outputDim, A.hidden])], {
    N: A.outputDim, K: A.hidden, modality: 'audio',
  });
  addF32(f32, 'audio.outputBias', [p('model.audio_tower.output_proj.bias', [A.outputDim])], { modality: 'audio' });
  addQ4(q4, 'audio.proj', [p('model.embed_audio.embedding_projection.weight', [T.hidden, A.outputDim])], {
    N: T.hidden, K: A.outputDim, modality: 'audio',
  });

  return { q4, f32 };
}

// Compatibility for existing diagnostics that inspect the original E4B spec.
export const buildGemmaE4BSpecs = () => buildGemmaSpecs(DEFAULT_CFG);

// ---------------------------------------------------------------------------
// HTTP range reads and BF16 conversion.

async function fetchWithTimeout(url, opts, ms = STALL_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ms);
  try {
    return await fetch(url, { ...opts, signal: controller.signal });
  } catch (e) {
    if (e.name === 'AbortError') throw new Error('Connecting to Hugging Face timed out. Check your connection and try again.');
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

async function readChunk(reader, ms = STALL_TIMEOUT_MS) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error('Network read stalled — no data received in 20s. Check your connection and try again.')), ms);
  });
  try {
    return await Promise.race([reader.read(), timeout]);
  } finally {
    clearTimeout(timer);
  }
}

function bf16ToF32(u16) {
  const out = new Float32Array(u16.length);
  const outU32 = new Uint32Array(out.buffer);
  for (let i = 0; i < u16.length; i++) outU32[i] = u16[i] << 16;
  return out;
}

function rawToF32(raw, dtype) {
  const bytes = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
  if (dtype === 'BF16') return bf16ToF32(new Uint16Array(bytes));
  if (dtype === 'F32') return new Float32Array(bytes);
  throw new Error(`Unsupported checkpoint dtype: ${dtype}`);
}

async function fetchShardHeader(config) {
  const url = checkpointUrl(config);
  const prefix = await fetchWithTimeout(url, { headers: { Range: 'bytes=0-7' } });
  if (!prefix.ok) throw new Error(`HF fetch failed (${prefix.status}) for ${config.checkpoint}`);
  const prefixBuffer = await prefix.arrayBuffer();
  if (prefixBuffer.byteLength < 8) throw new Error('Gemma safetensors header prefix is truncated');
  const headerLength = Number(new DataView(prefixBuffer).getBigUint64(0, true));
  if (!Number.isSafeInteger(headerLength) || headerLength < 2 || headerLength > 8 * 2 ** 20) {
    throw new Error(`Invalid Gemma safetensors header size: ${headerLength}`);
  }
  const headerResponse = await fetchWithTimeout(url, { headers: { Range: `bytes=8-${7 + headerLength}` } });
  if (!headerResponse.ok) throw new Error(`HF fetch failed (${headerResponse.status}) for ${config.checkpoint}`);
  const header = JSON.parse(await headerResponse.text());
  const infos = {};
  for (const [name, tensor] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    if (!Array.isArray(tensor.shape) || !Array.isArray(tensor.data_offsets)) {
      throw new Error(`Malformed safetensors entry: ${name}`);
    }
    infos[name] = {
      dtype: tensor.dtype,
      shape: tensor.shape,
      start: 8 + headerLength + tensor.data_offsets[0],
      end: 8 + headerLength + tensor.data_offsets[1],
    };
  }
  return infos;
}

function validateHeader(infos, specs) {
  const used = new Set();
  const problems = [];
  for (const spec of [...specs.q4, ...specs.f32]) {
    for (const part of spec.parts) {
      const info = infos[part.name];
      if (!info) {
        problems.push(`missing ${part.name}`);
      } else if (info.dtype !== 'BF16') {
        problems.push(`${part.name}: expected BF16, got ${info.dtype}`);
      } else if (!sameShape(info.shape, part.shape)) {
        problems.push(`${part.name}: expected [${part.shape}], got [${info.shape}]`);
      }
      if (used.has(part.name)) problems.push(`duplicate schema reference ${part.name}`);
      used.add(part.name);
    }
  }
  const extras = Object.keys(infos).filter((name) => !used.has(name));
  if (extras.length) problems.push(`unmapped checkpoint tensors: ${extras.slice(0, 4).join(', ')}${extras.length > 4 ? ` (+${extras.length - 4})` : ''}`);
  if (problems.length) throw new Error(`Gemma checkpoint schema mismatch: ${problems.slice(0, 6).join('; ')}`);
}

// ---------------------------------------------------------------------------
// BF16 -> Q4 worker pool.

class WorkerPool {
  constructor(n = Math.min(6, Math.max(1, navigator.hardwareConcurrency || 4))) {
    this.pending = new Map();
    this.nextId = 0;
    this.rr = 0;
    this.size = n;
    this.workers = Array.from({ length: n }, () => this.#spawn());
  }

  #spawn() {
    const worker = new Worker('/webgpu-llm/quant-worker.js');
    worker.onmessage = (event) => {
      const { id, qdata, scales } = event.data;
      const pending = this.pending.get(id);
      if (!pending) return;
      this.pending.delete(id);
      pending.resolve({ qdata: new Uint32Array(qdata), scales: new Uint32Array(scales) });
    };
    worker.onerror = (event) => {
      const error = new Error(`Quantization worker failed: ${event.message || 'unknown error'}`);
      this.error = error;
      for (const [id, pending] of this.pending) {
        pending.reject(error);
        this.pending.delete(id);
      }
    };
    return worker;
  }

  quantize(u16, rows, K) {
    if (this.error) return Promise.reject(this.error);
    const id = this.nextId++;
    const worker = this.workers[this.rr++ % this.workers.length];
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      worker.postMessage({ id, u16: u16.buffer, rows, K }, [u16.buffer]);
    });
  }

  destroy() {
    for (const worker of this.workers) worker.terminate();
    this.workers = [];
  }
}

// ---------------------------------------------------------------------------

export class GemmaLoader {
  constructor(gpu, status = () => {}, config = DEFAULT_CFG) {
    this.gpu = gpu;
    this.status = status;
    this.cfg = config;
    this.specs = buildGemmaSpecs(config);
    this.q4ByName = new Map(this.specs.q4.map((spec) => [spec.name, spec]));
    this.f32ByName = new Map(this.specs.f32.map((spec) => [spec.name, spec]));
  }

  async opfsDir() {
    navigator.storage.persist?.().catch(() => {});
    const root = await navigator.storage.getDirectory();
    return root.getDirectoryHandle(this.cfg.cacheNamespace, { create: true });
  }

  async cacheValid() {
    try {
      const dir = await this.opfsDir();
      const text = await (await (await dir.getFileHandle('manifest.json')).getFile()).text();
      const manifest = JSON.parse(text);
      if (manifest.version !== this.cfg.cacheVersion
          || manifest.schemaVersion !== this.cfg.schemaVersion
          || manifest.repo !== this.cfg.repo
          || manifest.revision !== this.cfg.revision
          || manifest.checkpoint !== this.cfg.checkpoint
          || manifest.complete !== true
          || !Array.isArray(manifest.entries)) return null;
      return manifest;
    } catch {
      return null;
    }
  }

  async clearCache() {
    const root = await navigator.storage.getDirectory();
    try { await root.removeEntry(this.cfg.cacheNamespace, { recursive: true }); } catch {}
  }

  /** Loads all text, vision, and audio weights.  F32 entries are GPUBuffer;
   * Q4 entries are `{ q, s, N, K, shape }`, or row-sharded equivalents. */
  async load() {
    if (!this.gpu?.device) throw new Error(`A ready WebGPU device is required before loading ${this.cfg.label} weights.`);
    const dev = this.gpu.device;
    dev.pushErrorScope('out-of-memory');
    dev.pushErrorScope('validation');
    const weights = {};
    let validationError;
    let memoryError;
    try {
      for (const spec of this.specs.q4) this.#makeQ4Buffers(spec, weights);

      const cached = await this.cacheValid();
      const cachedNames = new Set((cached?.entries ?? []).map((entry) => entry.name));
      const missingQ4 = this.specs.q4.filter((spec) => !cachedNames.has(spec.name));
      const missingF32 = this.specs.f32.filter((spec) => !cachedNames.has(spec.name));
      if (cached) await this.#loadFromCache(cached, weights);
      if (!cached || missingQ4.length || missingF32.length) {
        await this.#downloadAndQuantize(weights, cached, missingQ4, missingF32);
      }
    } finally {
      validationError = await dev.popErrorScope().catch(() => null);
      memoryError = await dev.popErrorScope().catch(() => null);
    }
    if (memoryError) {
      throw new Error(`Out of GPU memory while loading ${this.cfg.label}. Full multimodal Q4 weights need several GB of GPU memory.`);
    }
    if (validationError) throw new Error(`GPU error while loading ${this.cfg.label} weights: ${validationError.message}`);
    return weights;
  }

  #makeQ4Buffers(spec, weights) {
    const { q: qbytes, s: sbytes } = q4Bytes(spec);
    const limit = Math.min(this.gpu.limits.maxStorageBufferBindingSize, this.gpu.limits.maxBufferSize);
    if (qbytes <= limit && sbytes <= limit) {
      weights[spec.name] = {
        q: this.gpu.storage(qbytes, `${spec.name}.q`),
        s: this.gpu.storage(sbytes, `${spec.name}.s`),
        N: spec.N,
        K: spec.K,
        shape: spec.shape,
      };
      return;
    }

    const qRowBytes = spec.K / 2;
    const sRowBytes = spec.K / 8;
    let rowsPerShard = Math.floor((limit * 0.9) / Math.max(qRowBytes, sRowBytes));
    // K is always a multiple of 32, so a row boundary is also a Q4 block
    // boundary.  Aligning the common case to 32 rows helps embedding kernels.
    rowsPerShard = Math.floor(rowsPerShard / 32) * 32 || rowsPerShard;
    if (rowsPerShard < 1) throw new Error(`${spec.name}: one Q4 row exceeds this GPU's storage binding limit.`);
    const shards = [];
    for (let start = 0; start < spec.N; start += rowsPerShard) {
      const rows = Math.min(rowsPerShard, spec.N - start);
      shards.push({
        start,
        rows,
        q: this.gpu.storage(rows * qRowBytes, `${spec.name}.q.${start}`),
        s: this.gpu.storage(rows * sRowBytes, `${spec.name}.s.${start}`),
      });
    }
    weights[spec.name] = { shards, N: spec.N, K: spec.K, shape: spec.shape };
  }

  #uploadQ4Rows(spec, weights, start, qdata, scales) {
    const entry = weights[spec.name];
    const qWordsPerRow = spec.K / 8;
    const sWordsPerRow = spec.K / 32;
    const rows = qdata.length / qWordsPerRow;
    const targets = entry.shards ?? [{ start: 0, rows: spec.N, q: entry.q, s: entry.s }];
    for (const target of targets) {
      const first = Math.max(start, target.start);
      const end = Math.min(start + rows, target.start + target.rows);
      if (first >= end) continue;
      const sourceRow = first - start;
      const targetRow = first - target.start;
      const count = end - first;
      this.gpu.upload(target.q, qdata.subarray(sourceRow * qWordsPerRow, (sourceRow + count) * qWordsPerRow), targetRow * qWordsPerRow * 4);
      this.gpu.upload(target.s, scales.subarray(sourceRow * sWordsPerRow, (sourceRow + count) * sWordsPerRow), targetRow * sWordsPerRow * 4);
    }
  }

  #track(bytes) { this._pendingGpuUploads = (this._pendingGpuUploads ?? 0) + bytes; }

  async #flush(force = false) {
    if (!force && (this._pendingGpuUploads ?? 0) < MAX_PENDING_GPU_UPLOAD_BYTES) return;
    if (this._flushPromise) {
      await this._flushPromise;
      if (force && (this._pendingGpuUploads ?? 0)) return this.#flush(true);
      return;
    }
    this._pendingGpuUploads = 0;
    this._flushPromise = this.gpu.device.queue.onSubmittedWorkDone().finally(() => { this._flushPromise = null; });
    await this._flushPromise;
    if (force && (this._pendingGpuUploads ?? 0)) await this.#flush(true);
  }

  async #loadFromCache(manifest, weights) {
    const dir = await this.opfsDir();
    const file = await (await dir.getFileHandle('weights.bin')).getFile();
    const known = manifest.entries.filter((entry) => this.q4ByName.has(entry.name) || this.f32ByName.has(entry.name));
    const total = known.reduce((n, entry) => n + (entry.kind === 'q4' ? entry.qbytes + entry.sbytes : entry.bytes), 0);
    let done = 0;

    for (const entry of known) {
      if (entry.kind === 'q4') {
        const spec = this.q4ByName.get(entry.name);
        const expected = q4Bytes(spec);
        if (entry.qbytes !== expected.q || entry.sbytes !== expected.s) throw new Error(`Cached Q4 layout is invalid for ${entry.name}`);
        const qRowBytes = spec.K / 2;
        const sRowBytes = spec.K / 8;
        const rowsPerChunk = Math.max(1, Math.floor(CACHE_UPLOAD_CHUNK_BYTES / (qRowBytes + sRowBytes)));
        for (let row = 0; row < spec.N; row += rowsPerChunk) {
          const rows = Math.min(rowsPerChunk, spec.N - row);
          const [qbuf, sbuf] = await Promise.all([
            file.slice(entry.off + row * qRowBytes, entry.off + (row + rows) * qRowBytes).arrayBuffer(),
            file.slice(entry.off + entry.qbytes + row * sRowBytes, entry.off + entry.qbytes + (row + rows) * sRowBytes).arrayBuffer(),
          ]);
          const qdata = new Uint32Array(qbuf);
          const scales = new Uint32Array(sbuf);
          this.#uploadQ4Rows(spec, weights, row, qdata, scales);
          this.#track(qdata.byteLength + scales.byteLength);
          await this.#flush();
        }
      } else {
        const spec = this.f32ByName.get(entry.name);
        if (entry.bytes !== spec.elements * 4) throw new Error(`Cached FP32 layout is invalid for ${entry.name}`);
        const buffer = this.gpu.storage(entry.bytes, entry.name);
        weights[entry.name] = buffer;
        for (let offset = 0; offset < entry.bytes; offset += CACHE_UPLOAD_CHUNK_BYTES) {
          const size = Math.min(CACHE_UPLOAD_CHUNK_BYTES, entry.bytes - offset);
          const data = new Float32Array(await file.slice(entry.off + offset, entry.off + offset + size).arrayBuffer());
          this.gpu.upload(buffer, data, offset);
          this.#track(data.byteLength);
          await this.#flush();
        }
      }
      done += entry.kind === 'q4' ? entry.qbytes + entry.sbytes : entry.bytes;
      this.status(`Loading cached ${this.cfg.label} weights… ${(done / 1e9).toFixed(2)} / ${(total / 1e9).toFixed(2)} GB`, 'cache', total ? done / total : 1);
    }
    await this.#flush(true);
  }

  async #fetchTensor(info, onBytes) {
    const url = checkpointUrl(this.cfg);
    const response = await fetchWithTimeout(url, { headers: { Range: `bytes=${info.start}-${info.end - 1}` } });
    if (!response.ok && response.status !== 206) throw new Error(`Range fetch failed: ${response.status}`);
    const reader = response.body?.getReader();
    if (!reader) throw new Error('Range response has no readable body');
    const chunks = [];
    try {
      for (;;) {
        const { done, value } = await readChunk(reader);
        if (done) break;
        chunks.push(value);
        onBytes?.(value.byteLength);
      }
    } catch (error) {
      await reader.cancel().catch(() => {});
      throw error;
    }
    const total = chunks.reduce((n, chunk) => n + chunk.byteLength, 0);
    if (total !== info.end - info.start) throw new Error(`Range fetch returned ${total} bytes, expected ${info.end - info.start}`);
    const raw = new Uint8Array(total);
    let offset = 0;
    for (const chunk of chunks) { raw.set(chunk, offset); offset += chunk.byteLength; }
    return raw;
  }

  // Calls onRows with bounded row-aligned BF16 pieces and returns the number
  // of source rows delivered.  Bounded chunks keep PLE quantization from
  // accumulating gigabytes of queued Worker messages.
  async #fetchTensorRows(info, rowBytes, onRows, onBytes) {
    const url = checkpointUrl(this.cfg);
    const response = await fetchWithTimeout(url, { headers: { Range: `bytes=${info.start}-${info.end - 1}` } });
    if (!response.ok && response.status !== 206) throw new Error(`Range fetch failed: ${response.status}`);
    const reader = response.body?.getReader();
    if (!reader) throw new Error('Range response has no readable body');
    const maxRows = Math.max(1, Math.floor(QUANT_INPUT_CHUNK_BYTES / rowBytes));
    let pending = new Uint8Array(0);
    let rowOffset = 0;
    let received = 0;
    try {
      for (;;) {
        const { done, value } = await readChunk(reader);
        if (done) break;
        received += value.byteLength;
        onBytes?.(value.byteLength);
        let data;
        if (pending.length) {
          data = new Uint8Array(pending.length + value.byteLength);
          data.set(pending);
          data.set(value, pending.length);
        } else {
          data = value;
        }
        let cursor = 0;
        while (data.byteLength - cursor >= rowBytes) {
          const rows = Math.min(maxRows, Math.floor((data.byteLength - cursor) / rowBytes));
          const bytes = rows * rowBytes;
          const piece = data.slice(cursor, cursor + bytes);
          await onRows(new Uint16Array(piece.buffer), rowOffset, rows);
          rowOffset += rows;
          cursor += bytes;
        }
        pending = data.slice(cursor);
      }
    } catch (error) {
      await reader.cancel().catch(() => {});
      throw error;
    }
    if (received !== info.end - info.start) throw new Error(`Range fetch returned ${received} bytes, expected ${info.end - info.start}`);
    if (pending.length) throw new Error('Tensor stream ended mid-row');
    return rowOffset;
  }

  async #runBounded(specs, lanes, run) {
    const queue = [...specs];
    let failure = null;
    const workers = Array.from({ length: Math.min(lanes, queue.length) }, async () => {
      while (!failure && queue.length) {
        const spec = queue.shift();
        try {
          await run(spec);
        } catch (error) {
          failure ??= error;
        }
      }
    });
    await Promise.all(workers);
    if (failure) throw failure;
  }

  async #downloadQ4Spec(spec, infos, weights, pool, entry, write, onBytes) {
    const qRowBytes = spec.K / 2;
    const sRowBytes = spec.K / 8;
    const qWordsPerRow = spec.K / 8;
    const sWordsPerRow = spec.K / 32;
    const maxInFlight = Math.max(2, pool.size * 2);
    const inFlight = new Set();
    let row = 0;
    try {
      for (const part of spec.parts) {
        const info = infos[part.name];
        const expectedRows = matrixRows(part.shape);
        const partRow = row;
        const deliveredRows = await this.#fetchTensorRows(info, spec.K * 2, async (u16, rowOffset, rows) => {
          const task = pool.quantize(u16, rows, spec.K).then(async ({ qdata, scales }) => {
            if (qdata.length !== rows * qWordsPerRow || scales.length !== rows * sWordsPerRow) {
              throw new Error(`Quantization worker returned an invalid layout for ${spec.name}`);
            }
            const start = partRow + rowOffset;
            this.#uploadQ4Rows(spec, weights, start, qdata, scales);
            this.#track(qdata.byteLength + scales.byteLength);
            await write({ type: 'write', position: entry.off + start * qRowBytes, data: qdata });
            await write({ type: 'write', position: entry.off + entry.qbytes + start * sRowBytes, data: scales });
            await this.#flush();
          });
          inFlight.add(task);
          task.then(() => inFlight.delete(task), () => inFlight.delete(task));
          if (inFlight.size >= maxInFlight) await Promise.race(inFlight);
        }, onBytes);
        if (deliveredRows !== expectedRows) throw new Error(`${part.name}: got ${deliveredRows} rows, expected ${expectedRows}`);
        row += expectedRows;
      }
      if (row !== spec.N) throw new Error(`${spec.name}: got ${row} Q4 rows, expected ${spec.N}`);
      await Promise.all(inFlight);
    } catch (error) {
      await Promise.allSettled(inFlight);
      throw error;
    }
  }

  async #downloadF32Spec(spec, infos, weights, entry, write, onBytes) {
    const parts = await Promise.all(spec.parts.map(async (part) => {
      const raw = await this.#fetchTensor(infos[part.name], onBytes);
      return rawToF32(raw, infos[part.name].dtype);
    }));
    const output = new Float32Array(spec.elements);
    let offset = 0;
    for (const part of parts) { output.set(part, offset); offset += part.length; }
    if (offset !== spec.elements) throw new Error(`FP32 transform produced an invalid layout for ${spec.name}`);
    const buffer = this.gpu.storage(output.byteLength, spec.name);
    weights[spec.name] = buffer;
    this.gpu.upload(buffer, output);
    this.#track(output.byteLength);
    await write({ type: 'write', position: entry.off, data: output });
    await this.#flush();
  }

  async #downloadAndQuantize(weights, existing, q4Specs = this.specs.q4, f32Specs = this.specs.f32) {
    const infos = await fetchShardHeader(this.cfg);
    validateHeader(infos, this.specs);
    const uniqueSourceNames = new Set([...q4Specs, ...f32Specs].flatMap((spec) => spec.parts.map((part) => part.name)));
    const totalDownloadBytes = [...uniqueSourceNames].reduce((n, name) => n + (infos[name].end - infos[name].start), 0);
    let downloadedBytes = 0;
    const started = performance.now();
    const onBytes = (bytes) => {
      downloadedBytes += bytes;
      const seconds = Math.max((performance.now() - started) / 1000, 0.001);
      const mbps = downloadedBytes / 1e6 / seconds;
      this.status(
        `Downloading + quantizing ${this.cfg.label}… ${(downloadedBytes / 1e9).toFixed(2)} / ${(totalDownloadBytes / 1e9).toFixed(2)} GB  (${mbps.toFixed(0)} MB/s)`,
        'download',
        totalDownloadBytes ? downloadedBytes / totalDownloadBytes : 1,
      );
    };

    const dir = await this.opfsDir();
    const fileHandle = await dir.getFileHandle('weights.bin', { create: true });
    const writable = await fileHandle.createWritable({ keepExistingData: !!existing });
    let writeTail = Promise.resolve();
    const write = (args) => {
      const result = writeTail.then(() => writable.write(args));
      writeTail = result.catch(() => {});
      return result;
    };

    const manifest = {
      version: this.cfg.cacheVersion,
      schemaVersion: this.cfg.schemaVersion,
      repo: this.cfg.repo,
      revision: this.cfg.revision,
      checkpoint: this.cfg.checkpoint,
      complete: false,
      entries: [...(existing?.entries ?? [])],
    };
    let offset = existing
      ? Math.max(0, ...existing.entries.map((entry) => entry.off + (entry.kind === 'q4' ? entry.qbytes + entry.sbytes : entry.bytes)))
      : 0;
    const q4Entries = new Map();
    const f32Entries = new Map();
    for (const spec of q4Specs) {
      const bytes = q4Bytes(spec);
      const entry = { name: spec.name, kind: 'q4', off: offset, qbytes: bytes.q, sbytes: bytes.s };
      q4Entries.set(spec.name, entry);
      manifest.entries.push(entry);
      offset += bytes.q + bytes.s;
    }
    for (const spec of f32Specs) {
      const entry = { name: spec.name, kind: 'f32', off: offset, bytes: spec.elements * 4 };
      f32Entries.set(spec.name, entry);
      manifest.entries.push(entry);
      offset += entry.bytes;
    }

    const pool = new WorkerPool();
    try {
      // Two range-reading lanes avoid unbounded raw-weight buffering while the
      // shared pool uses all available workers for BF16 -> Q4 conversion.
      await this.#runBounded(q4Specs, 2, (spec) => this.#downloadQ4Spec(
        spec, infos, weights, pool, q4Entries.get(spec.name), write, onBytes,
      ));
      await this.#runBounded(f32Specs, 8, (spec) => this.#downloadF32Spec(
        spec, infos, weights, f32Entries.get(spec.name), write, onBytes,
      ));
      await writeTail;
      await writable.close();
      manifest.complete = true;
      const manifestHandle = await dir.getFileHandle('manifest.json', { create: true });
      const manifestWritable = await manifestHandle.createWritable();
      await manifestWritable.write(JSON.stringify(manifest));
      await manifestWritable.close();
    } catch (error) {
      await writable.abort().catch(() => {});
      throw error;
    } finally {
      pool.destroy();
    }
    await this.#flush(true);
    this.status(`${this.cfg.label} weights ready.`, 'done', 1);
  }
}
