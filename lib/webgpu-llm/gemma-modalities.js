// Image, video, and audio towers for Gemma 4 E4B.
//
// The text decoder owns persistent state (and its FP8 KV caches); these
// encoders allocate short-lived buffers for one media item at a time.  That
// keeps a large video/audio request from permanently inflating the model's
// VRAM footprint and lets the public completion API preserve media ordering.
import { GEMMA_E4B_CFG as DEFAULT_CFG } from './gemma-config.js';
import * as KS from './gemma-kernels.js';

const F = Float32Array.BYTES_PER_ELEMENT;
const ceilDiv = (n, d) => Math.ceil(n / d);
const u32 = (...values) => new Uint32Array(values);

function q4Rows(weight, firstRow, rows) {
  if (!weight?.q || !weight?.s || weight.shards) {
    throw new Error('Gemma media projection unexpectedly uses a sharded Q4 matrix.');
  }
  const qRowBytes = weight.K / 2;
  const sRowBytes = weight.K / 8;
  const qOffset = firstRow * qRowBytes;
  const sOffset = firstRow * sRowBytes;
  if (qOffset % 256 || sOffset % 256) {
    throw new Error('Gemma media Q4 component offset is not WebGPU binding aligned.');
  }
  return [
    { buffer: weight.q, offset: qOffset, size: rows * qRowBytes },
    { buffer: weight.s, offset: sOffset, size: rows * sRowBytes },
  ];
}

function temporary(gpu, entries, label) {
  const buffers = [];
  const out = {};
  const uniforms = new Set([
    'uRows',
    'uRelativeRows',
    'uElements',
    'uPool',
    'uConvRows0',
    'uConvRows1',
    'uConv0',
    'uConv1',
  ]);
  for (const [name, bytes] of Object.entries(entries)) {
    const buffer = uniforms.has(name)
      ? gpu.uniform(bytes, `${label}.${name}`)
      : gpu.storage(bytes, `${label}.${name}`);
    buffers.push(buffer);
    out[name] = buffer;
  }
  out.destroy = () => {
    for (const buffer of buffers) buffer.destroy();
  };
  return out;
}

/**
 * Reference WebGPU media encoders.  Each projection remains Q4 and every
 * Gemma4ClippableLinear bounds tensor is consumed before/after its linear.
 */
export class GemmaModalities {
  constructor(gpu, weights, config = DEFAULT_CFG) {
    this.gpu = gpu;
    this.w = weights;
    this.cfg = config;
    this.#pipelines();
  }

  #pipelines() {
    const g = this.gpu;
    const cache = new Map();
    const named = (name, config, factory) => {
      const key = `gemma.media.${name}.${JSON.stringify(config)}`;
      let pipeline = cache.get(key);
      if (!pipeline) {
        pipeline = g.pipeline(key, factory);
        cache.set(key, pipeline);
      }
      return pipeline;
    };
    this.p = {
      gemm: (N, K, options = {}) => named('gemm', { N, K, ...options }, () => KS.gemmQ4({ N, K, ...options })),
      rms: (K, options = {}) => named('rms', { K, ...options }, () => KS.rmsnorm({ K, ...options })),
      ln: (K) => named('ln', { K }, () => KS.layernorm({ K })),
      add: named('add', {}, KS.add),
      clampIn: named('clip.in', {}, () => KS.clampByBounds({ OFFSET: 0 })),
      clampOut: named('clip.out', {}, () => KS.clampByBounds({ OFFSET: 2 })),
      geluPair: (K) => named('geluPair', { K }, () => KS.geluMulPair({ K })),
      gelu: (K) => named('gelu', { K }, () => KS.geluMul({ K })),
      relu: named('relu', {}, KS.relu),
      silu: named('silu', {}, KS.silu),
      glu: (C) => named('glu', { C }, () => KS.glu({ C })),
      conv: named('conv3x3', {}, () => KS.conv2d3x3({ STRIDE: 2, PAD: 1 })),
      depthwise: named('depthwise', {}, () => KS.depthwiseConv1d({ C: 1024, K: 5 })),
      addBias1536: named('addBias', {}, () => KS.addBias({ K: 1536 })),
      scaleHalf: named('half', {}, () => KS.scale({ SCALE: 0.5 })),
      headRms: named('vision.headRms', {}, () => KS.headRmsnorm({ HEADS: 12, HEAD_DIM: 64 })),
      headRmsUnit: named('vision.headRmsUnit', {}, () => KS.headRmsnorm({ HEADS: 12, HEAD_DIM: 64, WITH_GAMMA: 0 })),
      visionRope: named('vision.rope', {}, () => KS.vision2DRope({ HEADS: 12, HEAD_DIM: 64, THETA: 100 })),
      visionPos: named('vision.position', {}, () => KS.addVisionPositionsQ4({ H: 768, TABLE_SIZE: 10240 })),
      // Vision, like text, uses normalized Q/K with a trained unit attention
      // scaling factor rather than the conventional 1/sqrt(head_dim).
      visionAttn: named('vision.attention', {}, () => KS.denseAttention({ HEADS: 12, HEAD_DIM: 64, ATTENTION_SCALE: this.cfg.vision.attentionScale })),
      visionPool: named('vision.pool', {}, () => KS.visionPool({ H: 768, POOL: 3, SCALE: Math.sqrt(768) })),
      audioQk: named('audio.qk', {}, () => KS.audioQkScale({ HEADS: 8, HEAD_DIM: 128 })),
      audioRelative: named('audio.relative', {}, () => KS.audioRelativePositions({ HIDDEN: 1024, REL: 13 })),
      audioAttn: named('audio.attention', {}, () => KS.audioChunkAttention({
        HEADS: this.cfg.audio.heads,
        HEAD_DIM: this.cfg.audio.headDim,
        CHUNK: this.cfg.audio.attentionChunk,
        LEFT: this.cfg.audio.attentionPastHorizon,
        RIGHT: this.cfg.audio.attentionRight,
        SOFTCAP: this.cfg.audio.attentionLogitCap,
      })),
    };
  }

  #run(pass, pipeline, bind, x, y = 1) {
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(x, y);
  }

  #bind(pipeline, buffers) {
    return this.gpu.bind(pipeline, buffers);
  }

  #matmul(pass, rows, N, K, weight, input, output, uRows, firstRow = 0) {
    const matrix = q4Rows(weight, firstRow, N);
    const pipeline = this.p.gemm(N, K);
    this.#run(
      pass,
      pipeline,
      this.#bind(pipeline, [...matrix, input, output, uRows]),
      ceilDiv(N, 64),
      ceilDiv(rows, 4),
    );
  }

  #clippedMatmul(pass, {
    rows, N, K, weight, firstRow = 0, bounds, input, output, scratch, uRows, uElements,
  }) {
    const clipIn = this.#bind(this.p.clampIn, [input, bounds, scratch.clip, uElements]);
    this.#run(pass, this.p.clampIn, clipIn, ceilDiv(rows * K, 256));
    this.#matmul(pass, rows, N, K, weight, scratch.clip, scratch.raw, uRows, firstRow);
    const clipOut = this.#bind(this.p.clampOut, [scratch.raw, bounds, output, uElements]);
    this.#run(pass, this.p.clampOut, clipOut, ceilDiv(rows * N, 256));
  }

  #add(pass, a, b, out, elements, uElements) {
    this.#run(pass, this.p.add, this.#bind(this.p.add, [a, b, out, uElements]), ceilDiv(elements, 256));
  }

  #rms(pass, rows, K, input, gamma, output, uRows, unit = false) {
    const pipeline = this.p.rms(K, unit ? { WITH_GAMMA: 0 } : {});
    this.#run(pass, pipeline, this.#bind(pipeline, [input, gamma, output, uRows]), rows);
  }

  #layernorm(pass, rows, K, input, gamma, output, uRows) {
    const pipeline = this.p.ln(K);
    this.#run(pass, pipeline, this.#bind(pipeline, [input, gamma, output, uRows]), rows);
  }

  #visionLayer(pass, index, rows, buffers, uRows, uElements) {
    const w = this.w;
    const H = this.cfg.vision.hidden;
    const I = this.cfg.vision.interm;
    const L = (name) => w[`vision.L${index}.${name}`];
    const current = buffers.x;
    const alternate = buffers.xAlt;

    this.#rms(pass, rows, H, current, L('ln1'), buffers.xn, uRows);
    const qkv = L('qkv');
    this.#clippedMatmul(pass, { rows, N: H, K: H, weight: qkv, firstRow: 0, bounds: L('q.clip'), input: buffers.xn, output: buffers.q, scratch: buffers, uRows, uElements });
    this.#clippedMatmul(pass, { rows, N: H, K: H, weight: qkv, firstRow: H, bounds: L('k.clip'), input: buffers.xn, output: buffers.k, scratch: buffers, uRows, uElements });
    this.#clippedMatmul(pass, { rows, N: H, K: H, weight: qkv, firstRow: 2 * H, bounds: L('v.clip'), input: buffers.xn, output: buffers.v, scratch: buffers, uRows, uElements });
    this.#run(pass, this.p.headRms, this.#bind(this.p.headRms, [buffers.q, L('qNorm'), buffers.qNorm, uRows]), rows, 12);
    this.#run(pass, this.p.headRms, this.#bind(this.p.headRms, [buffers.k, L('kNorm'), buffers.kNorm, uRows]), rows, 12);
    this.#run(pass, this.p.headRmsUnit, this.#bind(this.p.headRmsUnit, [buffers.v, L('qNorm'), buffers.vNorm, uRows]), rows, 12);
    this.#run(pass, this.p.visionRope, this.#bind(this.p.visionRope, [buffers.qNorm, buffers.kNorm, buffers.positions, buffers.qRot, buffers.kRot, uRows]), ceilDiv(rows * H, 256));
    this.#run(pass, this.p.visionAttn, this.#bind(this.p.visionAttn, [buffers.qRot, buffers.kRot, buffers.vNorm, buffers.attn, uRows]), rows, 12);
    this.#clippedMatmul(pass, { rows, N: H, K: H, weight: L('o'), bounds: L('o.clip'), input: buffers.attn, output: buffers.tmp, scratch: buffers, uRows, uElements });
    this.#rms(pass, rows, H, buffers.tmp, L('postAttnNorm'), buffers.tmpNorm, uRows);
    this.#add(pass, current, buffers.tmpNorm, alternate, rows * H, uElements);

    this.#rms(pass, rows, H, alternate, L('preFfnNorm'), buffers.xn, uRows);
    const gateUp = L('gateup');
    this.#clippedMatmul(pass, { rows, N: I, K: H, weight: gateUp, firstRow: 0, bounds: L('gate.clip'), input: buffers.xn, output: buffers.gate, scratch: buffers, uRows, uElements });
    this.#clippedMatmul(pass, { rows, N: I, K: H, weight: gateUp, firstRow: I, bounds: L('up.clip'), input: buffers.xn, output: buffers.up, scratch: buffers, uRows, uElements });
    const pair = this.p.geluPair(I);
    this.#run(pass, pair, this.#bind(pair, [buffers.gate, buffers.up, buffers.act, uRows]), ceilDiv(rows * I, 256));
    this.#clippedMatmul(pass, { rows, N: H, K: I, weight: L('down'), bounds: L('down.clip'), input: buffers.act, output: buffers.tmp, scratch: buffers, uRows, uElements });
    this.#rms(pass, rows, H, buffers.tmp, L('postFfnNorm'), buffers.tmpNorm, uRows);
    this.#add(pass, alternate, buffers.tmpNorm, current, rows * H, uElements);
  }

  /** Encode one preprocessed image into Gemma language-model soft embeddings. */
  async encodeImage(input) {
    const V = this.cfg.vision;
    if (!input || !(input.pixelValues instanceof Float32Array) || !(input.positionIds instanceof Int32Array)) {
      throw new Error('Gemma image input must contain pixelValues and positionIds tensors.');
    }
    const rows = input.patchCount;
    const patchW = input.width / V.patchSize;
    const patchH = input.height / V.patchSize;
    if (!Number.isInteger(rows) || rows < 1 || patchW * patchH !== rows || patchW % 3 || patchH % 3) {
      throw new Error('Gemma image patches must form an unpadded 3×3-poolable grid.');
    }
    const softRows = (patchW / 3) * (patchH / 3);
    if (softRows !== input.numSoftTokens) throw new Error('Gemma image soft-token count does not match its patch grid.');
    const maxWidth = 2 * V.interm;
    const b = temporary(this.gpu, {
      pixels: rows * V.hidden * F,
      positions: rows * 2 * Int32Array.BYTES_PER_ELEMENT,
      x: rows * V.hidden * F,
      xAlt: rows * V.hidden * F,
      xn: rows * V.hidden * F,
      q: rows * V.hidden * F,
      k: rows * V.hidden * F,
      v: rows * V.hidden * F,
      qNorm: rows * V.hidden * F,
      kNorm: rows * V.hidden * F,
      vNorm: rows * V.hidden * F,
      qRot: rows * V.hidden * F,
      kRot: rows * V.hidden * F,
      attn: rows * V.hidden * F,
      tmp: rows * V.hidden * F,
      tmpNorm: rows * V.hidden * F,
      gate: rows * V.interm * F,
      up: rows * V.interm * F,
      act: rows * V.interm * F,
      clip: rows * maxWidth * F,
      raw: rows * maxWidth * F,
      pool: softRows * V.hidden * F,
      poolNorm: softRows * V.hidden * F,
      projected: softRows * this.cfg.text.hidden * F,
      uRows: 16,
      uElements: 16,
      uPool: 16,
    }, 'gemma.image');
    try {
      const q = this.gpu.device.queue;
      // Gemma 4 deliberately skips ImageNet normalization, but its patch
      // embedder converts [0,1] RGB to [-1,1] before the learned projection.
      const pixels = new Float32Array(rows * V.hidden);
      const sourcePixels = input.pixelValues.subarray(0, rows * V.hidden);
      for (let i = 0; i < pixels.length; i++) pixels[i] = 2 * (sourcePixels[i] - 0.5);
      q.writeBuffer(b.pixels, 0, pixels);
      q.writeBuffer(b.positions, 0, input.positionIds.subarray(0, rows * 2));
      q.writeBuffer(b.uRows, 0, u32(rows, 0, 0, 0));
      q.writeBuffer(b.uElements, 0, u32(rows * maxWidth, 0, 0, 0));
      q.writeBuffer(b.uPool, 0, u32(patchW, patchH, patchW / 3, patchH / 3));
      const enc = this.gpu.device.createCommandEncoder();
      const pass = enc.beginComputePass();
      this.#matmul(pass, rows, V.hidden, V.hidden, this.w['vision.patchProj'], b.pixels, b.x, b.uRows);
      this.#run(pass, this.p.visionPos, this.#bind(this.p.visionPos, [b.x, b.positions, this.w['vision.positionEmb'].q, this.w['vision.positionEmb'].s, b.xAlt, b.uRows]), ceilDiv(V.hidden, 256), rows);
      // Swap the position-augmented input into x without aliasing a bind group.
      const swap = b.x; b.x = b.xAlt; b.xAlt = swap;
      for (let i = 0; i < V.layers; i++) this.#visionLayer(pass, i, rows, b, b.uRows, b.uElements);
      this.#run(pass, this.p.visionPool, this.#bind(this.p.visionPool, [b.x, b.pool, b.uPool]), ceilDiv(V.hidden, 256), softRows);
      // The pooler has a different row count, so use a dedicated uniform for
      // both its RMSNorm and the language projection.
      // A queue write before submit applies to the entire command; emit the
      // final two stages in a second encoder with the soft-row uniform.
      pass.end();
      q.submit([enc.finish()]);
      q.writeBuffer(b.uRows, 0, u32(softRows, 0, 0, 0));
      const project = this.gpu.device.createCommandEncoder();
      const projectPass = project.beginComputePass();
      this.#rms(projectPass, softRows, V.hidden, b.pool, this.w['vision.L0.ln1'], b.poolNorm, b.uRows, true);
      this.#matmul(projectPass, softRows, this.cfg.text.hidden, V.hidden, this.w['vision.proj'], b.poolNorm, b.projected, b.uRows);
      projectPass.end();
      q.submit([project.finish()]);
      const final = await this.gpu.readback(b.projected, softRows * this.cfg.text.hidden * F);
      return new Float32Array(final);
    } finally {
      b.destroy();
    }
  }

  /** Encode each sampled video frame through the vision tower in frame order. */
  async encodeVideo(input) {
    if (!input || !Number.isInteger(input.frameCount) || input.frameCount < 1) {
      throw new Error('Gemma video input must contain at least one sampled frame.');
    }
    const pieces = [];
    const pixelsPerFrame = input.maxPatches * this.cfg.vision.hidden;
    const positionsPerFrame = input.maxPatches * 2;
    for (let frame = 0; frame < input.frameCount; frame++) {
      const embedded = await this.encodeImage({
        pixelValues: input.pixelValues.subarray(frame * pixelsPerFrame, (frame + 1) * pixelsPerFrame),
        positionIds: input.positionIds.subarray(frame * positionsPerFrame, (frame + 1) * positionsPerFrame),
        patchCount: input.patchCountPerFrame,
        numSoftTokens: input.numSoftTokensPerFrame,
        width: input.width,
        height: input.height,
      });
      pieces.push(embedded);
    }
    const width = this.cfg.text.hidden;
    const out = new Float32Array(pieces.reduce((n, piece) => n + piece.length, 0));
    let offset = 0;
    for (const piece of pieces) { out.set(piece, offset); offset += piece.length; }
    if (out.length % width) throw new Error('Gemma video encoder returned an invalid embedding width.');
    return out;
  }

  /** Encode a browser-preprocessed 16-kHz log-mel audio feature matrix. */
  async encodeAudio(input) {
    const A = this.cfg.audio;
    if (!input || !(input.inputFeatures instanceof Float32Array) || !Number.isInteger(input.frameCount) || input.frameCount < 1) {
      throw new Error('Gemma audio input must contain a non-empty log-mel feature matrix.');
    }
    const frames = input.frameCount;
    const h0 = ceilDiv(frames, 2);
    const h1 = ceilDiv(h0, 2);
    const rows = ceilDiv(h1, A.attentionChunk) * A.attentionChunk;
    const maxWidth = A.interm;
    const maxElements = Math.max(rows * maxWidth, h0 * 64 * 128, h1 * 32 * 32);
    const b = temporary(this.gpu, {
      mel: frames * 128 * F,
      conv0: h0 * 64 * 128 * F,
      conv0Norm: h0 * 64 * 128 * F,
      conv1: rows * 32 * 32 * F,
      conv1Norm: rows * 32 * 32 * F,
      x: rows * A.hidden * F,
      xAlt: rows * A.hidden * F,
      xn: rows * A.hidden * F,
      q: rows * A.hidden * F,
      k: rows * A.hidden * F,
      v: rows * A.hidden * F,
      qNorm: rows * A.hidden * F,
      kNorm: rows * A.hidden * F,
      attn: rows * A.hidden * F,
      relative: 13 * A.hidden * F,
      relativeK: 13 * A.hidden * F,
      valid: rows * Uint32Array.BYTES_PER_ELEMENT,
      tmp: rows * A.hidden * F,
      tmpNorm: rows * A.hidden * F,
      scaled: rows * A.hidden * F,
      ff: rows * A.interm * F,
      ffAct: rows * A.interm * F,
      lconvStart: rows * (2 * A.hidden) * F,
      lconv: rows * A.hidden * F,
      clip: rows * maxWidth * F,
      raw: rows * maxWidth * F,
      out1536: rows * A.outputDim * F,
      outBias: rows * A.outputDim * F,
      outNorm: rows * A.outputDim * F,
      projected: rows * this.cfg.text.hidden * F,
      uRows: 16,
      // The relative-position projection has a fixed 13 rows, unlike the
      // chunk-padded audio sequence. Give its GEMM an independent row count.
      uRelativeRows: 16,
      uElements: 16,
      uConvRows0: 16,
      uConvRows1: 16,
      uConv0: 32,
      uConv1: 32,
    }, 'gemma.audio');
    try {
      const q = this.gpu.device.queue;
      const valid = new Uint32Array(rows);
      for (let r = 0; r < h1; r++) {
        const source = Math.min(frames - 1, r * 4);
        valid[r] = input.inputFeaturesMask ? (input.inputFeaturesMask[source] ? 1 : 0) : 1;
      }
      q.writeBuffer(b.mel, 0, input.inputFeatures.subarray(0, frames * 128));
      q.writeBuffer(b.valid, 0, valid);
      q.writeBuffer(b.uRows, 0, u32(rows, 0, 0, 0));
      q.writeBuffer(b.uRelativeRows, 0, u32(13, 0, 0, 0));
      q.writeBuffer(b.uElements, 0, u32(maxElements, 0, 0, 0));
      q.writeBuffer(b.uConvRows0, 0, u32(h0 * 64, 0, 0, 0));
      q.writeBuffer(b.uConvRows1, 0, u32(h1 * 32, 0, 0, 0));
      q.writeBuffer(b.uConv0, 0, u32(128, frames, 1, 64, h0, 128, 0, 0));
      q.writeBuffer(b.uConv1, 0, u32(64, h0, 128, 32, h1, 32, 0, 0));

      const enc = this.gpu.device.createCommandEncoder();
      const pass = enc.beginComputePass();
      this.#run(pass, this.p.conv, this.#bind(this.p.conv, [b.mel, this.w['audio.subsample.conv0'], b.conv0, b.uConv0]), ceilDiv(h0 * 64 * 128, 256));
      this.#layernorm(pass, h0 * 64, 128, b.conv0, this.w['audio.subsample.norm0'], b.conv0Norm, b.uConvRows0);
      this.#run(pass, this.p.relu, this.#bind(this.p.relu, [b.conv0Norm, b.conv0, b.uElements]), ceilDiv(h0 * 64 * 128, 256));
      this.#run(pass, this.p.conv, this.#bind(this.p.conv, [b.conv0, this.w['audio.subsample.conv1'], b.conv1, b.uConv1]), ceilDiv(h1 * 32 * 32, 256));
      this.#layernorm(pass, h1 * 32, 32, b.conv1, this.w['audio.subsample.norm1'], b.conv1Norm, b.uConvRows1);
      this.#run(pass, this.p.relu, this.#bind(this.p.relu, [b.conv1Norm, b.conv1, b.uElements]), ceilDiv(h1 * 32 * 32, 256));
      this.#matmul(pass, rows, A.hidden, A.hidden, this.w['audio.subsample.inputProj'], b.conv1, b.x, b.uRows);
      this.#run(pass, this.p.audioRelative, this.#bind(this.p.audioRelative, [b.relative]), ceilDiv(13 * A.hidden, 256));

      let current = b.x;
      let alternate = b.xAlt;
      const swap = () => { const old = current; current = alternate; alternate = old; };
      for (let i = 0; i < A.layers; i++) {
        const L = (name) => this.w[`audio.L${i}.${name}`];
        // First feed-forward branch.
        this.#rms(pass, rows, A.hidden, current, L('ff1.preNorm'), b.xn, b.uRows);
        this.#clippedMatmul(pass, { rows, N: A.interm, K: A.hidden, weight: L('ff1.fc1'), bounds: L('ff1.fc1.clip'), input: b.xn, output: b.ff, scratch: b, uRows: b.uRows, uElements: b.uElements });
        this.#run(pass, this.p.silu, this.#bind(this.p.silu, [b.ff, b.ffAct, b.uElements]), ceilDiv(rows * A.interm, 256));
        this.#clippedMatmul(pass, { rows, N: A.hidden, K: A.interm, weight: L('ff1.fc2'), bounds: L('ff1.fc2.clip'), input: b.ffAct, output: b.tmp, scratch: b, uRows: b.uRows, uElements: b.uElements });
        this.#rms(pass, rows, A.hidden, b.tmp, L('ff1.postNorm'), b.tmpNorm, b.uRows);
        this.#run(pass, this.p.scaleHalf, this.#bind(this.p.scaleHalf, [b.tmpNorm, b.scaled, b.uElements]), ceilDiv(rows * A.hidden, 256));
        this.#add(pass, current, b.scaled, alternate, rows * A.hidden, b.uElements); swap();

        // Local relative attention.
        this.#rms(pass, rows, A.hidden, current, L('normPreAttn'), b.xn, b.uRows);
        const qkv = L('attn.qkv');
        this.#clippedMatmul(pass, { rows, N: A.hidden, K: A.hidden, weight: qkv, firstRow: 0, bounds: L('attn.q.clip'), input: b.xn, output: b.q, scratch: b, uRows: b.uRows, uElements: b.uElements });
        this.#clippedMatmul(pass, { rows, N: A.hidden, K: A.hidden, weight: qkv, firstRow: A.hidden, bounds: L('attn.k.clip'), input: b.xn, output: b.k, scratch: b, uRows: b.uRows, uElements: b.uElements });
        this.#clippedMatmul(pass, { rows, N: A.hidden, K: A.hidden, weight: qkv, firstRow: 2 * A.hidden, bounds: L('attn.v.clip'), input: b.xn, output: b.v, scratch: b, uRows: b.uRows, uElements: b.uElements });
        this.#run(pass, this.p.audioQk, this.#bind(this.p.audioQk, [b.q, b.k, L('attn.perDimScale'), b.qNorm, b.kNorm, b.uRows]), ceilDiv(A.hidden, 256), rows);
        this.#matmul(pass, 13, A.hidden, A.hidden, L('attn.relativeK'), b.relative, b.relativeK, b.uRelativeRows);
        this.#run(pass, this.p.audioAttn, this.#bind(this.p.audioAttn, [b.qNorm, b.kNorm, b.v, b.relativeK, b.valid, b.attn, b.uRows]), rows / A.attentionChunk, A.heads);
        this.#clippedMatmul(pass, { rows, N: A.hidden, K: A.hidden, weight: L('attn.o'), bounds: L('attn.o.clip'), input: b.attn, output: b.tmp, scratch: b, uRows: b.uRows, uElements: b.uElements });
        this.#rms(pass, rows, A.hidden, b.tmp, L('normPostAttn'), b.tmpNorm, b.uRows);
        this.#add(pass, current, b.tmpNorm, alternate, rows * A.hidden, b.uElements); swap();

        // LightConv branch.
        this.#rms(pass, rows, A.hidden, current, L('lconv.preNorm'), b.xn, b.uRows);
        this.#clippedMatmul(pass, { rows, N: 2 * A.hidden, K: A.hidden, weight: L('lconv.start'), bounds: L('lconv.start.clip'), input: b.xn, output: b.lconvStart, scratch: b, uRows: b.uRows, uElements: b.uElements });
        const glu = this.p.glu(A.hidden);
        this.#run(pass, glu, this.#bind(glu, [b.lconvStart, b.lconv, b.uRows]), ceilDiv(A.hidden, 256), rows);
        this.#run(pass, this.p.depthwise, this.#bind(this.p.depthwise, [b.lconv, L('lconv.depthwise'), b.tmp, b.uRows]), ceilDiv(A.hidden, 256), rows);
        this.#rms(pass, rows, A.hidden, b.tmp, L('lconv.convNorm'), b.tmpNorm, b.uRows);
        this.#run(pass, this.p.silu, this.#bind(this.p.silu, [b.tmpNorm, b.lconv, b.uElements]), ceilDiv(rows * A.hidden, 256));
        this.#clippedMatmul(pass, { rows, N: A.hidden, K: A.hidden, weight: L('lconv.end'), bounds: L('lconv.end.clip'), input: b.lconv, output: b.tmp, scratch: b, uRows: b.uRows, uElements: b.uElements });
        this.#add(pass, current, b.tmp, alternate, rows * A.hidden, b.uElements); swap();

        // Second feed-forward branch and output norm.
        this.#rms(pass, rows, A.hidden, current, L('ff2.preNorm'), b.xn, b.uRows);
        this.#clippedMatmul(pass, { rows, N: A.interm, K: A.hidden, weight: L('ff2.fc1'), bounds: L('ff2.fc1.clip'), input: b.xn, output: b.ff, scratch: b, uRows: b.uRows, uElements: b.uElements });
        this.#run(pass, this.p.silu, this.#bind(this.p.silu, [b.ff, b.ffAct, b.uElements]), ceilDiv(rows * A.interm, 256));
        this.#clippedMatmul(pass, { rows, N: A.hidden, K: A.interm, weight: L('ff2.fc2'), bounds: L('ff2.fc2.clip'), input: b.ffAct, output: b.tmp, scratch: b, uRows: b.uRows, uElements: b.uElements });
        this.#rms(pass, rows, A.hidden, b.tmp, L('ff2.postNorm'), b.tmpNorm, b.uRows);
        this.#run(pass, this.p.scaleHalf, this.#bind(this.p.scaleHalf, [b.tmpNorm, b.scaled, b.uElements]), ceilDiv(rows * A.hidden, 256));
        this.#add(pass, current, b.scaled, alternate, rows * A.hidden, b.uElements); swap();
        this.#rms(pass, rows, A.hidden, current, L('normOut'), alternate, b.uRows); swap();
      }
      this.#matmul(pass, rows, A.outputDim, A.hidden, this.w['audio.outputProj'], current, b.out1536, b.uRows);
      this.#run(pass, this.p.addBias1536, this.#bind(this.p.addBias1536, [b.out1536, this.w['audio.outputBias'], b.outBias, b.uRows]), ceilDiv(A.outputDim, 256), rows);
      this.#rms(pass, rows, A.outputDim, b.outBias, this.w['audio.L0.normOut'], b.outNorm, b.uRows, true);
      this.#matmul(pass, rows, this.cfg.text.hidden, A.outputDim, this.w['audio.proj'], b.outNorm, b.projected, b.uRows);
      pass.end();
      q.submit([enc.finish()]);
      const raw = new Float32Array(await this.gpu.readback(b.projected, rows * this.cfg.text.hidden * F));
      const count = valid.subarray(0, h1).reduce((n, value) => n + value, 0);
      const out = new Float32Array(count * this.cfg.text.hidden);
      let dst = 0;
      for (let row = 0; row < h1; row++) {
        if (!valid[row]) continue;
        out.set(raw.subarray(row * this.cfg.text.hidden, (row + 1) * this.cfg.text.hidden), dst * this.cfg.text.hidden);
        dst++;
      }
      if (!out.length) throw new Error('Gemma audio preprocessing produced no valid soft tokens.');
      return out;
    } finally {
      b.destroy();
    }
  }
}
