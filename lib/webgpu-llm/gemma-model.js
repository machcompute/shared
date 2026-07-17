// Google Gemma 4 E4B execution graph.
//
// This intentionally mirrors the public Model API used by the Qwen runtime:
// `reset`, `prefill`, `decodeBatch`, and the presence-penalty helpers have the
// same semantics.  The graph itself is separate because Gemma's decoder uses
// alternating sliding/global attention, PLE injection, Q4 weights, and an
// symmetric int8 KV cache.
import {
  GEMMA_E4B_CFG as DEFAULT_CFG,
  gemmaSharedKvSource,
  isGemmaDeclaredKvSharedLayer,
  isGemmaFullAttention,
} from './gemma-config.js';
import * as KS from './gemma-kernels.js';
import { rmsnormQ, sampler as samplerKernel, topk } from './kernels.js';
import { GemmaModalities } from './gemma-modalities.js';
import { updateTokenMaskState } from './token-mask-state.js';

const F = Float32Array.BYTES_PER_ELEMENT;
const TOPK = 20;
const TOPK_BLOCK = 1024;
const TOPK_BLOCKS = Math.ceil(DEFAULT_CFG.text.vocab / TOPK_BLOCK);
const MAX_DECODE_BATCH = 8;
const PENALTY_WINDOW = 512;
const MAX_STOP_IDS = 4;
const STOP_PAD = 0xffffffff;
// gemvQ4 emits four logits per X workgroup. WebGPU caps each dispatch
// dimension at 65,535, so the tied 262,144-row embedding head must be split.
const MAX_GEMV_ROWS = 4 * 65_535;

const ceilDiv = (n, d) => Math.ceil(n / d);
const u32 = (...values) => new Uint32Array(values);

/** Absolute fixed-size key chunks touched by a prompt/decode dispatch. */
export function gemmaAttentionChunkRange(basePos, rows, window = 0, chunkSize = 256) {
  const newestPosition = basePos + rows - 1;
  const firstPosition = window > 0
    ? Math.max(0, basePos + 1 - window)
    : 0;
  const base = Math.floor(firstPosition / chunkSize);
  const last = Math.floor(newestPosition / chunkSize);
  return { base, count: last - base + 1 };
}

function q4Parts(weight) {
  if (weight?.shards) return weight.shards;
  if (weight?.q && weight?.s) {
    return [{ start: 0, rows: weight.N, q: weight.q, s: weight.s }];
  }
  throw new Error('Gemma runtime received an invalid Q4 weight entry.');
}

/**
 * Return aligned views into an unsharded, row-major Q4 matrix.  Gemma's
 * concatenated QKV/gate-up tensors are deliberately split this way so each
 * component retains its own clippable-linear calibration bounds.
 */
function q4Rows(weight, firstRow, rows) {
  if (!weight?.q || !weight?.s || weight.shards) {
    throw new Error('A sliced Gemma Q4 matrix unexpectedly uses shards.');
  }
  const qRowBytes = weight.K / 2;
  const sRowBytes = weight.K / 8;
  const qOffset = firstRow * qRowBytes;
  const sOffset = firstRow * sRowBytes;
  if (qOffset % 256 || sOffset % 256) {
    throw new Error('Gemma Q4 component slice is not WebGPU binding aligned.');
  }
  return [
    { buffer: weight.q, offset: qOffset, size: rows * qRowBytes },
    { buffer: weight.s, offset: sOffset, size: rows * sRowBytes },
  ];
}

function q4PartRows(part, K, firstRow, rows) {
  const qRowBytes = K / 2;
  const sRowBytes = K / 8;
  const qOffset = firstRow * qRowBytes;
  const sOffset = firstRow * sRowBytes;
  if (qOffset % 256 || sOffset % 256) {
    throw new Error('Gemma Q4 head slice is not WebGPU binding aligned.');
  }
  return [
    { buffer: part.q, offset: qOffset, size: rows * qRowBytes },
    { buffer: part.s, offset: sOffset, size: rows * sRowBytes },
  ];
}

function ropeTable(maxCtx, pairs, theta) {
  const out = new Float32Array(maxCtx * pairs * 2);
  for (let pos = 0; pos < maxCtx; pos++) {
    const base = pos * pairs * 2;
    for (let pair = 0; pair < pairs; pair++) {
      const angle = pos * Math.pow(theta, -pair / pairs);
      out[base + pair] = Math.cos(angle);
      out[base + pairs + pair] = Math.sin(angle);
    }
  }
  return out;
}

/**
 * Reference-quality E4B text decoder.  It intentionally uses a small prefill
 * chunk (rather than Qwen's large fused batches) because the model's full
 * attention layers have a different head width and int8 cache layout.
 */
export class GemmaModel {
  constructor(gpu, weights, opts = {}) {
    this.gpu = gpu;
    this.w = weights;
    this.cfg = opts.config ?? DEFAULT_CFG;
    this.embeddingWidth = this.cfg.text.hidden;
    this.maxCtx = opts.maxCtx ?? 8192;
    // The tiled prefill GEMM reuses each dequantized weight across 32 prompt
    // rows. E2B's smaller hidden width benefits from a deeper queue; E4B uses
    // a more conservative 64 rows to bound compositor stalls and scratch VRAM.
    const chunk = opts.chunk ?? (this.cfg.text.hidden <= 1536 ? 128 : 64);
    if (!Number.isInteger(chunk) || chunk < 32 || chunk > 256 || chunk % 32 !== 0) {
      throw new Error(`Gemma prefill chunk must be a multiple of 32 from 32 to 256; got ${chunk}`);
    }
    this.chunk = chunk;
    this.pos = 0;
    this.BATCH = 8;
    this.hasMtp = false;
    this.spec = false;
    const subgroupMin = this.gpu.info?.subgroupMinSize;
    const subgroupMax = this.gpu.info?.subgroupMaxSize;
    this.directGemvSubgroups = !!this.gpu.subgroups && subgroupMin === 32 && subgroupMax === 32;
    this.gemvRowLanes = this.directGemvSubgroups ? 32 : 64;
    this.gemvRowsPerWorkgroup = 256 / this.gemvRowLanes;
    this.i8 = this.gpu.dp4a ? 1 : 0;
    this.fusedGateUp = this.i8 && this.directGemvSubgroups;
    this.attentionHeadBatch = 4;
    this.decodeAttentionHeadBatch = 2;
    this.decodeAttentionChunk = 128;
    this.genIds = [];
    this.allowedCount = 0;
    this.tokenMaskKind = 'all';
    this.tokenMaskIds = null;
    this.#alloc();
    this.#pipelines();
    this.#binds();
    this.modalities = new GemmaModalities(gpu, weights, this.cfg);
  }

  #alloc() {
    const g = this.gpu;
    const T = this.cfg.text;
    const C = this.chunk;
    const maxInterm = T.interm * (T.doubleWideMlpFrom === undefined ? 1 : 2);
    const S = (bytes, label) => g.storage(bytes, label);
    const maxQkv = (T.heads + T.kvHeads * 2) * T.globalHeadDim;
    const maxQ = T.heads * T.globalHeadDim;

    this.b = {
      x: S(C * T.hidden * F, 'gemma.x'),
      xAlt: S(C * T.hidden * F, 'gemma.xAlt'),
      xn: S(C * T.hidden * F, 'gemma.xn'),
      // Decode-only RMSNorm output. Q4 GEMVs consume these packed int8
      // activations through DP4a; prefill keeps using the f32 tiled GEMM.
      xq: S(T.hidden, 'gemma.xq'),
      xsm: S(T.hidden / 4, 'gemma.xsm'),
      qkv: S(C * maxQkv * F, 'gemma.qkv'),
      q: S(C * maxQ * F, 'gemma.q'),
      attn: S(C * maxQ * F, 'gemma.attn'),
      tmp: S(C * T.hidden * F, 'gemma.tmp'),
      tmpNorm: S(C * T.hidden * F, 'gemma.tmpNorm'),
      scaled: S(C * T.hidden * F, 'gemma.scaled'),
      gateUp: S(C * (2 * maxInterm) * F, 'gemma.gateUp'),
      act: S(C * maxInterm * F, 'gemma.act'),
      pleIdentity: S(C * T.plePackedDim * F, 'gemma.pleIdentity'),
      pleProjected: S(C * T.plePackedDim * F, 'gemma.pleProjected'),
      pleContext: S(C * T.plePackedDim * F, 'gemma.pleContext'),
      ple: S(C * T.plePackedDim * F, 'gemma.ple'),
      pleTextMask: S(C * Uint32Array.BYTES_PER_ELEMENT, 'gemma.pleTextMask'),
      pleSmall: S(C * T.pleDim * F, 'gemma.pleSmall'),
      pleGate: S(C * T.pleDim * F, 'gemma.pleGate'),
      // Flash-attention chunk partials: per (row, Q head, decode key chunk) an
      // unnormalized accumulator plus its online-softmax max and denominator.
      // Sized for the widest (global) head; sliding layers use a shorter
      // stride within the same scratch buffer.
      attnPartials: S(
        C * T.heads * Math.ceil(this.maxCtx / this.decodeAttentionChunk) * (T.globalHeadDim + 2) * F,
        'gemma.attnPartials',
      ),
      head: S(T.hidden * F, 'gemma.head'),
      headNorm: S(T.hidden * F, 'gemma.headNorm'),
      logits: S(T.vocab * F, 'gemma.logits'),
      cand: S(TOPK_BLOCKS * TOPK * 2 * F, 'gemma.candidates'),
      allowed: S(T.vocab * Uint32Array.BYTES_PER_ELEMENT, 'gemma.allowedTokens'),
      tokens: S(C * Uint32Array.BYTES_PER_ELEMENT, 'gemma.tokens'),
      // Attention kernels use `{ basePos, rows }`; all regular matrix/norm
      // kernels use `{ rows, ... }`. They cannot share the same uniform.
      u: g.uniform(16, 'gemma.u'),
      uAttnLocal: g.uniform(16, 'gemma.uAttnLocal'),
      uCompute: g.uniform(16, 'gemma.uCompute'),
      uRows: g.uniform(16, 'gemma.uRows'),
      uTokens: g.uniform(16, 'gemma.uTokens'),
      uElements: g.uniform(16, 'gemma.uElements'),
      uPleElements: g.uniform(16, 'gemma.uPleElements'),
      uAllowed: g.uniform(16, 'gemma.uAllowed'),
      samplerParams: g.uniform(48, 'gemma.samplerParams'),
      sampled: S(MAX_DECODE_BATCH * Uint32Array.BYTES_PER_ELEMENT, 'gemma.sampled'),
      samplerCtl: S(8, 'gemma.samplerCtl'),
      recent: S((PENALTY_WINDOW + 1) * Uint32Array.BYTES_PER_ELEMENT, 'gemma.recent'),
      // Source positions copied into `u` between decode passes. GPU copies in
      // one command encoder preserve distinct positions without CPU submits.
      decodePositions: S(MAX_DECODE_BATCH * 16, 'gemma.decodePositions'),
      decodeLocalPositions: S(MAX_DECODE_BATCH * 16, 'gemma.decodeLocalPositions'),
      ropeLocal: S(this.maxCtx * T.slidingRopeDim * F, 'gemma.ropeLocal'),
      ropeGlobal: S(this.maxCtx * (T.fullRotaryDim * 2) * F, 'gemma.ropeGlobal'),
    };
    this.candBytes = TOPK_BLOCKS * TOPK * 2 * F;
    this.zeroPle = new Float32Array(T.plePackedDim);
    this.candRead = g.buf(
      this.candBytes,
      GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      'gemma.candidateRead',
    );
    this.sampleRead = g.buf(
      MAX_DECODE_BATCH * 4 + 8,
      GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      'gemma.sampleRead',
    );

    // Each cache contains packed int8 values (four lanes/u32) followed by one
    // f32 scale for every (token, KV-head).  The scale is stored in the same
    // u32 buffer as its IEEE-754 bit pattern.
    this.state = [];
    const cacheOwners = new Map();
    for (let i = 0; i < T.layers; i++) {
      const kind = isGemmaFullAttention(this.cfg, i) ? 'full_attention' : 'sliding_attention';
      const owner = isGemmaDeclaredKvSharedLayer(this.cfg, i)
        ? gemmaSharedKvSource(this.cfg, kind)
        : i;
      const existing = cacheOwners.get(owner);
      if (existing) {
        this.state.push(existing);
        continue;
      }
      const headDim = isGemmaFullAttention(this.cfg, owner)
        ? T.globalHeadDim
        : T.slidingHeadDim;
      const words = this.maxCtx * T.kvHeads * (headDim / 4 + 1);
      const state = {
        k: S(words * Uint32Array.BYTES_PER_ELEMENT, `gemma.kvK${owner}`),
        v: S(words * Uint32Array.BYTES_PER_ELEMENT, `gemma.kvV${owner}`),
        headDim,
      };
      cacheOwners.set(owner, state);
      this.state.push(state);
    }
    g.upload(this.b.ropeLocal, ropeTable(this.maxCtx, T.slidingRopeDim / 2, T.slidingRopeTheta));
    g.upload(this.b.ropeGlobal, ropeTable(this.maxCtx, T.fullRotaryDim / 2, T.fullRopeTheta));
  }

  #pipelines() {
    const g = this.gpu;
    const T = this.cfg.text;
    const cache = new Map();
    const named = (prefix, values, factory) => {
      const key = `${prefix}:${JSON.stringify(values)}`;
      let pipeline = cache.get(key);
      if (!pipeline) {
        pipeline = g.pipeline(`gemma.${key}`, factory);
        cache.set(key, pipeline);
      }
      return pipeline;
    };
    this.p = {
      gemm: (N, K, options = {}) => named('gemm', { N, K, ...options }, () => KS.gemmQ4({ N, K, ...options })),
      gemv: (N, K, options = {}) => named(
        'gemv',
        {
          N,
          K,
          subgroups: g.subgroups ? 1 : 0,
          rowLanes: this.gemvRowLanes,
          directSubgroup: this.directGemvSubgroups ? 1 : 0,
          ...options,
        },
        () => KS.gemvQ4({
          N,
          K,
          SUBGROUPS: g.subgroups ? 1 : 0,
          ROW_LANES: this.gemvRowLanes,
          DIRECT_SUBGROUP: this.directGemvSubgroups ? 1 : 0,
          ...options,
        })
      ),
      gateUpFused: (INTERM, K) => named(
        'gateUpFused',
        { INTERM, K },
        () => KS.gemvQ4GateUpI8({ INTERM, K }),
      ),
      gather: (start, rows, K, scale) => named('gather', { start, rows, K, scale }, () => KS.gatherQ4({ START: start, NUM: rows, K, SCALE: scale })),
      rms: (K, options = {}) => named('rms', { K, ...options }, () => KS.rmsnorm({ K, ...options })),
      add: named('add', {}, KS.add),
      gelu: (K) => named('gelu', { K }, () => KS.geluMul({ K })),
      scale: named('scale', {}, KS.scaleByScalar),
      pleProjectionScale: named('pleProjectionScale', {}, () => KS.scale({ SCALE: T.pleModelProjectionScale })),
      pleCombine: named('pleCombine', {}, () => KS.pleCombineMasked({ LAYERS: T.layers, DIM: T.pleDim })),
      topk: (filter = 'all') => named('topk', { filter }, () => topk({
        VOCAB: T.vocab,
        KTOP: TOPK,
        SOFTCAP: T.finalLogitSoftcap,
        ALLOWED: filter === 'allow',
        DENIED: filter === 'deny',
      })),
      sampler: named('sampler', {}, () => samplerKernel({ BLOCKS: TOPK_BLOCKS, KTOP: TOPK, WIN: PENALTY_WINDOW })),
    };
    this.p.pleNorm = this.p.rms(T.pleDim);
    this.p.hiddenNorm = this.p.rms(T.hidden);
    this.p.finalNorm = this.p.rms(T.hidden);
    this.p.hiddenNormQ = this.i8
      ? named('rmsQ', { K: T.hidden }, () => rmsnormQ({ K: T.hidden }))
      : null;
    this.p.hiddenNormAdd = named('rmsAdd', { K: T.hidden }, () => KS.rmsnormAdd({ K: T.hidden }));
    this.p.hiddenNormAddQ = this.i8
      ? named('rmsAddQ', { K: T.hidden }, () => KS.rmsnormAddQ({ K: T.hidden }))
      : null;
    this.p.hiddenNormAddScale = named('rmsAddScale', { K: T.hidden }, () => KS.rmsnormAddScale({ K: T.hidden }));
    this.p.pleModel = this.p.gemm(T.plePackedDim, T.hidden);
    this.p.pleModelDecode = this.p.gemv(T.plePackedDim, T.hidden);

    this.p.kv = {
      local: named('kv.local', {}, () => KS.textKvPrep({
        HEADS: T.heads,
        KV_HEADS: T.kvHeads,
        HEAD_DIM: T.slidingHeadDim,
        ROTARY_PAIRS: T.slidingRopeDim / 2,
        MAXCTX: this.maxCtx,
      })),
      global: named('kv.global', {}, () => KS.textKvPrep({
        HEADS: T.heads,
        KV_HEADS: T.kvHeads,
        HEAD_DIM: T.globalHeadDim,
        ROTARY_PAIRS: T.fullRotaryDim / 2,
        MAXCTX: this.maxCtx,
      })),
    };
    this.p.qPrep = {
      local: named('qPrep.local', {}, () => KS.textQPrep({
        HEADS: T.heads,
        KV_HEADS: T.kvHeads,
        HEAD_DIM: T.slidingHeadDim,
        ROTARY_PAIRS: T.slidingRopeDim / 2,
        MAXCTX: this.maxCtx,
      })),
      global: named('qPrep.global', {}, () => KS.textQPrep({
        HEADS: T.heads,
        KV_HEADS: T.kvHeads,
        HEAD_DIM: T.globalHeadDim,
        ROTARY_PAIRS: T.fullRotaryDim / 2,
        MAXCTX: this.maxCtx,
      })),
    };
    this.p.attn = {
      local: named('attn.local', { subgroups: g.subgroups ? 1 : 0 }, () => KS.textFlashAttention({
        HEADS: T.heads,
        KV_HEADS: T.kvHeads,
        HEAD_DIM: T.slidingHeadDim,
        MAXCTX: this.maxCtx,
        WINDOW: T.slidingWindow,
        SUBGROUPS: g.subgroups ? 1 : 0,
        HEAD_BATCH: this.attentionHeadBatch,
        // Gemma's Q/K RMS normalization is trained with a unit attention
        // scaling factor (unlike conventional 1/sqrt(head_dim) attention).
        ATTENTION_SCALE: T.attentionScale,
      })),
      global: named('attn.global', { subgroups: g.subgroups ? 1 : 0 }, () => KS.textFlashAttention({
        HEADS: T.heads,
        KV_HEADS: T.kvHeads,
        HEAD_DIM: T.globalHeadDim,
        MAXCTX: this.maxCtx,
        WINDOW: 0,
        SUBGROUPS: g.subgroups ? 1 : 0,
        HEAD_BATCH: this.attentionHeadBatch,
        ATTENTION_SCALE: T.attentionScale,
      })),
    };
    this.p.attnCombine = {
      local: named('attnCombine.local', {}, () => KS.textFlashCombine({
        HEADS: T.heads,
        HEAD_DIM: T.slidingHeadDim,
        MAXCTX: this.maxCtx,
        WINDOW: T.slidingWindow,
      })),
      global: named('attnCombine.global', {}, () => KS.textFlashCombine({
        HEADS: T.heads,
        HEAD_DIM: T.globalHeadDim,
        MAXCTX: this.maxCtx,
        WINDOW: 0,
      })),
    };
    this.p.attnDecode = {
      local: named('attnDecode.local', { subgroups: g.subgroups ? 1 : 0 }, () => KS.textFlashAttention({
        HEADS: T.heads,
        KV_HEADS: T.kvHeads,
        HEAD_DIM: T.slidingHeadDim,
        MAXCTX: this.maxCtx,
        WINDOW: T.slidingWindow,
        SUBGROUPS: g.subgroups ? 1 : 0,
        HEAD_BATCH: this.decodeAttentionHeadBatch,
        CHUNK_SIZE: this.decodeAttentionChunk,
        ATTENTION_SCALE: T.attentionScale,
      })),
      global: named('attnDecode.global', { subgroups: g.subgroups ? 1 : 0 }, () => KS.textFlashAttention({
        HEADS: T.heads,
        KV_HEADS: T.kvHeads,
        HEAD_DIM: T.globalHeadDim,
        MAXCTX: this.maxCtx,
        WINDOW: 0,
        SUBGROUPS: g.subgroups ? 1 : 0,
        HEAD_BATCH: this.decodeAttentionHeadBatch,
        CHUNK_SIZE: this.decodeAttentionChunk,
        ATTENTION_SCALE: T.attentionScale,
      })),
    };
    this.p.attnCombineDecode = {
      local: named('attnCombineDecode.local', {}, () => KS.textFlashCombine({
        HEADS: T.heads,
        HEAD_DIM: T.slidingHeadDim,
        MAXCTX: this.maxCtx,
        WINDOW: T.slidingWindow,
        CHUNK_SIZE: this.decodeAttentionChunk,
      })),
      global: named('attnCombineDecode.global', {}, () => KS.textFlashCombine({
        HEADS: T.heads,
        HEAD_DIM: T.globalHeadDim,
        MAXCTX: this.maxCtx,
        WINDOW: 0,
        CHUNK_SIZE: this.decodeAttentionChunk,
      })),
    };
  }

  #binds() {
    const g = this.gpu;
    const b = this.b;
    const w = this.w;
    const T = this.cfg.text;
    const P = this.p;

    this.embBinds = q4Parts(w['text.emb']).map((part) =>
      g.bind(P.gather(part.start, part.rows, T.hidden, T.embedScale), [
        b.tokens, part.q, part.s, b.x, b.uCompute,
      ]),
    );
    this.pleEmbBinds = q4Parts(w['text.pleEmb']).map((part) =>
      g.bind(P.gather(part.start, part.rows, T.plePackedDim, T.pleEmbedScale), [
        b.tokens, part.q, part.s, b.pleIdentity, b.uCompute,
      ]),
    );

    this.pleModelBind = this.#q4Bind(P.pleModel, w['text.pleModelProj'], [b.x, b.pleProjected, b.uCompute]);
    this.pleModelDecodeBind = this.#q4Bind(P.pleModelDecode, w['text.pleModelProj'], [b.x, b.pleProjected]);
    // `rmsnorm` cannot safely bind the same buffer for read/write on all
    // implementations.  Keep the normalized context in `ple` temporarily,
    // then combine it with the identity table into `pleProjected`.
    this.pleProjectionScaleBind = g.bind(P.pleProjectionScale, [b.pleProjected, b.pleContext, b.uPleElements]);
    this.pleNormBind = g.bind(P.pleNorm, [b.pleContext, w['text.pleProjectionNorm'], b.ple, b.uRows]);
    this.pleCombineBind = g.bind(P.pleCombine, [b.pleIdentity, b.ple, b.pleTextMask, b.pleProjected, b.uTokens]);

    this.layers = [];
    for (let i = 0; i < T.layers; i++) {
      const full = isGemmaFullAttention(this.cfg, i);
      const kind = full ? 'global' : 'local';
      const headDim = full ? T.globalHeadDim : T.slidingHeadDim;
      const qRows = T.heads * headDim;
      const interm = i >= (T.doubleWideMlpFrom ?? Infinity) ? T.interm * 2 : T.interm;
      const sharedKv = isGemmaDeclaredKvSharedLayer(this.cfg, i);
      const qkvRows = sharedKv ? qRows : qRows + 2 * T.kvHeads * headDim;
      // The decoder pipeline labels are `local`/`global`, while the
      // checkpoint config records shared-KV owners as
      // `sliding_attention`/`full_attention`. Keep that translation explicit
      // so the declared shared tail binds its configured int8 cache owners.
      const sharedSource = sharedKv
        ? gemmaSharedKvSource(this.cfg,
            full ? 'full_attention' : 'sliding_attention'
          )
        : i;
      // Gemma applies `layer_scalar` to the entire layer result after the PLE
      // residual. This fixed x/xAlt pattern avoids writable alias bindings and
      // leaves the next layer's state in x again.
      const current = b.x;
      const other = b.xAlt;
      const L = (name) => w[`text.L${i}.${name}`];
      const qkvPipeline = P.gemm(qkvRows, T.hidden);
      const oPipeline = P.gemm(T.hidden, qRows);
      const gateUpPipeline = P.gemm(2 * interm, T.hidden);
      const downPipeline = P.gemm(T.hidden, interm);
      const pleGatePipeline = P.gemm(T.pleDim, T.hidden);
      const pleProjPipeline = P.gemm(T.hidden, T.pleDim);
      const normDecodeOptions = this.i8 ? { I8: 1 } : {};
      const normDecodePipeline = (rows) => P.gemv(rows, T.hidden, normDecodeOptions);
      const downDecodeOptions = this.directGemvSubgroups
        ? { ROW_LANES: 64, DIRECT_SUBGROUP: 0 }
        : {};
      const gateUpDecodeOptions = normDecodeOptions;
      const pleGateMulPipeline = this.gpu.pipeline(`gemma.pleGateMul.${i}`, () => KS.pleGateMul({ LAYERS: T.layers, DIM: T.pleDim, LAYER: i }));
      const e = {
        index: i,
        full,
        current,
        output: other,
        sharedKv,
        qRows,
        qkvRows,
        headDim,
        interm,
        ln1: g.bind(P.hiddenNorm, [current, L('ln1'), b.xn, b.uCompute]),
        qkvPipeline,
        qkv: sharedKv
          ? this.#q4RowsBind(qkvPipeline, L('qkv'), 0, qRows, [b.xn, b.qkv, b.uCompute])
          : this.#q4Bind(qkvPipeline, L('qkv'), [b.xn, b.qkv, b.uCompute]),
        ln1Q: this.i8 ? g.bind(P.hiddenNormQ, [current, L('ln1'), b.xq, b.xsm]) : null,
        qkvDecodePipeline: normDecodePipeline(qkvRows),
        qkvDecode: sharedKv
          ? this.#q4RowsBind(
              normDecodePipeline(qRows),
              L('qkv'), 0, qRows,
              this.i8 ? [b.xq, b.qkv, b.xsm] : [b.xn, b.qkv],
            )
          : this.#q4Bind(
              normDecodePipeline(qkvRows),
              L('qkv'),
              this.i8 ? [b.xq, b.qkv, b.xsm] : [b.xn, b.qkv],
            ),
        prep: sharedKv
          ? g.bind(P.qPrep[kind], [b.qkv, L('qNorm'), full ? b.ropeGlobal : b.ropeLocal, b.q, b.u])
          : g.bind(P.kv[kind], [
              b.qkv, L('qNorm'), L('kNorm'), full ? b.ropeGlobal : b.ropeLocal,
              b.q, this.state[i].k, this.state[i].v, b.u,
            ]),
        attn: g.bind(P.attn[kind], [
          b.q, this.state[sharedSource].k, this.state[sharedSource].v, b.attnPartials,
          full ? b.u : b.uAttnLocal,
        ]),
        attnCombine: g.bind(P.attnCombine[kind], [
          b.attnPartials, b.attn, full ? b.u : b.uAttnLocal,
        ]),
        attnDecode: g.bind(P.attnDecode[kind], [
          b.q, this.state[sharedSource].k, this.state[sharedSource].v,
          b.attnPartials, full ? b.u : b.uAttnLocal,
        ]),
        attnCombineDecode: g.bind(P.attnCombineDecode[kind], [
          b.attnPartials, b.attn, full ? b.u : b.uAttnLocal,
        ]),
        oPipeline,
        o: this.#q4Bind(oPipeline, L('o'), [b.attn, b.tmp, b.uCompute]),
        oDecodePipeline: P.gemv(T.hidden, qRows),
        oDecode: this.#q4Bind(P.gemv(T.hidden, qRows), L('o'), [b.attn, b.tmp]),
        postAttnNormAdd: g.bind(P.hiddenNormAdd, [b.tmp, L('postAttnNorm'), current, other, b.uCompute]),
        postAttnNormAddQ: this.i8 ? g.bind(P.hiddenNormAddQ, [
          b.tmp, L('postAttnNorm'), current, other, L('preFfnNorm'),
          b.xq, b.xsm, b.uCompute,
        ]) : null,
        preFfnNorm: g.bind(P.hiddenNorm, [other, L('preFfnNorm'), b.xn, b.uCompute]),
        preFfnNormQ: this.i8
          ? g.bind(P.hiddenNormQ, [other, L('preFfnNorm'), b.xq, b.xsm])
          : null,
        gateUpPipeline,
        gateUp: this.#q4Bind(gateUpPipeline, L('gateup'), [b.xn, b.gateUp, b.uCompute]),
        gateUpDecodeRowsPerWorkgroup: this.gemvRowsPerWorkgroup,
        gateUpDecodePipeline: P.gemv(2 * interm, T.hidden, gateUpDecodeOptions),
        gateUpDecode: this.#q4Bind(
          P.gemv(2 * interm, T.hidden, gateUpDecodeOptions),
          L('gateup'),
          this.i8 ? [b.xq, b.gateUp, b.xsm] : [b.xn, b.gateUp],
        ),
        gateUpFusedPipeline: this.fusedGateUp
          ? P.gateUpFused(interm, T.hidden)
          : null,
        gateUpFused: null,
        gelu: g.bind(P.gelu(interm), [b.gateUp, b.act, b.uCompute]),
        downPipeline,
        down: this.#q4Bind(downPipeline, L('down'), [b.act, b.tmp, b.uCompute]),
        // Portable decode shares prefill's materialized GELU. Fixed subgroup
        // adapters use the paired gate/up projection above instead.
        downDecodeRowsPerWorkgroup: this.directGemvSubgroups ? 4 : this.gemvRowsPerWorkgroup,
        downDecodePipeline: P.gemv(T.hidden, interm, downDecodeOptions),
        downDecode: this.#q4Bind(P.gemv(T.hidden, interm, downDecodeOptions), L('down'), [b.act, b.tmp]),
        postFfnNormAdd: g.bind(P.hiddenNormAdd, [b.tmp, L('postFfnNorm'), other, current, b.uCompute]),
        pleGatePipeline,
        pleGate: this.#q4Bind(pleGatePipeline, L('pleGate'), [current, b.pleGate, b.uCompute]),
        pleGateDecodePipeline: P.gemv(T.pleDim, T.hidden),
        pleGateDecode: this.#q4Bind(P.gemv(T.pleDim, T.hidden), L('pleGate'), [current, b.pleGate]),
        pleGateMulPipeline,
        pleGateMul: g.bind(pleGateMulPipeline, [b.pleGate, b.pleProjected, b.pleSmall, b.uCompute]),
        pleProjPipeline,
        pleProj: this.#q4Bind(pleProjPipeline, L('pleProj'), [b.pleSmall, b.tmp, b.uCompute]),
        pleProjDecodePipeline: P.gemv(T.hidden, T.pleDim),
        pleProjDecode: this.#q4Bind(P.gemv(T.hidden, T.pleDim), L('pleProj'), [b.pleSmall, b.tmp]),
        pleNormAddScale: g.bind(P.hiddenNormAddScale, [b.tmp, L('pleNorm'), current, L('layerScalar'), b.uCompute]),
      };
      if (this.fusedGateUp) {
        e.gateUpFused = this.#q4Bind(e.gateUpFusedPipeline, L('gateup'), [b.xq, b.act, b.xsm]);
      }
      this.layers.push(e);
    }
    this.finalX = b.x;
    this.finalNormBind = g.bind(P.finalNorm, [b.head, w['text.norm'], b.headNorm, b.uCompute]);
    this.finalNormQBind = this.i8
      ? g.bind(P.hiddenNormQ, [b.head, w['text.norm'], b.xq, b.xsm])
      : null;
    this.lmRuns = [];
    for (const part of q4Parts(w['text.emb'])) {
      for (let firstRow = 0; firstRow < part.rows; firstRow += MAX_GEMV_ROWS) {
        const rows = Math.min(MAX_GEMV_ROWS, part.rows - firstRow);
        const decodePipeline = P.gemv(rows, T.hidden, {
          OSTRIDE: T.vocab,
          OUTOFF: part.start + firstRow,
          ...(this.i8 ? { I8: 1 } : {}),
        });
        this.lmRuns.push({
          rows,
          pipeline: decodePipeline,
          bind: g.bind(decodePipeline, [
            ...q4PartRows(part, T.hidden, firstRow, rows),
            this.i8 ? b.xq : b.headNorm,
            b.logits,
            ...(this.i8 ? [b.xsm] : []),
          ]),
        });
      }
    }
    this.topkBind = g.bind(P.topk('all'), [b.logits, b.cand]);
    this.topkAllowedBind = g.bind(P.topk('allow'), [b.logits, b.cand, b.allowed, b.uAllowed]);
    this.topkDeniedBind = g.bind(P.topk('deny'), [b.logits, b.cand, b.allowed, b.uAllowed]);
    this.samplerBind = g.bind(P.sampler, [b.cand, b.samplerParams, b.tokens, b.sampled, b.samplerCtl, b.recent]);
  }

  #q4Bind(pipeline, weight, tail) {
    if (!weight?.q || !weight?.s || weight.shards) {
      throw new Error('Gemma dense projection weights must be unsharded Q4 matrices.');
    }
    return this.gpu.bind(pipeline, [weight.q, weight.s, ...tail]);
  }

  #q4RowsBind(pipeline, weight, firstRow, rows, tail) {
    return this.gpu.bind(pipeline, [...q4Rows(weight, firstRow, rows), ...tail]);
  }

  #setU(basePos, rows) {
    this.gpu.device.queue.writeBuffer(this.b.u, 0, u32(basePos, rows, 0, 0));
    const chunkSize = rows === 1 ? this.decodeAttentionChunk : 256;
    const local = gemmaAttentionChunkRange(basePos, rows, this.cfg.text.slidingWindow, chunkSize);
    this.gpu.device.queue.writeBuffer(this.b.uAttnLocal, 0, u32(basePos, rows, local.base, 0));
    this.gpu.device.queue.writeBuffer(this.b.uCompute, 0, u32(rows, 0, 0, 0));
    this.gpu.device.queue.writeBuffer(this.b.uElements, 0, u32(rows * this.cfg.text.hidden, 0, 0, 0));
  }

  #run(pass, pipeline, bind, x, y = 1, z = 1) {
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(x, y, z);
  }

  #encodeEmbeddings(enc) {
    const b = this.b;
    const pass = enc.beginComputePass();
    for (let i = 0; i < this.embBinds.length; i++) {
      const part = q4Parts(this.w['text.emb'])[i];
      this.#run(pass, this.p.gather(part.start, part.rows, this.cfg.text.hidden, this.cfg.text.embedScale), this.embBinds[i], ceilDiv(this.cfg.text.hidden, 256), 1);
    }
    for (let i = 0; i < this.pleEmbBinds.length; i++) {
      const part = q4Parts(this.w['text.pleEmb'])[i];
      this.#run(pass, this.p.gather(part.start, part.rows, this.cfg.text.plePackedDim, this.cfg.text.pleEmbedScale), this.pleEmbBinds[i], ceilDiv(this.cfg.text.plePackedDim, 256), 1);
    }
    pass.end();
  }

  #submitEmbeddings(tokens, basePos, overrides) {
    const b = this.b;
    const q = this.gpu.device.queue;
    q.writeBuffer(b.tokens, 0, new Uint32Array(tokens));
    this.#setU(basePos, tokens.length);
    const enc = this.gpu.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    for (let i = 0; i < this.embBinds.length; i++) {
      const part = q4Parts(this.w['text.emb'])[i];
      this.#run(pass, this.p.gather(part.start, part.rows, this.cfg.text.hidden, this.cfg.text.embedScale), this.embBinds[i], ceilDiv(this.cfg.text.hidden, 256), tokens.length);
    }
    for (let i = 0; i < this.pleEmbBinds.length; i++) {
      const part = q4Parts(this.w['text.pleEmb'])[i];
      this.#run(pass, this.p.gather(part.start, part.rows, this.cfg.text.plePackedDim, this.cfg.text.pleEmbedScale), this.pleEmbBinds[i], ceilDiv(this.cfg.text.plePackedDim, 256), tokens.length);
    }
    pass.end();
    q.submit([enc.finish()]);

    const textMask = new Uint32Array(tokens.length);
    textMask.fill(1);
    if (overrides) {
      for (let row = 0; row < tokens.length; row++) {
        const vector = overrides.get?.(basePos + row);
        if (!vector) continue;
        if (!(vector instanceof Float32Array) || vector.length !== this.cfg.text.hidden) {
          throw new Error(`Gemma multimodal embedding override must be a ${this.cfg.text.hidden}-wide Float32Array.`);
        }
        q.writeBuffer(b.x, row * this.cfg.text.hidden * F, vector);
        // Gemma's reference PLE path omits token-identity embeddings for
        // multimodal positions (there is no real input ID for a soft token).
        q.writeBuffer(b.pleIdentity, row * this.cfg.text.plePackedDim * F, this.zeroPle);
        textMask[row] = 0;
      }
    }
    q.writeBuffer(b.pleTextMask, 0, textMask);
  }

  #forward(rows, basePos, withHead, encoder = null) {
    const g = this.gpu;
    const b = this.b;
    const P = this.p;
    const T = this.cfg.text;
    const q = g.device.queue;
    q.writeBuffer(b.uRows, 0, u32(rows * T.layers, 0, 0, 0));
    q.writeBuffer(b.uTokens, 0, u32(rows, 0, 0, 0));
    q.writeBuffer(b.uPleElements, 0, u32(rows * T.plePackedDim, 0, 0, 0));
    const enc = encoder ?? g.device.createCommandEncoder();
    let pass = enc.beginComputePass();
    const run = this.#run.bind(this, pass);
    const mm = (pipe, bind, n) => run(
      pipe,
      bind,
      ceilDiv(n, KS.GEMM_Q4_TILE_N),
      ceilDiv(rows, KS.GEMM_Q4_TILE_T),
    );
    const decode = rows === 1;
    const dense = (prefillPipe, prefillBind, decodePipe, decodeBind, n) => {
      if (decode) run(decodePipe, decodeBind, ceilDiv(n, this.gemvRowsPerWorkgroup));
      else mm(prefillPipe, prefillBind, n);
    };
    dense(P.pleModel, this.pleModelBind, P.pleModelDecode, this.pleModelDecodeBind, T.plePackedDim);
    run(P.pleProjectionScale, this.pleProjectionScaleBind, ceilDiv(rows * T.plePackedDim, 256));
    run(P.pleNorm, this.pleNormBind, rows * T.layers);
    run(P.pleCombine, this.pleCombineBind, ceilDiv(rows * T.plePackedDim, 256));

    const attentionChunk = decode ? this.decodeAttentionChunk : 256;
    const globalChunks = gemmaAttentionChunkRange(basePos, rows, 0, attentionChunk);
    const localChunks = gemmaAttentionChunkRange(basePos, rows, T.slidingWindow, attentionChunk);
    for (const e of this.layers) {
      run(decode && this.i8 ? P.hiddenNormQ : P.hiddenNorm, decode && this.i8 ? e.ln1Q : e.ln1, rows);
      dense(e.qkvPipeline, e.qkv, e.qkvDecodePipeline, e.qkvDecode, e.qkvRows);
      const kind = e.full ? 'global' : 'local';
      const chunks = e.full ? globalChunks : localChunks;
      run(e.sharedKv ? P.qPrep[kind] : P.kv[kind], e.prep, rows, T.kvHeads);
      if (decode) {
        run(
          P.attnDecode[kind],
          e.attnDecode,
          1,
          T.heads / this.decodeAttentionHeadBatch,
          chunks.count,
        );
      } else {
        run(
          P.attn[kind],
          e.attn,
          rows,
          T.kvHeads * (T.heads / T.kvHeads / this.attentionHeadBatch),
          chunks.count,
        );
      }
      run(
        decode ? P.attnCombineDecode[kind] : P.attnCombine[kind],
        decode ? e.attnCombineDecode : e.attnCombine,
        rows,
        T.heads,
      );
      dense(e.oPipeline, e.o, e.oDecodePipeline, e.oDecode, T.hidden);
      if (decode && this.i8) {
        run(P.hiddenNormAddQ, e.postAttnNormAddQ, rows);
      } else {
        run(P.hiddenNormAdd, e.postAttnNormAdd, rows);
        run(P.hiddenNorm, e.preFfnNorm, rows);
      }
      if (decode) {
        if (this.fusedGateUp) {
          run(e.gateUpFusedPipeline, e.gateUpFused, ceilDiv(e.interm, 8));
        } else {
          run(
            e.gateUpDecodePipeline,
            e.gateUpDecode,
            ceilDiv(2 * e.interm, e.gateUpDecodeRowsPerWorkgroup),
          );
        }
      } else {
        mm(e.gateUpPipeline, e.gateUp, 2 * e.interm);
      }
      if (!decode || !this.fusedGateUp) {
        run(P.gelu(e.interm), e.gelu, ceilDiv(e.interm, 256), rows);
      }
      if (decode) {
        run(e.downDecodePipeline, e.downDecode, ceilDiv(T.hidden, e.downDecodeRowsPerWorkgroup));
      } else {
        mm(e.downPipeline, e.down, T.hidden);
      }
      run(P.hiddenNormAdd, e.postFfnNormAdd, rows);
      dense(e.pleGatePipeline, e.pleGate, e.pleGateDecodePipeline, e.pleGateDecode, T.pleDim);
      run(e.pleGateMulPipeline, e.pleGateMul, 1, rows);
      dense(e.pleProjPipeline, e.pleProj, e.pleProjDecodePipeline, e.pleProjDecode, T.hidden);
      run(P.hiddenNormAddScale, e.pleNormAddScale, rows);
    }
    pass.end();

    if (!withHead) {
      if (!encoder) q.submit([enc.finish()]);
      return null;
    }
    enc.copyBufferToBuffer(this.finalX, (rows - 1) * T.hidden * F, b.head, 0, T.hidden * F);
    pass = enc.beginComputePass();
    const headRun = this.#run.bind(this, pass);
    headRun(this.i8 ? P.hiddenNormQ : P.finalNorm, this.i8 ? this.finalNormQBind : this.finalNormBind, 1);
    for (const run of this.lmRuns) {
      headRun(run.pipeline, run.bind, ceilDiv(run.rows, this.gemvRowsPerWorkgroup));
    }
    const topkBind = this.tokenMaskKind === 'allow'
      ? this.topkAllowedBind
      : this.tokenMaskKind === 'deny'
        ? this.topkDeniedBind
        : this.topkBind;
    headRun(P.topk(this.tokenMaskKind), topkBind, TOPK_BLOCKS);
    pass.end();
    if (encoder) return null;
    enc.copyBufferToBuffer(b.cand, 0, this.candRead, 0, this.candBytes);
    q.submit([enc.finish()]);
    return this.#readCandidates();
  }

  async #readCandidates() {
    await this.candRead.mapAsync(GPUMapMode.READ);
    const copy = this.candRead.getMappedRange().slice(0);
    this.candRead.unmap();
    return {
      ids: new Uint32Array(copy),
      vals: new Float32Array(copy),
    };
  }

  resetPenaltyWindow() {
    this.genIds = [];
  }

  notePenaltyToken(id) {
    this.genIds.push(id);
  }

  recentSet() {
    return new Set(this.genIds.slice(-512));
  }

  /** Discard an unused tail returned by batched decode.
   * KV cache entries are position-addressed, so lowering `pos` is sufficient;
   * later decoding overwrites the speculative slots. `sampledCount` is how
   * many of the discarded tokens entered the penalty window — one fewer than
   * `count` when the tail ends in a stop token, which is never pushed. */
  rewindDecode(count, sampledCount = count) {
    count = Math.max(0, Math.floor(count));
    sampledCount = Math.max(0, Math.floor(sampledCount));
    if (count) this.pos = Math.max(0, this.pos - count);
    if (sampledCount) this.genIds.splice(Math.max(0, this.genIds.length - sampledCount), sampledCount);
  }

  async reset() {
    // KV entries are indexed by position, so making position zero is enough
    // to make all previous int8 values unreachable.  Avoiding a multi-GB clear
    // keeps reloads and fresh conversations responsive.
    this.#setTokenMask(undefined);
    this.pos = 0;
    this.genIds = [];
    await this.gpu.device.queue.onSubmittedWorkDone();
  }

  /** Feed prompt IDs; optional absolute-position overrides inject media rows. */
  async prefill(tokens, onProgress = () => {}, overrides = null) {
    if (!Array.isArray(tokens) && !(tokens instanceof Uint32Array)) {
      throw new Error('Gemma prefill expects token IDs.');
    }
    if (!tokens.length) throw new Error('Gemma prefill received an empty prompt.');
    let off = 0;
    let cands = null;
    while (off < tokens.length) {
      const rows = Math.min(this.chunk, tokens.length - off);
      if (this.pos + rows > this.maxCtx) throw new Error('Context window exceeded');
      const part = Array.from(tokens.slice(off, off + rows));
      const last = off + rows === tokens.length;
      this.#submitEmbeddings(part, this.pos, overrides);
      const result = this.#forward(rows, this.pos, last);
      if (last) cands = await result;
      // Match the Qwen scheduler: keep prompt work bounded to one submitted
      // chunk. Gemma chunks are much heavier, and unbounded queueing starves
      // Chrome's compositor while making logical context race ahead of GPU.
      else await this.gpu.device.queue.onSubmittedWorkDone();
      this.pos += rows;
      off += rows;
      onProgress(off, tokens.length);
    }
    return cands;
  }

  /**
   * Drop any tool-constraint token mask left over from a previous request.
   * The mask is model state so batched decodes can reuse it, and reset()
   * clears it — but suffix continuations skip reset(), and a prefill head
   * sampling through a stale one-token mask (typically just the tool-call
   * close, a stop id) ends the new turn instantly.
   */
  clearAllowedTokenIds() {
    this.#setTokenMask(undefined);
  }

  #setTokenMask(mask) {
    // ToolConstraint reuses immutable masks. Avoid re-uploading a ~1 MiB
    // vocabulary allow-list (or even a small deny-list) on every token.
    updateTokenMaskState(this, mask);
  }

  async decode(token, params = {}) {
    this.#setTokenMask(params.tokenMask);
    return this.prefill([token]);
  }

  #packSamplerParams(params) {
    const rawStops = params.stopIds ?? (params.eosId === undefined ? [] : [params.eosId]);
    if (!Array.isArray(rawStops)) throw new Error('stopIds must be an array');
    if (rawStops.length > MAX_STOP_IDS) throw new Error(`stopIds supports at most ${MAX_STOP_IDS} ids`);
    const stops = rawStops.map((id) => {
      if (!Number.isInteger(id)) throw new Error(`invalid stop token id: ${id}`);
      return id >>> 0;
    });
    const buf = new ArrayBuffer(48);
    const u = new Uint32Array(buf);
    const f = new Float32Array(buf);
    u[0] = (params.seed ?? (Math.random() * 0xffffffff)) >>> 0;
    u[1] = stops.length;
    u[2] = Math.max(1, Math.min(TOPK, Number.isFinite(params.topK) ? Math.floor(params.topK) : TOPK));
    f[3] = Number.isFinite(params.temperature) ? Math.max(0, params.temperature) : 0.6;
    f[4] = Number.isFinite(params.topP) ? Math.min(1, Math.max(0, params.topP)) : 0.95;
    f[5] = Number.isFinite(params.presencePenalty) ? Math.min(10, Math.max(0, params.presencePenalty)) : 0;
    for (let i = 0; i < MAX_STOP_IDS; i++) u[8 + i] = stops[i] ?? STOP_PAD;
    return { buf, stops, stopSet: new Set(stops) };
  }

  async decodeBatch(firstToken, k, params = {}) {
    const g = this.gpu;
    const b = this.b;
    const q = g.device.queue;
    k = Math.min(Math.max(1, Math.floor(k)), this.BATCH, MAX_DECODE_BATCH, this.maxCtx - this.pos);
    if (k < 1) throw new Error('Context window exceeded');
    this.#setTokenMask(params.tokenMask);
    const packed = this.#packSamplerParams(params);
    q.writeBuffer(b.tokens, 0, u32(firstToken));
    q.writeBuffer(b.samplerParams, 0, packed.buf);
    q.writeBuffer(b.samplerCtl, 0, u32(0, 0));

    const recentIds = this.genIds.slice(-PENALTY_WINDOW);
    const recent = new Uint32Array(PENALTY_WINDOW + 1);
    recent[0] = recentIds.length;
    recent.set(recentIds, 1);
    q.writeBuffer(b.recent, 0, recent);
    q.writeBuffer(b.pleTextMask, 0, u32(1));
    this.#setU(this.pos, 1);

    const positions = new Uint32Array(k * 4);
    const localPositions = new Uint32Array(k * 4);
    for (let i = 0; i < k; i++) {
      const position = this.pos + i;
      const local = gemmaAttentionChunkRange(
        position,
        1,
        this.cfg.text.slidingWindow,
        this.decodeAttentionChunk,
      );
      positions[i * 4] = position;
      positions[i * 4 + 1] = 1;
      localPositions[i * 4] = position;
      localPositions[i * 4 + 1] = 1;
      localPositions[i * 4 + 2] = local.base;
    }
    q.writeBuffer(b.decodePositions, 0, positions);
    q.writeBuffer(b.decodeLocalPositions, 0, localPositions);

    const enc = g.device.createCommandEncoder();
    for (let i = 0; i < k; i++) {
      enc.copyBufferToBuffer(b.decodePositions, i * 16, b.u, 0, 16);
      enc.copyBufferToBuffer(b.decodeLocalPositions, i * 16, b.uAttnLocal, 0, 16);
      this.#encodeEmbeddings(enc);
      this.#forward(1, this.pos + i, true, enc);
      const pass = enc.beginComputePass();
      this.#run(pass, this.p.sampler, this.samplerBind, 1);
      pass.end();
    }
    enc.copyBufferToBuffer(b.sampled, 0, this.sampleRead, 0, MAX_DECODE_BATCH * 4);
    enc.copyBufferToBuffer(b.samplerCtl, 0, this.sampleRead, MAX_DECODE_BATCH * 4, 8);
    q.submit([enc.finish()]);

    await this.sampleRead.mapAsync(GPUMapMode.READ);
    const copy = this.sampleRead.getMappedRange().slice(0);
    this.sampleRead.unmap();
    const allIds = Array.from(new Uint32Array(copy, 0, k));
    const stopAt = allIds.findIndex((id) => packed.stopSet.has(id >>> 0));
    const stopped = stopAt >= 0;
    const ids = stopped ? allIds.slice(0, stopAt + 1) : allIds;
    const stopId = stopped ? ids[ids.length - 1] : null;
    const fed = stopped ? stopAt + 1 : k;
    this.pos += fed;
    this.genIds.push(...(stopped ? ids.slice(0, -1) : ids));
    return {
      ids,
      fed,
      eos: stopped && params.eosId !== undefined && stopId === (params.eosId >>> 0),
      stopped,
      stopId,
    };
  }

  /** Encode an image into one text-hidden-width row per Gemma image soft token. */
  encodeImage(input) {
    return this.modalities.encodeImage(input);
  }

  /** Encode a sampled video as frame-ordered vision soft-token rows. */
  encodeVideo(input) {
    return this.modalities.encodeVideo(input);
  }

  /** Encode log-mel audio into one text-hidden-width row per Gemma audio token. */
  encodeAudio(input) {
    return this.modalities.encodeAudio(input);
  }
}
