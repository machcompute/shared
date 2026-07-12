// Google Gemma 4 E4B execution graph.
//
// This intentionally mirrors the public Model API used by the Qwen runtime:
// `reset`, `prefill`, `decodeBatch`, and the presence-penalty helpers have the
// same semantics.  The graph itself is separate because Gemma's decoder uses
// alternating sliding/global attention, PLE injection, Q4 weights, and an
// emulated FP8 E4M3FN KV cache.
import {
  GEMMA_E4B_CFG as DEFAULT_CFG,
  gemmaSharedKvSource,
  isGemmaDeclaredKvSharedLayer,
  isGemmaFullAttention,
} from './gemma-config.js';
import * as KS from './gemma-kernels.js';
import { topk } from './kernels.js';
import { sample } from './tokenizer.js';
import { GemmaModalities } from './gemma-modalities.js';

const F = Float32Array.BYTES_PER_ELEMENT;
const TOPK = 20;
const TOPK_BLOCK = 1024;
const TOPK_BLOCKS = Math.ceil(DEFAULT_CFG.text.vocab / TOPK_BLOCK);
// gemvQ4 emits four logits per X workgroup. WebGPU caps each dispatch
// dimension at 65,535, so the tied 262,144-row embedding head must be split.
const MAX_GEMV_ROWS = 4 * 65_535;

const ceilDiv = (n, d) => Math.ceil(n / d);
const u32 = (...values) => new Uint32Array(values);

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
 * attention layers have a different head width and FP8 cache layout.
 */
export class GemmaModel {
  constructor(gpu, weights, opts = {}) {
    this.gpu = gpu;
    this.w = weights;
    this.cfg = opts.config ?? DEFAULT_CFG;
    this.embeddingWidth = this.cfg.text.hidden;
    this.maxCtx = opts.maxCtx ?? 8192;
    this.chunk = opts.chunk ?? 16;
    this.pos = 0;
    this.BATCH = 1;
    this.hasMtp = false;
    this.spec = false;
    this.genIds = [];
    this.allowedCount = 0;
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
      head: S(T.hidden * F, 'gemma.head'),
      headNorm: S(T.hidden * F, 'gemma.headNorm'),
      logits: S(T.vocab * F, 'gemma.logits'),
      cappedLogits: S(T.vocab * F, 'gemma.cappedLogits'),
      cand: S(TOPK_BLOCKS * TOPK * 2 * F, 'gemma.candidates'),
      allowed: S(T.vocab * Uint32Array.BYTES_PER_ELEMENT, 'gemma.allowedTokens'),
      tokens: S(C * Uint32Array.BYTES_PER_ELEMENT, 'gemma.tokens'),
      // Attention kernels use `{ basePos, rows }`; all regular matrix/norm
      // kernels use `{ rows, ... }`. They cannot share the same uniform.
      u: g.uniform(16, 'gemma.u'),
      uCompute: g.uniform(16, 'gemma.uCompute'),
      uRows: g.uniform(16, 'gemma.uRows'),
      uTokens: g.uniform(16, 'gemma.uTokens'),
      uElements: g.uniform(16, 'gemma.uElements'),
      uPleElements: g.uniform(16, 'gemma.uPleElements'),
      uAllowed: g.uniform(16, 'gemma.uAllowed'),
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

    // Each cache contains packed FP8 bytes (four lanes/u32) followed by one
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
      gemv: (N, K, options = {}) => named('gemv', { N, K, ...options }, () => KS.gemvQ4({ N, K, ...options })),
      gather: (start, rows, K, scale) => named('gather', { start, rows, K, scale }, () => KS.gatherQ4({ START: start, NUM: rows, K, SCALE: scale })),
      rms: (K, options = {}) => named('rms', { K, ...options }, () => KS.rmsnorm({ K, ...options })),
      add: named('add', {}, KS.add),
      gelu: (K) => named('gelu', { K }, () => KS.geluMul({ K })),
      scale: named('scale', {}, KS.scaleByScalar),
      softcap: named('softcap', {}, () => KS.softcap({ CAP: T.finalLogitSoftcap })),
      pleProjectionScale: named('pleProjectionScale', {}, () => KS.scale({ SCALE: T.pleModelProjectionScale })),
      pleCombine: named('pleCombine', {}, () => KS.pleCombineMasked({ LAYERS: T.layers, DIM: T.pleDim })),
      topk: (allowed = false) => named('topk', { allowed }, () => topk({ VOCAB: T.vocab, KTOP: TOPK, ALLOWED: allowed })),
    };
    this.p.pleNorm = this.p.rms(T.pleDim);
    this.p.hiddenNorm = this.p.rms(T.hidden);
    this.p.finalNorm = this.p.rms(T.hidden);
    this.p.pleModel = this.p.gemm(T.plePackedDim, T.hidden);

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
      local: named('attn.local', {}, () => KS.textCausalAttention({
        HEADS: T.heads,
        KV_HEADS: T.kvHeads,
        HEAD_DIM: T.slidingHeadDim,
        MAXCTX: this.maxCtx,
        WINDOW: T.slidingWindow,
        // Gemma's Q/K RMS normalization is trained with a unit attention
        // scaling factor (unlike conventional 1/sqrt(head_dim) attention).
        ATTENTION_SCALE: T.attentionScale,
      })),
      global: named('attn.global', {}, () => KS.textCausalAttention({
        HEADS: T.heads,
        KV_HEADS: T.kvHeads,
        HEAD_DIM: T.globalHeadDim,
        MAXCTX: this.maxCtx,
        WINDOW: 0,
        ATTENTION_SCALE: T.attentionScale,
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
      // so the declared shared tail binds its configured FP8 cache owners.
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
      const e = {
        current,
        output: other,
        sharedKv,
        qRows,
        qkvRows,
        headDim,
        interm,
        ln1: g.bind(P.hiddenNorm, [current, L('ln1'), b.xn, b.uCompute]),
        qkv: sharedKv
          ? this.#q4RowsBind(P.gemm(qRows, T.hidden), L('qkv'), 0, qRows, [b.xn, b.qkv, b.uCompute])
          : this.#q4Bind(P.gemm(qkvRows, T.hidden), L('qkv'), [b.xn, b.qkv, b.uCompute]),
        prep: sharedKv
          ? g.bind(P.qPrep[kind], [b.qkv, L('qNorm'), full ? b.ropeGlobal : b.ropeLocal, b.q, b.u])
          : g.bind(P.kv[kind], [
              b.qkv, L('qNorm'), L('kNorm'), full ? b.ropeGlobal : b.ropeLocal,
              b.q, this.state[i].k, this.state[i].v, b.u,
            ]),
        attn: g.bind(P.attn[kind], [
          b.q, this.state[sharedSource].k, this.state[sharedSource].v, b.attn, b.u,
        ]),
        o: this.#q4Bind(P.gemm(T.hidden, qRows), L('o'), [b.attn, b.tmp, b.uCompute]),
        postAttnNorm: g.bind(P.hiddenNorm, [b.tmp, L('postAttnNorm'), b.tmpNorm, b.uCompute]),
        addAttn: g.bind(P.add, [current, b.tmpNorm, other, b.uElements]),
        preFfnNorm: g.bind(P.hiddenNorm, [other, L('preFfnNorm'), b.xn, b.uCompute]),
        gateUp: this.#q4Bind(P.gemm(2 * interm, T.hidden), L('gateup'), [b.xn, b.gateUp, b.uCompute]),
        gelu: g.bind(P.gelu(interm), [b.gateUp, b.act, b.uCompute]),
        down: this.#q4Bind(P.gemm(T.hidden, interm), L('down'), [b.act, b.tmp, b.uCompute]),
        postFfnNorm: g.bind(P.hiddenNorm, [b.tmp, L('postFfnNorm'), b.tmpNorm, b.uCompute]),
        addFfn: g.bind(P.add, [other, b.tmpNorm, current, b.uElements]),
        pleGate: this.#q4Bind(P.gemm(T.pleDim, T.hidden), L('pleGate'), [current, b.pleGate, b.uCompute]),
        pleGateMul: g.bind(
          this.gpu.pipeline(`gemma.pleGateMul.${i}`, () => KS.pleGateMul({ LAYERS: T.layers, DIM: T.pleDim, LAYER: i })),
          [b.pleGate, b.pleProjected, b.pleSmall, b.uCompute],
        ),
        pleProj: this.#q4Bind(P.gemm(T.hidden, T.pleDim), L('pleProj'), [b.pleSmall, b.tmp, b.uCompute]),
        pleNorm: g.bind(P.hiddenNorm, [b.tmp, L('pleNorm'), b.tmpNorm, b.uCompute]),
        addPle: g.bind(P.add, [current, b.tmpNorm, other, b.uElements]),
        layerScale: g.bind(P.scale, [other, L('layerScalar'), current, b.uElements]),
      };
      this.layers.push(e);
    }
    this.finalX = b.x;
    this.finalNormBind = g.bind(P.finalNorm, [b.head, w['text.norm'], b.headNorm, b.uCompute]);
    this.lmRuns = [];
    for (const part of q4Parts(w['text.emb'])) {
      for (let firstRow = 0; firstRow < part.rows; firstRow += MAX_GEMV_ROWS) {
        const rows = Math.min(MAX_GEMV_ROWS, part.rows - firstRow);
        const pipeline = P.gemv(rows, T.hidden, {
          OSTRIDE: T.vocab,
          OUTOFF: part.start + firstRow,
        });
        this.lmRuns.push({
          rows,
          pipeline,
          bind: g.bind(pipeline, [
            ...q4PartRows(part, T.hidden, firstRow, rows),
            b.headNorm,
            b.logits,
          ]),
        });
      }
    }
    this.softcapBind = g.bind(P.softcap, [b.logits, b.cappedLogits, b.uElements]);
    this.topkBind = g.bind(P.topk(false), [b.cappedLogits, b.cand]);
    this.topkAllowedBind = g.bind(P.topk(true), [b.cappedLogits, b.cand, b.allowed, b.uAllowed]);
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
    this.gpu.device.queue.writeBuffer(this.b.uCompute, 0, u32(rows, 0, 0, 0));
    this.gpu.device.queue.writeBuffer(this.b.uElements, 0, u32(rows * this.cfg.text.hidden, 0, 0, 0));
  }

  #run(pass, pipeline, bind, x, y = 1) {
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(x, y);
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

  #forward(rows, basePos, withHead) {
    const g = this.gpu;
    const b = this.b;
    const P = this.p;
    const T = this.cfg.text;
    const q = g.device.queue;
    q.writeBuffer(b.uRows, 0, u32(rows * T.layers, 0, 0, 0));
    q.writeBuffer(b.uTokens, 0, u32(rows, 0, 0, 0));
    q.writeBuffer(b.uPleElements, 0, u32(rows * T.plePackedDim, 0, 0, 0));
    const enc = g.device.createCommandEncoder();
    let pass = enc.beginComputePass();
    const run = this.#run.bind(this, pass);
    const mm = (pipe, bind, n) => run(pipe, bind, ceilDiv(n, 64), ceilDiv(rows, 4));
    const elem = ceilDiv(rows * T.hidden, 256);

    mm(P.pleModel, this.pleModelBind, T.plePackedDim);
    run(P.pleProjectionScale, this.pleProjectionScaleBind, ceilDiv(rows * T.plePackedDim, 256));
    run(P.pleNorm, this.pleNormBind, rows * T.layers);
    run(P.pleCombine, this.pleCombineBind, ceilDiv(rows * T.plePackedDim, 256));

    for (const e of this.layers) {
      run(P.hiddenNorm, e.ln1, rows);
      mm(P.gemm(e.qkvRows, T.hidden), e.qkv, e.qkvRows);
      const kind = e.headDim === T.globalHeadDim ? 'global' : 'local';
      run(e.sharedKv ? P.qPrep[kind] : P.kv[kind], e.prep, rows, T.kvHeads);
      run(P.attn[kind], e.attn, rows, T.kvHeads);
      mm(P.gemm(T.hidden, e.qRows), e.o, T.hidden);
      run(P.hiddenNorm, e.postAttnNorm, rows);
      run(P.add, e.addAttn, elem);
      run(P.hiddenNorm, e.preFfnNorm, rows);
      mm(P.gemm(2 * e.interm, T.hidden), e.gateUp, 2 * e.interm);
      run(P.gelu(e.interm), e.gelu, ceilDiv(e.interm, 256), rows);
      mm(P.gemm(T.hidden, e.interm), e.down, T.hidden);
      run(P.hiddenNorm, e.postFfnNorm, rows);
      run(P.add, e.addFfn, elem);
      mm(P.gemm(T.pleDim, T.hidden), e.pleGate, T.pleDim);
      run(this.gpu.pipeline(`gemma.pleGateMul.${this.layers.indexOf(e)}`, () => KS.pleGateMul({ LAYERS: T.layers, DIM: T.pleDim, LAYER: this.layers.indexOf(e) })), e.pleGateMul, 1, rows);
      mm(P.gemm(T.hidden, T.pleDim), e.pleProj, T.hidden);
      run(P.hiddenNorm, e.pleNorm, rows);
      run(P.add, e.addPle, elem);
      run(P.scale, e.layerScale, elem);
    }
    pass.end();

    if (!withHead) {
      q.submit([enc.finish()]);
      return null;
    }
    enc.copyBufferToBuffer(this.finalX, (rows - 1) * T.hidden * F, b.head, 0, T.hidden * F);
    pass = enc.beginComputePass();
    const headRun = this.#run.bind(this, pass);
    headRun(P.finalNorm, this.finalNormBind, 1);
    for (const run of this.lmRuns) {
      headRun(run.pipeline, run.bind, ceilDiv(run.rows, 4));
    }
    q.writeBuffer(b.uElements, 0, u32(T.vocab, 0, 0, 0));
    headRun(P.softcap, this.softcapBind, ceilDiv(T.vocab, 256));
    headRun(P.topk(this.allowedCount > 0), this.allowedCount > 0 ? this.topkAllowedBind : this.topkBind, TOPK_BLOCKS);
    pass.end();
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

  async reset() {
    // KV entries are indexed by position, so making position zero is enough
    // to make all previous FP8 values unreachable.  Avoiding a multi-GB clear
    // keeps reloads and fresh conversations responsive.
    this.pos = 0;
    this.genIds = [];
    this.allowedCount = 0;
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
      else await this.gpu.device.queue.onSubmittedWorkDone();
      this.pos += rows;
      off += rows;
      onProgress(off, tokens.length);
    }
    return cands;
  }

  #setAllowedTokenIds(ids) {
    const count = ids?.length ?? 0;
    if (count > this.cfg.text.vocab) throw new Error('Tool constraint has too many allowed token IDs.');
    if (count) this.gpu.device.queue.writeBuffer(this.b.allowed, 0, ids);
    this.gpu.device.queue.writeBuffer(this.b.uAllowed, 0, u32(count, 0, 0, 0));
    this.allowedCount = count;
  }

  async decode(token, params = {}) {
    this.#setAllowedTokenIds(params.allowedTokenIds);
    return this.prefill([token]);
  }

  async decodeBatch(firstToken, _k, params = {}) {
    const cands = await this.decode(firstToken, params);
    const id = sample(cands, {
      temperature: params.temperature,
      topP: params.topP,
      topK: params.topK,
      presencePenalty: params.presencePenalty,
      recentIds: this.recentSet(),
    });
    const stops = new Set((params.stopIds ?? (params.eosId === undefined ? [] : [params.eosId])).map((value) => value >>> 0));
    const stopped = stops.has(id >>> 0);
    if (!stopped) this.genIds.push(id);
    return {
      ids: [id],
      fed: 1,
      eos: params.eosId !== undefined && id === params.eosId,
      stopped,
      stopId: stopped ? id : null,
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
