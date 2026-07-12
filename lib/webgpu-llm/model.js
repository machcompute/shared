// Forward-pass orchestration for the Qwen3.5-4B hybrid stack.
//
// Decode path (T=1) is heavily fused to minimize dispatch count (~6/layer):
//   in-proj GEMV (fused RMSNorm) → conv(+state shift) → deltaRule(+prep) →
//   out GEMV (fused gated-norm) → gateup GEMV (fused RMSNorm) → down GEMV (fused SwiGLU)
// Full-attention layers use partitioned flash-decoding (attnPart + attnReduce).
// Prefill path (chunked GEMM) keeps separate small kernels; both paths are
// validated to produce identical results (test2.html).
import { CFG, RT, PSIZE, TOPK_WGS, isFullAttn } from './config.js';
import * as KS from './kernels.js';

const F = Float32Array.BYTES_PER_ELEMENT;
const MAX_STOP_IDS = 4;
const STOP_PAD = 0xFFFFFFFF;
const ATTN_SPLIT_PARTITIONS = Number.POSITIVE_INFINITY;

export class Model {
  constructor(gpu, weights, opts = {}) {
    this.gpu = gpu;
    this.w = weights;
    this.maxCtx = opts.maxCtx ?? RT.maxCtx;
    this.chunk = opts.chunk ?? RT.chunk;
    this.pmax = Math.ceil(this.maxCtx / PSIZE);
    this.pos = 0;          // tokens committed to state
    this.hasMtp = !!weights['mtp.in'];
    this.spec = this.hasMtp; // MTP self-speculative decoding (toggleable)
    this.mtpExact = 0;       // exact MTP cache entries (slots 0..mtpExact-1)
    this._restoreTo = 0;     // pending state restore (0 = none) before next round
    this._aPrev = 1;         // accepted count of previous round (hN row of h_{P-1})
    this.#alloc();
    this.#pipelines();
    this.#binds();
  }

  #alloc() {
    const g = this.gpu, C = this.chunk, H = CFG.hidden;
    const S = (n, l) => g.storage(n, l);
    // int8 KV cache: [maxCtx][4][64] u32 data + one f32 scale per (pos, head)
    const KVBYTES = this.maxCtx * 4 * (64 + 1) * 4;
    this.b = {
      x: S(C * H * F, 'x'), xn: S(C * H * F, 'xn'),
      qkvz: S(C * CFG.inL * F, 'qkvz'),        // fused in-projection out (linear & full layers)
      qkvc: S(C * 8192 * F, 'qkvc'),           // post-conv
      qn: S(C * 2048 * F, 'qn'), kn: S(C * 2048 * F, 'kn'), gb: S(C * 64 * F, 'gb'),
      core: S(C * 4096 * F, 'core'), gn: S(C * 4096 * F, 'gn'),
      gu: S(C * 18432 * F, 'gu'), act: S(C * 9216 * F, 'act'),
      hx: S(H * F, 'hx'), hn: S(H * F, 'hn'),
      xq: S(3 * H, 'xq'), xsm: S(3 * (H / 16) * 4, 'xsm'), // int8 activations (decode row 0; verify rows 0..2)
      parts: S(4 * this.pmax * 4 * 258 * F, 'parts'),
      logits: S(3 * CFG.vocab * F, 'logits'),
      cand: S(3 * TOPK_WGS * RT.topkK * 2 * 4, 'cand'),
      allowed: S(CFG.vocab * 4, 'allowedTokens'),
      tokens: S(C * 4, 'tokens'),
      u: g.uniform(16, 'u'),
      uAllowed: g.uniform(16, 'uAllowed'),
      rope: S(this.maxCtx * 64 * F, 'rope'),
    };
    this.candBytes = TOPK_WGS * RT.topkK * 2 * 4;
    this.candRead = g.buf(3 * this.candBytes + 8, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, 'candRead');
    // batched decode (GPU-side sampling, up to BATCH tokens per submit)
    this.BATCH = 8;
    this.b.sp = g.uniform(48, 'sp');
    // presence-penalty window: [count, ring of RT.ppWindow generated token ids]
    this.b.recent = S((1 + RT.ppWindow) * 4, 'recent');
    this.genIds = [];
    this.b.ctl = S(8, 'ctl');                 // [stopFlag, iterCounter]
    this.b.sampled = S(this.BATCH * 4, 'sampled');
    this.b.uIter = Array.from({ length: this.BATCH }, (_, i) => g.uniform(16, `uIter${i}`));
    this.sampRead = g.buf(this.BATCH * 4 + 8, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, 'sampRead');
    if (this.hasMtp) {
      Object.assign(this.b, {
        kcM: S(KVBYTES, 'kcM'),
        vcM: S(KVBYTES, 'vcM'),
        fcIn: S(C * 5120 * F, 'fcIn'),
        mtpH: S(C * H * F, 'mtpH'),
        mtpHn: S(H * F, 'mtpHn'),
        embK: S(C * H * F, 'embK'),
        hN: S((C + 1) * H * F, 'hN'),
        draftIds: S(8, 'draftIds'),
        uA: g.uniform(16, 'uA'), uB: g.uniform(16, 'uB'),
        uC0: g.uniform(16, 'uC0'), uC1: g.uniform(16, 'uC1'), uPf: g.uniform(16, 'uPf'),
        posBuf: S(32, 'posBuf'),          // [pos, aPrev, mtpExact, snapValid, round, ...]
        specLog: S(4 * 4 * 4, 'specLog'), // [R=4][a, next, d0, d1]
      });
      this.R = 4; // chained speculative rounds per submit
      this.specRead = g.buf(4 * 16 + 32 + 8, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, 'specRead');
    }
    this.state = [];
    for (let i = 0; i < CFG.layers; i++) {
      if (isFullAttn(i)) {
        this.state.push({
          kc: S(KVBYTES, `kc${i}`),
          vc: S(KVBYTES, `vc${i}`),
        });
      } else {
        this.state.push({
          ds: S(32 * 128 * 128 * F, `ds${i}`),
          cs: S(3 * 8192 * F, `cs${i}`),
          ...(this.hasMtp ? {
            dsnap: S(3 * 32 * 128 * 128 * F, `dsnap${i}`),
            csnap: S(3 * 3 * 8192 * F, `csnap${i}`),
          } : {}),
        });
      }
    }
    const rt = new Float32Array(this.maxCtx * 64);
    for (let p = 0; p < this.maxCtx; p++) {
      for (let i = 0; i < 32; i++) {
        const ang = p * Math.pow(CFG.ropeTheta, -i / 32);
        rt[p * 64 + i] = Math.cos(ang);
        rt[p * 64 + 32 + i] = Math.sin(ang);
      }
    }
    g.upload(this.b.rope, rt);
  }

  #pipelines() {
    const g = this.gpu;
    const SG = g.subgroups ? 1 : 0;
    const P = {};
    const gemv = (N, K, opt = {}) =>
      g.pipeline(`gemv_${N}_${K}_${JSON.stringify(opt)}_${SG}`, () => KS.gemvQ4({ N, K, SUBGROUPS: SG, RESIDUAL: 0, ...opt }));
    const gemm = (N, K, R) => g.pipeline(`gemm_${N}_${K}_${R}`, () => KS.gemmQ4({ N, K, RESIDUAL: R }));
    // decode GEMVs (plain reads bufXn written by the rmsnorm dispatch; prologue-fused
    // norm variants measured ~2x slower on large N, so only cheap fusions are kept).
    // With DP4a, the rmsnorm-fed GEMVs read int8 activations from rmsnormQ instead.
    this.i8 = g.dp4a ? 1 : 0;
    const XM = this.i8 ? { MODE: 'i8' } : {};
    P.inL = gemv(CFG.inL, 2560, XM);
    P.inF = gemv(CFG.inF, 2560, XM);
    P.outG = gemv(2560, 4096, { MODE: 'gnorm', RESIDUAL: 1, ZOFF: 8192 });
    P.o = gemv(2560, 4096, { RESIDUAL: 1 });
    P.gateup = gemv(18432, 2560, XM);
    P.down = gemv(2560, 9216, { MODE: 'glu', RESIDUAL: 1 });
    P.rmsQ = this.i8 ? g.pipeline('rmsnormQ_2560', () => KS.rmsnormQ({ K: 2560 })) : null;
    // prefill GEMMs
    P.m_inL = gemm(CFG.inL, 2560, 0);
    P.m_inF = gemm(CFG.inF, 2560, 0);
    P.m_out = gemm(2560, 4096, 1);
    P.m_gateup = gemm(18432, 2560, 0);
    P.m_down = gemm(2560, 9216, 1);
    // shared / prefill kernels
    P.rms = g.pipeline('rmsnorm_2560', () => KS.rmsnorm({ K: 2560 }));
    P.conv = g.pipeline('conv1d', () => KS.conv1d({ SHIFT: 0 }));
    P.convS = g.pipeline('conv1d_shift', () => KS.conv1d({ SHIFT: 1 }));
    P.convShift = g.pipeline('convShift', () => KS.convShift({}));
    P.dprep = g.pipeline('deltaPrep', KS.deltaPrep);
    P.delta = g.pipeline('deltaRule0', () => KS.deltaRule({ PREP: 0 }));
    P.deltaP = g.pipeline('deltaRule1', () => KS.deltaRule({ PREP: 1 }));
    P.gnorm = g.pipeline('gatedNorm', KS.gatedNorm);
    P.aprep = g.pipeline(`attnPrep_${this.maxCtx}`, () => KS.attnPrep({ MAXCTX: this.maxCtx }));
    P.attn = g.pipeline(`attention_${this.maxCtx}`, () => KS.attention({ MAXCTX: this.maxCtx }));
    P.attnPart = g.pipeline(`attnPart_${this.maxCtx}`, () => KS.attnPart({ PMAX: this.pmax, MAXCTX: this.maxCtx }));
    P.attnPrepD = g.pipeline(`attnPrepD_${this.maxCtx}`, () => KS.attnDecodePrep({ MAXCTX: this.maxCtx }));
    P.attnPartD = g.pipeline(`attnPartD_${this.maxCtx}_${SG}`, () => KS.attnPartPrepared({ PMAX: this.pmax, MAXCTX: this.maxCtx, SUBGROUPS: SG }));
    P.attnRed = g.pipeline('attnReduce', () => KS.attnReduce({ PMAX: this.pmax }));
    P.silumul = g.pipeline('siluMul', KS.siluMul);
    P.topk = g.pipeline('topk', () => KS.topk({ VOCAB: CFG.vocab, KTOP: RT.topkK }));
    P.topkAllowed = g.pipeline('topk_allowed', () => KS.topk({ VOCAB: CFG.vocab, KTOP: RT.topkK, ALLOWED: true }));
    // batched decode: GPU sampler + stop-frozen state kernels
    P.samp = g.pipeline('sampler', () => KS.sampler({ BLOCKS: TOPK_WGS, KTOP: RT.topkK, WIN: RT.ppWindow }));
    P.convSF = g.pipeline('conv1d_shift_flag', () => KS.conv1d({ SHIFT: 1, FLAG: 1 }));
    P.deltaPF = g.pipeline('deltaRule1_flag', () => KS.deltaRule({ PREP: 1, FLAG: 1 }));
    P.lm = this.w.embShards.map((sh) =>
      g.pipeline(`lm_${sh.start}_${sh.rows}_${SG}_${this.i8}`, () => KS.gemvQ4({ N: sh.rows, K: 2560, RESIDUAL: 0, OUTOFF: sh.start, SUBGROUPS: SG, ...XM })));
    P.gather = this.w.embShards.map((sh) =>
      g.pipeline(`gather_${sh.start}_${sh.rows}`, () => KS.gather({ START: sh.start, NUM: sh.rows, K: 2560 })));
    if (this.hasMtp) {
      P.preE = g.pipeline('rms_preE', () => KS.rmsnorm({ K: 2560, OSTRIDE: 5120, OUTOFF: 0 }));
      P.preH = g.pipeline('rms_preH', () => KS.rmsnorm({ K: 2560, OSTRIDE: 5120, OUTOFF: 2560 }));
      P.fc = gemv(2560, 5120);
      P.m_fc = gemm(2560, 5120, 0);
      P.am1 = g.pipeline('argmax1', () => KS.argmaxMerge({ SLOT: 1, BLOCKS: TOPK_WGS, KTOP: RT.topkK }));
      P.am2 = g.pipeline('argmax2', () => KS.argmaxMerge({ SLOT: 2, BLOCKS: TOPK_WGS, KTOP: RT.topkK }));
      // GPU-chained speculation: storage-position variants + GPU acceptance
      P.accept = g.pipeline('acceptSpec', () => KS.acceptSpec({ BLOCKS: TOPK_WGS, KTOP: RT.topkK, WIN: RT.ppWindow }));
      P.attnPartA = g.pipeline(`attnPartA_${this.maxCtx}`, () => KS.attnPart({ PMAX: this.pmax, MAXCTX: this.maxCtx, SPOS: 1, OFFS: -1 }));
      P.attnPartB = g.pipeline(`attnPartB_${this.maxCtx}`, () => KS.attnPart({ PMAX: this.pmax, MAXCTX: this.maxCtx, SPOS: 1, OFFS: 0 }));
      P.attnPrepDA = g.pipeline(`attnPrepDA_${this.maxCtx}`, () => KS.attnDecodePrep({ MAXCTX: this.maxCtx, SPOS: 1, OFFS: -1 }));
      P.attnPrepDB = g.pipeline(`attnPrepDB_${this.maxCtx}`, () => KS.attnDecodePrep({ MAXCTX: this.maxCtx, SPOS: 1, OFFS: 0 }));
      P.attnPartDA = g.pipeline(`attnPartDA_${this.maxCtx}_${SG}`, () => KS.attnPartPrepared({ PMAX: this.pmax, MAXCTX: this.maxCtx, SPOS: 1, OFFS: -1, SUBGROUPS: SG }));
      P.attnPartDB = g.pipeline(`attnPartDB_${this.maxCtx}_${SG}`, () => KS.attnPartPrepared({ PMAX: this.pmax, MAXCTX: this.maxCtx, SPOS: 1, OFFS: 0, SUBGROUPS: SG }));
      P.attnRedA = g.pipeline('attnRedA', () => KS.attnReduce({ PMAX: this.pmax, SPOS: 1, OFFS: -1 }));
      P.attnRedB = g.pipeline('attnRedB', () => KS.attnReduce({ PMAX: this.pmax, SPOS: 1, OFFS: 0 }));
      P.aprepV = g.pipeline(`aprepV_${this.maxCtx}`, () => KS.attnPrep({ MAXCTX: this.maxCtx, SPOS: 'verify' }));
      P.aprepCu = [0, 1].map((j) => g.pipeline(`aprepCu${j}_${this.maxCtx}`, () => KS.attnPrep({ MAXCTX: this.maxCtx, SPOS: 'catchup', J: j })));
      P.attnV = g.pipeline(`attnV_${this.maxCtx}`, () => KS.attention({ MAXCTX: this.maxCtx, SPOS: 1 }));
      P.deltaLZ = g.pipeline('deltaLZ', () => KS.deltaRule({ PREP: 1, SNAP: 1, LAZY: 1 }));
      P.convLZ = g.pipeline('convLZ', () => KS.conv1d({ SNAP: 1, LAZY: 1 }));
      P.convShiftF = g.pipeline('convShiftF', () => KS.convShift({ TBAKE: 3, FLAG: 1 }));
      P.preHD = g.pipeline('preHD', () => KS.rmsnorm({ K: 2560, OSTRIDE: 5120, OUTOFF: 2560, DROW: 1 }));
      // multi-token GEMVs for the T=3 verify pass (GEMM tiles would idle the GPU);
      // mirrors the fused decode path: i8 inputs, GLU/gnorm fusion
      const vt = (N, K, opt = {}) => g.pipeline(`gemvT_${N}_${K}_${JSON.stringify(opt)}_${SG}`,
        () => KS.gemvT({ N, K, TN: 3, SUBGROUPS: SG, ...opt }));
      const XT = this.i8 ? { MODE: 'i8' } : {};
      P.vt = {
        inL: vt(CFG.inL, 2560, XT),
        inF: vt(CFG.inF, 2560, XT),
        out: vt(2560, 4096, { MODE: 'gnorm', RESIDUAL: 1, ZOFF: 8192 }),
        o: vt(2560, 4096, { RESIDUAL: 1 }),
        gateup: vt(18432, 2560, XT),
        down: vt(2560, 9216, { MODE: 'glu', RESIDUAL: 1 }),
        lm: this.w.embShards.map((sh) => vt(sh.rows, 2560, { OSTRIDE: CFG.vocab, OUTOFF: sh.start, ...XT })),
      };
      P.convSnap = g.pipeline('conv1d_snap', () => KS.conv1d({ SNAP: 1 }));
      P.deltaPS = g.pipeline('deltaRule_ps', () => KS.deltaRule({ PREP: 1, SNAP: 1 }));
      P.copyRows = g.pipeline('copyRows', () => /*wgsl*/`
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let i = wid.x * 256u + lid.x;
  dst[i] = src[i];
}`);
    }
    this.p = P;
  }

  #binds() {
    const g = this.gpu, b = this.b, w = this.w, P = this.p;
    this.lb = [];
    for (let i = 0; i < CFG.layers; i++) {
      const st = this.state[i];
      const L = (n) => w[`L${i}.${n}`];
      const e = { i, full: isFullAttn(i) };
      e.ln1 = g.bind(P.rms, [b.x, L('ln1'), b.xn]);
      e.ln2 = g.bind(P.rms, [b.x, L('ln2'), b.xn]);
      if (this.i8) {
        e.ln1Q = g.bind(P.rmsQ, [b.x, L('ln1'), b.xq, b.xsm]);
        e.ln2Q = g.bind(P.rmsQ, [b.x, L('ln2'), b.xq, b.xsm]);
      }
      const xin = this.i8 ? [b.xq] : [b.xn];
      const xtail = this.i8 ? [b.xsm] : [];
      if (e.full) {
        e.in = g.bind(P.inF, [L('in').q, L('in').s, ...xin, b.qkvz, ...xtail]);
        e.m_in = g.bind(P.m_inF, [L('in').q, L('in').s, b.xn, b.qkvz, b.u]);
        if (this.hasMtp) {
          e.vt_in = g.bind(P.vt.inF, [L('in').q, L('in').s, ...xin, b.qkvz, ...xtail]);
          e.vt_o = g.bind(P.vt.o, [L('o').q, L('o').s, b.gn, b.x]);
          e.aprepV = g.bind(P.aprepV, [b.qkvz, L('qnw'), L('knw'), b.rope, b.core, st.kc, st.vc, b.posBuf]);
          e.attnV = g.bind(P.attnV, [b.core, st.kc, st.vc, b.qkvz, b.gn, b.posBuf]);
        }
        e.attnPart = g.bind(P.attnPart, [b.qkvz, L('qnw'), L('knw'), b.rope, st.kc, st.vc, b.parts, b.u]);
        e.attnPrepD = g.bind(P.attnPrepD, [b.qkvz, L('qnw'), L('knw'), b.rope, b.core, st.kc, st.vc, b.u]);
        e.attnPartD = g.bind(P.attnPartD, [b.core, st.kc, st.vc, b.parts, b.u]);
        e.attnRed = g.bind(P.attnRed, [b.parts, b.qkvz, b.gn, b.u]);
        e.attnPartI = b.uIter.map((u) => g.bind(P.attnPart, [b.qkvz, L('qnw'), L('knw'), b.rope, st.kc, st.vc, b.parts, u]));
        e.attnPrepDI = b.uIter.map((u) => g.bind(P.attnPrepD, [b.qkvz, L('qnw'), L('knw'), b.rope, b.core, st.kc, st.vc, u]));
        e.attnPartDI = b.uIter.map((u) => g.bind(P.attnPartD, [b.core, st.kc, st.vc, b.parts, u]));
        e.attnRedI = b.uIter.map((u) => g.bind(P.attnRed, [b.parts, b.qkvz, b.gn, u]));
        e.aprep = g.bind(P.aprep, [b.qkvz, L('qnw'), L('knw'), b.rope, b.core, st.kc, st.vc, b.u]);
        e.attn = g.bind(P.attn, [b.core, st.kc, st.vc, b.qkvz, b.gn, b.u]);
        e.o = g.bind(P.o, [L('o').q, L('o').s, b.gn, b.x]);
        e.m_o = g.bind(P.m_out, [L('o').q, L('o').s, b.gn, b.x, b.u]);
      } else {
        e.in = g.bind(P.inL, [L('in').q, L('in').s, ...xin, b.qkvz, ...xtail]);
        e.m_in = g.bind(P.m_inL, [L('in').q, L('in').s, b.xn, b.qkvz, b.u]);
        e.conv = g.bind(P.conv, [L('convw'), st.cs, b.qkvz, b.qkvc]);
        if (this.hasMtp) {
          e.convSnap = g.bind(P.convSnap, [L('convw'), st.cs, b.qkvz, b.qkvc, st.csnap]);
          e.deltaPS = g.bind(P.deltaPS, [st.ds, L('adt'), b.qkvc, b.qkvz, b.core, b.u, st.dsnap]);
          e.vt_in = g.bind(P.vt.inL, [L('in').q, L('in').s, ...xin, b.qkvz, ...xtail]);
          e.vt_out = g.bind(P.vt.out, [L('out').q, L('out').s, b.core, b.x, b.qkvz, L('gnw')]);
          e.convLZ = g.bind(P.convLZ, [L('convw'), st.cs, b.qkvz, b.qkvc, st.csnap, b.posBuf, b.ctl]);
          e.convShiftF = g.bind(P.convShiftF, [b.qkvz, st.cs, b.ctl]);
          e.deltaLZ = g.bind(P.deltaLZ, [st.ds, L('adt'), b.qkvc, b.qkvz, b.core, st.dsnap, b.posBuf, b.ctl]);
        }
        e.convS = g.bind(P.convS, [L('convw'), st.cs, b.qkvz, b.qkvc]);
        e.convSF = g.bind(P.convSF, [L('convw'), st.cs, b.qkvz, b.qkvc, b.ctl]);
        e.convShift = g.bind(P.convShift, [b.qkvz, st.cs, b.u]);
        e.dprep = g.bind(P.dprep, [b.qkvc, b.qkvz, L('adt'), b.qn, b.kn, b.gb]);
        e.delta = g.bind(P.delta, [st.ds, b.qn, b.kn, b.qkvc, b.gb, b.core, b.u]);
        e.deltaP = g.bind(P.deltaP, [st.ds, L('adt'), b.qkvc, b.qkvz, b.core, b.u]);
        e.deltaPF = g.bind(P.deltaPF, [st.ds, L('adt'), b.qkvc, b.qkvz, b.core, b.u, b.ctl]);
        e.gnorm = g.bind(P.gnorm, [b.core, b.qkvz, L('gnw'), b.gn]);
        e.outG = g.bind(P.outG, [L('out').q, L('out').s, b.core, b.x, b.qkvz, L('gnw')]);
        e.m_out = g.bind(P.m_out, [L('out').q, L('out').s, b.gn, b.x, b.u]);
      }
      e.gateup = g.bind(P.gateup, [L('gateup').q, L('gateup').s, ...xin, b.gu, ...xtail]);
      e.m_gateup = g.bind(P.m_gateup, [L('gateup').q, L('gateup').s, b.xn, b.gu, b.u]);
      e.silumul = g.bind(P.silumul, [b.gu, b.act]);
      e.down = g.bind(P.down, [L('down').q, L('down').s, b.gu, b.x]);
      e.m_down = g.bind(P.m_down, [L('down').q, L('down').s, b.act, b.x, b.u]);
      if (this.hasMtp) {
        e.vt_gateup = g.bind(P.vt.gateup, [L('gateup').q, L('gateup').s, ...xin, b.gu, ...xtail]);
        e.vt_down = g.bind(P.vt.down, [L('down').q, L('down').s, b.gu, b.x]);
      }
      this.lb.push(e);
    }
    this.hb = this.i8 ? {
      normD: g.bind(P.rmsQ, [b.x, w.norm, b.xq, b.xsm]),
      normP: g.bind(P.rmsQ, [b.hx, w.norm, b.xq, b.xsm]),
      lm: w.embShards.map((sh, s) => g.bind(P.lm[s], [sh.q, sh.s, b.xq, b.logits, b.xsm])),
      lmPre: w.embShards.map((sh, s) => g.bind(P.lm[s], [sh.q, sh.s, b.xq, b.logits, b.xsm])),
      gather: w.embShards.map((sh, s) => g.bind(P.gather[s], [sh.q, sh.s, b.tokens, b.x])),
      topk: g.bind(P.topk, [b.logits, b.cand]),
      topkAllowed: g.bind(P.topkAllowed, [b.logits, b.cand, b.allowed, b.uAllowed]),
      samp: g.bind(P.samp, [b.cand, b.sp, b.tokens, b.sampled, b.ctl, b.recent]),
      vhead: this.hasMtp ? g.bind(P.rms, [b.x, w.norm, b.hN]) : null,
      vheadQ: this.hasMtp ? g.bind(P.rmsQ, [b.x, w.norm, b.xq, b.xsm]) : null,
      lmT: this.hasMtp ? w.embShards.map((sh, s) => g.bind(P.vt.lm[s], [sh.q, sh.s, b.xq, b.logits, b.xsm])) : null,
    } : {
      normD: g.bind(P.rms, [b.x, w.norm, b.xn]),      // decode: final norm on row 0
      normP: g.bind(P.rms, [b.hx, w.norm, b.hn]),     // prefill: norm the copied last row
      lm: w.embShards.map((sh, s) => g.bind(P.lm[s], [sh.q, sh.s, b.xn, b.logits])),
      lmPre: w.embShards.map((sh, s) => g.bind(P.lm[s], [sh.q, sh.s, b.hn, b.logits])),
      gather: w.embShards.map((sh, s) => g.bind(P.gather[s], [sh.q, sh.s, b.tokens, b.x])),
      topk: g.bind(P.topk, [b.logits, b.cand]),
      topkAllowed: g.bind(P.topkAllowed, [b.logits, b.cand, b.allowed, b.uAllowed]),
      samp: g.bind(P.samp, [b.cand, b.sp, b.tokens, b.sampled, b.ctl, b.recent]),
      vhead: this.hasMtp ? g.bind(P.rms, [b.x, w.norm, b.hN]) : null,
      vheadQ: null,
      lmT: this.hasMtp ? w.embShards.map((sh, s) => g.bind(P.vt.lm[s], [sh.q, sh.s, b.hN, b.logits])) : null,
    };
    if (this.hasMtp) this.#mtpBinds();
  }

  #mtpBinds() {
    const g = this.gpu, b = this.b, P = this.p, C = this.chunk;
    const M = (n) => this.w[`mtp.${n}`];
    const R = (buf, row) => ({ buffer: buf, offset: row * 10240, size: 10240 });
    const xin = this.i8 ? [b.xq] : [b.xn];
    const xtail = this.i8 ? [b.xsm] : [];
    const mb = {};
    const layer = (u) => ({
      part: g.bind(P.attnPart, [b.qkvz, M('qnw'), M('knw'), b.rope, b.kcM, b.vcM, b.parts, u]),
      prepD: g.bind(P.attnPrepD, [b.qkvz, M('qnw'), M('knw'), b.rope, b.core, b.kcM, b.vcM, u]),
      partD: g.bind(P.attnPartD, [b.core, b.kcM, b.vcM, b.parts, u]),
      red: g.bind(P.attnRed, [b.parts, b.qkvz, b.gn, u]),
    });
    mb.A = layer(b.uA);
    mb.B = layer(b.uB);
    mb.ln1 = this.i8 ? g.bind(P.rmsQ, [b.mtpH, M('ln1'), b.xq, b.xsm]) : g.bind(P.rms, [b.mtpH, M('ln1'), b.xn]);
    mb.ln2 = this.i8 ? g.bind(P.rmsQ, [b.mtpH, M('ln2'), b.xq, b.xsm]) : g.bind(P.rms, [b.mtpH, M('ln2'), b.xn]);
    mb.in = g.bind(P.inF, [M('in').q, M('in').s, ...xin, b.qkvz, ...xtail]);
    mb.o = g.bind(P.o, [M('o').q, M('o').s, b.gn, b.mtpH]);
    mb.gateup = g.bind(P.gateup, [M('gateup').q, M('gateup').s, ...xin, b.gu, ...xtail]);
    mb.down = g.bind(P.down, [M('down').q, M('down').s, b.gu, b.mtpH]);
    mb.normF = g.bind(P.rms, [b.mtpH, M('norm'), b.mtpHn]); // f32 hidden for the draft chain
    mb.normQ = this.i8 ? g.bind(P.rmsQ, [b.mtpH, M('norm'), b.xq, b.xsm])
                       : g.bind(P.rms, [b.mtpH, M('norm'), b.xn]); // feeds the shared lm binds
    mb.preE = [0, 1, 2].map((r) => g.bind(P.preE, [R(b.x, r), M('preE'), b.fcIn]));
    mb.preH_hN = [0, 1, 2].map((r) => g.bind(P.preH, [R(b.hN, r), M('preH'), b.fcIn]));
    mb.preH_chain = g.bind(P.preH, [b.mtpHn, M('preH'), b.fcIn]);
    mb.fc = g.bind(P.fc, [M('fc').q, M('fc').s, b.fcIn, b.mtpH]);
    mb.am1 = g.bind(P.am1, [b.cand, b.tokens, b.draftIds]);
    mb.am2 = g.bind(P.am2, [b.cand, b.tokens, b.draftIds]);
    mb.cu = [b.uC0, b.uC1].map((u) =>
      g.bind(P.aprep, [b.qkvz, M('qnw'), M('knw'), b.rope, b.core, b.kcM, b.vcM, u]));
    // chained speculation (storage-position) binds
    mb.AS = {
      part: g.bind(P.attnPartA, [b.qkvz, M('qnw'), M('knw'), b.rope, b.kcM, b.vcM, b.parts, b.posBuf]),
      prepD: g.bind(P.attnPrepDA, [b.qkvz, M('qnw'), M('knw'), b.rope, b.core, b.kcM, b.vcM, b.posBuf]),
      partD: g.bind(P.attnPartDA, [b.core, b.kcM, b.vcM, b.parts, b.posBuf]),
      red: g.bind(P.attnRedA, [b.parts, b.qkvz, b.gn, b.posBuf]),
    };
    mb.BS = {
      part: g.bind(P.attnPartB, [b.qkvz, M('qnw'), M('knw'), b.rope, b.kcM, b.vcM, b.parts, b.posBuf]),
      prepD: g.bind(P.attnPrepDB, [b.qkvz, M('qnw'), M('knw'), b.rope, b.core, b.kcM, b.vcM, b.posBuf]),
      partD: g.bind(P.attnPartDB, [b.core, b.kcM, b.vcM, b.parts, b.posBuf]),
      red: g.bind(P.attnRedB, [b.parts, b.qkvz, b.gn, b.posBuf]),
    };
    mb.cuS = [0, 1].map((j) =>
      g.bind(P.aprepCu[j], [b.qkvz, M('qnw'), M('knw'), b.rope, b.core, b.kcM, b.vcM, b.posBuf]));
    mb.preHD = g.bind(P.preHD, [b.hN, M('preH'), b.fcIn, b.posBuf]);
    mb.accept = g.bind(P.accept, [b.cand, b.sp, b.draftIds, b.tokens, b.posBuf, b.ctl, b.specLog, b.recent]);
    mb.pf = {
      copyEmb: g.bind(P.copyRows, [b.x, b.embK]),
      hNorm: g.bind(P.rms, [b.x, this.w.norm, { buffer: b.hN, offset: 10240, size: C * 10240 }]),
      preE: g.bind(P.preE, [b.embK, M('preE'), b.fcIn]),
      preH: g.bind(P.preH, [b.hN, M('preH'), b.fcIn]),
      fc: g.bind(P.m_fc, [M('fc').q, M('fc').s, b.fcIn, b.mtpH, b.u]),
      ln1: g.bind(P.rms, [b.mtpH, M('ln1'), b.xn]),
      in: g.bind(P.m_inF, [M('in').q, M('in').s, b.xn, b.qkvz, b.u]),
      prep: g.bind(P.aprep, [b.qkvz, M('qnw'), M('knw'), b.rope, b.core, b.kcM, b.vcM, b.uPf]),
    };
    this.mb = mb;
  }

  #setU(basePos, T) {
    this.gpu.device.queue.writeBuffer(this.b.u, 0, new Uint32Array([basePos, T, 0, 0]));
  }

  // ---- presence-penalty window (generated tokens, per assistant turn) ----
  resetPenaltyWindow() { this.genIds = []; }
  notePenaltyToken(id) { this.genIds.push(id); }
  recentSet() { return new Set(this.genIds.slice(-RT.ppWindow)); }
  // Seed the GPU ring from the CPU-side list before each batched submit; the
  // sampler/acceptSpec kernels append tokens sampled within the batch.
  #writeRecent() {
    const W = RT.ppWindow;
    const arr = new Uint32Array(1 + W);
    const n = this.genIds.length;
    arr[0] = n;
    for (let i = Math.max(0, n - W); i < n; i++) arr[1 + (i % W)] = this.genIds[i];
    this.gpu.device.queue.writeBuffer(this.b.recent, 0, arr);
  }

  #packSamplerParams(sp = {}) {
    const rawStopIds = sp.stopIds !== undefined
      ? sp.stopIds
      : (sp.eosId !== undefined ? [sp.eosId] : [STOP_PAD]);
    if (!Array.isArray(rawStopIds)) throw new Error('stopIds must be an array');
    if (rawStopIds.length > MAX_STOP_IDS) throw new Error(`stopIds supports at most ${MAX_STOP_IDS} ids`);
    const stopIds = rawStopIds.map((id) => {
      if (!Number.isInteger(id)) throw new Error(`invalid stop token id: ${id}`);
      return id >>> 0;
    });
    const primaryStop = stopIds[0] ?? STOP_PAD;
    const eosId = (sp.eosId !== undefined ? sp.eosId : primaryStop) >>> 0;
    const buf = new ArrayBuffer(48);
    const u = new Uint32Array(buf);
    const f = new Float32Array(buf);
    u[0] = (sp.seed ?? 1) >>> 0;
    u[1] = stopIds.length;
    // Clamp: topK=0 or NaN params would make the sampler shader read out of
    // bounds / emit arbitrary tokens.
    u[2] = Math.max(1, Math.min(RT.topkK, Number.isFinite(sp.topK) ? sp.topK : RT.topkK));
    f[3] = Number.isFinite(sp.temperature) && sp.temperature > 0 ? sp.temperature : 0;
    f[4] = Number.isFinite(sp.topP) ? Math.min(1, Math.max(0, sp.topP)) : 1;
    f[5] = Number.isFinite(sp.presencePenalty) ? Math.min(10, Math.max(0, sp.presencePenalty)) : 0;
    for (let i = 0; i < MAX_STOP_IDS; i++) u[8 + i] = stopIds[i] ?? STOP_PAD;
    return { buf, stopIds, stopSet: new Set(stopIds), eosId };
  }

  #firstStopIndex(ids, packed) {
    return ids.findIndex((id) => packed.stopSet.has(id >>> 0));
  }

  #encodeDecode(pos) {
    const g = this.gpu, P = this.p, b = this.b;
    const nP = Math.floor(pos / PSIZE) + 1; // partitions covering pos+1 positions
    const enc = g.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    const run = (pipe, bind, x, y = 1) => { pass.setPipeline(pipe); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(x, y); };
    const rmsP = this.i8 ? P.rmsQ : P.rms;
    for (let s = 0; s < this.w.embShards.length; s++) run(P.gather[s], this.hb.gather[s], 1);
    for (const e of this.lb) {
      run(rmsP, this.i8 ? e.ln1Q : e.ln1, 1);
      if (e.full) {
        run(P.inF, e.in, CFG.inF / 4);
        if (!g.subgroups || nP < ATTN_SPLIT_PARTITIONS) run(P.attnPart, e.attnPart, nP, 4);
        else {
          run(P.attnPrepD, e.attnPrepD, 4);
          run(P.attnPartD, e.attnPartD, nP, 4);
        }
        run(P.attnRed, e.attnRed, 4, 4);
        run(P.o, e.o, 2560 / 4);
      } else {
        run(P.inL, e.in, CFG.inL / 4);
        run(P.convS, e.convS, 32, 1);
        run(P.deltaP, e.deltaP, 256);
        run(P.outG, e.outG, 2560 / 4);
      }
      run(rmsP, this.i8 ? e.ln2Q : e.ln2, 1);
      run(P.gateup, e.gateup, 18432 / 4);
      run(P.down, e.down, 2560 / 4);
    }
    run(rmsP, this.hb.normD, 1);
    for (let s = 0; s < this.w.embShards.length; s++) {
      run(P.lm[s], this.hb.lm[s], Math.ceil(this.w.embShards[s].rows / 4));
    }
    run(P.topk, this.hb.topk, TOPK_WGS);
    pass.end();
    enc.copyBufferToBuffer(b.cand, 0, this.candRead, 0, this.candBytes);
    return enc.finish();
  }

  // opts: withHead (normal next-token head), snap (per-token state snapshots, for
  // speculative verify), mtpPf (build MTP KV cache for these positions),
  // specHead (per-position logits + top-k, for verify), enc (continue an encoder)
  #encodePrefill(T, opts = {}) {
    const g = this.gpu, P = this.p, b = this.b, mb = this.mb;
    const enc = opts.enc ?? g.device.createCommandEncoder();
    let pass = enc.beginComputePass();
    const run = (pipe, bind, x, y = 1) => { pass.setPipeline(pipe); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(x, y); };
    const mm = (pipe, bind, N) => run(pipe, bind, Math.ceil(N / 128), Math.ceil(T / 32));
    for (let s = 0; s < this.w.embShards.length; s++) run(P.gather[s], this.hb.gather[s], T);
    if (opts.mtpPf) run(P.copyRows, mb.pf.copyEmb, (T * CFG.hidden) / 256); // keep embeddings
    for (const e of this.lb) {
      run(P.rms, e.ln1, T);
      if (e.full) {
        mm(P.m_inF, e.m_in, CFG.inF);
        run(P.aprep, e.aprep, T, 24);
        run(P.attn, e.attn, T, 4);
        mm(P.m_out, e.m_o, 2560);
      } else {
        mm(P.m_inL, e.m_in, CFG.inL);
        run(P.conv, e.conv, 32, T);
        run(P.convShift, e.convShift, 32);
        run(P.dprep, e.dprep, T, 32);
        run(P.delta, e.delta, 256);
        run(P.gnorm, e.gnorm, T, 32);
        mm(P.m_out, e.m_out, 2560);
      }
      run(P.rms, e.ln2, T);
      mm(P.m_gateup, e.m_gateup, 18432);
      run(P.silumul, e.silumul, 36, T);
      mm(P.m_down, e.m_down, 2560);
    }
    if (opts.mtpPf) {
      // batched MTP cache build: entry slot (chunkStart-1+j) = (h[j-1 shifted], emb tokens[j])
      run(P.rms, mb.pf.hNorm, T);        // hN rows 1..T = normed h of this chunk
      run(P.preE, mb.pf.preE, T);
      run(P.preH, mb.pf.preH, T);
      mm(P.m_fc, mb.pf.fc, 2560);
      run(P.rms, mb.pf.ln1, T);
      mm(P.m_inF, mb.pf.in, CFG.inF);
      run(P.aprep, mb.pf.prep, T, 24);   // uPf: basePos = chunkStart-1, skips row 0 on chunk 0
    }
    pass.end();
    if (opts.mtpPf) {
      // rotate: h of this chunk's last position → hN row 0 (bounce via hx;
      // same-buffer copies are not universally supported)
      enc.copyBufferToBuffer(b.hN, T * CFG.hidden * F, b.hx, 0, CFG.hidden * F);
      enc.copyBufferToBuffer(b.hx, 0, b.hN, 0, CFG.hidden * F);
    }
    if (opts.withHead) {
      enc.copyBufferToBuffer(b.x, (T - 1) * CFG.hidden * F, b.hx, 0, CFG.hidden * F);
      pass = enc.beginComputePass();
      run(this.i8 ? P.rmsQ : P.rms, this.hb.normP, 1);
      for (let s = 0; s < this.w.embShards.length; s++) {
        run(P.lm[s], this.hb.lmPre[s], Math.ceil(this.w.embShards[s].rows / 4));
      }
      run(P.topk, this.hb.topk, TOPK_WGS);
      pass.end();
      enc.copyBufferToBuffer(b.cand, 0, this.candRead, 0, this.candBytes);
    }
    return enc;
  }

  // Fused verify forward (T=3, speculative decoding): mirrors the decode path
  // (i8 GEMVs, inline delta prep, gnorm/GLU fusion) with per-token state
  // snapshots, plus per-position logits + top-k.
  #encodeVerify(enc) {
    const P = this.p, b = this.b;
    const T = 3;
    const pass = enc.beginComputePass();
    const run = (pipe, bind, x, y = 1) => { pass.setPipeline(pipe); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(x, y); };
    const rmsP = this.i8 ? P.rmsQ : P.rms;
    for (let s = 0; s < this.w.embShards.length; s++) run(P.gather[s], this.hb.gather[s], T);
    for (const e of this.lb) {
      run(rmsP, this.i8 ? e.ln1Q : e.ln1, T);
      if (e.full) {
        run(P.vt.inF, e.vt_in, CFG.inF / 4);
        run(P.aprep, e.aprep, T, 24);
        run(P.attn, e.attn, T, 4);
        run(P.vt.o, e.vt_o, 2560 / 4);
      } else {
        run(P.vt.inL, e.vt_in, CFG.inL / 4);
        run(P.convSnap, e.convSnap, 32, T);
        run(P.convShift, e.convShift, 32);
        run(P.deltaPS, e.deltaPS, 256);
        run(P.vt.out, e.vt_out, 2560 / 4);
      }
      run(rmsP, this.i8 ? e.ln2Q : e.ln2, T);
      run(P.vt.gateup, e.vt_gateup, 18432 / 4);
      run(P.vt.down, e.vt_down, 2560 / 4);
    }
    run(P.rms, this.hb.vhead, T);            // f32 normed h rows (next round's MTP inputs)
    if (this.i8) run(P.rmsQ, this.hb.vheadQ, T);
    for (let s = 0; s < this.w.embShards.length; s++) {
      run(P.vt.lm[s], this.hb.lmT[s], Math.ceil(this.w.embShards[s].rows / 4));
    }
    run(P.topk, this.hb.topk, TOPK_WGS, T);
    pass.end();
    enc.copyBufferToBuffer(b.cand, 0, this.candRead, 0, 3 * this.candBytes);
    enc.copyBufferToBuffer(b.draftIds, 0, this.candRead, 3 * this.candBytes, 8);
    return enc;
  }

  async #readBack() {
    await this.candRead.mapAsync(GPUMapMode.READ);
    const buf = this.candRead.getMappedRange().slice(0);
    this.candRead.unmap();
    return buf;
  }

  #candRow(buf, r) {
    const n = this.candBytes / 4;
    return { ids: new Uint32Array(buf, r * this.candBytes, n), vals: new Float32Array(buf, r * this.candBytes, n) };
  }

  // Apply a pending post-verify state rollback (partial draft acceptance).
  #applyRestore(enc) {
    if (!this._restoreTo) return;
    const a = this._restoreTo;
    for (const st of this.state) {
      if (!st.ds) continue;
      enc.copyBufferToBuffer(st.dsnap, (a - 1) * 2097152, st.ds, 0, 2097152);
      enc.copyBufferToBuffer(st.csnap, (a - 1) * 98304, st.cs, 0, 98304);
    }
    this._restoreTo = 0;
  }

  /** Feed prompt tokens; returns top-k candidates for the next token. */
  async prefill(tokens, onProgress = () => {}) {
    const q = this.gpu.device.queue;
    const mtpPf = this.spec && this.hasMtp;
    let off = 0;
    let cands = null;
    while (off < tokens.length) {
      const n = Math.min(this.chunk, tokens.length - off);
      const last = off + n >= tokens.length;
      if (this.pos + n > this.maxCtx) throw new Error('Context window exceeded');
      q.writeBuffer(this.b.tokens, 0, new Uint32Array(tokens.slice(off, off + n)));
      this.#setU(this.pos, n);
      if (mtpPf) {
        q.writeBuffer(this.b.uPf, 0, new Uint32Array([(this.pos - 1) >>> 0, n, this.pos === 0 ? 1 : 0, 0]));
      }
      const enc = this.gpu.device.createCommandEncoder();
      this.#applyRestore(enc);
      q.submit([this.#encodePrefill(n, { withHead: last, mtpPf, enc }).finish()]);
      if (last) cands = this.#candRow(await this.#readBack(), 0);
      else await q.onSubmittedWorkDone();
      this.pos += n;
      off += n;
      onProgress(off, tokens.length);
    }
    if (mtpPf) {
      this.mtpExact = this.pos - 1; // slots 0..pos-2 built; hN row 0 = h_{pos-1}
      this._aPrev = 1;
    }
    return cands;
  }

  /**
   * Batched decode: feeds `firstTok`, then chains up to k-1 further tokens in a
   * single GPU submit — each iteration's sampler writes the next input token
   * on-GPU (no CPU round-trip). After a stop token, state-mutating kernels
   * freeze via a flag, so no rollback is needed. Returns sampled tokens through
   * the first stop token, if hit, and how many tokens were actually fed into the state.
   */
  async decodeBatch(firstTok, k, sp) {
    const g = this.gpu, q = g.device.queue, P = this.p, b = this.b;
    k = Math.min(k, this.BATCH, this.maxCtx - this.pos);
    if (k < 1) throw new Error('Context window exceeded');
    const startPos = this.pos;
    const allowedCount = sp.allowedTokenIds?.length ?? 0;
    if (allowedCount) {
      if (allowedCount > CFG.vocab) throw new Error('Tool constraint has too many allowed token IDs.');
      q.writeBuffer(b.allowed, 0, sp.allowedTokenIds);
    }
    q.writeBuffer(b.uAllowed, 0, new Uint32Array([allowedCount, 0, 0, 0]));
    q.writeBuffer(b.tokens, 0, new Uint32Array([firstTok]));
    q.writeBuffer(b.ctl, 0, new Uint32Array([0, 0]));
    q.writeBuffer(b.u, 0, new Uint32Array([startPos, 1, 0, 0])); // deltaP reads only u.T
    const packed = this.#packSamplerParams(sp);
    q.writeBuffer(b.sp, 0, packed.buf);
    this.#writeRecent();
    for (let i = 0; i < k; i++) {
      q.writeBuffer(b.uIter[i], 0, new Uint32Array([startPos + i, 1, 0, 0]));
    }

    const enc = g.device.createCommandEncoder();
    this.#applyRestore(enc);
    const pass = enc.beginComputePass();
    const run = (pipe, bind, x, y = 1) => { pass.setPipeline(pipe); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(x, y); };
    const rmsP = this.i8 ? P.rmsQ : P.rms;
    for (let i = 0; i < k; i++) {
      const nP = Math.floor((startPos + i) / PSIZE) + 1;
      for (let s = 0; s < this.w.embShards.length; s++) run(P.gather[s], this.hb.gather[s], 1);
      for (const e of this.lb) {
        run(rmsP, this.i8 ? e.ln1Q : e.ln1, 1);
        if (e.full) {
          run(P.inF, e.in, CFG.inF / 4);
          if (!g.subgroups || nP < ATTN_SPLIT_PARTITIONS) run(P.attnPart, e.attnPartI[i], nP, 4);
          else {
            run(P.attnPrepD, e.attnPrepDI[i], 4);
            run(P.attnPartD, e.attnPartDI[i], nP, 4);
          }
          run(P.attnRed, e.attnRedI[i], 4, 4);
          run(P.o, e.o, 2560 / 4);
        } else {
          run(P.inL, e.in, CFG.inL / 4);
          run(P.convSF, e.convSF, 32, 1);
          run(P.deltaPF, e.deltaPF, 256);
          run(P.outG, e.outG, 2560 / 4);
        }
        run(rmsP, this.i8 ? e.ln2Q : e.ln2, 1);
        run(P.gateup, e.gateup, 18432 / 4);
        run(P.down, e.down, 2560 / 4);
      }
      run(rmsP, this.hb.normD, 1);
      for (let s = 0; s < this.w.embShards.length; s++) {
        run(P.lm[s], this.hb.lm[s], Math.ceil(this.w.embShards[s].rows / 4));
      }
      run(allowedCount ? P.topkAllowed : P.topk, allowedCount ? this.hb.topkAllowed : this.hb.topk, TOPK_WGS, 1);
      run(P.samp, this.hb.samp, 1);
    }
    pass.end();
    enc.copyBufferToBuffer(b.sampled, 0, this.sampRead, 0, this.BATCH * 4);
    enc.copyBufferToBuffer(b.ctl, 0, this.sampRead, this.BATCH * 4, 8);
    q.submit([enc.finish()]);

    await this.sampRead.mapAsync(GPUMapMode.READ);
    const buf = this.sampRead.getMappedRange().slice(0);
    this.sampRead.unmap();
    const ids = Array.from(new Uint32Array(buf, 0, k));
    const stopAt = this.#firstStopIndex(ids, packed);
    const stopped = stopAt >= 0;
    const stopId = stopped ? ids[stopAt] : null;
    const fed = stopped ? stopAt + 1 : k; // iterations after a stop sample left state frozen
    this.pos = startPos + fed;
    // mirror the GPU's ring appends (sampled non-stop tokens)
    this.genIds.push(...(stopped ? ids.slice(0, stopAt) : ids.slice(0, k)));
    return {
      ids: stopped ? ids.slice(0, stopAt + 1) : ids,
      fed,
      eos: stopped && stopId === packed.eosId,
      stopped,
      stopId,
    };
  }

  /** Feed one committed token; returns candidates for the next. */
  async decode(token) {
    if (this.pos + 1 > this.maxCtx) throw new Error('Context window exceeded');
    const q = this.gpu.device.queue;
    q.writeBuffer(this.b.tokens, 0, new Uint32Array([token]));
    this.#setU(this.pos, 1);
    if (this._restoreTo) {
      const enc = this.gpu.device.createCommandEncoder();
      this.#applyRestore(enc);
      q.submit([enc.finish()]);
    }
    q.submit([this.#encodeDecode(this.pos)]);
    this.pos += 1;
    return this.#candRow(await this.#readBack(), 0);
  }

  /**
   * MTP self-speculative round: commit `t1` (already sampled), draft 2 more
   * tokens with the MTP head, verify all 3 in one batched forward.
   * Returns { extra: tokens accepted beyond t1, next: sampled next token }.
   */
  async specRound(t1, sampleFn) {
    if (!this.hasMtp) throw new Error('MTP weights missing; speculative decoding is unavailable');
    const g = this.gpu, q = g.device.queue, P = this.p, b = this.b, mb = this.mb;
    const pos = this.pos; // committed count before this round
    if (pos + 3 > this.maxCtx) throw new Error('Context window exceeded');
    const aPrev = this._aPrev;
    const cu = Math.max(0, Math.min(2, (pos - 1) - this.mtpExact));

    q.writeBuffer(b.tokens, 0, new Uint32Array([t1]));
    this.#setU(pos, 3);
    q.writeBuffer(b.uA, 0, new Uint32Array([pos - 1, 1, 0, 0]));
    q.writeBuffer(b.uB, 0, new Uint32Array([pos, 1, 0, 0]));
    q.writeBuffer(b.uC0, 0, new Uint32Array([this.mtpExact, 1, 0, 0]));
    q.writeBuffer(b.uC1, 0, new Uint32Array([this.mtpExact + 1, 1, 0, 0]));

    const enc = g.device.createCommandEncoder();
    this.#applyRestore(enc);
    let pass = enc.beginComputePass();
    const run = (pipe, bind, x, y = 1) => { pass.setPipeline(pipe); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(x, y); };
    const rmsP = this.i8 ? P.rmsQ : P.rms;
    const gathers = (n) => { for (let s = 0; s < this.w.embShards.length; s++) run(P.gather[s], this.hb.gather[s], n); };
    const mtpFront = (preHBind, embRow) => { // fc input assembly + fc
      gathers(embRow + 1);
      run(P.preE, mb.preE[embRow], 1);
      run(P.preH, preHBind, 1);
      run(P.fc, mb.fc, 2560 / 4);
    };
    const mtpLayer = (lb, nP) => {
      run(rmsP, mb.ln1, 1);
      run(P.inF, mb.in, CFG.inF / 4);
      if (!g.subgroups || nP < ATTN_SPLIT_PARTITIONS) run(P.attnPart, lb.part, nP, 4);
      else {
        run(P.attnPrepD, lb.prepD, 4);
        run(P.attnPartD, lb.partD, nP, 4);
      }
      run(P.attnRed, lb.red, 4, 4);
      run(P.o, mb.o, 2560 / 4);
      run(rmsP, mb.ln2, 1);
      run(P.gateup, mb.gateup, 18432 / 4);
      run(P.down, mb.down, 2560 / 4);
    };
    const draftHead = (am) => {
      run(this.i8 ? P.rmsQ : P.rms, mb.normQ, 1);
      for (let s = 0; s < this.w.embShards.length; s++) run(P.lm[s], this.hb.lm[s], Math.ceil(this.w.embShards[s].rows / 4));
      run(P.topk, this.hb.topk, TOPK_WGS, 1);
      run(am === 1 ? P.am1 : P.am2, am === 1 ? mb.am1 : mb.am2, 1);
    };

    // catch-up: exact MTP cache entries for tokens accepted last round
    for (let j = 0; j < cu; j++) {
      mtpFront(mb.preH_hN[j], j + 1); // token = prev round's tokens[j+1] (still in buffer)
      run(rmsP, mb.ln1, 1);
      run(P.inF, mb.in, CFG.inF / 4);
      run(P.aprep, mb.cu[j], 1, 24);
    }
    // draft A: slot pos-1, exact inputs (h_{pos-1}, t1)
    mtpFront(mb.preH_hN[aPrev - 1], 0);
    mtpLayer(mb.A, Math.floor((pos - 1) / PSIZE) + 1);
    run(P.rms, mb.normF, 1);
    draftHead(1);
    // draft B: slot pos, chained hidden + drafted token
    mtpFront(mb.preH_chain, 1);
    mtpLayer(mb.B, Math.floor(pos / PSIZE) + 1);
    run(P.rms, mb.normF, 1);
    draftHead(2);
    pass.end();

    // verify all 3 chunk tokens through the main model (with state snapshots)
    q.submit([this.#encodeVerify(enc).finish()]);

    const buf = await this.#readBack();
    const dr = new Uint32Array(buf, 3 * this.candBytes, 2);
    const extra = [];
    let a = 1;
    let next = sampleFn(this.#candRow(buf, 0));
    if (next === dr[0]) {
      extra.push(next);
      a = 2;
      next = sampleFn(this.#candRow(buf, 1));
      if (next === dr[1]) {
        extra.push(next);
        a = 3;
        next = sampleFn(this.#candRow(buf, 2));
      }
    }
    this.pos = pos + a;
    this.mtpExact = pos;             // draft A appended the exact entry for slot pos-1
    this._restoreTo = a < 3 ? a : 0; // partial accept → roll back delta/conv states
    this._aPrev = a;
    return { extra, next, accepted: a };
  }

  /**
   * GPU-chained speculative decoding: R rounds of (catch-up → 2 MTP drafts →
   * fused 3-token verify → GPU acceptance) in ONE submit. Acceptance, position
   * tracking and snapshot-restore all run on the GPU; readback is ~100 bytes.
   */
  async specChain(firstTok, R, sp) {
    if (!this.hasMtp) throw new Error('MTP weights missing; speculative decoding is unavailable');
    const g = this.gpu, q = g.device.queue, P = this.p, b = this.b, mb = this.mb;
    const pos0 = this.pos;
    R = Math.min(R, this.R, Math.floor((this.maxCtx - pos0 - 1) / 3));
    if (R < 1) throw new Error('Context window exceeded');
    q.writeBuffer(b.tokens, 0, new Uint32Array([firstTok]));
    q.writeBuffer(b.ctl, 0, new Uint32Array([0, 0]));
    q.writeBuffer(b.posBuf, 0, new Uint32Array([pos0, this._aPrev, this.mtpExact, this._restoreTo ? 1 : 0, 0, 0, 0, 0]));
    this._restoreTo = 0; // consumed by the lazy snapshot reads
    const packed = this.#packSamplerParams(sp);
    q.writeBuffer(b.sp, 0, packed.buf);
    this.#writeRecent();

    const nPmax = Math.min(this.pmax, Math.floor((pos0 + 3 * R) / PSIZE) + 1); // over-dispatch safe
    const enc = g.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    const run = (pipe, bind, x, y = 1) => { pass.setPipeline(pipe); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(x, y); };
    const rmsP = this.i8 ? P.rmsQ : P.rms;
    const gathers = (n) => { for (let s = 0; s < this.w.embShards.length; s++) run(P.gather[s], this.hb.gather[s], n); };
    const draftHead = (am) => {
      run(this.i8 ? P.rmsQ : P.rms, mb.normQ, 1);
      for (let s = 0; s < this.w.embShards.length; s++) run(P.lm[s], this.hb.lm[s], Math.ceil(this.w.embShards[s].rows / 4));
      run(P.topk, this.hb.topk, TOPK_WGS, 1);
      run(am === 1 ? P.am1 : P.am2, am === 1 ? mb.am1 : mb.am2, 1);
    };

    for (let r = 0; r < R; r++) {
      // catch-up: MTP cache entries for the previous round's accepted tokens (self-guarded)
      for (let j = 0; j < 2; j++) {
        gathers(j + 2);
        run(P.preE, mb.preE[j + 1], 1);
        run(P.preH, mb.preH_hN[j], 1);
        run(P.fc, mb.fc, 2560 / 4);
        run(rmsP, mb.ln1, 1);
        run(P.inF, mb.in, CFG.inF / 4);
        run(P.aprepCu[j], mb.cuS[j], 1, 24);
      }
      // draft A (exact inputs: h of last accepted token, round input token)
      gathers(1);
      run(P.preE, mb.preE[0], 1);
      run(P.preHD, mb.preHD, 1);
      run(P.fc, mb.fc, 2560 / 4);
      run(rmsP, mb.ln1, 1);
      run(P.inF, mb.in, CFG.inF / 4);
      if (!g.subgroups || nPmax < ATTN_SPLIT_PARTITIONS) run(P.attnPartA, mb.AS.part, nPmax, 4);
      else {
        run(P.attnPrepDA, mb.AS.prepD, 4);
        run(P.attnPartDA, mb.AS.partD, nPmax, 4);
      }
      run(P.attnRedA, mb.AS.red, 4, 4);
      run(P.o, mb.o, 2560 / 4);
      run(rmsP, mb.ln2, 1);
      run(P.gateup, mb.gateup, 18432 / 4);
      run(P.down, mb.down, 2560 / 4);
      run(P.rms, mb.normF, 1);
      draftHead(1);
      // draft B (chained hidden, drafted token)
      gathers(2);
      run(P.preE, mb.preE[1], 1);
      run(P.preH, mb.preH_chain, 1);
      run(P.fc, mb.fc, 2560 / 4);
      run(rmsP, mb.ln1, 1);
      run(P.inF, mb.in, CFG.inF / 4);
      if (!g.subgroups || nPmax < ATTN_SPLIT_PARTITIONS) run(P.attnPartB, mb.BS.part, nPmax, 4);
      else {
        run(P.attnPrepDB, mb.BS.prepD, 4);
        run(P.attnPartDB, mb.BS.partD, nPmax, 4);
      }
      run(P.attnRedB, mb.BS.red, 4, 4);
      run(P.o, mb.o, 2560 / 4);
      run(rmsP, mb.ln2, 1);
      run(P.gateup, mb.gateup, 18432 / 4);
      run(P.down, mb.down, 2560 / 4);
      draftHead(2);
      // fused verify (T=3), lazy snapshot-restore + stop freeze
      gathers(3);
      for (const e of this.lb) {
        run(rmsP, this.i8 ? e.ln1Q : e.ln1, 3);
        if (e.full) {
          run(P.vt.inF, e.vt_in, CFG.inF / 4);
          run(P.aprepV, e.aprepV, 3, 24);
          run(P.attnV, e.attnV, 3, 4);
          run(P.vt.o, e.vt_o, 2560 / 4);
        } else {
          run(P.vt.inL, e.vt_in, CFG.inL / 4);
          run(P.convLZ, e.convLZ, 32, 3);
          run(P.convShiftF, e.convShiftF, 32);
          run(P.deltaLZ, e.deltaLZ, 256);
          run(P.vt.out, e.vt_out, 2560 / 4);
        }
        run(rmsP, this.i8 ? e.ln2Q : e.ln2, 3);
        run(P.vt.gateup, e.vt_gateup, 18432 / 4);
        run(P.vt.down, e.vt_down, 2560 / 4);
      }
      run(P.rms, this.hb.vhead, 3);
      if (this.i8) run(P.rmsQ, this.hb.vheadQ, 3);
      for (let s = 0; s < this.w.embShards.length; s++) {
        run(P.vt.lm[s], this.hb.lmT[s], Math.ceil(this.w.embShards[s].rows / 4));
      }
      run(P.topk, this.hb.topk, TOPK_WGS, 3);
      run(P.accept, mb.accept, 1);
    }
    pass.end();
    enc.copyBufferToBuffer(b.specLog, 0, this.specRead, 0, R * 16);
    enc.copyBufferToBuffer(b.posBuf, 0, this.specRead, this.R * 16, 32);
    enc.copyBufferToBuffer(b.ctl, 0, this.specRead, this.R * 16 + 32, 8);
    q.submit([enc.finish()]);

    await this.specRead.mapAsync(GPUMapMode.READ);
    const buf = this.specRead.getMappedRange().slice(0);
    this.specRead.unmap();
    const log = new Uint32Array(buf, 0, R * 4);
    const pb = new Uint32Array(buf, this.R * 16, 8);
    const stopped = new Uint32Array(buf, this.R * 16 + 32, 2)[0] === 1;
    this.pos = pb[0];
    this._aPrev = pb[1];
    this.mtpExact = pb[2];
    this._restoreTo = pb[3] ? pb[1] : 0; // bridge for non-chained paths
    const rounds = [];
    let stopId = null;
    for (let r = 0; r < R; r++) {
      const a = log[r * 4];
      if (a === 0) break; // frozen after a stop token
      const next = log[r * 4 + 1];
      if (stopId === null && packed.stopSet.has(next >>> 0)) stopId = next;
      const d0 = log[r * 4 + 2], d1 = log[r * 4 + 3];
      // mirror acceptSpec's ring appends: pick[0..a-1] = accepted drafts + next
      const picks = a >= 3 ? [d0, d1, next] : a >= 2 ? [d0, next] : [next];
      this.genIds.push(...picks.filter((id) => !packed.stopSet.has(id >>> 0)));
      rounds.push({ a, next, d0, d1 });
    }
    return { rounds, eos: stopped && stopId === packed.eosId, stopped, stopId };
  }

  /** Reset all sequence state (delta states, conv states; KV entries become unreachable). */
  async reset() {
    const enc = this.gpu.device.createCommandEncoder();
    for (const st of this.state) {
      if (st.ds) { enc.clearBuffer(st.ds); enc.clearBuffer(st.cs); }
    }
    this.gpu.device.queue.submit([enc.finish()]);
    this.pos = 0;
    this.genIds = [];
    this.mtpExact = 0;
    this._restoreTo = 0;
    this._aPrev = 1;
    await this.gpu.device.queue.onSubmittedWorkDone();
  }
}
