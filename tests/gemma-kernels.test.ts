import { describe, expect, it } from "vitest";
import {
  GEMM_Q4_TILE_N,
  GEMM_Q4_TILE_T,
  gemvQ4,
  gemvQ4GateUpI8,
  gemmQ4,
  rmsnormAddQ,
  textFlashAttention,
} from "../lib/webgpu-llm/gemma-kernels.js";
import { gemmaAttentionChunkRange } from "../lib/webgpu-llm/gemma-model.js";
import { topk } from "../lib/webgpu-llm/kernels.js";

describe("Gemma tiled Q4 prefill GEMM", () => {
  it("shares a 32x128 tile through workgroup memory", () => {
    const source = gemmQ4({ N: 2560, K: 2560 });

    expect(GEMM_Q4_TILE_N).toBe(128);
    expect(GEMM_Q4_TILE_T).toBe(32);
    expect(source).toContain("@compute @workgroup_size(256)");
    expect(source).toContain("var<workgroup> xTile: array<f32, 544>");
    expect(source).toContain("var<workgroup> wTile: array<f32, 2176>");
    expect(source).toContain("let n0 = wid.x * 128u");
    expect(source).toContain("let t0 = wid.y * 32u");
  });

  it("preserves affine Q4 dequantization and residual output variants", () => {
    const assign = gemmQ4({ N: 128, K: 32 });
    const residual = gemmQ4({ N: 128, K: 32, RESIDUAL: 1, OSTRIDE: 256, OUTOFF: 64 });

    expect(assign).toContain("sm.x * q + sm.y");
    expect(assign).toContain("out[dst] = acc[i * 4u + j]");
    expect(residual).toContain("out[dst] += acc[i * 4u + j]");
    expect(residual).toContain("promptRow * 256u + 64u + outputRow");
  });
});

describe("Gemma decode GEMV subgroup layout", () => {
  it("maps one output row directly to each fixed 32-lane subgroup", () => {
    const source = gemvQ4({
      N: 2560,
      K: 2560,
      SUBGROUPS: 1,
      ROW_LANES: 32,
      DIRECT_SUBGROUP: 1,
    });

    expect(source).toContain("let row = wid.x * 8u + localRow");
    expect(source).toContain("if (sgi == 0u && row < 2560u)");
    expect(source).not.toContain("let subgroupCount");
  });

  it("fuses the residual and quantized pre-FFN norms", () => {
    const source = rmsnormAddQ({ K: 2560 });

    expect(source).toContain("var<workgroup> ysh: array<f32, 2560>");
    expect(source).toContain("out[base + i] = y");
    expect(source).toContain("pack2x16float(vec2f(sc, f32(sum)))");
  });

  it("keeps the portable 64-lane reduction as the default", () => {
    const source = gemvQ4({ N: 2560, K: 2560, SUBGROUPS: 1 });

    expect(source).toContain("let row = wid.x * 4u + localRow");
    expect(source).toContain("let subgroupCount");
  });

  it("uses packed int8 activations for DP4a decode projections", () => {
    const source = gemvQ4({
      N: 20480,
      K: 2560,
      SUBGROUPS: 1,
      ROW_LANES: 32,
      DIRECT_SUBGROUP: 1,
      I8: 1,
    });

    expect(source).toContain("requires packed_4x8_integer_dot_product");
    expect(source).toContain("dot4I8Packed");
    expect(source).toContain("var<storage, read> xsm: array<u32>");
    expect(source).not.toContain("fn loadX");
  });

  it("fuses paired gate/up subgroup rows directly into the MLP activation", () => {
    const source = gemvQ4GateUpI8({ INTERM: 10240, K: 2560 });

    expect(source).toContain("let output = wid.x * 8u + localPair");
    expect(source).toContain("select(0u, 10240u");
    expect(source).toContain("subgroupShuffleXor(acc, 8u)");
    expect(source).toContain("gemma_gelu_tanh(gate) * up");
    expect(source).not.toContain("array<vec4f>");
  });


});

describe("Gemma active attention chunks", () => {
  it("limits sliding attention while retaining absolute chunk indices", () => {
    expect(gemmaAttentionChunkRange(0, 64, 512)).toEqual({ base: 0, count: 1 });
    expect(gemmaAttentionChunkRange(4096, 1, 512)).toEqual({ base: 14, count: 3 });
    expect(gemmaAttentionChunkRange(4096, 1, 0)).toEqual({ base: 0, count: 17 });
    expect(gemmaAttentionChunkRange(4096, 1, 512, 128)).toEqual({ base: 28, count: 5 });
  });

  it("supports paired-head 128-key decode chunks", () => {
    const source = textFlashAttention({
      HEADS: 8,
      KV_HEADS: 2,
      HEAD_DIM: 256,
      MAXCTX: 8192,
      WINDOW: 512,
      SUBGROUPS: 1,
      HEAD_BATCH: 2,
      CHUNK_SIZE: 128,
    });

    expect(source).toContain("let chunkBase = chunk * 128u");
    expect(source).toContain("let valid = lid.x < 128u");
    expect(source).toContain("var<workgroup> probs: array<vec2f, 256>");
  });

  it("offsets dispatched workgroups by the active absolute chunk", () => {
    const source = textFlashAttention({
      HEADS: 8,
      KV_HEADS: 2,
      HEAD_DIM: 256,
      MAXCTX: 8192,
      WINDOW: 512,
      SUBGROUPS: 1,
    });

    expect(source).toContain("chunkBase: u32");
    expect(source).toContain("let chunk = u.chunkBase + wid.z");
    expect(source).toContain("fn gemma_kvq_decode4");
    expect(source).toContain("i32(w << 24u) >> 24u");
    expect(source).toContain("let sgMax = subgroupMax");
    expect(source).toContain("let sgSum = subgroupAdd");
  });
});

describe("Gemma fused logit softcap", () => {
  it("applies the trained cap before candidates reach the sampler", () => {
    const source = topk({ VOCAB: 262144, KTOP: 20, SOFTCAP: 30 });

    expect(source).toContain("30.00000000 * tanh(logits[rowOff + g] / 30.00000000)");
  });
});
