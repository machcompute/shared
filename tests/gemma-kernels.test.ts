import { describe, expect, it } from "vitest";
import {
  GEMM_Q4_TILE_N,
  GEMM_Q4_TILE_T,
  gemvQ4,
  gemmQ4,
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

  it("keeps the portable 64-lane reduction as the default", () => {
    const source = gemvQ4({ N: 2560, K: 2560, SUBGROUPS: 1 });

    expect(source).toContain("let row = wid.x * 4u + localRow");
    expect(source).toContain("let subgroupCount");
  });
});

describe("Gemma active attention chunks", () => {
  it("limits sliding attention while retaining absolute chunk indices", () => {
    expect(gemmaAttentionChunkRange(0, 64, 512)).toEqual({ base: 0, count: 1 });
    expect(gemmaAttentionChunkRange(4096, 1, 512)).toEqual({ base: 14, count: 3 });
    expect(gemmaAttentionChunkRange(4096, 1, 0)).toEqual({ base: 0, count: 17 });
  });

  it("offsets dispatched workgroups by the active absolute chunk", () => {
    const source = textFlashAttention({
      HEADS: 8,
      KV_HEADS: 2,
      HEAD_DIM: 256,
      MAXCTX: 8192,
      WINDOW: 512,
    });

    expect(source).toContain("chunkBase: u32");
    expect(source).toContain("let chunk = u.chunkBase + wid.z");
  });
});

describe("Gemma fused logit softcap", () => {
  it("applies the trained cap before candidates reach the sampler", () => {
    const source = topk({ VOCAB: 262144, KTOP: 20, SOFTCAP: 30 });

    expect(source).toContain("30.00000000 * tanh(logits[rowOff + g] / 30.00000000)");
  });
});
