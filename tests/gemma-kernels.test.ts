import { describe, expect, it } from "vitest";
import { textFlashAttention } from "../lib/webgpu-llm/gemma-kernels.js";
import { gemmaAttentionChunkRange } from "../lib/webgpu-llm/gemma-model.js";
import { topk } from "../lib/webgpu-llm/kernels.js";

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
