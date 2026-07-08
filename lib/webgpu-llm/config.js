// Qwen3.5-4B text-model architecture constants (from config.json on HF).
// Hybrid stack: 24x GatedDeltaNet ("linear_attention") + 8x full GQA attention,
// full attention every 4th layer (indices 3, 7, 11, ... 31).
export const CFG = {
  repo: 'Qwen/Qwen3.5-4B',
  hidden: 2560,
  layers: 32,
  interm: 9216,
  vocab: 248320,
  eps: 1e-6,

  // full attention
  heads: 16,
  headDim: 256,
  kvHeads: 4,
  ropeDim: 64,          // partial_rotary_factor 0.25 * 256
  ropeTheta: 1e7,
  qgDim: 16 * 256 * 2,  // 8192: q_proj emits [q(256) | gate(256)] interleaved per head
  kvDim: 2 * 4 * 256,   // 2048: k rows then v rows (we concat k_proj+v_proj at load)

  // linear attention (Gated DeltaNet)
  vHeads: 32,
  kHeads: 16,
  kDim: 128,
  vDim: 128,
  convK: 4,
  qkvDim: 16 * 128 * 2 + 32 * 128, // 8192: [q 16x128 | k 16x128 | v 32x128]
  zDim: 32 * 128,                  // 4096
  convDim: 8192,

  // fused input projections (concatenated row-wise at load time)
  inL: 8192 + 4096 + 32 + 32, // 12352: [qkv | z | b | a]
  inF: 8192 + 1024 + 1024,    // 10240: [q&gate | k | v]

  // tokens
  eosText: '<|im_end|>',
};

export const PSIZE = 512; // attention partition length (decode)

export const isFullAttn = (i) => (i % 4) === 3;

// Runtime defaults
export const RT = {
  maxCtx: 65536,     // KV cache length (full-attn layers only)
  chunk: 256,       // prefill chunk size
  topkBlock: 1024,  // vocab slice per top-k workgroup
  topkK: 20,        // candidates per block
  ppWindow: 512,    // presence-penalty window (recent generated tokens)
};
export const TOPK_WGS = Math.ceil(CFG.vocab / RT.topkBlock); // 243
