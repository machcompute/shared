// Gemma 4 E4B architecture constants, copied from the public model config and
// cross-checked against model.safetensors. This is deliberately separate from
// config.js because the two registered runtimes have distinct architectures.

export const GEMMA_E4B_CFG = {
  id: 'google/gemma-4-E4B',
  label: 'Gemma 4 E4B',
  repo: 'google/gemma-4-E4B-it',
  revision: 'main',
  checkpoint: 'model.safetensors',
  cacheNamespace: 'gemma4-e4b-it-q4-v1',
  cacheVersion: 2,

  schemaVersion: 2,

  text: {
    hidden: 2560,
    layers: 42,
    interm: 10240,
    vocab: 262144,
    eps: 1e-6,
    heads: 8,
    kvHeads: 2,
    slidingHeadDim: 256,
    globalHeadDim: 512,
    slidingWindow: 512,
    // Sliding attention rotates its entire 256-wide head.  Full attention has
    // 512-wide heads and proportional RoPE: only 128 channels rotate
    // (the remaining channels receive cos=1 / sin=0).
    slidingRopeDim: 256,
    fullRopeDim: 512,
    fullRotaryDim: 128,
    fullRotaryFactor: 0.25,
    slidingRopeTheta: 10000,
    fullRopeTheta: 1000000,
    pleDim: 256,
    plePackedDim: 10752, // 42 layers * 256 PLE channels
    embedScale: Math.sqrt(2560),
    pleEmbedScale: Math.sqrt(256),
    pleInputScale: Math.SQRT1_2,
    pleModelProjectionScale: 1 / Math.sqrt(2560),
    declaredKvSharedLayers: 18,
    hiddenActivation: 'gelu_pytorch_tanh',
    attentionScale: 1,
    finalLogitSoftcap: 30,
  },

  vision: {
    hidden: 768,
    layers: 16,
    interm: 3072,
    heads: 12,
    headDim: 64,
    patchSize: 16,
    positionEmbeddingSize: 10240,
    poolingKernel: 3,
    softTokensPerImage: 280,
    eps: 1e-6,
    ropeTheta: 100,
    hiddenActivation: 'gelu_pytorch_tanh',
    attentionScale: 1,
    useClippedLinears: true,
    standardize: false,
  },

  audio: {
    hidden: 1024,
    layers: 12,
    interm: 4096,
    heads: 8,
    headDim: 128,
    outputDim: 1536,
    convKernel: 5,
    subsampleChannels: [128, 32],
    attentionChunk: 12,
    // The Hugging Face config counts the query position in this value.  The
    // blocked attention shader therefore consumes 12 prior positions.
    attentionLeft: 13,
    attentionPastHorizon: 12,
    attentionRight: 0,
    eps: 1e-6,
    hiddenActivation: 'silu',
    attentionLogitCap: 50,
    residualWeight: 0.5,
    gradientClipping: 1e10,
    useClippedLinears: true,
  },

  tokens: {
    pad: 0,
    eos: 1,
    bos: 2,
    boi: 255999,
    boa: 256000,
    image: 258880,
    audio: 258881,
    eoi: 258882,
    eoa: 258883,
    video: 258884,
  },
};

// E2B keeps E4B's multimodal towers but uses the smaller dense text decoder.
// Values are pinned to google/gemma-4-E2B-it's config.json so the loader can
// validate the checkpoint rather than infer its shape from tensor names.
export const GEMMA_E2B_CFG = {
  id: 'google/gemma-4-E2B',
  label: 'Gemma 4 E2B',
  repo: 'google/gemma-4-E2B-it',
  revision: '9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf',
  checkpoint: 'model.safetensors',
  cacheNamespace: 'gemma4-e2b-it-q4-v1',
  cacheVersion: 2,
  schemaVersion: 2,
  text: {
    hidden: 1536,
    layers: 35,
    interm: 6144,
    // `use_double_wide_mlp` applies to the shared-KV tail (layers 15–34).
    doubleWideMlpFrom: 15,
    vocab: 262144,
    eps: 1e-6,
    heads: 8,
    kvHeads: 1,
    slidingHeadDim: 256,
    globalHeadDim: 512,
    slidingWindow: 512,
    slidingRopeDim: 256,
    fullRopeDim: 512,
    fullRotaryDim: 128,
    fullRotaryFactor: 0.25,
    slidingRopeTheta: 10000,
    fullRopeTheta: 1000000,
    pleDim: 256,
    plePackedDim: 8960,
    embedScale: Math.sqrt(1536),
    pleEmbedScale: Math.sqrt(256),
    pleInputScale: Math.SQRT1_2,
    pleModelProjectionScale: 1 / Math.sqrt(1536),
    declaredKvSharedLayers: 20,
    hiddenActivation: 'gelu_pytorch_tanh',
    attentionScale: 1,
    finalLogitSoftcap: 30,
  },
  // The media encoders are architecture-identical across E2B and E4B.
  vision: GEMMA_E4B_CFG.vision,
  audio: GEMMA_E4B_CFG.audio,
  tokens: GEMMA_E4B_CFG.tokens,
  layerTypes: Array.from({ length: 35 }, (_, i) => ((i + 1) % 5 === 0 ? 'full_attention' : 'sliding_attention')),
  declaredSharedKvSource: { sliding_attention: 13, full_attention: 14 },
};

// Full/global attention occurs every sixth layer, starting at index 5.  The
// other layers use the 512-token sliding window.
export const GEMMA_E4B_LAYER_TYPES = Array.from(
  { length: GEMMA_E4B_CFG.text.layers },
  (_, i) => ((i + 1) % 6 === 0 ? 'full_attention' : 'sliding_attention'),
);

export const isGemmaE4BFullAttention = (layer) => GEMMA_E4B_LAYER_TYPES[layer] === 'full_attention';

// The published config advertises sharing K/V states for the last 18 layers.
// The public safetensors file nonetheless contains K/V tensors for *all* 42
// layers.  Keep both facts visible to the runtime: the loader is checkpoint
// complete and a later model implementation can deliberately choose either
// the declared sharing policy or the per-layer checkpoint weights.
export const GEMMA_E4B_DECLARED_KV_SHARED_FROM = GEMMA_E4B_CFG.text.layers - GEMMA_E4B_CFG.text.declaredKvSharedLayers; // 24
export const GEMMA_E4B_DECLARED_SHARED_KV_SOURCE = {
  sliding_attention: 22,
  full_attention: 23,
};

export const GEMMA_E4B_CHECKPOINT_HAS_PER_LAYER_KV = true;

export const isGemmaE4BDeclaredKvSharedLayer = (layer) => layer >= GEMMA_E4B_DECLARED_KV_SHARED_FROM;

export const gemmaLayerTypes = (config) => config.layerTypes ?? GEMMA_E4B_LAYER_TYPES;

export const isGemmaFullAttention = (config, layer) =>
  gemmaLayerTypes(config)[layer] === 'full_attention';

export const isGemmaDeclaredKvSharedLayer = (config, layer) =>
  layer >= config.text.layers - config.text.declaredKvSharedLayers;

export const gemmaSharedKvSource = (config, kind) =>
  (config.declaredSharedKvSource ?? GEMMA_E4B_DECLARED_SHARED_KV_SOURCE)[kind];
