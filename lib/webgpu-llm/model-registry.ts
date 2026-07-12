/**
 * The engine deliberately exposes a fixed model registry instead of accepting
 * arbitrary Hub IDs.  Aside from being safer, this keeps cache schemas,
 * tokenizer implementations and WebGPU kernel contracts model-specific.
 */
export const QWEN_MODEL_ID = "Qwen/Qwen3.5-4B" as const;
export const GEMMA_E4B_MODEL_ID = "google/gemma-4-E4B" as const;
export const GEMMA_E2B_MODEL_ID = "google/gemma-4-E2B" as const;

export type ModelId = typeof QWEN_MODEL_ID | typeof GEMMA_E4B_MODEL_ID | typeof GEMMA_E2B_MODEL_ID;
export type ModelModality = "text" | "image" | "audio" | "video";

export interface ModelProfile {
  id: ModelId;
  label: string;
  modalities: readonly ModelModality[];
  maxContext: number;
}

export const MODEL_PROFILES: Record<ModelId, ModelProfile> = {
  [QWEN_MODEL_ID]: {
    id: QWEN_MODEL_ID,
    label: "Qwen 3.5 4B",
    modalities: ["text"],
    maxContext: 65_536,
  },
  [GEMMA_E4B_MODEL_ID]: {
    id: GEMMA_E4B_MODEL_ID,
    label: "Gemma 4 E4B",
    modalities: ["text", "image", "audio", "video"],
    maxContext: 131_072,
  },
  [GEMMA_E2B_MODEL_ID]: {
    id: GEMMA_E2B_MODEL_ID,
    label: "Gemma 4 E2B",
    modalities: ["text", "image", "audio", "video"],
    maxContext: 131_072,
  },
};

export const DEFAULT_MODEL_ID: ModelId = QWEN_MODEL_ID;

export function isModelId(value: unknown): value is ModelId {
  return value === QWEN_MODEL_ID || value === GEMMA_E4B_MODEL_ID || value === GEMMA_E2B_MODEL_ID;
}

export function isGemmaModelId(value: ModelId): value is typeof GEMMA_E4B_MODEL_ID | typeof GEMMA_E2B_MODEL_ID {
  return value === GEMMA_E4B_MODEL_ID || value === GEMMA_E2B_MODEL_ID;
}

export function getModelProfile(id: ModelId): ModelProfile {
  return MODEL_PROFILES[id];
}

export function availableModels(): ModelProfile[] {
  return Object.values(MODEL_PROFILES);
}

export function assertModelRegistered(id: ModelId): ModelProfile {
  return getModelProfile(id);
}
