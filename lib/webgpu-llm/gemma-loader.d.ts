import type { GPU } from './gpu.js';
import type { GemmaConfig } from './gemma-config.js';
import type { WeightEntry } from './loader.js';

export type GemmaWeightsMap = Record<string, GPUBuffer | WeightEntry | number> & { __nativeBytes?: number };

export declare function buildGemmaGGUFMap(config?: GemmaConfig): ReadonlyArray<unknown>;
export declare const buildGemmaSpecs: typeof buildGemmaGGUFMap;
export declare function buildGemmaE4BSpecs(): ReadonlyArray<unknown>;

export declare class GemmaLoader {
  constructor(gpu: GPU | null, status?: (msg: string, phase?: string, frac?: number | null) => void, config?: GemmaConfig);
  cacheValid(): Promise<object | null>;
  clearCache(): Promise<void>;
  load(): Promise<GemmaWeightsMap>;
}
