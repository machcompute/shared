import type { GPU } from "./gpu.js";

export interface WeightEntry {
  q?: GPUBuffer;
  s?: GPUBuffer;
  size?: number;
  N?: number;
  K?: number;
}

export interface CacheManifest {
  version: number;
  complete: boolean;
  entries: unknown[];
}

export type WeightsMap = Record<string, WeightEntry> & {
  embShards?: Array<{ start: number; rows: number; q: GPUBuffer; s: GPUBuffer }>;
};

export declare class Loader {
  constructor(gpu: GPU | null, status: (msg: string, phase?: string, frac?: number | null) => void);
  cacheValid(): Promise<CacheManifest | null>;
  clearCache(): Promise<void>;
  load(): Promise<WeightsMap>;
}
