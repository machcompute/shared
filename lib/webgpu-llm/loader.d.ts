import type { GPU } from "./gpu.js";

export interface WeightEntry {
  buffer?: GPUBuffer;
  dimensions?: number[];
  N: number;
  K: number;
  type: number;
  typeName: string;
  blockSize: number;
  typeSize: number;
  rowStride: number;
  byteLength: number;
  start?: number;
  rows?: number;
  outOffset?: number;
  shards?: WeightEntry[];
  segments?: WeightEntry[];
}

export interface CacheManifest {
  schema: number;
  namespace: string;
  complete: boolean;
  files: Array<{
    filename: string;
    revision: string;
    byteLength: number;
    sha256: string;
    dataOffset: number;
  }>;
}

export type WeightsMap = Record<string, WeightEntry> & {
  embShards?: WeightEntry[];
  __nativeBytes?: number;
};

export declare function buildQwenGGUFMap(): ReadonlyArray<unknown>;

export declare class Loader {
  constructor(gpu: GPU | null, status: (msg: string, phase?: string, frac?: number | null) => void);
  cacheValid(): Promise<CacheManifest | null>;
  clearCache(): Promise<void>;
  load(): Promise<WeightsMap>;
}
