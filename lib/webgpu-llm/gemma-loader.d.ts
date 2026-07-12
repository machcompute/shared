import type { GPU } from './gpu.js';
import type { GemmaConfig } from './gemma-config.js';

export interface GemmaTensorPart {
  name: string;
  shape: number[];
}

export interface GemmaQ4Spec {
  name: string;
  parts: GemmaTensorPart[];
  N: number;
  K: number;
  shape: number[];
  layout?: Record<string, string | number>;
  modality: 'text' | 'vision' | 'audio';
  transform: 'bf16-q4';
}

export interface GemmaF32Spec {
  name: string;
  parts: GemmaTensorPart[];
  shape: number[];
  elements: number;
  modality: 'text' | 'vision' | 'audio';
  transform: 'bf16-f32';
}

export interface GemmaQ4Shard {
  start: number;
  rows: number;
  q: GPUBuffer;
  s: GPUBuffer;
}

export interface GemmaQ4Weight {
  q?: GPUBuffer;
  s?: GPUBuffer;
  shards?: GemmaQ4Shard[];
  N: number;
  K: number;
  shape: number[];
}

export type GemmaWeightsMap = Record<string, GPUBuffer | GemmaQ4Weight>;

export interface GemmaCacheEntry {
  name: string;
  kind: 'q4' | 'f32';
  off: number;
  qbytes?: number;
  sbytes?: number;
  bytes?: number;
}

export interface GemmaCacheManifest {
  version: number;
  schemaVersion: number;
  repo: string;
  revision: string;
  checkpoint: string;
  complete: boolean;
  entries: GemmaCacheEntry[];
}

export declare function buildGemmaE4BSpecs(): {
  q4: GemmaQ4Spec[];
  f32: GemmaF32Spec[];
};
export declare function buildGemmaSpecs(config?: GemmaConfig): {
  q4: GemmaQ4Spec[];
  f32: GemmaF32Spec[];
};

export declare class GemmaLoader {
  constructor(gpu: GPU | null, status?: (msg: string, phase?: string, frac?: number | null) => void, config?: GemmaConfig);
  cacheValid(): Promise<GemmaCacheManifest | null>;
  clearCache(): Promise<void>;
  load(): Promise<GemmaWeightsMap>;
}
