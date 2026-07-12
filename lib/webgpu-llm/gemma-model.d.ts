import type { GPU } from './gpu.js';
import type { GemmaWeightsMap } from './gemma-loader.js';
import type { GemmaConfig } from './gemma-config.js';
import type { GemmaAudioInput, GemmaVideoInput, GemmaVisionInput } from './gemma-media';

export interface GemmaCandidates {
  ids: Uint32Array;
  vals: Float32Array;
}

export interface GemmaDecodeParams {
  temperature?: number;
  topP?: number;
  topK?: number;
  presencePenalty?: number;
  eosId?: number;
  stopIds?: number[];
  allowedTokenIds?: Uint32Array;
}

export interface GemmaDecodeBatchResult {
  ids: number[];
  fed: number;
  eos: boolean;
  stopped: boolean;
  stopId: number | null;
}

/** Gemma's Q4 text decoder with independent FP8 E4M3FN KV caches. */
export declare class GemmaModel {
  constructor(gpu: GPU, weights: GemmaWeightsMap, opts?: { maxCtx?: number; chunk?: number; config?: GemmaConfig });
  pos: number;
  maxCtx: number;
  /** Width of text and projected media embeddings for the loaded checkpoint. */
  embeddingWidth: number;
  BATCH: number;
  hasMtp: false;
  spec: false;
  reset(): Promise<void>;
  resetPenaltyWindow(): void;
  notePenaltyToken(id: number): void;
  recentSet(): Set<number>;
  rewindDecode(count: number): void;
  prefill(
    tokenIds: number[] | Uint32Array,
    onProgress?: (done: number, total: number) => void,
    overrides?: Map<number, Float32Array> | null,
  ): Promise<GemmaCandidates>;
  decode(tokenId: number): Promise<GemmaCandidates>;
  decodeBatch(tokenId: number, k: number, params?: GemmaDecodeParams): Promise<GemmaDecodeBatchResult>;
  encodeImage(input: GemmaVisionInput): Promise<Float32Array>;
  encodeVideo(input: GemmaVideoInput): Promise<Float32Array>;
  encodeAudio(input: GemmaAudioInput): Promise<Float32Array>;
}
