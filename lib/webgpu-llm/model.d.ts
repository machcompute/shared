import type { GPU } from "./gpu.js";
import type { WeightsMap } from "./loader.js";

export interface SampleParams {
  temperature?: number;
  topP?: number;
  topK?: number;
  presencePenalty?: number;
  // Either eosId (single, back-compat) or stopIds (up to 4) — the kernel
  // freezes state as soon as any one of them is sampled.
  eosId?: number;
  stopIds?: number[];
  seed?: number;
}

export interface Candidates {
  ids: number[];
  vals: number[];
}

export interface DecodeBatchResult {
  ids: number[];
  fed: number;
  eos: boolean;
  stopped: boolean;
  stopId: number | null;
}

export interface SpecRound {
  a: number;
  d0?: number;
  d1?: number;
  next: number | null;
}

export interface SpecChainResult {
  rounds: SpecRound[];
  eos: boolean;
  stopped: boolean;
  stopId: number | null;
}

export declare class Model {
  constructor(gpu: GPU, weights: WeightsMap, opts: { maxCtx: number });
  pos: number;
  maxCtx: number;
  hasMtp: boolean;
  spec: boolean;
  BATCH: number;
  reset(): Promise<void>;
  resetPenaltyWindow(): void;
  notePenaltyToken(id: number): void;
  recentSet(): Set<number>;
  prefill(tokenIds: number[], onProgress?: (done: number, total: number) => void): Promise<Candidates>;
  decodeBatch(firstToken: number, k: number, params: SampleParams): Promise<DecodeBatchResult>;
  decode(token: number): Promise<unknown>;
  specChain(firstToken: number, rounds: number, params: SampleParams): Promise<SpecChainResult>;
}
