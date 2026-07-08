export declare class Tokenizer {
  static load(status?: (msg: string) => void): Promise<Tokenizer>;
  eos: number;
  chatTemplate: string;
  special: Map<string, number>;
  encode(text: string): number[];
  sanitize(text: string): string;
  makeDecoder(): (id: number) => string;
}

export declare function sample(
  cands: { ids: number[]; vals: number[] },
  opts?: {
    temperature?: number;
    topK?: number;
    topP?: number;
    presencePenalty?: number;
    recentIds?: Set<number> | null;
  }
): number;
