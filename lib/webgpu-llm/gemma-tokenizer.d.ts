import type { GemmaConfig } from './gemma-config.js';
export declare class GemmaTokenizer {
  static load(status?: (message: string) => void, config?: GemmaConfig): Promise<GemmaTokenizer>;
  eos: number;
  bos: number;
  pad: number;
  image: number;
  audio: number;
  video: number;
  turnEnd: number;
  toolCallOpen: number | undefined;
  toolCallClose: number | undefined;
  chatTemplate: string | null;
  special: Map<string, number>;
  sanitize(text: string): string;
  encode(text: string): number[];
  makeDecoder(): (id: number) => string;
  vocabSize(): number;
  specialTokenId(text: string): number | undefined;
  tokenText(id: number): string | undefined;
}
