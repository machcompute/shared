export declare const CFG: {
  repo: string;
  hidden: number;
  layers: number;
  interm: number;
  vocab: number;
  eps: number;
  heads: number;
  headDim: number;
  kvHeads: number;
  ropeDim: number;
  ropeTheta: number;
  qgDim: number;
  kvDim: number;
  vHeads: number;
  kHeads: number;
  kDim: number;
  vDim: number;
  convK: number;
  qkvDim: number;
  zDim: number;
  convDim: number;
  inL: number;
  inF: number;
  eosText: string;
};

export declare const PSIZE: number;

export declare function isFullAttn(i: number): boolean;

export declare const RT: {
  maxCtx: number;
  chunk: number;
  topkBlock: number;
  topkK: number;
  ppWindow: number;
};

export declare const TOPK_WGS: number;
