export declare class GPU {
  adapter: GPUAdapter;
  device: GPUDevice;
  limits: GPUSupportedLimits;
  features: string[];
  subgroups: boolean;
  dp4a: boolean;
  info: { vendor?: string; architecture?: string; [key: string]: unknown };
  pipelines: Map<string, GPUComputePipeline>;

  init(statusCb?: (msg: string, kind?: string) => void): Promise<this>;
  buf(size: number, usage: number, label?: string): GPUBuffer;
  storage(size: number, label?: string): GPUBuffer;
  uniform(size: number, label?: string): GPUBuffer;
  upload(buffer: GPUBuffer, data: ArrayBufferView | ArrayBuffer, offset?: number): void;
  pipeline(key: string, wgslFn: () => string): GPUComputePipeline;
  bind(pipeline: GPUComputePipeline, buffers: unknown[], label?: string): GPUBindGroup;
  readback(buffer: GPUBuffer, size: number): Promise<ArrayBuffer>;
  destroy(): void;
}
