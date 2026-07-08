// The largest var<workgroup> footprint among our kernels (attnPart:
// qsh 4096 + scores 8192 + red4 4096 + arr/red 2048 + mnew/lchunk 32).
const REQUIRED_WG_STORAGE = 18464;

// Thin WebGPU runtime: device init, buffer helpers, pipeline cache.
export class GPU {
  async init(statusCb = () => {}) {
    if (!navigator.gpu) throw new Error('WebGPU is not available in this browser. Use Chrome/Edge 121+ (on Linux you may need chrome://flags → "Vulkan" + "Unsafe WebGPU").');
    this.adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!this.adapter) throw new Error('No WebGPU adapter found.');
    const a = this.adapter.limits;
    if (a.maxComputeWorkgroupStorageSize < REQUIRED_WG_STORAGE) {
      throw new Error(`This GPU offers ${a.maxComputeWorkgroupStorageSize} bytes of compute workgroup memory; the attention kernels need ${REQUIRED_WG_STORAGE}.`);
    }
    // Ask for as much as the adapter offers where we care.
    const want = {
      maxStorageBufferBindingSize: a.maxStorageBufferBindingSize,
      maxBufferSize: a.maxBufferSize,
      maxComputeWorkgroupStorageSize: Math.min(32768, a.maxComputeWorkgroupStorageSize),
      maxComputeInvocationsPerWorkgroup: 256, // spec-guaranteed minimum
      maxStorageBuffersPerShaderStage: Math.min(10, a.maxStorageBuffersPerShaderStage),
    };
    this.features = [];
    for (const f of ['timestamp-query']) if (this.adapter.features.has(f)) this.features.push(f);
    // subgroup reduction path: only safe when subgroups can't straddle a 64-lane row group
    const ai = this.adapter.info || {};
    this.subgroups = this.adapter.features.has('subgroups')
      && ai.subgroupMinSize >= 8 && ai.subgroupMaxSize <= 64;
    if (this.subgroups) this.features.push('subgroups');
    // int8 packed dot product (DP4a) — WGSL language feature, no device feature needed
    this.dp4a = !!navigator.gpu.wgslLanguageFeatures?.has('packed_4x8_integer_dot_product');
    this.device = await this.adapter.requestDevice({ requiredLimits: want, requiredFeatures: this.features });
    this.limits = this.device.limits;
    this.device.lost.then((info) => {
      // A deliberate destroy() (e.g. model reload) also resolves `lost`.
      if (this.destroyed || info.reason === 'destroyed') return;
      statusCb(`GPU device lost: ${info.message}`, 'error');
    });
    this.device.addEventListener?.('uncapturederror', (e) => console.error('[webgpu]', e.error?.message || e));
    this.device.onuncapturederror = (e) => console.error('[webgpu]', e.error?.message || e);
    this.pipelines = new Map();
    this.info = this.adapter.info || {};
    return this;
  }

  buf(size, usage, label) {
    return this.device.createBuffer({ size: Math.ceil(size / 4) * 4, usage, label });
  }
  storage(size, label) {
    return this.buf(size, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, label);
  }
  uniform(size, label) {
    return this.buf(size, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label);
  }
  upload(buffer, data, offset = 0) {
    this.device.queue.writeBuffer(buffer, offset, data.buffer ?? data, data.byteOffset ?? 0, data.byteLength ?? undefined);
  }

  // Pipeline cache keyed by (name + baked-constant signature).
  pipeline(key, wgslFn) {
    let p = this.pipelines.get(key);
    if (!p) {
      const code = wgslFn();
      const shaderModule = this.device.createShaderModule({ code, label: key });
      p = this.device.createComputePipeline({ layout: 'auto', compute: { module: shaderModule, entryPoint: 'main' }, label: key });
      this.pipelines.set(key, p);
    }
    return p;
  }

  // buffers: GPUBuffer or {buffer, offset, size} (offset must be 256-aligned)
  bind(pipeline, buffers, label) {
    return this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((b, i) => ({ binding: i, resource: b.buffer ? b : { buffer: b } })),
      label,
    });
  }

  // Frees every buffer/pipeline created on this device and silences its
  // lost handler. The instance is unusable afterwards.
  destroy() {
    this.destroyed = true;
    this.device?.destroy();
  }

  async readback(buffer, size) {
    const staging = this.buf(size, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, 'staging-tmp');
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buffer, 0, staging, 0, size);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = staging.getMappedRange().slice(0);
    staging.unmap(); staging.destroy();
    return out;
  }
}
