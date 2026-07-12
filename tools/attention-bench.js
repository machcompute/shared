import * as K from '../lib/webgpu-llm/kernels.js';

const MAXCTX = 32768;
const PMAX = Math.ceil(MAXCTX / 512);
const CASES = [512, 1024, 2048, 4096, 8192, 16384, 32768];
const WARMUP = 4;
const ITERS = 100;
const out = document.querySelector('#out');

function rng(seed = 1) {
  return () => {
    seed = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    seed ^= seed + Math.imul(seed ^ (seed >>> 7), 61 | seed);
    return ((seed ^ (seed >>> 14)) >>> 0) / 4294967296;
  };
}

function data(n, seed, scale = 0.2) {
  const a = new Float32Array(n), random = rng(seed);
  for (let i = 0; i < n; i++) a[i] = (random() * 2 - 1) * scale;
  return a;
}

function buffer(device, bytes, usage, initial, label) {
  const b = device.createBuffer({ size: Math.ceil(bytes / 4) * 4, usage, label });
  if (initial) device.queue.writeBuffer(b, 0, initial);
  return b;
}

function pipeline(device, code, label) {
  return device.createComputePipeline({
    label,
    layout: 'auto',
    compute: { module: device.createShaderModule({ code, label }), entryPoint: 'main' },
  });
}

function bind(device, p, buffers) {
  return device.createBindGroup({
    layout: p.getBindGroupLayout(0),
    entries: buffers.map((b, binding) => ({ binding, resource: { buffer: b } })),
  });
}

async function read(device, src, bytes) {
  const dst = buffer(device, bytes, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(src, 0, dst, 0, bytes);
  device.queue.submit([enc.finish()]);
  await dst.mapAsync(GPUMapMode.READ);
  const value = new Float32Array(dst.getMappedRange().slice(0));
  dst.destroy();
  return value;
}

function error(a, b) {
  let maxAbs = 0, maxRel = 0, rms = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    maxAbs = Math.max(maxAbs, d);
    maxRel = Math.max(maxRel, d / Math.max(1e-4, Math.abs(b[i])));
    rms += d * d;
  }
  return { maxAbs, maxRel, rms: Math.sqrt(rms / a.length) };
}

async function main() {
  if (!navigator.gpu) throw new Error('WebGPU unavailable');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter');
  const features = ['timestamp-query', 'subgroups'].filter(feature => adapter.features.has(feature));
  const device = await adapter.requestDevice({
    requiredFeatures: features,
    requiredLimits: { maxComputeWorkgroupStorageSize: Math.min(32768, adapter.limits.maxComputeWorkgroupStorageSize) },
  });
  device.pushErrorScope('validation');
  const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const U = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
  const qg = buffer(device, 10240 * 4, S, data(10240, 1), 'qg');
  const qw = buffer(device, 256 * 4, S, data(256, 2, 0.1).map(x => x + 1), 'qw');
  const kw = buffer(device, 256 * 4, S, data(256, 3, 0.1).map(x => x + 1), 'kw');
  const ropeData = new Float32Array(MAXCTX * 64);
  for (let p = 0; p < MAXCTX; p++) for (let i = 0; i < 32; i++) {
    const x = p * Math.pow(1e7, -i / 32);
    ropeData[p * 64 + i] = Math.cos(x);
    ropeData[p * 64 + 32 + i] = Math.sin(x);
  }
  const rope = buffer(device, ropeData.byteLength, S, ropeData, 'rope');
  const kvBytes = MAXCTX * 4 * 65 * 4;
  const kc = buffer(device, kvBytes, S, null, 'kc');
  const vc = buffer(device, kvBytes, S, null, 'vc');
  const kcRef = buffer(device, kvBytes, S, null, 'kc-ref');
  const vcRef = buffer(device, kvBytes, S, null, 'vc-ref');
  const parts = buffer(device, 4 * PMAX * 4 * 258 * 4, S, null, 'parts');
  const qr = buffer(device, 4096 * 4, S, null, 'qr');
  const actual = buffer(device, 4096 * 4, S, null, 'actual');
  const optimized = buffer(device, 4096 * 4, S, null, 'optimized');
  const expected = buffer(device, 4096 * 4, S, null, 'expected');
  const uniform = buffer(device, 16, U, null, 'uniform');

  const partP = pipeline(device, K.attnPart({ PMAX, MAXCTX }), 'attnPart-baseline');
  const decodePrepP = pipeline(device, K.attnDecodePrep({ MAXCTX }), 'attnDecodePrep');
  const preparedPartP = pipeline(device, K.attnPartPrepared({ PMAX, MAXCTX, SUBGROUPS: features.includes('subgroups') ? 1 : 0 }), 'attnPartPrepared');
  const reduceP = pipeline(device, K.attnReduce({ PMAX }), 'attnReduce');
  const optimizedReduceP = pipeline(device, K.attnReduce({ PMAX }), 'attnReduce-optimized');
  const prepP = pipeline(device, K.attnPrep({ MAXCTX }), 'attnPrep');
  const streamP = pipeline(device, K.attention({ MAXCTX }), 'attention');
  // Compile the storage-position variants used by chained MTP speculation too.
  pipeline(device, K.attnDecodePrep({ MAXCTX, SPOS: 1, OFFS: -1 }), 'attnDecodePrep-spos-a');
  pipeline(device, K.attnDecodePrep({ MAXCTX, SPOS: 1, OFFS: 0 }), 'attnDecodePrep-spos-b');
  pipeline(device, K.attnPartPrepared({ PMAX, MAXCTX, SPOS: 1, OFFS: -1, SUBGROUPS: features.includes('subgroups') ? 1 : 0 }), 'attnPartPrepared-spos-a');
  pipeline(device, K.attnPartPrepared({ PMAX, MAXCTX, SPOS: 1, OFFS: 0, SUBGROUPS: features.includes('subgroups') ? 1 : 0 }), 'attnPartPrepared-spos-b');
  const partB = bind(device, partP, [qg, qw, kw, rope, kc, vc, parts, uniform]);
  const reduceB = bind(device, reduceP, [parts, qg, actual, uniform]);
  const decodePrepB = bind(device, decodePrepP, [qg, qw, kw, rope, qr, kc, vc, uniform]);
  const preparedPartB = bind(device, preparedPartP, [qr, kc, vc, parts, uniform]);
  const optimizedReduceB = bind(device, optimizedReduceP, [parts, qg, optimized, uniform]);
  const prepB = bind(device, prepP, [qg, qw, kw, rope, qr, kcRef, vcRef, uniform]);
  const streamB = bind(device, streamP, [qr, kcRef, vcRef, qg, expected, uniform]);
  const pipelineError = await device.popErrorScope();
  if (pipelineError) throw new Error(`WebGPU validation: ${pipelineError.message}`);

  function dispatch(pass, pos, mode) {
    if (mode === 'partitioned') {
      pass.setPipeline(partP); pass.setBindGroup(0, partB); pass.dispatchWorkgroups(Math.floor(pos / 512) + 1, 4);
      pass.setPipeline(reduceP); pass.setBindGroup(0, reduceB); pass.dispatchWorkgroups(4, 4);
    } else if (mode === 'optimized') {
      pass.setPipeline(decodePrepP); pass.setBindGroup(0, decodePrepB); pass.dispatchWorkgroups(4);
      pass.setPipeline(preparedPartP); pass.setBindGroup(0, preparedPartB); pass.dispatchWorkgroups(Math.floor(pos / 512) + 1, 4);
      pass.setPipeline(optimizedReduceP); pass.setBindGroup(0, optimizedReduceB); pass.dispatchWorkgroups(4, 4);
    } else {
      pass.setPipeline(prepP); pass.setBindGroup(0, prepB); pass.dispatchWorkgroups(1, 24);
      pass.setPipeline(streamP); pass.setBindGroup(0, streamB); pass.dispatchWorkgroups(1, 4);
    }
  }

  function encode(pos, mode) {
    device.queue.writeBuffer(uniform, 0, new Uint32Array([pos, 1, 0, 0]));
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    dispatch(pass, pos, mode);
    pass.end();
    return enc.finish();
  }

  async function time(pos, mode) {
    for (let i = 0; i < WARMUP; i++) device.queue.submit([encode(pos, mode)]);
    await device.queue.onSubmittedWorkDone();
    if (features.includes('timestamp-query')) {
      const querySet = device.createQuerySet({ type: 'timestamp', count: 2 });
      const queryBuffer = buffer(device, 16, GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC);
      const readBuffer = buffer(device, 16, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
      device.queue.writeBuffer(uniform, 0, new Uint32Array([pos, 1, 0, 0]));
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass({
        timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 },
      });
      for (let i = 0; i < ITERS; i++) dispatch(pass, pos, mode);
      pass.end();
      enc.resolveQuerySet(querySet, 0, 2, queryBuffer, 0);
      enc.copyBufferToBuffer(queryBuffer, 0, readBuffer, 0, 16);
      device.queue.submit([enc.finish()]);
      await readBuffer.mapAsync(GPUMapMode.READ);
      const stamps = new BigUint64Array(readBuffer.getMappedRange());
      const ms = Number(stamps[1] - stamps[0]) / 1e6 / ITERS;
      readBuffer.unmap(); readBuffer.destroy(); queryBuffer.destroy(); querySet.destroy();
      return ms;
    }
    const start = performance.now();
    for (let i = 0; i < ITERS; i++) device.queue.submit([encode(pos, mode)]);
    await device.queue.onSubmittedWorkDone();
    return (performance.now() - start) / ITERS;
  }

  // Seed both caches identically using prep calls at each position. Correctness
  // uses 512 positions; timing uses deterministic random cache bytes afterwards.
  for (let pos = 0; pos < 512; pos++) {
    device.queue.writeBuffer(uniform, 0, new Uint32Array([pos, 1, 0, 0]));
    const enc = device.createCommandEncoder(), pass = enc.beginComputePass();
    pass.setPipeline(prepP); pass.setBindGroup(0, prepB); pass.dispatchWorkgroups(1, 24); pass.end();
    device.queue.submit([enc.finish()]);
  }
  await device.queue.onSubmittedWorkDone();
  // Copy the reference cache into the partitioned cache before its append.
  let enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(kcRef, 0, kc, 0, kvBytes);
  enc.copyBufferToBuffer(vcRef, 0, vc, 0, kvBytes);
  device.queue.submit([enc.finish()]);
  device.queue.submit([encode(511, 'partitioned'), encode(511, 'optimized'), encode(511, 'streaming')]);
  await device.queue.onSubmittedWorkDone();
  const expectedValues = await read(device, expected, 4096 * 4);
  const checks = {
    partitioned: error(await read(device, actual, 4096 * 4), expectedValues),
    optimized: error(await read(device, optimized, 4096 * 4), expectedValues),
  };
  device.queue.submit([encode(MAXCTX - 1, 'partitioned'), encode(MAXCTX - 1, 'optimized')]);
  await device.queue.onSubmittedWorkDone();
  const longCheck = error(
    await read(device, optimized, 4096 * 4),
    await read(device, actual, 4096 * 4),
  );

  const rows = [];
  // Bring the discrete GPU out of its idle power state before sub-millisecond
  // measurements; otherwise the first mode at each run is systematically noisy.
  await time(MAXCTX - 1, 'streaming');
  for (const n of CASES) {
    const baselineMs = await time(n - 1, 'partitioned');
    const splitMs = await time(n - 1, 'optimized');
    rows.push({
      context: n,
      baselineMs,
      splitSubgroupMs: splitMs,
      productionMs: n < 16384 ? baselineMs : splitMs,
      streamingMs: await time(n - 1, 'streaming'),
    });
  }
  return {
    adapter: {
      vendor: adapter.info?.vendor,
      architecture: adapter.info?.architecture,
      device: adapter.info?.device,
      description: adapter.info?.description,
      subgroupMinSize: adapter.info?.subgroupMinSize,
      subgroupMaxSize: adapter.info?.subgroupMaxSize,
    },
    timestampQuery: features.includes('timestamp-query'),
    subgroups: features.includes('subgroups'),
    iterations: ITERS,
    correctnessAt512: checks,
    splitVsBaselineAtMaxContext: longCheck,
    rows,
  };
}

main().then(result => {
  window.__attentionBench = result;
  out.textContent = JSON.stringify(result, null, 2);
}).catch(error => {
  window.__attentionBench = { error: error.stack || String(error) };
  out.textContent = window.__attentionBench.error;
});
