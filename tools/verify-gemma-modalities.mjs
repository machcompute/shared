#!/usr/bin/env node

// Real Dawn smoke coverage for the native GGUF multimodal projectors. This
// intentionally feeds already-preprocessed tensors so browser media decoding
// stays outside the kernel verification boundary.

import path from 'node:path';
import { pathToFileURL } from 'node:url';
import {
  DEFAULT_CACHE_DIR,
  MODEL_SPECS,
  ROOT,
  installNativeWebGPU,
  installNodePlatform,
  loadRuntime,
  makeProgressReporter,
} from './webgpu-model-throughput.mjs';

const models = process.argv.slice(2).length ? process.argv.slice(2) : ['gemma-e2b', 'gemma-e4b'];

function deterministicValues(length, scale = 1) {
  const values = new Float32Array(length);
  for (let i = 0; i < length; i++) values[i] = scale * Math.sin(i * 0.017 + 0.3);
  return values;
}

function verifyOutput(label, values, width) {
  if (!(values instanceof Float32Array) || !values.length || values.length % width) {
    throw new Error(`${label} returned an invalid embedding shape: ${values?.length ?? 'missing'} / ${width}`);
  }
  let sum = 0;
  let squareSum = 0;
  let maxAbs = 0;
  for (const value of values) {
    if (!Number.isFinite(value)) throw new Error(`${label} returned a non-finite embedding value`);
    sum += value;
    squareSum += value * value;
    maxAbs = Math.max(maxAbs, Math.abs(value));
  }
  if (!(squareSum > 0)) throw new Error(`${label} returned only zero embeddings`);
  return {
    rows: values.length / width,
    width,
    mean: sum / values.length,
    rms: Math.sqrt(squareSum / values.length),
    maxAbs,
  };
}

function imageFixture(hidden) {
  const patchWidth = 3;
  const patchHeight = 3;
  const patchCount = patchWidth * patchHeight;
  const positions = new Int32Array(patchCount * 2);
  for (let y = 0; y < patchHeight; y++) {
    for (let x = 0; x < patchWidth; x++) {
      const row = y * patchWidth + x;
      positions[row * 2] = y;
      positions[row * 2 + 1] = x;
    }
  }
  const pixels = deterministicValues(patchCount * hidden, 0.45);
  for (let i = 0; i < pixels.length; i++) pixels[i] += 0.5;
  return {
    pixelValues: pixels,
    positionIds: positions,
    patchCount,
    numSoftTokens: 1,
    width: 48,
    height: 48,
  };
}

function audioFixture() {
  const frameCount = 12;
  return {
    inputFeatures: deterministicValues(frameCount * 128, 0.7),
    inputFeaturesMask: new Uint8Array(frameCount).fill(1),
    frameCount,
  };
}

async function runModel(modelName) {
  const spec = MODEL_SPECS[modelName];
  if (!spec || spec.kind !== 'gemma') throw new Error(`Expected a Gemma model name, got ${modelName}`);
  const options = {
    modelName,
    model: spec,
    context: 1024,
    prefillChunk: 64,
    batchSize: 8,
    cacheDir: path.resolve(process.env.WEBGPU_CACHE_DIR ?? DEFAULT_CACHE_DIR),
    backend: process.env.WEBGPU_BACKEND ?? null,
    adapter: process.env.WEBGPU_ADAPTER ?? null,
  };
  const restoreWebGPU = installNativeWebGPU(options);
  let gpu = null;
  try {
    const { GPU } = await import(pathToFileURL(path.join(ROOT, 'lib/webgpu-llm/gpu.js')));
    gpu = await new GPU().init((message, kind) => process.stderr.write(`[${modelName}:gpu:${kind ?? 'status'}] ${message}\n`));
    const model = await loadRuntime(gpu, options, makeProgressReporter());
    const image = imageFixture(model.cfg.vision.hidden);

    const imageStart = performance.now();
    const imageEmbedding = await model.encodeImage(image);
    const imageMs = performance.now() - imageStart;

    const videoStart = performance.now();
    const videoEmbedding = await model.encodeVideo({
      ...image,
      pixelValues: new Float32Array(image.pixelValues),
      positionIds: new Int32Array(image.positionIds),
      frameCount: 1,
      maxPatches: image.patchCount,
      patchCountPerFrame: image.patchCount,
      numSoftTokensPerFrame: image.numSoftTokens,
    });
    const videoMs = performance.now() - videoStart;

    const audioStart = performance.now();
    const audioEmbedding = await model.encodeAudio(audioFixture());
    const audioMs = performance.now() - audioStart;

    const width = model.cfg.text.hidden;
    return {
      model: spec.id,
      image: { ...verifyOutput(`${modelName} image`, imageEmbedding, width), milliseconds: imageMs },
      video: { ...verifyOutput(`${modelName} video`, videoEmbedding, width), milliseconds: videoMs },
      audio: { ...verifyOutput(`${modelName} audio`, audioEmbedding, width), milliseconds: audioMs },
    };
  } finally {
    gpu?.destroy();
    restoreWebGPU();
  }
}

const restorePlatform = installNodePlatform(path.resolve(process.env.WEBGPU_CACHE_DIR ?? DEFAULT_CACHE_DIR));
try {
  const results = [];
  for (const model of models) results.push(await runModel(model));
  console.log(JSON.stringify({ ok: true, results }, null, 2));
} finally {
  restorePlatform();
}
