#!/usr/bin/env node

import { spawnSync } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const fixtures = JSON.parse(await readFile(new URL('../tests/fixtures/gguf/llama-cpp-goldens.json', import.meta.url), 'utf8'));
const requested = process.argv[2] ?? null;
const cases = requested ? fixtures.cases.filter((entry) => entry.model === requested) : fixtures.cases;
if (!cases.length) throw new Error(`Unknown GGUF golden model: ${requested}`);

const results = [];
for (const fixture of cases) {
  const args = [
    'tools/webgpu-model-throughput.mjs',
    '--model', fixture.model,
    '--prompt-ids', fixture.promptTokenIds.join(','),
    '--decode-tokens', String(fixture.generatedTokenIds.length - 1),
    '--warmup-tokens', '0',
    '--runs', '1',
    '--context', '1024',
    '--batch-size', '1',
    '--prefill-chunk', '64',
  ];
  const run = spawnSync(process.execPath, args, { cwd: root, encoding: 'utf8', maxBuffer: 8 * 1024 * 1024 });
  if (run.status !== 0) {
    process.stderr.write(run.stderr);
    throw new Error(`${fixture.model} Dawn golden run failed with status ${run.status}`);
  }
  const output = JSON.parse(run.stdout);
  const actual = output.runs[0].generatedTokenIds;
  if (actual.length !== fixture.generatedTokenIds.length || actual.some((id, i) => id !== fixture.generatedTokenIds[i])) {
    throw new Error(`${fixture.model} differs from llama.cpp ${fixtures.llamaCppCommit}: expected ${fixture.generatedTokenIds}, got ${actual}`);
  }
  results.push({ model: fixture.model, generatedTokenIds: actual, matches: true });
}

process.stdout.write(`${JSON.stringify({ llamaCppCommit: fixtures.llamaCppCommit, results }, null, 2)}\n`);
