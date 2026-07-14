/** Decode one token at a time whenever a tool opener may appear. */
export function qwenDecodeBatchSize(hasTools: boolean, configuredBatchSize: number): number {
  return hasTools ? 1 : configuredBatchSize;
}

/**
 * A sampled length-limit token has been emitted to the client but has not
 * entered the model state yet. Only terminal stop tokens and completed tool
 * calls therefore leave a reusable GPU prefix.
 */
export function canCommitCompletionPrefix(options: {
  aborted: boolean;
  stopped: boolean;
  hasToolCalls: boolean;
}): boolean {
  return !options.aborted && (options.stopped || options.hasToolCalls);
}
