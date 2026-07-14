const u32 = (...values) => new Uint32Array(values);

/**
 * Apply an immutable token mask to a Gemma model-like state object. Keeping
 * this bookkeeping separate makes the upload de-duplication testable without
 * allocating a real WebGPU model.
 */
export function updateTokenMaskState(state, mask) {
  const kind = mask?.kind ?? 'all';
  if (kind !== 'all' && kind !== 'allow' && kind !== 'deny') {
    throw new Error(`Unknown token mask kind: ${kind}`);
  }
  const ids = kind === 'all' ? null : mask?.tokenIds;
  if (kind !== 'all' && !(ids instanceof Uint32Array)) {
    throw new Error(`A ${kind} token mask requires Uint32Array tokenIds.`);
  }
  const count = ids?.length ?? 0;
  if (count > state.cfg.text.vocab) {
    throw new Error('Tool constraint has too many token IDs.');
  }
  if (kind === state.tokenMaskKind && ids === state.tokenMaskIds) return;
  if (count && ids !== state.tokenMaskIds) {
    state.gpu.device.queue.writeBuffer(state.b.allowed, 0, ids);
  }
  if (count !== state.allowedCount) {
    state.gpu.device.queue.writeBuffer(state.b.uAllowed, 0, u32(count, 0, 0, 0));
  }
  state.allowedCount = count;
  state.tokenMaskKind = kind;
  state.tokenMaskIds = ids;
}
