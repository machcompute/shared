// Minimal, strict GGUF v3 reader and native GGML tensor helpers.
//
// GGUF stores dimensions in innermost-first order.  A 2-D matrix therefore
// has dimensions [columns, rows] and is laid out row-major in the data area.

export const GGUF_MAGIC = 0x46554747;
export const GGUF_VERSION = 3;

export const GGUF_VALUE_TYPE = Object.freeze({
  UINT8: 0, INT8: 1, UINT16: 2, INT16: 3, UINT32: 4, INT32: 5,
  FLOAT32: 6, BOOL: 7, STRING: 8, ARRAY: 9, UINT64: 10, INT64: 11,
  FLOAT64: 12,
});

export const GGML_TYPE = Object.freeze({
  F32: 0,
  Q4_0: 2,
  Q4_1: 3,
  Q8_0: 8,
  Q5_K: 13,
  Q6_K: 14,
  BF16: 30,
});

export const GGML_TYPE_NAME = Object.freeze(Object.fromEntries(
  Object.entries(GGML_TYPE).map(([name, value]) => [value, name]),
));

export const GGML_LAYOUT = Object.freeze({
  [GGML_TYPE.F32]: Object.freeze({ blockSize: 1, typeSize: 4 }),
  [GGML_TYPE.BF16]: Object.freeze({ blockSize: 1, typeSize: 2 }),
  [GGML_TYPE.Q4_0]: Object.freeze({ blockSize: 32, typeSize: 18 }),
  [GGML_TYPE.Q4_1]: Object.freeze({ blockSize: 32, typeSize: 20 }),
  [GGML_TYPE.Q8_0]: Object.freeze({ blockSize: 32, typeSize: 34 }),
  [GGML_TYPE.Q5_K]: Object.freeze({ blockSize: 256, typeSize: 176 }),
  [GGML_TYPE.Q6_K]: Object.freeze({ blockSize: 256, typeSize: 210 }),
});

export class GGUFNeedMoreDataError extends Error {
  constructor(requiredBytes) {
    super(`GGUF header is truncated; need at least ${requiredBytes} bytes`);
    this.name = 'GGUFNeedMoreDataError';
    this.requiredBytes = requiredBytes;
  }
}

function safeNumber(value, label) {
  if (value > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error(`${label} exceeds JavaScript's safe integer range`);
  return Number(value);
}

class Reader {
  constructor(input) {
    this.bytes = input instanceof Uint8Array
      ? input
      : new Uint8Array(input.buffer ?? input, input.byteOffset ?? 0, input.byteLength ?? input.length);
    this.view = new DataView(this.bytes.buffer, this.bytes.byteOffset, this.bytes.byteLength);
    this.offset = 0;
    this.decoder = new TextDecoder('utf-8', { fatal: true });
  }

  need(bytes) {
    const required = this.offset + bytes;
    if (required > this.bytes.byteLength) throw new GGUFNeedMoreDataError(required);
  }

  u8() { this.need(1); return this.view.getUint8(this.offset++); }
  i8() { this.need(1); return this.view.getInt8(this.offset++); }
  u16() { this.need(2); const v = this.view.getUint16(this.offset, true); this.offset += 2; return v; }
  i16() { this.need(2); const v = this.view.getInt16(this.offset, true); this.offset += 2; return v; }
  u32() { this.need(4); const v = this.view.getUint32(this.offset, true); this.offset += 4; return v; }
  i32() { this.need(4); const v = this.view.getInt32(this.offset, true); this.offset += 4; return v; }
  f32() { this.need(4); const v = this.view.getFloat32(this.offset, true); this.offset += 4; return v; }
  u64() { this.need(8); const v = this.view.getBigUint64(this.offset, true); this.offset += 8; return v; }
  i64() { this.need(8); const v = this.view.getBigInt64(this.offset, true); this.offset += 8; return v; }
  f64() { this.need(8); const v = this.view.getFloat64(this.offset, true); this.offset += 8; return v; }

  string() {
    const length = safeNumber(this.u64(), 'GGUF string length');
    this.need(length);
    const value = this.decoder.decode(this.bytes.subarray(this.offset, this.offset + length));
    this.offset += length;
    return value;
  }
}

function readValue(reader, type, depth = 0) {
  if (depth > 8) throw new Error('GGUF metadata arrays are nested too deeply');
  switch (type) {
    case GGUF_VALUE_TYPE.UINT8: return reader.u8();
    case GGUF_VALUE_TYPE.INT8: return reader.i8();
    case GGUF_VALUE_TYPE.UINT16: return reader.u16();
    case GGUF_VALUE_TYPE.INT16: return reader.i16();
    case GGUF_VALUE_TYPE.UINT32: return reader.u32();
    case GGUF_VALUE_TYPE.INT32: return reader.i32();
    case GGUF_VALUE_TYPE.FLOAT32: return reader.f32();
    case GGUF_VALUE_TYPE.BOOL: {
      const value = reader.u8();
      if (value > 1) throw new Error(`Invalid GGUF boolean value: ${value}`);
      return value === 1;
    }
    case GGUF_VALUE_TYPE.STRING: return reader.string();
    case GGUF_VALUE_TYPE.ARRAY: {
      const elementType = reader.u32();
      const length = safeNumber(reader.u64(), 'GGUF array length');
      const values = new Array(length);
      for (let i = 0; i < length; i++) values[i] = readValue(reader, elementType, depth + 1);
      return values;
    }
    case GGUF_VALUE_TYPE.UINT64: return reader.u64();
    case GGUF_VALUE_TYPE.INT64: return reader.i64();
    case GGUF_VALUE_TYPE.FLOAT64: return reader.f64();
    default: throw new Error(`Unsupported GGUF metadata value type: ${type}`);
  }
}

const alignUp = (value, alignment) => Math.ceil(value / alignment) * alignment;

export function ggmlTensorByteLength(dimensions, type) {
  const layout = GGML_LAYOUT[type];
  if (!layout) throw new Error(`Unsupported GGML tensor type ${type} (${GGML_TYPE_NAME[type] ?? 'unknown'})`);
  const elements = dimensions.reduce((n, dim) => n * dim, 1);
  if (elements % layout.blockSize !== 0) {
    throw new Error(`Tensor element count ${elements} is not divisible by ${GGML_TYPE_NAME[type]} block size ${layout.blockSize}`);
  }
  return elements / layout.blockSize * layout.typeSize;
}

export function parseGGUF(input, { fileSize = null } = {}) {
  const reader = new Reader(input);
  const magic = reader.u32();
  if (magic !== GGUF_MAGIC) throw new Error('Invalid GGUF magic');
  const version = reader.u32();
  if (version !== GGUF_VERSION) throw new Error(`Unsupported GGUF version ${version}; expected v${GGUF_VERSION}`);
  const tensorCount = safeNumber(reader.u64(), 'GGUF tensor count');
  const metadataCount = safeNumber(reader.u64(), 'GGUF metadata count');
  if (tensorCount > 1_000_000 || metadataCount > 1_000_000) throw new Error('Implausible GGUF table size');

  const metadata = Object.create(null);
  const metadataTypes = Object.create(null);
  for (let i = 0; i < metadataCount; i++) {
    const key = reader.string();
    if (Object.hasOwn(metadata, key)) throw new Error(`Duplicate GGUF metadata key: ${key}`);
    const type = reader.u32();
    metadataTypes[key] = type;
    metadata[key] = readValue(reader, type);
  }

  const tensors = [];
  const tensorsByName = new Map();
  for (let i = 0; i < tensorCount; i++) {
    const name = reader.string();
    if (tensorsByName.has(name)) throw new Error(`Duplicate GGUF tensor: ${name}`);
    const dimensionCount = reader.u32();
    if (dimensionCount < 1 || dimensionCount > 4) throw new Error(`Invalid dimension count ${dimensionCount} for ${name}`);
    const dimensions = new Array(dimensionCount);
    for (let d = 0; d < dimensionCount; d++) dimensions[d] = safeNumber(reader.u64(), `${name} dimension`);
    const type = reader.u32();
    if (!GGML_LAYOUT[type]) throw new Error(`Unsupported GGML tensor type ${type} for ${name}`);
    const relativeOffset = safeNumber(reader.u64(), `${name} data offset`);
    const byteLength = ggmlTensorByteLength(dimensions, type);
    const tensor = { name, dimensions, type, typeName: GGML_TYPE_NAME[type], relativeOffset, byteLength };
    tensors.push(tensor);
    tensorsByName.set(name, tensor);
  }

  const alignment = Number(metadata['general.alignment'] ?? 32);
  if (!Number.isInteger(alignment) || alignment <= 0 || (alignment & (alignment - 1)) !== 0) {
    throw new Error(`Invalid GGUF alignment: ${alignment}`);
  }
  const tableEnd = reader.offset;
  const dataOffset = alignUp(tableEnd, alignment);
  for (const tensor of tensors) {
    tensor.offset = dataOffset + tensor.relativeOffset;
    tensor.end = tensor.offset + tensor.byteLength;
    if (tensor.relativeOffset % alignment !== 0) throw new Error(`Misaligned tensor offset for ${tensor.name}`);
    if (fileSize !== null && tensor.end > fileSize) throw new Error(`Tensor ${tensor.name} extends past the GGUF file`);
  }

  return { version, tensorCount, metadataCount, metadata, metadataTypes, tensors, tensorsByName, alignment, tableEnd, dataOffset };
}

export function tensorMatrixShape(tensor) {
  if (tensor.dimensions.length !== 2) throw new Error(`${tensor.name} is not a matrix`);
  return { K: tensor.dimensions[0], N: tensor.dimensions[1] };
}

export function nativeWeightDescriptor(tensor, buffer, extra = {}) {
  const layout = GGML_LAYOUT[tensor.type];
  const K = tensor.dimensions[0];
  const N = tensor.dimensions.length === 1 ? 1 : tensor.dimensions.slice(1).reduce((n, dim) => n * dim, 1);
  const rowStride = K / layout.blockSize * layout.typeSize;
  return Object.freeze({
    buffer,
    dimensions: tensor.dimensions.slice(),
    N,
    K,
    type: tensor.type,
    typeName: tensor.typeName,
    blockSize: layout.blockSize,
    typeSize: layout.typeSize,
    rowStride,
    byteLength: tensor.byteLength,
    ...extra,
  });
}
