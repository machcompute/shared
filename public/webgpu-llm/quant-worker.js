// Worker: bf16 rows -> Q4 blocks of 32 (scale+min as packed f16 pair).
// In:  { id, u16: ArrayBuffer of bf16 values, rows, K }
// Out: { id, qdata: ArrayBuffer(u32 rows*K/8), scales: ArrayBuffer(u32 rows*K/32) }

const f32buf = new Float32Array(1);
const u32buf = new Uint32Array(f32buf.buffer);
function f16bits(v) {
  f32buf[0] = v;
  const x = u32buf[0];
  const sign = (x >>> 16) & 0x8000;
  let e = (x >>> 23) & 0xff;
  let m = x & 0x7fffff;
  if (e === 0xff) return sign | 0x7c00 | (m ? 1 : 0);
  let he = e - 127 + 15;
  if (he >= 31) return sign | 0x7c00;              // overflow -> inf
  if (he <= 0) {                                    // subnormal
    if (he < -10) return sign;
    m |= 0x800000;
    const shift = 14 - he;
    let hm = m >>> shift;
    if ((m >>> (shift - 1)) & 1) hm += 1;           // round
    return sign | hm;
  }
  let hm = m >>> 13;
  if (m & 0x1000) {                                 // round to nearest
    hm += 1;
    if (hm === 0x400) { hm = 0; he += 1; if (he >= 31) return sign | 0x7c00; }
  }
  return sign | (he << 10) | hm;
}

function f16toF32(h) {
  const sign = (h & 0x8000) ? -1 : 1;
  const e = (h >>> 10) & 0x1f;
  const m = h & 0x3ff;
  if (e === 0) return sign * m * Math.pow(2, -24);
  if (e === 31) return m ? NaN : sign * Infinity;
  return sign * (1024 + m) * Math.pow(2, e - 25);
}

onmessage = (ev) => {
  const { id, u16: u16buf, rows, K } = ev.data;
  const u16 = new Uint16Array(u16buf);
  const n = rows * K;
  // bf16 -> f32
  const u32 = new Uint32Array(n);
  for (let i = 0; i < n; i++) u32[i] = u16[i] << 16;
  const f = new Float32Array(u32.buffer);

  const nblocks = n >> 5;
  const qdata = new Uint32Array(n >> 3);
  const scales = new Uint32Array(nblocks);
  for (let b = 0; b < nblocks; b++) {
    const base = b << 5;
    let mn = f[base], mx = f[base];
    for (let j = 1; j < 32; j++) {
      const v = f[base + j];
      if (v < mn) mn = v; else if (v > mx) mx = v;
    }
    const s16 = f16bits((mx - mn) / 15);
    const m16 = f16bits(mn);
    scales[b] = (m16 << 16) | s16;
    // quantize against the f16-rounded scale/min the GPU will dequantize with
    const scale = f16toF32(s16);
    mn = f16toF32(m16);
    const inv = scale > 0 ? 1 / scale : 0;
    for (let w = 0; w < 4; w++) {
      let word = 0;
      const wb = base + (w << 3);
      for (let j = 0; j < 8; j++) {
        let q = Math.round((f[wb + j] - mn) * inv);
        if (q < 0) q = 0; else if (q > 15) q = 15;
        word |= q << (j << 2);
      }
      qdata[(b << 2) + w] = word >>> 0;
    }
  }
  postMessage({ id, qdata: qdata.buffer, scales: scales.buffer }, [qdata.buffer, scales.buffer]);
};
