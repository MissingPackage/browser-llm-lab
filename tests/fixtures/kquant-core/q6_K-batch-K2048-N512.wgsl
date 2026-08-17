
@group(0) @binding(0) var<storage, read> blocks: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
fn blkw(i: u32) -> u32 { return blocks[i]; }
const SB_PER_ROW = 8u;
const LPU = 8u;
const CHUNKS = 4u;
const UNITS_PER_SB = 8u;
const UNITS = 64u;
var<workgroup> partial: array<f32, 64>;
fn sbyte(base: u32, i: u32) -> u32 {
  return (blkw(base + (i >> 2u)) >> ((i & 3u) * 8u)) & 0xffu;
}
fn sint8(base: u32, i: u32) -> f32 {
  return f32((i32(sbyte(base, i)) << 24u) >> 24u);
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let r = wid.x + wid.y * 32768u;
  let xR = wid.z * 2048u;
  if (r >= 512u) { return; }
  let t = lid.x;
  var acc = 0.0;
  for (var u = t; u < UNITS; u = u + 64u) {
    let sb = u / UNITS_PER_SB;
    let rem = u % UNITS_PER_SB;
    let n = rem / CHUNKS;                       // gruppo da 128
    let lo = (rem % CHUNKS) * LPU;              // primo l dell'unita'
    let wb = (r * SB_PER_ROW + sb) * 53u;
    let d = unpack2x16float(blkw(wb + 52u)).x; // d f16 a byte offset 208
    let xBase = sb * 256u;
    {
      let qlO = n * 64u;        // byte offset dentro ql (0 o 64)
      let qhO = 128u + n * 32u; // byte offset qh
      let scO = 192u + n * 8u;  // byte offset scales
      for (var l = lo; l < lo + LPU; l = l + 1u) {
        let is = l >> 4u;
        let qlA = sbyte(wb, qlO + l);
        let qlB = sbyte(wb, qlO + l + 32u);
        let qh = sbyte(wb, qhO + l);
        let q1 = f32((qlA & 0xfu) | (((qh >> 0u) & 3u) << 4u)) - 32.0;
        let q2 = f32((qlB & 0xfu) | (((qh >> 2u) & 3u) << 4u)) - 32.0;
        let q3 = f32((qlA >> 4u) | (((qh >> 4u) & 3u) << 4u)) - 32.0;
        let q4 = f32((qlB >> 4u) | (((qh >> 6u) & 3u) << 4u)) - 32.0;
        let e = xBase + n * 128u + l;
        acc = acc + d * (sint8(wb, scO + is) * q1 * x[xR + e]
                       + sint8(wb, scO + is + 2u) * q2 * x[xR + e + 32u]
                       + sint8(wb, scO + is + 4u) * q3 * x[xR + e + 64u]
                       + sint8(wb, scO + is + 6u) * q4 * x[xR + e + 96u]);
      }
    }
  }
  partial[t] = acc;
  workgroupBarrier();
  var stride = 32u;
  while (stride > 0u) {
    if (t < stride) { partial[t] = partial[t] + partial[t + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  // coda accumulante (accum): vedi la nota sopra la funzione
  if (t == 0u) { y[wid.z * 512u + r] = partial[0]; }
}