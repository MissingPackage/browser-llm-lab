
struct Sel { id: u32, slot: u32, w: f32, flags: u32 }
struct MoeIdx { selIdx: u32, tableBase: u32, moeLayer: u32, pad: u32 }
@group(0) @binding(0) var<storage, read> arena0: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> arena1: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> arena2: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<storage, read> selBuf: array<Sel>;
@group(0) @binding(6) var<uniform> moeIdx: MoeIdx;
fn ld4(b: u32, i: u32) -> vec4<u32> {
  switch b {
    case 0u: { return arena0[i]; }
    case 1u: { return arena1[i]; }
    case 2u: { return arena2[i]; }
    default: { return arena0[i]; }
  }
}
fn ldw(b: u32, w: u32) -> u32 { return ld4(b, w >> 2u)[w & 3u]; }
const SLAB_W = 4096u;
const SLABS_PER_BUF = 8u;
var<private> gBase: u32;
var<private> gBi: u32;
fn blkw(i: u32) -> u32 { return ldw(gBi, gBase + i); }
const SB_PER_ROW = 8u;
const LPU = 16u;
const CHUNKS = 2u;
const UNITS_PER_SB = 8u;
const UNITS = 64u;
var<workgroup> partial: array<f32, 64>;
fn sbyte(base: u32, i: u32) -> u32 {
  return (blkw(base + (i >> 2u)) >> ((i & 3u) * 8u)) & 0xffu;
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let r = wid.x + wid.y * 32768u;
  if (r >= 512u) { return; }
  let sel = selBuf[moeIdx.selIdx];
  let ok = sel.slot != 0xffffffffu;
  let slot = select(0u, sel.slot, ok);
  let bi = slot / SLABS_PER_BUF;
  let base = (slot % SLABS_PER_BUF) * SLAB_W;
  gBi = bi;
  gBase = base + 1152u;
  if (!ok) { return; }
  let t = lid.x;
  var acc = 0.0;
  for (var u = t; u < UNITS; u = u + 64u) {
    let sb = u / UNITS_PER_SB;
    let rem = u % UNITS_PER_SB;
    let j = rem / CHUNKS;                   // gruppo da 64 elementi
    let lo = (rem % CHUNKS) * LPU;          // primo l dell'unita'
    let wb = (r * SB_PER_ROW + sb) * 36u;   // base word del superblocco
    let dm = unpack2x16float(blkw(wb));   // (d, dmin)
    let xBase = sb * 256u;
    let is = 2u * j;
    var sc1: u32; var mn1: u32; var sc2: u32; var mn2: u32;
    if (is < 4u) {
      sc1 = sbyte(wb, 4u + is) & 63u;
      mn1 = sbyte(wb, 4u + is + 4u) & 63u;
      sc2 = sbyte(wb, 4u + is + 1u) & 63u;
      mn2 = sbyte(wb, 4u + is + 5u) & 63u;
    } else {
      sc1 = (sbyte(wb, 4u + is + 4u) & 0xfu) | ((sbyte(wb, 4u + is - 4u) >> 6u) << 4u);
      mn1 = (sbyte(wb, 4u + is + 4u) >> 4u) | ((sbyte(wb, 4u + is) >> 6u) << 4u);
      sc2 = (sbyte(wb, 4u + is + 5u) & 0xfu) | ((sbyte(wb, 4u + is - 3u) >> 6u) << 4u);
      mn2 = (sbyte(wb, 4u + is + 5u) >> 4u) | ((sbyte(wb, 4u + is + 1u) >> 6u) << 4u);
    }
    let d1 = dm.x * f32(sc1); let min1 = dm.y * f32(mn1);
    let d2 = dm.x * f32(sc2); let min2 = dm.y * f32(mn2);
    var dot1 = 0.0; var sx1 = 0.0; var dot2 = 0.0; var sx2 = 0.0;
    for (var l = lo; l < lo + LPU; l = l + 1u) {
      let ql = sbyte(wb, 16u + j * 32u + l);   // qs a byte offset 16 (niente qh)
      let x1 = x[xBase + j * 64u + l];
      let x2 = x[xBase + j * 64u + 32u + l];
      dot1 = dot1 + f32(ql & 0xfu) * x1; sx1 = sx1 + x1;
      dot2 = dot2 + f32(ql >> 4u) * x2; sx2 = sx2 + x2;
    }
    acc = acc + d1 * dot1 - min1 * sx1 + d2 * dot2 - min2 * sx2;
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
  if (t == 0u) { y[r] = partial[0]; }
}