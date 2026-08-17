
@group(0) @binding(0) var<storage, read> blocks: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
const SB_PER_ROW = 8u;
var<workgroup> partial: array<f32, 64>;
fn sbyte(base: u32, i: u32) -> u32 {
  return (blocks[base + (i >> 2u)] >> ((i & 3u) * 8u)) & 0xffu;
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let r = wid.x + wid.y * 32768u;
  let xR = wid.z * 2048u;
  if (r >= 4096u) { return; }
  let t = lid.x;
  var acc = 0.0;
  for (var sb = t; sb < SB_PER_ROW; sb = sb + 64u) {
    let wb = (r * SB_PER_ROW + sb) * 44u;   // base word del superblocco
    let dm = unpack2x16float(blocks[wb]);   // (d, dmin)
    let xBase = sb * 256u;
    var u1 = 1u; var u2 = 2u;
    for (var j = 0u; j < 4u; j = j + 1u) {  // 4 gruppi da 64 elementi
      let is = 2u * j;
      // get_scale_min_k4(is) e (is+1) — scales a byte offset 4 dentro il superblocco
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
      for (var l = 0u; l < 32u; l = l + 1u) {
        let ql = sbyte(wb, 48u + j * 32u + l);   // qs a byte offset 48
        let qh = sbyte(wb, 16u + l);             // qh a byte offset 16
        let x1 = x[xR + xBase + j * 64u + l];
        let x2 = x[xR + xBase + j * 64u + 32u + l];
        var q1 = f32(ql & 0xfu);
        if ((qh & u1) != 0u) { q1 = q1 + 16.0; }
        var q2 = f32(ql >> 4u);
        if ((qh & u2) != 0u) { q2 = q2 + 16.0; }
        dot1 = dot1 + q1 * x1; sx1 = sx1 + x1;
        dot2 = dot2 + q2 * x2; sx2 = sx2 + x2;
      }
      acc = acc + d1 * dot1 - min1 * sx1 + d2 * dot2 - min2 * sx2;
      u1 = u1 << 2u; u2 = u2 << 2u;
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
  if (t == 0u) { y[wid.z * 4096u + r] = partial[0]; }
}