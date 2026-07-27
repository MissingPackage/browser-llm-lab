//----------------------------------------
// Function: batch_prefill_ragged_kv_kernel
//----------------------------------------
@group(0) @binding(0) var<storage, read> k : array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> k_rope_pos_offset : array<i32>;
@group(0) @binding(2) var<storage, read> kv_indptr : array<i32>;
@group(0) @binding(3) var<storage, read_write> lse : array<f32>;
@group(0) @binding(4) var<storage, read_write> output : array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> q : array<vec2<f32>>;
@group(0) @binding(6) var<storage, read> q_indptr : array<i32>;
@group(0) @binding(7) var<storage, read> q_rope_position : array<i32>;
@group(0) @binding(8) var<storage, read> v : array<vec2<f32>>;

struct PODArgs {
  batch_size: i32,
  causal: i32,
  k_rope_pos_offset_elem_offset: i32,
  kv_indptr_elem_offset: i32,
  q_indptr_elem_offset: i32,
  q_rope_position_elem_offset: i32,
  rope_scale: f32,
  rope_theta: f32,
  rotary_mode: i32,
  sm_scale: f32,
  packGridDimX: u32
}
@group(0) @binding(9) var<uniform> podArgs : PODArgs;

var<workgroup> m_smem : array<f32, 16>;
var<workgroup> d_smem : array<f32, 16>;
var<workgroup> Q_smem : array<f32, 1024>;
var<workgroup> K_smem : array<f32, 1024>;
var<workgroup> V_smem : array<vec2<f32>, 512>;
var<workgroup> m_prev_smem : array<f32, 16>;
@compute @workgroup_size(32, 4, 1)
fn batch_prefill_ragged_kv_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  var tile_id : array<i32, 1>;
  var batch_idx : array<i32, 1>;
  var batch_rows : array<i32, 1>;
  var batch_tiles : array<i32, 1>;
  var kv_chunk_len : array<i32, 1>;
  var O_local : array<vec2<f32>, 4>;
  var S_local : array<vec2<f32>, 1>;
  var m_prev : array<f32, 1>;
  var m_new : array<f32, 1>;
  var d_new : array<f32, 1>;
  tile_id[0i] = v__1;
  batch_idx[0i] = 0i;
  batch_rows[0i] = ((q_indptr[(podArgs.q_indptr_elem_offset + 1i)] * 7i) - (q_indptr[podArgs.q_indptr_elem_offset] * 7i));
  batch_tiles[0i] = ((batch_rows[0i] + 15i)>>4u);
  while (true) {
    if (!(((batch_idx[0i] < podArgs.batch_size)))) { break; }
    while (true) {
      if (!(((batch_tiles[0i] <= tile_id[0i]) && (batch_idx[0i] < podArgs.batch_size)))) { break; }
      tile_id[0i] = (tile_id[0i] - batch_tiles[0i]);
      batch_idx[0i] = (batch_idx[0i] + 1i);
      if (batch_idx[0i] < podArgs.batch_size) {
        let b_idx : i32 = batch_idx[0i];
        batch_rows[0i] = ((q_indptr[((b_idx + podArgs.q_indptr_elem_offset) + 1i)] * 7i) - (q_indptr[(b_idx + podArgs.q_indptr_elem_offset)] * 7i));
        batch_tiles[0i] = ((batch_rows[0i] + 15i)>>4u);
      }
    }
    if ((batch_idx[0i] < podArgs.batch_size)) {
      let b_idx_1 : i32 = batch_idx[0i];
      let q_indptr_val : i32 = q_indptr[(b_idx_1 + podArgs.q_indptr_elem_offset)];
      let LH_start : i32 = (tile_id[0i] * 16i);
      kv_chunk_len[0i] = (kv_indptr[((b_idx_1 + podArgs.kv_indptr_elem_offset) + 1i)] - kv_indptr[(b_idx_1 + podArgs.kv_indptr_elem_offset)]);
      workgroupBarrier();
      let row : i32 = ((i32(threadIdx.y) * 32i) + i32(threadIdx.x));
      if (((i32(threadIdx.y) * 2i) + (i32(threadIdx.x)>>4u)) < 1i) {
        m_smem[((i32(threadIdx.y) * 32i) + i32(threadIdx.x))] = -5.000000e+04f;
        d_smem[((i32(threadIdx.y) * 32i) + i32(threadIdx.x))] = 1.000000e+00f;
      }
      for (var li_1 : i32 = 0i; li_1 < 2i; li_1++) {
        O_local[(li_1 * 2i)] = vec2<f32>(0.000000e+00f, 0.000000e+00f);
        O_local[((li_1 * 2i) + 1i)] = vec2<f32>(0.000000e+00f, 0.000000e+00f);
      }
      workgroupBarrier();
      for (var li_0 : i32 = 0i; li_0 < 2i; li_0++) {
        for (var lj_0_0 : i32 = 0i; lj_0_0 < 2i; lj_0_0++) {
          let cur_L : i32 = (((((((li_0 * 8i) + (i32(threadIdx.y) * 2i)) + (i32(threadIdx.x)>>4u)) + LH_start) / 7i) + ((((((li_0 * 8i) + (i32(threadIdx.y) * 2i)) + (i32(threadIdx.x)>>4u)) + LH_start) % 7i)>>31u)) + q_indptr_val);
          let cur_H_qo : i32 = ((i32(blockIdx.y) * 7i) + ((((((li_0 * 8i) + (i32(threadIdx.y) * 2i)) + (i32(threadIdx.x)>>4u)) + LH_start) % 7i) + (7i & ((((((li_0 * 8i) + (i32(threadIdx.y) * 2i)) + (i32(threadIdx.x)>>4u)) + LH_start) % 7i)>>31u))));
          if ((((((((li_0 * 8i) + (i32(threadIdx.y) * 2i)) + (i32(threadIdx.x)>>4u)) + LH_start) / 7i) + ((((((li_0 * 8i) + (i32(threadIdx.y) * 2i)) + (i32(threadIdx.x)>>4u)) + LH_start) % 7i)>>31u)) + q_indptr_val) < q_indptr[((b_idx_1 + podArgs.q_indptr_elem_offset) + 1i)]) {
            var condval : vec2<f32>;
            if ((podArgs.rotary_mode == 1i)) {
              let freq : vec2<f32> = (vec2<f32>((f32(q_rope_position[((((((((li_0 * 8i) + (i32(threadIdx.y) * 2i)) + (i32(threadIdx.x)>>4u)) + LH_start) / 7i) + ((((((li_0 * 8i) + (i32(threadIdx.y) * 2i)) + (i32(threadIdx.x)>>4u)) + LH_start) % 7i)>>31u)) + q_indptr_val) + podArgs.q_rope_position_elem_offset)]) * podArgs.rope_scale), (f32(q_rope_position[((((((((li_0 * 8i) + (i32(threadIdx.y) * 2i)) + (i32(threadIdx.x)>>4u)) + LH_start) / 7i) + ((((((li_0 * 8i) + (i32(threadIdx.y) * 2i)) + (i32(threadIdx.x)>>4u)) + LH_start) % 7i)>>31u)) + q_indptr_val) + podArgs.q_rope_position_elem_offset)]) * podArgs.rope_scale)) / pow(vec2<f32>(podArgs.rope_theta, podArgs.rope_theta), (vec2<f32>(vec2<i32>((((i32(threadIdx.x) & 15i) * 4i))+(2i*0), (((i32(threadIdx.x) & 15i) * 4i))+(2i*1))) / vec2<f32>(6.400000e+01f, 6.400000e+01f))));
              var condval_1 : vec2<f32>;
              if ((lj_0_0 < 1i)) {
                condval_1 = (q[((((((((((((((li_0 * 8i) + (i32(threadIdx.y) * 2i)) + (i32(threadIdx.x)>>4u)) + LH_start) / 7i) + ((((((li_0 * 8i) + (i32(threadIdx.y) * 2i)) + (i32(threadIdx.x)>>4u)) + LH_start) % 7i)>>31u)) * 896i) + (q_indptr_val * 896i)) + (i32(blockIdx.y) * 448i)) + (((((((li_0 * 8i) + (i32(threadIdx.y) * 2i)) + (i32(threadIdx.x)>>4u)) + LH_start) % 7i) + (7i & ((((((li_0 * 8i) + (i32(threadIdx.y) * 2i)) + (i32(threadIdx.x)>>4u)) + LH_start) % 7i)>>31u))) * 64i)) + (lj_0_0 * 32i)) + ((i32(threadIdx.x) & 15i) * 2i)) + 32i) / 2i)] * vec2<f32>(-1.000000e+00f, -1.000000e+00f));
} else {
                condval_1 = q[((((((((((((((li_0 * 8i) + (i32(threadIdx.y) * 2i)) + (i32(threadIdx.x)>>4u)) + LH_start) / 7i) + ((((((li_0 * 8i) + (i32(threadIdx.y) * 2i)) + (i32(threadIdx.x)>>4u)) + LH_start) % 7i)>>31u)) * 896i) + (q_indptr_val * 896i)) + (i32(blockIdx.y) * 448i)) + (((((((li_0 * 8i) + (i32(threadIdx.y) * 2i)) + (i32(threadIdx.x)>>4u)) + LH_start) % 7i) + (7i & ((((((li_0 * 8i) + (i32(threadIdx.y) * 2i)) + (i32(threadIdx.x)>>4u)) + LH_start) % 7i)>>31u))) * 64i)) + (lj_0_0 * 32i)) + ((i32(threadIdx.x) & 15i) * 2i)) - 32i) / 2i)];
}
              condval = fma(sin(freq), condval_1, (cos(freq) * q[(((((((((((((li_0 * 8i) + (i32(threadIdx.y) * 2i)) + (i32(threadIdx.x)>>4u)) + LH_start) / 7i) + ((((((li_0 * 8i) + (i32(threadIdx.y) * 2i)) + (i32(threadIdx.x)>>4u)) + LH_start) % 7i)>>31u)) * 896i) + (q_indptr_val * 896i)) + (i32(blockIdx.y) * 448i)) + (((((((li_0 * 8i) + (i32(threadIdx.y) * 2i)) + (i32(threadIdx.x)>>4u)) + LH_start) % 7i) + (7i & ((((((li_0 * 8i) + (i32(threadIdx.y) * 2i)) + (i32(threadIdx.x)>>4u)) + LH_start) % 7i)>>31u))) * 64i)) + (lj_0_0 * 32i)) + ((i32(threadIdx.x) & 15i) * 2i)) / 2i)]));
} else {
              condval = q[(((((((((((((li_0 * 8i) + (i32(threadIdx.y) * 2i)) + (i32(threadIdx.x)>>4u)) + LH_start) / 7i) + ((((((li_0 * 8i) + (i32(threadIdx.y) * 2i)) + (i32(threadIdx.x)>>4u)) + LH_start) % 7i)>>31u)) * 896i) + (q_indptr_val * 896i)) + (i32(blockIdx.y) * 448i)) + (((((((li_0 * 8i) + (i32(threadIdx.y) * 2i)) + (i32(threadIdx.x)>>4u)) + LH_start) % 7i) + (7i & ((((((li_0 * 8i) + (i32(threadIdx.y) * 2i)) + (i32(threadIdx.x)>>4u)) + LH_start) % 7i)>>31u))) * 64i)) + (lj_0_0 * 32i)) + ((i32(threadIdx.x) & 15i) * 2i)) / 2i)];
}
            let v__2 : i32 = (((((li_0 * 512i) + (i32(threadIdx.y) * 128i)) + ((i32(threadIdx.x)>>4u) * 64i)) + (lj_0_0 * 32i)) + ((i32(threadIdx.x) & 15i) * 2i));
            Q_smem[v__2 + 0] = condval[0];
            Q_smem[v__2 + 1] = condval[1];
          } else {
            let v__3 : i32 = (((((li_0 * 512i) + (i32(threadIdx.y) * 128i)) + ((i32(threadIdx.x)>>4u) * 64i)) + (lj_0_0 * 32i)) + ((i32(threadIdx.x) & 15i) * 2i));
            Q_smem[v__3 + 0] = vec2<f32>(0.000000e+00f, 0.000000e+00f)[0];
            Q_smem[v__3 + 1] = vec2<f32>(0.000000e+00f, 0.000000e+00f)[1];
          }
        }
      }
      workgroupBarrier();
      for (var iterator : i32 = 0i; iterator < ((kv_chunk_len[0i] + 15i)>>4u); iterator++) {
        let L_kv_start : i32 = (iterator * 16i);
        let L_kv_base : i32 = kv_indptr[(b_idx_1 + podArgs.kv_indptr_elem_offset)];
        workgroupBarrier();
        for (var lz_0 : i32 = 0i; lz_0 < 2i; lz_0++) {
          for (var ly_0_0 : i32 = 0i; ly_0_0 < 2i; ly_0_0++) {
            let cur_L_1 : i32 = ((((iterator * 16i) + (lz_0 * 8i)) + (i32(threadIdx.y) * 2i)) + (i32(threadIdx.x)>>4u));
            if (((((iterator * 16i) + (lz_0 * 8i)) + (i32(threadIdx.y) * 2i)) + (i32(threadIdx.x)>>4u)) < kv_chunk_len[0i]) {
              var condval_2 : vec2<f32>;
              if ((podArgs.rotary_mode == 1i)) {
                let freq_1 : vec2<f32> = (vec2<f32>((f32((((((iterator * 16i) + (lz_0 * 8i)) + (i32(threadIdx.y) * 2i)) + (i32(threadIdx.x)>>4u)) + k_rope_pos_offset[(b_idx_1 + podArgs.k_rope_pos_offset_elem_offset)])) * podArgs.rope_scale), (f32((((((iterator * 16i) + (lz_0 * 8i)) + (i32(threadIdx.y) * 2i)) + (i32(threadIdx.x)>>4u)) + k_rope_pos_offset[(b_idx_1 + podArgs.k_rope_pos_offset_elem_offset)])) * podArgs.rope_scale)) / pow(vec2<f32>(podArgs.rope_theta, podArgs.rope_theta), (vec2<f32>(vec2<i32>((((i32(threadIdx.x) & 15i) * 4i))+(2i*0), (((i32(threadIdx.x) & 15i) * 4i))+(2i*1))) / vec2<f32>(6.400000e+01f, 6.400000e+01f))));
                var condval_3 : vec2<f32>;
                if ((ly_0_0 < 1i)) {
                  condval_3 = (k[((((((((((iterator * 2048i) + (lz_0 * 1024i)) + (i32(threadIdx.y) * 256i)) + ((i32(threadIdx.x)>>4u) * 128i)) + (L_kv_base * 128i)) + (i32(blockIdx.y) * 64i)) + (ly_0_0 * 32i)) + ((i32(threadIdx.x) & 15i) * 2i)) + 32i) / 2i)] * vec2<f32>(-1.000000e+00f, -1.000000e+00f));
} else {
                  condval_3 = k[((((((((((iterator * 2048i) + (lz_0 * 1024i)) + (i32(threadIdx.y) * 256i)) + ((i32(threadIdx.x)>>4u) * 128i)) + (L_kv_base * 128i)) + (i32(blockIdx.y) * 64i)) + (ly_0_0 * 32i)) + ((i32(threadIdx.x) & 15i) * 2i)) - 32i) / 2i)];
}
                condval_2 = fma(sin(freq_1), condval_3, (cos(freq_1) * k[(((((((((iterator * 2048i) + (lz_0 * 1024i)) + (i32(threadIdx.y) * 256i)) + ((i32(threadIdx.x)>>4u) * 128i)) + (L_kv_base * 128i)) + (i32(blockIdx.y) * 64i)) + (ly_0_0 * 32i)) + ((i32(threadIdx.x) & 15i) * 2i)) / 2i)]));
} else {
                condval_2 = k[(((((((((iterator * 2048i) + (lz_0 * 1024i)) + (i32(threadIdx.y) * 256i)) + ((i32(threadIdx.x)>>4u) * 128i)) + (L_kv_base * 128i)) + (i32(blockIdx.y) * 64i)) + (ly_0_0 * 32i)) + ((i32(threadIdx.x) & 15i) * 2i)) / 2i)];
}
              let v__4 : vec2<i32> = vec2<i32>(((((((ly_0_0 * 512i) + ((i32(threadIdx.x) & 15i) * 32i)) + (lz_0 * 8i)) + (i32(threadIdx.y) * 2i)) + (i32(threadIdx.x)>>4u)))+(16i*0), ((((((ly_0_0 * 512i) + ((i32(threadIdx.x) & 15i) * 32i)) + (lz_0 * 8i)) + (i32(threadIdx.y) * 2i)) + (i32(threadIdx.x)>>4u)))+(16i*1));
              K_smem[v__4[0]] = condval_2[0];
              K_smem[v__4[1]] = condval_2[1];
            } else {
              let v__5 : vec2<i32> = vec2<i32>(((((((ly_0_0 * 512i) + ((i32(threadIdx.x) & 15i) * 32i)) + (lz_0 * 8i)) + (i32(threadIdx.y) * 2i)) + (i32(threadIdx.x)>>4u)))+(16i*0), ((((((ly_0_0 * 512i) + ((i32(threadIdx.x) & 15i) * 32i)) + (lz_0 * 8i)) + (i32(threadIdx.y) * 2i)) + (i32(threadIdx.x)>>4u)))+(16i*1));
              K_smem[v__5[0]] = vec2<f32>(0.000000e+00f, 0.000000e+00f)[0];
              K_smem[v__5[1]] = vec2<f32>(0.000000e+00f, 0.000000e+00f)[1];
            }
          }
        }
        workgroupBarrier();
        for (var lz_0_1 : i32 = 0i; lz_0_1 < 2i; lz_0_1++) {
          for (var ly_0_0_1 : i32 = 0i; ly_0_0_1 < 2i; ly_0_0_1++) {
            let cur_L_2 : i32 = ((((iterator * 16i) + (lz_0_1 * 8i)) + (i32(threadIdx.y) * 2i)) + (i32(threadIdx.x)>>4u));
            if (((((iterator * 16i) + (lz_0_1 * 8i)) + (i32(threadIdx.y) * 2i)) + (i32(threadIdx.x)>>4u)) < kv_chunk_len[0i]) {
              V_smem[(((((lz_0_1 * 256i) + (i32(threadIdx.y) * 64i)) + ((i32(threadIdx.x)>>4u) * 32i)) + (ly_0_0_1 * 16i)) + (i32(threadIdx.x) & 15i))] = v[(((((((((iterator * 2048i) + (lz_0_1 * 1024i)) + (i32(threadIdx.y) * 256i)) + ((i32(threadIdx.x)>>4u) * 128i)) + (L_kv_base * 128i)) + (i32(blockIdx.y) * 64i)) + (ly_0_0_1 * 32i)) + ((i32(threadIdx.x) & 15i) * 2i)) / 2i)];
            } else {
              V_smem[(((((lz_0_1 * 256i) + (i32(threadIdx.y) * 64i)) + ((i32(threadIdx.x)>>4u) * 32i)) + (ly_0_0_1 * 16i)) + (i32(threadIdx.x) & 15i))] = vec2<f32>(0.000000e+00f, 0.000000e+00f);
            }
          }
        }
        workgroupBarrier();
        S_local[0i] = vec2<f32>(0.000000e+00f, 0.000000e+00f);
        for (var lk_0 : i32 = 0i; lk_0 < 4i; lk_0++) {
          for (var lk_1 : i32 = 0i; lk_1 < 16i; lk_1++) {
            let v__6 : i32 = (((lk_0 * 256i) + (lk_1 * 16i)) + ((i32(threadIdx.x) & 7i) * 2i));
            S_local[0i] = fma(((vec2<f32>(Q_smem[((((i32(threadIdx.y) * 256i) + ((i32(threadIdx.x)>>3u) * 64i)) + (lk_0 * 16i)) + lk_1)], Q_smem[((((i32(threadIdx.y) * 256i) + ((i32(threadIdx.x)>>3u) * 64i)) + (lk_0 * 16i)) + lk_1)]) * vec2<f32>(K_smem[v__6 + 0], K_smem[v__6 + 1])) * vec2<f32>(podArgs.sm_scale, podArgs.sm_scale)), vec2<f32>(1.442695e+00f, 1.442695e+00f), S_local[0i]);
          }
        }
        workgroupBarrier();
        let v__7 : i32 = ((i32(threadIdx.y) * 64i) + (i32(threadIdx.x) * 2i));
        K_smem[v__7 + 0] = S_local[0i][0];
        K_smem[v__7 + 1] = S_local[0i][1];
        workgroupBarrier();
        let row_1 : i32 = ((i32(threadIdx.y) * 32i) + i32(threadIdx.x));
        if (((i32(threadIdx.y) * 2i) + (i32(threadIdx.x)>>4u)) < 1i) {
          m_prev[0i] = m_smem[((i32(threadIdx.y) * 32i) + i32(threadIdx.x))];
          m_new[0i] = m_smem[((i32(threadIdx.y) * 32i) + i32(threadIdx.x))];
          let row_ : i32 = (((((i32(threadIdx.y) * 32i) + LH_start) + i32(threadIdx.x)) / 7i) + (((((i32(threadIdx.y) * 32i) + LH_start) + i32(threadIdx.x)) % 7i)>>31u));
          for (var j : i32 = 0i; j < 16i; j++) {
            var condval_4 : bool;
            if ((0i < podArgs.causal)) {
              condval_4 = (((iterator * 16i) + j) <= ((((((((i32(threadIdx.y) * 32i) + LH_start) + i32(threadIdx.x)) / 7i) + (((((i32(threadIdx.y) * 32i) + LH_start) + i32(threadIdx.x)) % 7i)>>31u)) + kv_chunk_len[0i]) + q_indptr[(b_idx_1 + podArgs.q_indptr_elem_offset)]) - q_indptr[((b_idx_1 + podArgs.q_indptr_elem_offset) + 1i)]));
} else {
              condval_4 = (((iterator * 16i) + j) < kv_chunk_len[0i]);
}
            if (condval_4) {
              m_new[0i] = max(m_new[0i], K_smem[(((i32(threadIdx.y) * 512i) + (i32(threadIdx.x) * 16i)) + j)]);
            }
          }
          d_new[0i] = (d_smem[((i32(threadIdx.y) * 32i) + i32(threadIdx.x))] * exp2((m_prev[0i] - m_new[0i])));
        }
        let row_2 : i32 = ((i32(threadIdx.y) * 32i) + i32(threadIdx.x));
        workgroupBarrier();
        for (var j_1 : i32 = 0i; j_1 < 16i; j_1++) {
          if (((i32(threadIdx.y) * 2i) + (i32(threadIdx.x)>>4u)) < 1i) {
            let row__1 : i32 = (((((i32(threadIdx.y) * 32i) + LH_start) + i32(threadIdx.x)) / 7i) + (((((i32(threadIdx.y) * 32i) + LH_start) + i32(threadIdx.x)) % 7i)>>31u));
            var condval_5 : bool;
            if ((0i < podArgs.causal)) {
              condval_5 = (((iterator * 16i) + j_1) <= ((((((((i32(threadIdx.y) * 32i) + LH_start) + i32(threadIdx.x)) / 7i) + (((((i32(threadIdx.y) * 32i) + LH_start) + i32(threadIdx.x)) % 7i)>>31u)) + kv_chunk_len[0i]) + q_indptr[(b_idx_1 + podArgs.q_indptr_elem_offset)]) - q_indptr[((b_idx_1 + podArgs.q_indptr_elem_offset) + 1i)]));
} else {
              condval_5 = (((iterator * 16i) + j_1) < kv_chunk_len[0i]);
}
            if (condval_5) {
              K_smem[(((i32(threadIdx.y) * 512i) + (i32(threadIdx.x) * 16i)) + j_1)] = exp2((K_smem[(((i32(threadIdx.y) * 512i) + (i32(threadIdx.x) * 16i)) + j_1)] - m_new[0i]));
            } else {
              K_smem[(((i32(threadIdx.y) * 512i) + (i32(threadIdx.x) * 16i)) + j_1)] = exp2((-5.000000e+04f - m_new[0i]));
            }
          }
        }
        let row_3 : i32 = ((i32(threadIdx.y) * 32i) + i32(threadIdx.x));
        workgroupBarrier();
        if (((i32(threadIdx.y) * 2i) + (i32(threadIdx.x)>>4u)) < 1i) {
          for (var j_2 : i32 = 0i; j_2 < 16i; j_2++) {
            d_new[0i] = (d_new[0i] + K_smem[(((i32(threadIdx.y) * 512i) + (i32(threadIdx.x) * 16i)) + j_2)]);
          }
          m_smem[((i32(threadIdx.y) * 32i) + i32(threadIdx.x))] = m_new[0i];
          d_smem[((i32(threadIdx.y) * 32i) + i32(threadIdx.x))] = d_new[0i];
          m_prev_smem[((i32(threadIdx.y) * 32i) + i32(threadIdx.x))] = m_prev[0i];
        }
        workgroupBarrier();
        O_local[0i] = (O_local[0i] * vec2<f32>(exp2((m_prev_smem[((i32(threadIdx.y) * 4i) + ((i32(threadIdx.x)>>4u) * 2i))] - m_smem[((i32(threadIdx.y) * 4i) + ((i32(threadIdx.x)>>4u) * 2i))])), exp2((m_prev_smem[((i32(threadIdx.y) * 4i) + ((i32(threadIdx.x)>>4u) * 2i))] - m_smem[((i32(threadIdx.y) * 4i) + ((i32(threadIdx.x)>>4u) * 2i))]))));
        O_local[1i] = (O_local[1i] * vec2<f32>(exp2((m_prev_smem[((i32(threadIdx.y) * 4i) + ((i32(threadIdx.x)>>4u) * 2i))] - m_smem[((i32(threadIdx.y) * 4i) + ((i32(threadIdx.x)>>4u) * 2i))])), exp2((m_prev_smem[((i32(threadIdx.y) * 4i) + ((i32(threadIdx.x)>>4u) * 2i))] - m_smem[((i32(threadIdx.y) * 4i) + ((i32(threadIdx.x)>>4u) * 2i))]))));
        O_local[2i] = (O_local[2i] * vec2<f32>(exp2((m_prev_smem[(((i32(threadIdx.y) * 4i) + ((i32(threadIdx.x)>>4u) * 2i)) + 1i)] - m_smem[(((i32(threadIdx.y) * 4i) + ((i32(threadIdx.x)>>4u) * 2i)) + 1i)])), exp2((m_prev_smem[(((i32(threadIdx.y) * 4i) + ((i32(threadIdx.x)>>4u) * 2i)) + 1i)] - m_smem[(((i32(threadIdx.y) * 4i) + ((i32(threadIdx.x)>>4u) * 2i)) + 1i)]))));
        O_local[3i] = (O_local[3i] * vec2<f32>(exp2((m_prev_smem[(((i32(threadIdx.y) * 4i) + ((i32(threadIdx.x)>>4u) * 2i)) + 1i)] - m_smem[(((i32(threadIdx.y) * 4i) + ((i32(threadIdx.x)>>4u) * 2i)) + 1i)])), exp2((m_prev_smem[(((i32(threadIdx.y) * 4i) + ((i32(threadIdx.x)>>4u) * 2i)) + 1i)] - m_smem[(((i32(threadIdx.y) * 4i) + ((i32(threadIdx.x)>>4u) * 2i)) + 1i)]))));
        for (var lk_1_1 : i32 = 0i; lk_1_1 < 16i; lk_1_1++) {
          O_local[0i] = fma(vec2<f32>(K_smem[(((i32(threadIdx.y) * 64i) + ((i32(threadIdx.x)>>4u) * 32i)) + lk_1_1)], K_smem[(((i32(threadIdx.y) * 64i) + ((i32(threadIdx.x)>>4u) * 32i)) + lk_1_1)]), V_smem[((lk_1_1 * 32i) + ((i32(threadIdx.x) & 15i) * 2i))], O_local[0i]);
          O_local[1i] = fma(vec2<f32>(K_smem[(((i32(threadIdx.y) * 64i) + ((i32(threadIdx.x)>>4u) * 32i)) + lk_1_1)], K_smem[(((i32(threadIdx.y) * 64i) + ((i32(threadIdx.x)>>4u) * 32i)) + lk_1_1)]), V_smem[(((lk_1_1 * 32i) + ((i32(threadIdx.x) & 15i) * 2i)) + 1i)], O_local[1i]);
          O_local[2i] = fma(vec2<f32>(K_smem[((((i32(threadIdx.y) * 64i) + ((i32(threadIdx.x)>>4u) * 32i)) + lk_1_1) + 16i)], K_smem[((((i32(threadIdx.y) * 64i) + ((i32(threadIdx.x)>>4u) * 32i)) + lk_1_1) + 16i)]), V_smem[((lk_1_1 * 32i) + ((i32(threadIdx.x) & 15i) * 2i))], O_local[2i]);
          O_local[3i] = fma(vec2<f32>(K_smem[((((i32(threadIdx.y) * 64i) + ((i32(threadIdx.x)>>4u) * 32i)) + lk_1_1) + 16i)], K_smem[((((i32(threadIdx.y) * 64i) + ((i32(threadIdx.x)>>4u) * 32i)) + lk_1_1) + 16i)]), V_smem[(((lk_1_1 * 32i) + ((i32(threadIdx.x) & 15i) * 2i)) + 1i)], O_local[3i]);
        }
      }
      for (var li_1_1 : i32 = 0i; li_1_1 < 2i; li_1_1++) {
        let cur_L_3 : i32 = (((((((i32(threadIdx.y) * 4i) + ((i32(threadIdx.x)>>4u) * 2i)) + LH_start) + li_1_1) / 7i) + ((((((i32(threadIdx.y) * 4i) + ((i32(threadIdx.x)>>4u) * 2i)) + LH_start) + li_1_1) % 7i)>>31u)) + q_indptr[(b_idx_1 + podArgs.q_indptr_elem_offset)]);
        let cur_H_qo_1 : i32 = ((i32(blockIdx.y) * 7i) + ((((((i32(threadIdx.y) * 4i) + ((i32(threadIdx.x)>>4u) * 2i)) + LH_start) + li_1_1) % 7i) + (7i & ((((((i32(threadIdx.y) * 4i) + ((i32(threadIdx.x)>>4u) * 2i)) + LH_start) + li_1_1) % 7i)>>31u))));
        if (cur_L_3 < q_indptr[((b_idx_1 + podArgs.q_indptr_elem_offset) + 1i)]) {
          output[(((((cur_L_3 * 896i) + (i32(blockIdx.y) * 448i)) + (((((((i32(threadIdx.y) * 4i) + ((i32(threadIdx.x)>>4u) * 2i)) + LH_start) + li_1_1) % 7i) + (7i & ((((((i32(threadIdx.y) * 4i) + ((i32(threadIdx.x)>>4u) * 2i)) + LH_start) + li_1_1) % 7i)>>31u))) * 64i)) + ((i32(threadIdx.x) & 15i) * 4i)) / 2i)] = (O_local[(li_1_1 * 2i)] / vec2<f32>(d_smem[(((i32(threadIdx.y) * 4i) + ((i32(threadIdx.x)>>4u) * 2i)) + li_1_1)], d_smem[(((i32(threadIdx.y) * 4i) + ((i32(threadIdx.x)>>4u) * 2i)) + li_1_1)]));
        }
        let cur_L_4 : i32 = (((((((i32(threadIdx.y) * 4i) + ((i32(threadIdx.x)>>4u) * 2i)) + LH_start) + li_1_1) / 7i) + ((((((i32(threadIdx.y) * 4i) + ((i32(threadIdx.x)>>4u) * 2i)) + LH_start) + li_1_1) % 7i)>>31u)) + q_indptr[(b_idx_1 + podArgs.q_indptr_elem_offset)]);
        let cur_H_qo_2 : i32 = ((i32(blockIdx.y) * 7i) + ((((((i32(threadIdx.y) * 4i) + ((i32(threadIdx.x)>>4u) * 2i)) + LH_start) + li_1_1) % 7i) + (7i & ((((((i32(threadIdx.y) * 4i) + ((i32(threadIdx.x)>>4u) * 2i)) + LH_start) + li_1_1) % 7i)>>31u))));
        if (cur_L_4 < q_indptr[((b_idx_1 + podArgs.q_indptr_elem_offset) + 1i)]) {
          output[((((((cur_L_4 * 896i) + (i32(blockIdx.y) * 448i)) + (((((((i32(threadIdx.y) * 4i) + ((i32(threadIdx.x)>>4u) * 2i)) + LH_start) + li_1_1) % 7i) + (7i & ((((((i32(threadIdx.y) * 4i) + ((i32(threadIdx.x)>>4u) * 2i)) + LH_start) + li_1_1) % 7i)>>31u))) * 64i)) + ((i32(threadIdx.x) & 15i) * 4i)) + 2i) / 2i)] = (O_local[((li_1_1 * 2i) + 1i)] / vec2<f32>(d_smem[(((i32(threadIdx.y) * 4i) + ((i32(threadIdx.x)>>4u) * 2i)) + li_1_1)], d_smem[(((i32(threadIdx.y) * 4i) + ((i32(threadIdx.x)>>4u) * 2i)) + li_1_1)]));
        }
      }
      if (((i32(threadIdx.y) * 2i) + (i32(threadIdx.x)>>4u)) < 1i) {
        let cur_L_5 : i32 = ((((((i32(threadIdx.y) * 32i) + LH_start) + i32(threadIdx.x)) / 7i) + (((((i32(threadIdx.y) * 32i) + LH_start) + i32(threadIdx.x)) % 7i)>>31u)) + q_indptr[(b_idx_1 + podArgs.q_indptr_elem_offset)]);
        let cur_H_qo_3 : i32 = ((i32(blockIdx.y) * 7i) + (((((i32(threadIdx.y) * 32i) + LH_start) + i32(threadIdx.x)) % 7i) + (7i & (((((i32(threadIdx.y) * 32i) + LH_start) + i32(threadIdx.x)) % 7i)>>31u))));
        if (cur_L_5 < q_indptr[((b_idx_1 + podArgs.q_indptr_elem_offset) + 1i)]) {
          lse[(((cur_L_5 * 14i) + (i32(blockIdx.y) * 7i)) + (((((i32(threadIdx.y) * 32i) + LH_start) + i32(threadIdx.x)) % 7i) + (7i & (((((i32(threadIdx.y) * 32i) + LH_start) + i32(threadIdx.x)) % 7i)>>31u))))] = (m_smem[((i32(threadIdx.y) * 32i) + i32(threadIdx.x))] + log2(d_smem[((i32(threadIdx.y) * 32i) + i32(threadIdx.x))]));
        }
      }
      tile_id[0i] = (tile_id[0i] + 16i);
    }
  }
}

