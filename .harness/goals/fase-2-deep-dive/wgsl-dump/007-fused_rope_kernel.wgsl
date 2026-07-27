//----------------------------------------
// Function: fused_rope_kernel
//----------------------------------------
@group(0) @binding(0) var<storage, read_write> k : array<f32>;
@group(0) @binding(1) var<storage, read> position_map : array<i32>;
@group(0) @binding(2) var<storage, read_write> q : array<f32>;
@group(0) @binding(3) var<storage, read> qkv : array<f32>;
@group(0) @binding(4) var<storage, read_write> v : array<f32>;

struct PODArgs {
  apply_rope: i32,
  position_map_elem_offset: i32,
  seq_len: i32,
  packGridDimX: u32
}
@group(0) @binding(5) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(256, 1, 1)
fn fused_rope_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (((v__1 * 256i) + i32(threadIdx.x)) < (podArgs.seq_len * 1152i)) {
    if ((((v__1 * 256i) + i32(threadIdx.x)) % 1152i) < 896i) {
      var condval : f32;
      if ((0i < podArgs.apply_rope)) {
        let freq : f32 = (f32(position_map[((((v__1 * 2i) + (i32(threadIdx.x)>>7u)) / 9i) + podArgs.position_map_elem_offset)]) / pow(1.000000e+06f, (f32(((i32(threadIdx.x) & 31i) * 2i)) / 6.400000e+01f)));
        var condval_1 : f32;
        if (((i32(threadIdx.x) & 63i) < 32i)) {
          condval_1 = (qkv[(((v__1 * 256i) + i32(threadIdx.x)) + 32i)] * -1.000000e+00f);
} else {
          condval_1 = qkv[(((v__1 * 256i) + i32(threadIdx.x)) - 32i)];
}
        condval = fma(sin(freq), condval_1, (cos(freq) * qkv[((v__1 * 256i) + i32(threadIdx.x))]));
} else {
        condval = qkv[((v__1 * 256i) + i32(threadIdx.x))];
}
      q[((((((v__1 * 2i) + (i32(threadIdx.x)>>7u)) / 9i) * 896i) + ((((v__1 * 4i) + (i32(threadIdx.x)>>6u)) % 18i) * 64i)) + (i32(threadIdx.x) & 63i))] = condval;
    } else {
      if ((((v__1 * 256i) + i32(threadIdx.x)) % 1152i) < 1024i) {
        var condval_2 : f32;
        if ((0i < podArgs.apply_rope)) {
          let freq_1 : f32 = (f32(position_map[((((v__1 * 2i) + (i32(threadIdx.x)>>7u)) / 9i) + podArgs.position_map_elem_offset)]) / pow(1.000000e+06f, (f32(((i32(threadIdx.x) & 31i) * 2i)) / 6.400000e+01f)));
          var condval_3 : f32;
          if (((i32(threadIdx.x) & 63i) < 32i)) {
            condval_3 = (qkv[(((v__1 * 256i) + i32(threadIdx.x)) + 32i)] * -1.000000e+00f);
} else {
            condval_3 = qkv[(((v__1 * 256i) + i32(threadIdx.x)) - 32i)];
}
          condval_2 = fma(sin(freq_1), condval_3, (cos(freq_1) * qkv[((v__1 * 256i) + i32(threadIdx.x))]));
} else {
          condval_2 = qkv[((v__1 * 256i) + i32(threadIdx.x))];
}
        k[(((((((v__1 * 2i) + (i32(threadIdx.x)>>7u)) / 9i) * 128i) + ((((v__1 * 4i) + (i32(threadIdx.x)>>6u)) % 18i) * 64i)) + (i32(threadIdx.x) & 63i)) - 896i)] = condval_2;
      } else {
        v[(((((((v__1 * 2i) + (i32(threadIdx.x)>>7u)) / 9i) * 128i) + ((((v__1 * 4i) + (i32(threadIdx.x)>>6u)) % 18i) * 64i)) + (i32(threadIdx.x) & 63i)) - 1024i)] = qkv[((v__1 * 256i) + i32(threadIdx.x))];
      }
    }
  }
}

