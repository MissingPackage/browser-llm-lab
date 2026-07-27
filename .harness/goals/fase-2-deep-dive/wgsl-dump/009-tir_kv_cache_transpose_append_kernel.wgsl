//----------------------------------------
// Function: tir_kv_cache_transpose_append_kernel
//----------------------------------------
@group(0) @binding(0) var<storage, read> k_data : array<f32>;
@group(0) @binding(1) var<storage, read_write> pages : array<f32>;
@group(0) @binding(2) var<storage, read> position_map : array<i32>;
@group(0) @binding(3) var<storage, read> v_data : array<f32>;

struct PODArgs {
  ntoken: i32,
  pages_elem_offset: i32,
  position_map_elem_offset: i32,
  packGridDimX: u32
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(256, 1, 1)
fn tir_kv_cache_transpose_append_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (position_map[(((v__1 * 2i) + (i32(threadIdx.x)>>7u)) + podArgs.position_map_elem_offset)] != -1i) {
    if (((v__1 * 256i) + i32(threadIdx.x)) < (podArgs.ntoken * 128i)) {
      let position : i32 = position_map[(((v__1 * 2i) + (i32(threadIdx.x)>>7u)) + podArgs.position_map_elem_offset)];
      pages[((((((position>>4u) * 4096i) + (((i32(threadIdx.x) & 127i)>>6u) * 1024i)) + ((position & 15i) * 64i)) + podArgs.pages_elem_offset) + (i32(threadIdx.x) & 63i))] = k_data[((v__1 * 256i) + i32(threadIdx.x))];
    }
    if (((v__1 * 256i) + i32(threadIdx.x)) < (podArgs.ntoken * 128i)) {
      let position_1 : i32 = position_map[(((v__1 * 2i) + (i32(threadIdx.x)>>7u)) + podArgs.position_map_elem_offset)];
      pages[(((((((position_1>>4u) * 4096i) + (((i32(threadIdx.x) & 127i)>>6u) * 1024i)) + ((position_1 & 15i) * 64i)) + podArgs.pages_elem_offset) + (i32(threadIdx.x) & 63i)) + 2048i)] = v_data[((v__1 * 256i) + i32(threadIdx.x))];
    }
  }
}

