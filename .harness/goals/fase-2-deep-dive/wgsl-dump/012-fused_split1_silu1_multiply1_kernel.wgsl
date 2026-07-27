//----------------------------------------
// Function: fused_split1_silu1_multiply1_kernel
//----------------------------------------
@group(0) @binding(0) var<storage, read_write> T_multiply : array<f32>;
@group(0) @binding(1) var<storage, read> lv99 : array<f32>;

struct PODArgs {
  seq_len: i32,
  packGridDimX: u32
}
@group(0) @binding(2) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(256, 1, 1)
fn fused_split1_silu1_multiply1_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  T_multiply[((v__1 * 256i) + i32(threadIdx.x))] = ((lv99[((((v__1 / 19i) * 9728i) + ((v__1 % 19i) * 256i)) + i32(threadIdx.x))] * (1.000000e+00f / (1.000000e+00f + exp((0.000000e+00f - lv99[((((v__1 / 19i) * 9728i) + ((v__1 % 19i) * 256i)) + i32(threadIdx.x))]))))) * lv99[(((((v__1 / 19i) * 9728i) + ((v__1 % 19i) * 256i)) + i32(threadIdx.x)) + 4864i)]);
}

