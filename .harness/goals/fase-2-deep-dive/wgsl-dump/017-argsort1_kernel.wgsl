//----------------------------------------
// Function: argsort1_kernel
//----------------------------------------
@group(0) @binding(0) var<storage, read_write> out_buf : array<i32>;
@group(0) @binding(1) var<storage, read> probs : array<f32>;
@group(0) @binding(2) var<storage, read_write> value_buf : array<f32>;

struct PODArgs {
  batch_size: i32,
  elem_offset: i32,
  vocab_size: i32,
  packGridDimX: u32
}
@group(0) @binding(3) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(256, 1, 1)
fn argsort1_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (((v__1 * 256i) + i32(threadIdx.x)) < podArgs.vocab_size) {
    value_buf[(((v__1 * 256i) + (i32(blockIdx.y) * podArgs.vocab_size)) + i32(threadIdx.x))] = probs[((((v__1 * 256i) + (i32(blockIdx.y) * podArgs.vocab_size)) + i32(threadIdx.x)) + podArgs.elem_offset)];
    out_buf[(((v__1 * 256i) + (i32(blockIdx.y) * podArgs.vocab_size)) + i32(threadIdx.x))] = ((v__1 * 256i) + i32(threadIdx.x));
  }
}

