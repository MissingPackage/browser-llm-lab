//----------------------------------------
// Function: fused_dequantize_take1_kernel
//----------------------------------------
@group(0) @binding(0) var<storage, read_write> T_take : array<f32>;
@group(0) @binding(1) var<storage, read> input_ids : array<i32>;
@group(0) @binding(2) var<storage, read> model_embed_tokens_q_scale : array<f32>;
@group(0) @binding(3) var<storage, read> model_embed_tokens_q_weight : array<u32>;

struct PODArgs {
  seq_len: i32,
  packGridDimX: u32
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

@compute @workgroup_size(256, 1, 1)
fn fused_dequantize_take1_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  if (((v__1 * 256i) + i32(threadIdx.x)) < (podArgs.seq_len * 896i)) {
    T_take[((v__1 * 256i) + i32(threadIdx.x))] = ((f32(((model_embed_tokens_q_weight[((input_ids[(((v__1 * 2i) + (i32(threadIdx.x)>>7u)) / 7i)] * 112i) + (((v__1 * 32i) + (i32(threadIdx.x)>>3u)) % 112i))]>>u32(((i32(threadIdx.x) & 7i) * 4i))) & 15u)) - 7.000000e+00f) * model_embed_tokens_q_scale[((input_ids[(((v__1 * 2i) + (i32(threadIdx.x)>>7u)) / 7i)] * 28i) + (((v__1 * 8i) + (i32(threadIdx.x)>>5u)) % 28i))]);
  }
}

