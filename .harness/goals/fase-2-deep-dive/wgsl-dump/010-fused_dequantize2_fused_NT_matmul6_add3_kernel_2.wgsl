//----------------------------------------
// Function: fused_dequantize2_fused_NT_matmul6_add3_kernel_2
//----------------------------------------
@group(0) @binding(0) var<storage, read_write> T_add : array<f32>;
@group(0) @binding(1) var<storage, read> input_embeds : array<f32>;
@group(0) @binding(2) var<storage, read> model_layers_0_self_attn_o_proj_q_scale3 : array<f32>;
@group(0) @binding(3) var<storage, read> model_layers_0_self_attn_o_proj_q_weight3 : array<u32>;
@group(0) @binding(4) var<storage, read> reshape195 : array<f32>;

struct PODArgs {
  seq_len: i32,
  packGridDimX: u32
}
@group(0) @binding(5) var<uniform> podArgs : PODArgs;

var<workgroup> reshape195_reindex_pad_shared : array<f32, 256>;
var<workgroup> dequantize_reindex_shared : array<f32, 256>;
@compute @workgroup_size(8, 8, 1)
fn fused_dequantize2_fused_NT_matmul6_add3_kernel_2(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  var NT_matmul_intermediate_reindex_pad_local : array<f32, 16>;
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  for (var var_1 : i32 = 0i; var_1 < 1i; var_1++) {
    NT_matmul_intermediate_reindex_pad_local[0i] = 0.000000e+00f;
    NT_matmul_intermediate_reindex_pad_local[1i] = 0.000000e+00f;
    NT_matmul_intermediate_reindex_pad_local[2i] = 0.000000e+00f;
    NT_matmul_intermediate_reindex_pad_local[3i] = 0.000000e+00f;
    NT_matmul_intermediate_reindex_pad_local[4i] = 0.000000e+00f;
    NT_matmul_intermediate_reindex_pad_local[5i] = 0.000000e+00f;
    NT_matmul_intermediate_reindex_pad_local[6i] = 0.000000e+00f;
    NT_matmul_intermediate_reindex_pad_local[7i] = 0.000000e+00f;
    NT_matmul_intermediate_reindex_pad_local[8i] = 0.000000e+00f;
    NT_matmul_intermediate_reindex_pad_local[9i] = 0.000000e+00f;
    NT_matmul_intermediate_reindex_pad_local[10i] = 0.000000e+00f;
    NT_matmul_intermediate_reindex_pad_local[11i] = 0.000000e+00f;
    NT_matmul_intermediate_reindex_pad_local[12i] = 0.000000e+00f;
    NT_matmul_intermediate_reindex_pad_local[13i] = 0.000000e+00f;
    NT_matmul_intermediate_reindex_pad_local[14i] = 0.000000e+00f;
    NT_matmul_intermediate_reindex_pad_local[15i] = 0.000000e+00f;
    for (var ax3_0 : i32 = 0i; ax3_0 < 112i; ax3_0++) {
      workgroupBarrier();
      if ((((v__1 * 32i) + (i32(threadIdx.y) * 4i)) + (i32(threadIdx.x)>>1u)) < podArgs.seq_len) {
        reshape195_reindex_pad_shared[((i32(threadIdx.y) * 32i) + (i32(threadIdx.x) * 4i))] = reshape195[(((((v__1 * 28672i) + (i32(threadIdx.y) * 3584i)) + ((i32(threadIdx.x)>>1u) * 896i)) + (ax3_0 * 8i)) + ((i32(threadIdx.x) & 1i) * 4i))];
        reshape195_reindex_pad_shared[(((i32(threadIdx.y) * 32i) + (i32(threadIdx.x) * 4i)) + 1i)] = reshape195[((((((v__1 * 28672i) + (i32(threadIdx.y) * 3584i)) + ((i32(threadIdx.x)>>1u) * 896i)) + (ax3_0 * 8i)) + ((i32(threadIdx.x) & 1i) * 4i)) + 1i)];
        reshape195_reindex_pad_shared[(((i32(threadIdx.y) * 32i) + (i32(threadIdx.x) * 4i)) + 2i)] = reshape195[((((((v__1 * 28672i) + (i32(threadIdx.y) * 3584i)) + ((i32(threadIdx.x)>>1u) * 896i)) + (ax3_0 * 8i)) + ((i32(threadIdx.x) & 1i) * 4i)) + 2i)];
        reshape195_reindex_pad_shared[(((i32(threadIdx.y) * 32i) + (i32(threadIdx.x) * 4i)) + 3i)] = reshape195[((((((v__1 * 28672i) + (i32(threadIdx.y) * 3584i)) + ((i32(threadIdx.x)>>1u) * 896i)) + (ax3_0 * 8i)) + ((i32(threadIdx.x) & 1i) * 4i)) + 3i)];
      } else {
        reshape195_reindex_pad_shared[((i32(threadIdx.y) * 32i) + (i32(threadIdx.x) * 4i))] = 0.000000e+00f;
        reshape195_reindex_pad_shared[(((i32(threadIdx.y) * 32i) + (i32(threadIdx.x) * 4i)) + 1i)] = 0.000000e+00f;
        reshape195_reindex_pad_shared[(((i32(threadIdx.y) * 32i) + (i32(threadIdx.x) * 4i)) + 2i)] = 0.000000e+00f;
        reshape195_reindex_pad_shared[(((i32(threadIdx.y) * 32i) + (i32(threadIdx.x) * 4i)) + 3i)] = 0.000000e+00f;
      }
      dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + (i32(threadIdx.x) * 4i))] = ((f32(((model_layers_0_self_attn_o_proj_q_weight3[((((i32(blockIdx.y) * 3584i) + (i32(threadIdx.y) * 448i)) + ((i32(threadIdx.x)>>1u) * 112i)) + ax3_0)]>>u32(((i32(threadIdx.x) & 1i) * 16i))) & 15u)) - 7.000000e+00f) * model_layers_0_self_attn_o_proj_q_scale3[((((i32(blockIdx.y) * 896i) + (i32(threadIdx.y) * 112i)) + ((i32(threadIdx.x)>>1u) * 28i)) + (ax3_0>>2u))]);
      dequantize_reindex_shared[(((i32(threadIdx.y) * 32i) + (i32(threadIdx.x) * 4i)) + 1i)] = ((f32(((model_layers_0_self_attn_o_proj_q_weight3[((((i32(blockIdx.y) * 3584i) + (i32(threadIdx.y) * 448i)) + ((i32(threadIdx.x)>>1u) * 112i)) + ax3_0)]>>u32((((i32(threadIdx.x) & 1i) * 16i) + 4i))) & 15u)) - 7.000000e+00f) * model_layers_0_self_attn_o_proj_q_scale3[((((i32(blockIdx.y) * 896i) + (i32(threadIdx.y) * 112i)) + ((i32(threadIdx.x)>>1u) * 28i)) + (ax3_0>>2u))]);
      dequantize_reindex_shared[(((i32(threadIdx.y) * 32i) + (i32(threadIdx.x) * 4i)) + 2i)] = ((f32(((model_layers_0_self_attn_o_proj_q_weight3[((((i32(blockIdx.y) * 3584i) + (i32(threadIdx.y) * 448i)) + ((i32(threadIdx.x)>>1u) * 112i)) + ax3_0)]>>u32((((i32(threadIdx.x) & 1i) * 16i) + 8i))) & 15u)) - 7.000000e+00f) * model_layers_0_self_attn_o_proj_q_scale3[((((i32(blockIdx.y) * 896i) + (i32(threadIdx.y) * 112i)) + ((i32(threadIdx.x)>>1u) * 28i)) + (ax3_0>>2u))]);
      dequantize_reindex_shared[(((i32(threadIdx.y) * 32i) + (i32(threadIdx.x) * 4i)) + 3i)] = ((f32(((model_layers_0_self_attn_o_proj_q_weight3[((((i32(blockIdx.y) * 3584i) + (i32(threadIdx.y) * 448i)) + ((i32(threadIdx.x)>>1u) * 112i)) + ax3_0)]>>u32((((i32(threadIdx.x) & 1i) * 16i) + 12i))) & 15u)) - 7.000000e+00f) * model_layers_0_self_attn_o_proj_q_scale3[((((i32(blockIdx.y) * 896i) + (i32(threadIdx.y) * 112i)) + ((i32(threadIdx.x)>>1u) * 28i)) + (ax3_0>>2u))]);
      workgroupBarrier();
      NT_matmul_intermediate_reindex_pad_local[0i] = fma(reshape195_reindex_pad_shared[(i32(threadIdx.x) * 32i)], dequantize_reindex_shared[(i32(threadIdx.y) * 32i)], NT_matmul_intermediate_reindex_pad_local[0i]);
      NT_matmul_intermediate_reindex_pad_local[1i] = fma(reshape195_reindex_pad_shared[(i32(threadIdx.x) * 32i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 8i)], NT_matmul_intermediate_reindex_pad_local[1i]);
      NT_matmul_intermediate_reindex_pad_local[2i] = fma(reshape195_reindex_pad_shared[(i32(threadIdx.x) * 32i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 16i)], NT_matmul_intermediate_reindex_pad_local[2i]);
      NT_matmul_intermediate_reindex_pad_local[3i] = fma(reshape195_reindex_pad_shared[(i32(threadIdx.x) * 32i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 24i)], NT_matmul_intermediate_reindex_pad_local[3i]);
      NT_matmul_intermediate_reindex_pad_local[4i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 8i)], dequantize_reindex_shared[(i32(threadIdx.y) * 32i)], NT_matmul_intermediate_reindex_pad_local[4i]);
      NT_matmul_intermediate_reindex_pad_local[5i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 8i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 8i)], NT_matmul_intermediate_reindex_pad_local[5i]);
      NT_matmul_intermediate_reindex_pad_local[6i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 8i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 16i)], NT_matmul_intermediate_reindex_pad_local[6i]);
      NT_matmul_intermediate_reindex_pad_local[7i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 8i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 24i)], NT_matmul_intermediate_reindex_pad_local[7i]);
      NT_matmul_intermediate_reindex_pad_local[8i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 16i)], dequantize_reindex_shared[(i32(threadIdx.y) * 32i)], NT_matmul_intermediate_reindex_pad_local[8i]);
      NT_matmul_intermediate_reindex_pad_local[9i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 16i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 8i)], NT_matmul_intermediate_reindex_pad_local[9i]);
      NT_matmul_intermediate_reindex_pad_local[10i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 16i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 16i)], NT_matmul_intermediate_reindex_pad_local[10i]);
      NT_matmul_intermediate_reindex_pad_local[11i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 16i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 24i)], NT_matmul_intermediate_reindex_pad_local[11i]);
      NT_matmul_intermediate_reindex_pad_local[12i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 24i)], dequantize_reindex_shared[(i32(threadIdx.y) * 32i)], NT_matmul_intermediate_reindex_pad_local[12i]);
      NT_matmul_intermediate_reindex_pad_local[13i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 24i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 8i)], NT_matmul_intermediate_reindex_pad_local[13i]);
      NT_matmul_intermediate_reindex_pad_local[14i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 24i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 16i)], NT_matmul_intermediate_reindex_pad_local[14i]);
      NT_matmul_intermediate_reindex_pad_local[15i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 24i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 24i)], NT_matmul_intermediate_reindex_pad_local[15i]);
      NT_matmul_intermediate_reindex_pad_local[0i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 1i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 1i)], NT_matmul_intermediate_reindex_pad_local[0i]);
      NT_matmul_intermediate_reindex_pad_local[1i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 1i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 9i)], NT_matmul_intermediate_reindex_pad_local[1i]);
      NT_matmul_intermediate_reindex_pad_local[2i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 1i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 17i)], NT_matmul_intermediate_reindex_pad_local[2i]);
      NT_matmul_intermediate_reindex_pad_local[3i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 1i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 25i)], NT_matmul_intermediate_reindex_pad_local[3i]);
      NT_matmul_intermediate_reindex_pad_local[4i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 9i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 1i)], NT_matmul_intermediate_reindex_pad_local[4i]);
      NT_matmul_intermediate_reindex_pad_local[5i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 9i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 9i)], NT_matmul_intermediate_reindex_pad_local[5i]);
      NT_matmul_intermediate_reindex_pad_local[6i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 9i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 17i)], NT_matmul_intermediate_reindex_pad_local[6i]);
      NT_matmul_intermediate_reindex_pad_local[7i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 9i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 25i)], NT_matmul_intermediate_reindex_pad_local[7i]);
      NT_matmul_intermediate_reindex_pad_local[8i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 17i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 1i)], NT_matmul_intermediate_reindex_pad_local[8i]);
      NT_matmul_intermediate_reindex_pad_local[9i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 17i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 9i)], NT_matmul_intermediate_reindex_pad_local[9i]);
      NT_matmul_intermediate_reindex_pad_local[10i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 17i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 17i)], NT_matmul_intermediate_reindex_pad_local[10i]);
      NT_matmul_intermediate_reindex_pad_local[11i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 17i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 25i)], NT_matmul_intermediate_reindex_pad_local[11i]);
      NT_matmul_intermediate_reindex_pad_local[12i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 25i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 1i)], NT_matmul_intermediate_reindex_pad_local[12i]);
      NT_matmul_intermediate_reindex_pad_local[13i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 25i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 9i)], NT_matmul_intermediate_reindex_pad_local[13i]);
      NT_matmul_intermediate_reindex_pad_local[14i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 25i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 17i)], NT_matmul_intermediate_reindex_pad_local[14i]);
      NT_matmul_intermediate_reindex_pad_local[15i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 25i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 25i)], NT_matmul_intermediate_reindex_pad_local[15i]);
      NT_matmul_intermediate_reindex_pad_local[0i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 2i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 2i)], NT_matmul_intermediate_reindex_pad_local[0i]);
      NT_matmul_intermediate_reindex_pad_local[1i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 2i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 10i)], NT_matmul_intermediate_reindex_pad_local[1i]);
      NT_matmul_intermediate_reindex_pad_local[2i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 2i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 18i)], NT_matmul_intermediate_reindex_pad_local[2i]);
      NT_matmul_intermediate_reindex_pad_local[3i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 2i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 26i)], NT_matmul_intermediate_reindex_pad_local[3i]);
      NT_matmul_intermediate_reindex_pad_local[4i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 10i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 2i)], NT_matmul_intermediate_reindex_pad_local[4i]);
      NT_matmul_intermediate_reindex_pad_local[5i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 10i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 10i)], NT_matmul_intermediate_reindex_pad_local[5i]);
      NT_matmul_intermediate_reindex_pad_local[6i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 10i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 18i)], NT_matmul_intermediate_reindex_pad_local[6i]);
      NT_matmul_intermediate_reindex_pad_local[7i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 10i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 26i)], NT_matmul_intermediate_reindex_pad_local[7i]);
      NT_matmul_intermediate_reindex_pad_local[8i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 18i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 2i)], NT_matmul_intermediate_reindex_pad_local[8i]);
      NT_matmul_intermediate_reindex_pad_local[9i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 18i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 10i)], NT_matmul_intermediate_reindex_pad_local[9i]);
      NT_matmul_intermediate_reindex_pad_local[10i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 18i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 18i)], NT_matmul_intermediate_reindex_pad_local[10i]);
      NT_matmul_intermediate_reindex_pad_local[11i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 18i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 26i)], NT_matmul_intermediate_reindex_pad_local[11i]);
      NT_matmul_intermediate_reindex_pad_local[12i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 26i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 2i)], NT_matmul_intermediate_reindex_pad_local[12i]);
      NT_matmul_intermediate_reindex_pad_local[13i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 26i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 10i)], NT_matmul_intermediate_reindex_pad_local[13i]);
      NT_matmul_intermediate_reindex_pad_local[14i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 26i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 18i)], NT_matmul_intermediate_reindex_pad_local[14i]);
      NT_matmul_intermediate_reindex_pad_local[15i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 26i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 26i)], NT_matmul_intermediate_reindex_pad_local[15i]);
      NT_matmul_intermediate_reindex_pad_local[0i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 3i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 3i)], NT_matmul_intermediate_reindex_pad_local[0i]);
      NT_matmul_intermediate_reindex_pad_local[1i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 3i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 11i)], NT_matmul_intermediate_reindex_pad_local[1i]);
      NT_matmul_intermediate_reindex_pad_local[2i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 3i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 19i)], NT_matmul_intermediate_reindex_pad_local[2i]);
      NT_matmul_intermediate_reindex_pad_local[3i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 3i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 27i)], NT_matmul_intermediate_reindex_pad_local[3i]);
      NT_matmul_intermediate_reindex_pad_local[4i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 11i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 3i)], NT_matmul_intermediate_reindex_pad_local[4i]);
      NT_matmul_intermediate_reindex_pad_local[5i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 11i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 11i)], NT_matmul_intermediate_reindex_pad_local[5i]);
      NT_matmul_intermediate_reindex_pad_local[6i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 11i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 19i)], NT_matmul_intermediate_reindex_pad_local[6i]);
      NT_matmul_intermediate_reindex_pad_local[7i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 11i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 27i)], NT_matmul_intermediate_reindex_pad_local[7i]);
      NT_matmul_intermediate_reindex_pad_local[8i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 19i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 3i)], NT_matmul_intermediate_reindex_pad_local[8i]);
      NT_matmul_intermediate_reindex_pad_local[9i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 19i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 11i)], NT_matmul_intermediate_reindex_pad_local[9i]);
      NT_matmul_intermediate_reindex_pad_local[10i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 19i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 19i)], NT_matmul_intermediate_reindex_pad_local[10i]);
      NT_matmul_intermediate_reindex_pad_local[11i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 19i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 27i)], NT_matmul_intermediate_reindex_pad_local[11i]);
      NT_matmul_intermediate_reindex_pad_local[12i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 27i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 3i)], NT_matmul_intermediate_reindex_pad_local[12i]);
      NT_matmul_intermediate_reindex_pad_local[13i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 27i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 11i)], NT_matmul_intermediate_reindex_pad_local[13i]);
      NT_matmul_intermediate_reindex_pad_local[14i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 27i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 19i)], NT_matmul_intermediate_reindex_pad_local[14i]);
      NT_matmul_intermediate_reindex_pad_local[15i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 27i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 27i)], NT_matmul_intermediate_reindex_pad_local[15i]);
      NT_matmul_intermediate_reindex_pad_local[0i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 4i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 4i)], NT_matmul_intermediate_reindex_pad_local[0i]);
      NT_matmul_intermediate_reindex_pad_local[1i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 4i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 12i)], NT_matmul_intermediate_reindex_pad_local[1i]);
      NT_matmul_intermediate_reindex_pad_local[2i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 4i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 20i)], NT_matmul_intermediate_reindex_pad_local[2i]);
      NT_matmul_intermediate_reindex_pad_local[3i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 4i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 28i)], NT_matmul_intermediate_reindex_pad_local[3i]);
      NT_matmul_intermediate_reindex_pad_local[4i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 12i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 4i)], NT_matmul_intermediate_reindex_pad_local[4i]);
      NT_matmul_intermediate_reindex_pad_local[5i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 12i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 12i)], NT_matmul_intermediate_reindex_pad_local[5i]);
      NT_matmul_intermediate_reindex_pad_local[6i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 12i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 20i)], NT_matmul_intermediate_reindex_pad_local[6i]);
      NT_matmul_intermediate_reindex_pad_local[7i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 12i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 28i)], NT_matmul_intermediate_reindex_pad_local[7i]);
      NT_matmul_intermediate_reindex_pad_local[8i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 20i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 4i)], NT_matmul_intermediate_reindex_pad_local[8i]);
      NT_matmul_intermediate_reindex_pad_local[9i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 20i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 12i)], NT_matmul_intermediate_reindex_pad_local[9i]);
      NT_matmul_intermediate_reindex_pad_local[10i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 20i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 20i)], NT_matmul_intermediate_reindex_pad_local[10i]);
      NT_matmul_intermediate_reindex_pad_local[11i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 20i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 28i)], NT_matmul_intermediate_reindex_pad_local[11i]);
      NT_matmul_intermediate_reindex_pad_local[12i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 28i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 4i)], NT_matmul_intermediate_reindex_pad_local[12i]);
      NT_matmul_intermediate_reindex_pad_local[13i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 28i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 12i)], NT_matmul_intermediate_reindex_pad_local[13i]);
      NT_matmul_intermediate_reindex_pad_local[14i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 28i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 20i)], NT_matmul_intermediate_reindex_pad_local[14i]);
      NT_matmul_intermediate_reindex_pad_local[15i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 28i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 28i)], NT_matmul_intermediate_reindex_pad_local[15i]);
      NT_matmul_intermediate_reindex_pad_local[0i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 5i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 5i)], NT_matmul_intermediate_reindex_pad_local[0i]);
      NT_matmul_intermediate_reindex_pad_local[1i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 5i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 13i)], NT_matmul_intermediate_reindex_pad_local[1i]);
      NT_matmul_intermediate_reindex_pad_local[2i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 5i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 21i)], NT_matmul_intermediate_reindex_pad_local[2i]);
      NT_matmul_intermediate_reindex_pad_local[3i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 5i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 29i)], NT_matmul_intermediate_reindex_pad_local[3i]);
      NT_matmul_intermediate_reindex_pad_local[4i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 13i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 5i)], NT_matmul_intermediate_reindex_pad_local[4i]);
      NT_matmul_intermediate_reindex_pad_local[5i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 13i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 13i)], NT_matmul_intermediate_reindex_pad_local[5i]);
      NT_matmul_intermediate_reindex_pad_local[6i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 13i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 21i)], NT_matmul_intermediate_reindex_pad_local[6i]);
      NT_matmul_intermediate_reindex_pad_local[7i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 13i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 29i)], NT_matmul_intermediate_reindex_pad_local[7i]);
      NT_matmul_intermediate_reindex_pad_local[8i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 21i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 5i)], NT_matmul_intermediate_reindex_pad_local[8i]);
      NT_matmul_intermediate_reindex_pad_local[9i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 21i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 13i)], NT_matmul_intermediate_reindex_pad_local[9i]);
      NT_matmul_intermediate_reindex_pad_local[10i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 21i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 21i)], NT_matmul_intermediate_reindex_pad_local[10i]);
      NT_matmul_intermediate_reindex_pad_local[11i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 21i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 29i)], NT_matmul_intermediate_reindex_pad_local[11i]);
      NT_matmul_intermediate_reindex_pad_local[12i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 29i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 5i)], NT_matmul_intermediate_reindex_pad_local[12i]);
      NT_matmul_intermediate_reindex_pad_local[13i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 29i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 13i)], NT_matmul_intermediate_reindex_pad_local[13i]);
      NT_matmul_intermediate_reindex_pad_local[14i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 29i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 21i)], NT_matmul_intermediate_reindex_pad_local[14i]);
      NT_matmul_intermediate_reindex_pad_local[15i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 29i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 29i)], NT_matmul_intermediate_reindex_pad_local[15i]);
      NT_matmul_intermediate_reindex_pad_local[0i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 6i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 6i)], NT_matmul_intermediate_reindex_pad_local[0i]);
      NT_matmul_intermediate_reindex_pad_local[1i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 6i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 14i)], NT_matmul_intermediate_reindex_pad_local[1i]);
      NT_matmul_intermediate_reindex_pad_local[2i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 6i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 22i)], NT_matmul_intermediate_reindex_pad_local[2i]);
      NT_matmul_intermediate_reindex_pad_local[3i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 6i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 30i)], NT_matmul_intermediate_reindex_pad_local[3i]);
      NT_matmul_intermediate_reindex_pad_local[4i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 14i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 6i)], NT_matmul_intermediate_reindex_pad_local[4i]);
      NT_matmul_intermediate_reindex_pad_local[5i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 14i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 14i)], NT_matmul_intermediate_reindex_pad_local[5i]);
      NT_matmul_intermediate_reindex_pad_local[6i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 14i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 22i)], NT_matmul_intermediate_reindex_pad_local[6i]);
      NT_matmul_intermediate_reindex_pad_local[7i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 14i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 30i)], NT_matmul_intermediate_reindex_pad_local[7i]);
      NT_matmul_intermediate_reindex_pad_local[8i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 22i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 6i)], NT_matmul_intermediate_reindex_pad_local[8i]);
      NT_matmul_intermediate_reindex_pad_local[9i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 22i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 14i)], NT_matmul_intermediate_reindex_pad_local[9i]);
      NT_matmul_intermediate_reindex_pad_local[10i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 22i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 22i)], NT_matmul_intermediate_reindex_pad_local[10i]);
      NT_matmul_intermediate_reindex_pad_local[11i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 22i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 30i)], NT_matmul_intermediate_reindex_pad_local[11i]);
      NT_matmul_intermediate_reindex_pad_local[12i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 30i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 6i)], NT_matmul_intermediate_reindex_pad_local[12i]);
      NT_matmul_intermediate_reindex_pad_local[13i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 30i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 14i)], NT_matmul_intermediate_reindex_pad_local[13i]);
      NT_matmul_intermediate_reindex_pad_local[14i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 30i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 22i)], NT_matmul_intermediate_reindex_pad_local[14i]);
      NT_matmul_intermediate_reindex_pad_local[15i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 30i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 30i)], NT_matmul_intermediate_reindex_pad_local[15i]);
      NT_matmul_intermediate_reindex_pad_local[0i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 7i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 7i)], NT_matmul_intermediate_reindex_pad_local[0i]);
      NT_matmul_intermediate_reindex_pad_local[1i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 7i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 15i)], NT_matmul_intermediate_reindex_pad_local[1i]);
      NT_matmul_intermediate_reindex_pad_local[2i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 7i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 23i)], NT_matmul_intermediate_reindex_pad_local[2i]);
      NT_matmul_intermediate_reindex_pad_local[3i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 7i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 31i)], NT_matmul_intermediate_reindex_pad_local[3i]);
      NT_matmul_intermediate_reindex_pad_local[4i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 15i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 7i)], NT_matmul_intermediate_reindex_pad_local[4i]);
      NT_matmul_intermediate_reindex_pad_local[5i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 15i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 15i)], NT_matmul_intermediate_reindex_pad_local[5i]);
      NT_matmul_intermediate_reindex_pad_local[6i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 15i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 23i)], NT_matmul_intermediate_reindex_pad_local[6i]);
      NT_matmul_intermediate_reindex_pad_local[7i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 15i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 31i)], NT_matmul_intermediate_reindex_pad_local[7i]);
      NT_matmul_intermediate_reindex_pad_local[8i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 23i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 7i)], NT_matmul_intermediate_reindex_pad_local[8i]);
      NT_matmul_intermediate_reindex_pad_local[9i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 23i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 15i)], NT_matmul_intermediate_reindex_pad_local[9i]);
      NT_matmul_intermediate_reindex_pad_local[10i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 23i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 23i)], NT_matmul_intermediate_reindex_pad_local[10i]);
      NT_matmul_intermediate_reindex_pad_local[11i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 23i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 31i)], NT_matmul_intermediate_reindex_pad_local[11i]);
      NT_matmul_intermediate_reindex_pad_local[12i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 31i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 7i)], NT_matmul_intermediate_reindex_pad_local[12i]);
      NT_matmul_intermediate_reindex_pad_local[13i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 31i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 15i)], NT_matmul_intermediate_reindex_pad_local[13i]);
      NT_matmul_intermediate_reindex_pad_local[14i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 31i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 23i)], NT_matmul_intermediate_reindex_pad_local[14i]);
      NT_matmul_intermediate_reindex_pad_local[15i] = fma(reshape195_reindex_pad_shared[((i32(threadIdx.x) * 32i) + 31i)], dequantize_reindex_shared[((i32(threadIdx.y) * 32i) + 31i)], NT_matmul_intermediate_reindex_pad_local[15i]);
    }
    if (((v__1 * 32i) + (i32(threadIdx.x) * 4i)) < podArgs.seq_len) {
      T_add[((((v__1 * 28672i) + (i32(threadIdx.x) * 3584i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i))] = (NT_matmul_intermediate_reindex_pad_local[0i] + input_embeds[((((v__1 * 28672i) + (i32(threadIdx.x) * 3584i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i))]);
    }
    if (((v__1 * 32i) + (i32(threadIdx.x) * 4i)) < podArgs.seq_len) {
      T_add[(((((v__1 * 28672i) + (i32(threadIdx.x) * 3584i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 1i)] = (NT_matmul_intermediate_reindex_pad_local[1i] + input_embeds[(((((v__1 * 28672i) + (i32(threadIdx.x) * 3584i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 1i)]);
    }
    if (((v__1 * 32i) + (i32(threadIdx.x) * 4i)) < podArgs.seq_len) {
      T_add[(((((v__1 * 28672i) + (i32(threadIdx.x) * 3584i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 2i)] = (NT_matmul_intermediate_reindex_pad_local[2i] + input_embeds[(((((v__1 * 28672i) + (i32(threadIdx.x) * 3584i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 2i)]);
    }
    if (((v__1 * 32i) + (i32(threadIdx.x) * 4i)) < podArgs.seq_len) {
      T_add[(((((v__1 * 28672i) + (i32(threadIdx.x) * 3584i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 3i)] = (NT_matmul_intermediate_reindex_pad_local[3i] + input_embeds[(((((v__1 * 28672i) + (i32(threadIdx.x) * 3584i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 3i)]);
    }
    if ((((v__1 * 32i) + (i32(threadIdx.x) * 4i)) + 1i) < podArgs.seq_len) {
      T_add[(((((v__1 * 28672i) + (i32(threadIdx.x) * 3584i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 896i)] = (NT_matmul_intermediate_reindex_pad_local[4i] + input_embeds[(((((v__1 * 28672i) + (i32(threadIdx.x) * 3584i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 896i)]);
    }
    if ((((v__1 * 32i) + (i32(threadIdx.x) * 4i)) + 1i) < podArgs.seq_len) {
      T_add[(((((v__1 * 28672i) + (i32(threadIdx.x) * 3584i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 897i)] = (NT_matmul_intermediate_reindex_pad_local[5i] + input_embeds[(((((v__1 * 28672i) + (i32(threadIdx.x) * 3584i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 897i)]);
    }
    if ((((v__1 * 32i) + (i32(threadIdx.x) * 4i)) + 1i) < podArgs.seq_len) {
      T_add[(((((v__1 * 28672i) + (i32(threadIdx.x) * 3584i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 898i)] = (NT_matmul_intermediate_reindex_pad_local[6i] + input_embeds[(((((v__1 * 28672i) + (i32(threadIdx.x) * 3584i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 898i)]);
    }
    if ((((v__1 * 32i) + (i32(threadIdx.x) * 4i)) + 1i) < podArgs.seq_len) {
      T_add[(((((v__1 * 28672i) + (i32(threadIdx.x) * 3584i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 899i)] = (NT_matmul_intermediate_reindex_pad_local[7i] + input_embeds[(((((v__1 * 28672i) + (i32(threadIdx.x) * 3584i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 899i)]);
    }
    if ((((v__1 * 32i) + (i32(threadIdx.x) * 4i)) + 2i) < podArgs.seq_len) {
      T_add[(((((v__1 * 28672i) + (i32(threadIdx.x) * 3584i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 1792i)] = (NT_matmul_intermediate_reindex_pad_local[8i] + input_embeds[(((((v__1 * 28672i) + (i32(threadIdx.x) * 3584i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 1792i)]);
    }
    if ((((v__1 * 32i) + (i32(threadIdx.x) * 4i)) + 2i) < podArgs.seq_len) {
      T_add[(((((v__1 * 28672i) + (i32(threadIdx.x) * 3584i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 1793i)] = (NT_matmul_intermediate_reindex_pad_local[9i] + input_embeds[(((((v__1 * 28672i) + (i32(threadIdx.x) * 3584i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 1793i)]);
    }
    if ((((v__1 * 32i) + (i32(threadIdx.x) * 4i)) + 2i) < podArgs.seq_len) {
      T_add[(((((v__1 * 28672i) + (i32(threadIdx.x) * 3584i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 1794i)] = (NT_matmul_intermediate_reindex_pad_local[10i] + input_embeds[(((((v__1 * 28672i) + (i32(threadIdx.x) * 3584i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 1794i)]);
    }
    if ((((v__1 * 32i) + (i32(threadIdx.x) * 4i)) + 2i) < podArgs.seq_len) {
      T_add[(((((v__1 * 28672i) + (i32(threadIdx.x) * 3584i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 1795i)] = (NT_matmul_intermediate_reindex_pad_local[11i] + input_embeds[(((((v__1 * 28672i) + (i32(threadIdx.x) * 3584i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 1795i)]);
    }
    if ((((v__1 * 32i) + (i32(threadIdx.x) * 4i)) + 3i) < podArgs.seq_len) {
      T_add[(((((v__1 * 28672i) + (i32(threadIdx.x) * 3584i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 2688i)] = (NT_matmul_intermediate_reindex_pad_local[12i] + input_embeds[(((((v__1 * 28672i) + (i32(threadIdx.x) * 3584i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 2688i)]);
    }
    if ((((v__1 * 32i) + (i32(threadIdx.x) * 4i)) + 3i) < podArgs.seq_len) {
      T_add[(((((v__1 * 28672i) + (i32(threadIdx.x) * 3584i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 2689i)] = (NT_matmul_intermediate_reindex_pad_local[13i] + input_embeds[(((((v__1 * 28672i) + (i32(threadIdx.x) * 3584i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 2689i)]);
    }
    if ((((v__1 * 32i) + (i32(threadIdx.x) * 4i)) + 3i) < podArgs.seq_len) {
      T_add[(((((v__1 * 28672i) + (i32(threadIdx.x) * 3584i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 2690i)] = (NT_matmul_intermediate_reindex_pad_local[14i] + input_embeds[(((((v__1 * 28672i) + (i32(threadIdx.x) * 3584i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 2690i)]);
    }
    if ((((v__1 * 32i) + (i32(threadIdx.x) * 4i)) + 3i) < podArgs.seq_len) {
      T_add[(((((v__1 * 28672i) + (i32(threadIdx.x) * 3584i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 2691i)] = (NT_matmul_intermediate_reindex_pad_local[15i] + input_embeds[(((((v__1 * 28672i) + (i32(threadIdx.x) * 3584i)) + (i32(blockIdx.y) * 32i)) + (i32(threadIdx.y) * 4i)) + 2691i)]);
    }
  }
}

