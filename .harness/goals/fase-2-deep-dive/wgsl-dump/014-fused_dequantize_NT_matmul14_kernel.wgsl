//----------------------------------------
// Function: fused_dequantize_NT_matmul14_kernel
//----------------------------------------
@group(0) @binding(0) var<storage, read_write> NT_matmul : array<f32>;
@group(0) @binding(1) var<storage, read> model_embed_tokens_q_scale2 : array<f32>;
@group(0) @binding(2) var<storage, read> model_embed_tokens_q_weight2 : array<u32>;
@group(0) @binding(3) var<storage, read> rms_norm97 : array<f32>;

struct PODArgs {
  packGridDimX: u32
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

var<workgroup> red_buf0 : array<f32, 64>;
@compute @workgroup_size(64, 1, 1)
fn fused_dequantize_NT_matmul14_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  var NT_matmul_rf_local : array<f32, 1>;
  var model_embed_tokens_q_weight2_local : array<u32, 1>;
  var NT_matmul_rf_local_1 : array<f32, 1>;
  NT_matmul_rf_local[0i] = 0.000000e+00f;
  model_embed_tokens_q_weight2_local[0i] = model_embed_tokens_q_weight2[((v__1 * 112i) + i32(threadIdx.x))];
  NT_matmul_rf_local[0i] = fma(rms_norm97[(i32(threadIdx.x) * 8i)], ((f32(((model_embed_tokens_q_weight2_local[0i]>>0u) & 15u)) - 7.000000e+00f) * model_embed_tokens_q_scale2[((v__1 * 28i) + (i32(threadIdx.x)>>2u))]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm97[((i32(threadIdx.x) * 8i) + 1i)], ((f32(((model_embed_tokens_q_weight2_local[0i]>>4u) & 15u)) - 7.000000e+00f) * model_embed_tokens_q_scale2[((v__1 * 28i) + (i32(threadIdx.x)>>2u))]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm97[((i32(threadIdx.x) * 8i) + 2i)], ((f32(((model_embed_tokens_q_weight2_local[0i]>>8u) & 15u)) - 7.000000e+00f) * model_embed_tokens_q_scale2[((v__1 * 28i) + (i32(threadIdx.x)>>2u))]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm97[((i32(threadIdx.x) * 8i) + 3i)], ((f32(((model_embed_tokens_q_weight2_local[0i]>>12u) & 15u)) - 7.000000e+00f) * model_embed_tokens_q_scale2[((v__1 * 28i) + (i32(threadIdx.x)>>2u))]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm97[((i32(threadIdx.x) * 8i) + 4i)], ((f32(((model_embed_tokens_q_weight2_local[0i]>>16u) & 15u)) - 7.000000e+00f) * model_embed_tokens_q_scale2[((v__1 * 28i) + (i32(threadIdx.x)>>2u))]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm97[((i32(threadIdx.x) * 8i) + 5i)], ((f32(((model_embed_tokens_q_weight2_local[0i]>>20u) & 15u)) - 7.000000e+00f) * model_embed_tokens_q_scale2[((v__1 * 28i) + (i32(threadIdx.x)>>2u))]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm97[((i32(threadIdx.x) * 8i) + 6i)], ((f32(((model_embed_tokens_q_weight2_local[0i]>>24u) & 15u)) - 7.000000e+00f) * model_embed_tokens_q_scale2[((v__1 * 28i) + (i32(threadIdx.x)>>2u))]), NT_matmul_rf_local[0i]);
  NT_matmul_rf_local[0i] = fma(rms_norm97[((i32(threadIdx.x) * 8i) + 7i)], ((f32(((model_embed_tokens_q_weight2_local[0i]>>28u) & 15u)) - 7.000000e+00f) * model_embed_tokens_q_scale2[((v__1 * 28i) + (i32(threadIdx.x)>>2u))]), NT_matmul_rf_local[0i]);
  if (i32(threadIdx.x) < 48i) {
    model_embed_tokens_q_weight2_local[0i] = model_embed_tokens_q_weight2[(((v__1 * 112i) + i32(threadIdx.x)) + 64i)];
  }
  if (i32(threadIdx.x) < 48i) {
    NT_matmul_rf_local[0i] = fma(rms_norm97[((i32(threadIdx.x) * 8i) + 512i)], ((f32(((model_embed_tokens_q_weight2_local[0i]>>0u) & 15u)) - 7.000000e+00f) * model_embed_tokens_q_scale2[(((v__1 * 28i) + (i32(threadIdx.x)>>2u)) + 16i)]), NT_matmul_rf_local[0i]);
  }
  if (i32(threadIdx.x) < 48i) {
    NT_matmul_rf_local[0i] = fma(rms_norm97[((i32(threadIdx.x) * 8i) + 513i)], ((f32(((model_embed_tokens_q_weight2_local[0i]>>4u) & 15u)) - 7.000000e+00f) * model_embed_tokens_q_scale2[(((v__1 * 28i) + (i32(threadIdx.x)>>2u)) + 16i)]), NT_matmul_rf_local[0i]);
  }
  if (i32(threadIdx.x) < 48i) {
    NT_matmul_rf_local[0i] = fma(rms_norm97[((i32(threadIdx.x) * 8i) + 514i)], ((f32(((model_embed_tokens_q_weight2_local[0i]>>8u) & 15u)) - 7.000000e+00f) * model_embed_tokens_q_scale2[(((v__1 * 28i) + (i32(threadIdx.x)>>2u)) + 16i)]), NT_matmul_rf_local[0i]);
  }
  if (i32(threadIdx.x) < 48i) {
    NT_matmul_rf_local[0i] = fma(rms_norm97[((i32(threadIdx.x) * 8i) + 515i)], ((f32(((model_embed_tokens_q_weight2_local[0i]>>12u) & 15u)) - 7.000000e+00f) * model_embed_tokens_q_scale2[(((v__1 * 28i) + (i32(threadIdx.x)>>2u)) + 16i)]), NT_matmul_rf_local[0i]);
  }
  if (i32(threadIdx.x) < 48i) {
    NT_matmul_rf_local[0i] = fma(rms_norm97[((i32(threadIdx.x) * 8i) + 516i)], ((f32(((model_embed_tokens_q_weight2_local[0i]>>16u) & 15u)) - 7.000000e+00f) * model_embed_tokens_q_scale2[(((v__1 * 28i) + (i32(threadIdx.x)>>2u)) + 16i)]), NT_matmul_rf_local[0i]);
  }
  if (i32(threadIdx.x) < 48i) {
    NT_matmul_rf_local[0i] = fma(rms_norm97[((i32(threadIdx.x) * 8i) + 517i)], ((f32(((model_embed_tokens_q_weight2_local[0i]>>20u) & 15u)) - 7.000000e+00f) * model_embed_tokens_q_scale2[(((v__1 * 28i) + (i32(threadIdx.x)>>2u)) + 16i)]), NT_matmul_rf_local[0i]);
  }
  if (i32(threadIdx.x) < 48i) {
    NT_matmul_rf_local[0i] = fma(rms_norm97[((i32(threadIdx.x) * 8i) + 518i)], ((f32(((model_embed_tokens_q_weight2_local[0i]>>24u) & 15u)) - 7.000000e+00f) * model_embed_tokens_q_scale2[(((v__1 * 28i) + (i32(threadIdx.x)>>2u)) + 16i)]), NT_matmul_rf_local[0i]);
  }
  if (i32(threadIdx.x) < 48i) {
    NT_matmul_rf_local[0i] = fma(rms_norm97[((i32(threadIdx.x) * 8i) + 519i)], ((f32(((model_embed_tokens_q_weight2_local[0i]>>28u) & 15u)) - 7.000000e+00f) * model_embed_tokens_q_scale2[(((v__1 * 28i) + (i32(threadIdx.x)>>2u)) + 16i)]), NT_matmul_rf_local[0i]);
  }
  NT_matmul_rf_local_1[0i] = 0.000000e+00f;
  NT_matmul_rf_local_1[0i] = (NT_matmul_rf_local_1[0i] + NT_matmul_rf_local[0i]);
  workgroupBarrier();
  red_buf0[i32(threadIdx.x)] = NT_matmul_rf_local_1[0i];
  workgroupBarrier();
  if (i32(threadIdx.x) < 32i) {
    red_buf0[i32(threadIdx.x)] = (red_buf0[i32(threadIdx.x)] + red_buf0[(i32(threadIdx.x) + 32i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 16i) {
    red_buf0[i32(threadIdx.x)] = (red_buf0[i32(threadIdx.x)] + red_buf0[(i32(threadIdx.x) + 16i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 8i) {
    red_buf0[i32(threadIdx.x)] = (red_buf0[i32(threadIdx.x)] + red_buf0[(i32(threadIdx.x) + 8i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 4i) {
    red_buf0[i32(threadIdx.x)] = (red_buf0[i32(threadIdx.x)] + red_buf0[(i32(threadIdx.x) + 4i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 2i) {
    red_buf0[i32(threadIdx.x)] = (red_buf0[i32(threadIdx.x)] + red_buf0[(i32(threadIdx.x) + 2i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) < 1i) {
    red_buf0[i32(threadIdx.x)] = (red_buf0[i32(threadIdx.x)] + red_buf0[(i32(threadIdx.x) + 1i)]);
  }
  workgroupBarrier();
  if (i32(threadIdx.x) == 0i) {
    NT_matmul[v__1] = red_buf0[0i];
  }
}

