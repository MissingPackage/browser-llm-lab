//----------------------------------------
// Function: fused_dequantize4_fused_NT_matmul13_add5_kernel
//----------------------------------------
@group(0) @binding(0) var<storage, read_write> T_add : array<f32>;
@group(0) @binding(1) var<storage, read> lv291 : array<f32>;
@group(0) @binding(2) var<storage, read> lv292 : array<f32>;
@group(0) @binding(3) var<storage, read> model_layers_0_mlp_down_proj_q_scale2 : array<f32>;
@group(0) @binding(4) var<storage, read> model_layers_0_mlp_down_proj_q_weight2 : array<u32>;

struct PODArgs {
  packGridDimX: u32
}
@group(0) @binding(5) var<uniform> podArgs : PODArgs;

var<workgroup> red_buf0 : array<f32, 64>;
@compute @workgroup_size(64, 1, 1)
fn fused_dequantize4_fused_NT_matmul13_add5_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  var NT_matmul_rf_local : array<f32, 1>;
  var model_layers_0_mlp_down_proj_q_weight2_local : array<u32, 1>;
  var NT_matmul_rf_local_1 : array<f32, 1>;
  NT_matmul_rf_local[0i] = 0.000000e+00f;
  for (var ax1_0_fused_ax1_1_fused_0 : i32 = 0i; ax1_0_fused_ax1_1_fused_0 < 10i; ax1_0_fused_ax1_1_fused_0++) {
    if (((ax1_0_fused_ax1_1_fused_0 * 2i) + (i32(threadIdx.x)>>5u)) < 19i) {
      model_layers_0_mlp_down_proj_q_weight2_local[0i] = model_layers_0_mlp_down_proj_q_weight2[(((v__1 * 608i) + (ax1_0_fused_ax1_1_fused_0 * 64i)) + i32(threadIdx.x))];
    }
    if (((ax1_0_fused_ax1_1_fused_0 * 2i) + (i32(threadIdx.x)>>5u)) < 19i) {
      NT_matmul_rf_local[0i] = fma(lv292[((ax1_0_fused_ax1_1_fused_0 * 512i) + (i32(threadIdx.x) * 8i))], ((f32(((model_layers_0_mlp_down_proj_q_weight2_local[0i]>>0u) & 15u)) - 7.000000e+00f) * model_layers_0_mlp_down_proj_q_scale2[(((v__1 * 152i) + (ax1_0_fused_ax1_1_fused_0 * 16i)) + (i32(threadIdx.x)>>2u))]), NT_matmul_rf_local[0i]);
    }
    if (((ax1_0_fused_ax1_1_fused_0 * 2i) + (i32(threadIdx.x)>>5u)) < 19i) {
      NT_matmul_rf_local[0i] = fma(lv292[(((ax1_0_fused_ax1_1_fused_0 * 512i) + (i32(threadIdx.x) * 8i)) + 1i)], ((f32(((model_layers_0_mlp_down_proj_q_weight2_local[0i]>>4u) & 15u)) - 7.000000e+00f) * model_layers_0_mlp_down_proj_q_scale2[(((v__1 * 152i) + (ax1_0_fused_ax1_1_fused_0 * 16i)) + (i32(threadIdx.x)>>2u))]), NT_matmul_rf_local[0i]);
    }
    if (((ax1_0_fused_ax1_1_fused_0 * 2i) + (i32(threadIdx.x)>>5u)) < 19i) {
      NT_matmul_rf_local[0i] = fma(lv292[(((ax1_0_fused_ax1_1_fused_0 * 512i) + (i32(threadIdx.x) * 8i)) + 2i)], ((f32(((model_layers_0_mlp_down_proj_q_weight2_local[0i]>>8u) & 15u)) - 7.000000e+00f) * model_layers_0_mlp_down_proj_q_scale2[(((v__1 * 152i) + (ax1_0_fused_ax1_1_fused_0 * 16i)) + (i32(threadIdx.x)>>2u))]), NT_matmul_rf_local[0i]);
    }
    if (((ax1_0_fused_ax1_1_fused_0 * 2i) + (i32(threadIdx.x)>>5u)) < 19i) {
      NT_matmul_rf_local[0i] = fma(lv292[(((ax1_0_fused_ax1_1_fused_0 * 512i) + (i32(threadIdx.x) * 8i)) + 3i)], ((f32(((model_layers_0_mlp_down_proj_q_weight2_local[0i]>>12u) & 15u)) - 7.000000e+00f) * model_layers_0_mlp_down_proj_q_scale2[(((v__1 * 152i) + (ax1_0_fused_ax1_1_fused_0 * 16i)) + (i32(threadIdx.x)>>2u))]), NT_matmul_rf_local[0i]);
    }
    if (((ax1_0_fused_ax1_1_fused_0 * 2i) + (i32(threadIdx.x)>>5u)) < 19i) {
      NT_matmul_rf_local[0i] = fma(lv292[(((ax1_0_fused_ax1_1_fused_0 * 512i) + (i32(threadIdx.x) * 8i)) + 4i)], ((f32(((model_layers_0_mlp_down_proj_q_weight2_local[0i]>>16u) & 15u)) - 7.000000e+00f) * model_layers_0_mlp_down_proj_q_scale2[(((v__1 * 152i) + (ax1_0_fused_ax1_1_fused_0 * 16i)) + (i32(threadIdx.x)>>2u))]), NT_matmul_rf_local[0i]);
    }
    if (((ax1_0_fused_ax1_1_fused_0 * 2i) + (i32(threadIdx.x)>>5u)) < 19i) {
      NT_matmul_rf_local[0i] = fma(lv292[(((ax1_0_fused_ax1_1_fused_0 * 512i) + (i32(threadIdx.x) * 8i)) + 5i)], ((f32(((model_layers_0_mlp_down_proj_q_weight2_local[0i]>>20u) & 15u)) - 7.000000e+00f) * model_layers_0_mlp_down_proj_q_scale2[(((v__1 * 152i) + (ax1_0_fused_ax1_1_fused_0 * 16i)) + (i32(threadIdx.x)>>2u))]), NT_matmul_rf_local[0i]);
    }
    if (((ax1_0_fused_ax1_1_fused_0 * 2i) + (i32(threadIdx.x)>>5u)) < 19i) {
      NT_matmul_rf_local[0i] = fma(lv292[(((ax1_0_fused_ax1_1_fused_0 * 512i) + (i32(threadIdx.x) * 8i)) + 6i)], ((f32(((model_layers_0_mlp_down_proj_q_weight2_local[0i]>>24u) & 15u)) - 7.000000e+00f) * model_layers_0_mlp_down_proj_q_scale2[(((v__1 * 152i) + (ax1_0_fused_ax1_1_fused_0 * 16i)) + (i32(threadIdx.x)>>2u))]), NT_matmul_rf_local[0i]);
    }
    if (((ax1_0_fused_ax1_1_fused_0 * 2i) + (i32(threadIdx.x)>>5u)) < 19i) {
      NT_matmul_rf_local[0i] = fma(lv292[(((ax1_0_fused_ax1_1_fused_0 * 512i) + (i32(threadIdx.x) * 8i)) + 7i)], ((f32(((model_layers_0_mlp_down_proj_q_weight2_local[0i]>>28u) & 15u)) - 7.000000e+00f) * model_layers_0_mlp_down_proj_q_scale2[(((v__1 * 152i) + (ax1_0_fused_ax1_1_fused_0 * 16i)) + (i32(threadIdx.x)>>2u))]), NT_matmul_rf_local[0i]);
    }
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
    T_add[v__1] = (red_buf0[0i] + lv291[v__1]);
  }
}

