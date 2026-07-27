//----------------------------------------
// Function: rms_norm2_kernel
//----------------------------------------
@group(0) @binding(0) var<storage, read_write> T_cast : array<f32>;
@group(0) @binding(1) var<storage, read> input_embed : array<f32>;
@group(0) @binding(2) var<storage, read> model_layers_0_input_layernorm_weight2 : array<f32>;

struct PODArgs {
  packGridDimX: u32
}
@group(0) @binding(3) var<uniform> podArgs : PODArgs;

var<workgroup> red_buf0 : array<f32, 64>;
var<workgroup> T_multiply_red_shared : array<f32, 1>;
@compute @workgroup_size(64, 1, 1)
fn rms_norm2_kernel(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  var T_multiply_red_rf_local : array<f32, 1>;
  T_multiply_red_rf_local[0i] = 0.000000e+00f;
  T_multiply_red_rf_local[0i] = fma(input_embed[i32(threadIdx.x)], input_embed[i32(threadIdx.x)], T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(input_embed[(i32(threadIdx.x) + 64i)], input_embed[(i32(threadIdx.x) + 64i)], T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(input_embed[(i32(threadIdx.x) + 128i)], input_embed[(i32(threadIdx.x) + 128i)], T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(input_embed[(i32(threadIdx.x) + 192i)], input_embed[(i32(threadIdx.x) + 192i)], T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(input_embed[(i32(threadIdx.x) + 256i)], input_embed[(i32(threadIdx.x) + 256i)], T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(input_embed[(i32(threadIdx.x) + 320i)], input_embed[(i32(threadIdx.x) + 320i)], T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(input_embed[(i32(threadIdx.x) + 384i)], input_embed[(i32(threadIdx.x) + 384i)], T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(input_embed[(i32(threadIdx.x) + 448i)], input_embed[(i32(threadIdx.x) + 448i)], T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(input_embed[(i32(threadIdx.x) + 512i)], input_embed[(i32(threadIdx.x) + 512i)], T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(input_embed[(i32(threadIdx.x) + 576i)], input_embed[(i32(threadIdx.x) + 576i)], T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(input_embed[(i32(threadIdx.x) + 640i)], input_embed[(i32(threadIdx.x) + 640i)], T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(input_embed[(i32(threadIdx.x) + 704i)], input_embed[(i32(threadIdx.x) + 704i)], T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(input_embed[(i32(threadIdx.x) + 768i)], input_embed[(i32(threadIdx.x) + 768i)], T_multiply_red_rf_local[0i]);
  T_multiply_red_rf_local[0i] = fma(input_embed[(i32(threadIdx.x) + 832i)], input_embed[(i32(threadIdx.x) + 832i)], T_multiply_red_rf_local[0i]);
  workgroupBarrier();
  red_buf0[i32(threadIdx.x)] = T_multiply_red_rf_local[0i];
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
    T_multiply_red_shared[0i] = red_buf0[0i];
  }
  workgroupBarrier();
  T_cast[i32(threadIdx.x)] = (((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 8.960000e+02f) + 1.000000e-06f))) * input_embed[i32(threadIdx.x)]) * model_layers_0_input_layernorm_weight2[i32(threadIdx.x)]);
  T_cast[(i32(threadIdx.x) + 64i)] = (((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 8.960000e+02f) + 1.000000e-06f))) * input_embed[(i32(threadIdx.x) + 64i)]) * model_layers_0_input_layernorm_weight2[(i32(threadIdx.x) + 64i)]);
  T_cast[(i32(threadIdx.x) + 128i)] = (((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 8.960000e+02f) + 1.000000e-06f))) * input_embed[(i32(threadIdx.x) + 128i)]) * model_layers_0_input_layernorm_weight2[(i32(threadIdx.x) + 128i)]);
  T_cast[(i32(threadIdx.x) + 192i)] = (((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 8.960000e+02f) + 1.000000e-06f))) * input_embed[(i32(threadIdx.x) + 192i)]) * model_layers_0_input_layernorm_weight2[(i32(threadIdx.x) + 192i)]);
  T_cast[(i32(threadIdx.x) + 256i)] = (((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 8.960000e+02f) + 1.000000e-06f))) * input_embed[(i32(threadIdx.x) + 256i)]) * model_layers_0_input_layernorm_weight2[(i32(threadIdx.x) + 256i)]);
  T_cast[(i32(threadIdx.x) + 320i)] = (((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 8.960000e+02f) + 1.000000e-06f))) * input_embed[(i32(threadIdx.x) + 320i)]) * model_layers_0_input_layernorm_weight2[(i32(threadIdx.x) + 320i)]);
  T_cast[(i32(threadIdx.x) + 384i)] = (((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 8.960000e+02f) + 1.000000e-06f))) * input_embed[(i32(threadIdx.x) + 384i)]) * model_layers_0_input_layernorm_weight2[(i32(threadIdx.x) + 384i)]);
  T_cast[(i32(threadIdx.x) + 448i)] = (((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 8.960000e+02f) + 1.000000e-06f))) * input_embed[(i32(threadIdx.x) + 448i)]) * model_layers_0_input_layernorm_weight2[(i32(threadIdx.x) + 448i)]);
  T_cast[(i32(threadIdx.x) + 512i)] = (((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 8.960000e+02f) + 1.000000e-06f))) * input_embed[(i32(threadIdx.x) + 512i)]) * model_layers_0_input_layernorm_weight2[(i32(threadIdx.x) + 512i)]);
  T_cast[(i32(threadIdx.x) + 576i)] = (((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 8.960000e+02f) + 1.000000e-06f))) * input_embed[(i32(threadIdx.x) + 576i)]) * model_layers_0_input_layernorm_weight2[(i32(threadIdx.x) + 576i)]);
  T_cast[(i32(threadIdx.x) + 640i)] = (((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 8.960000e+02f) + 1.000000e-06f))) * input_embed[(i32(threadIdx.x) + 640i)]) * model_layers_0_input_layernorm_weight2[(i32(threadIdx.x) + 640i)]);
  T_cast[(i32(threadIdx.x) + 704i)] = (((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 8.960000e+02f) + 1.000000e-06f))) * input_embed[(i32(threadIdx.x) + 704i)]) * model_layers_0_input_layernorm_weight2[(i32(threadIdx.x) + 704i)]);
  T_cast[(i32(threadIdx.x) + 768i)] = (((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 8.960000e+02f) + 1.000000e-06f))) * input_embed[(i32(threadIdx.x) + 768i)]) * model_layers_0_input_layernorm_weight2[(i32(threadIdx.x) + 768i)]);
  T_cast[(i32(threadIdx.x) + 832i)] = (((1.000000e+00f / sqrt(((T_multiply_red_shared[0i] / 8.960000e+02f) + 1.000000e-06f))) * input_embed[(i32(threadIdx.x) + 832i)]) * model_layers_0_input_layernorm_weight2[(i32(threadIdx.x) + 832i)]);
}

