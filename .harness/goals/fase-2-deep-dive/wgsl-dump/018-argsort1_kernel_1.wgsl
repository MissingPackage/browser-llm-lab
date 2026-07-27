//----------------------------------------
// Function: argsort1_kernel_1
//----------------------------------------
@group(0) @binding(0) var<storage, read_write> out_buf : array<i32>;
@group(0) @binding(1) var<storage, read_write> out_swap_buf : array<i32>;
@group(0) @binding(2) var<storage, read_write> value_buf : array<f32>;
@group(0) @binding(3) var<storage, read_write> value_swap_buf : array<f32>;

struct PODArgs {
  batch_size: i32,
  vocab_size: i32,
  packGridDimX: u32
}
@group(0) @binding(4) var<uniform> podArgs : PODArgs;

var<workgroup> v : array<f32, 128>;
var<workgroup> v_1 : array<i32, 128>;
@compute @workgroup_size(64, 1, 1)
fn argsort1_kernel_1(
  @builtin(workgroup_id) blockIdx : vec3<u32>,
  @builtin(num_workgroups) gridDim : vec3<u32>,
  @builtin(local_invocation_id) threadIdx : vec3<u32>
) {
  if (blockIdx.z * gridDim.x + blockIdx.x > podArgs.packGridDimX) { return; }
  var v_2 : array<f32, 1>;
  var v_3 : array<f32, 1>;
  var v_4 : array<f32, 1>;
  var v_5 : array<i32, 1>;
  let v__1 : i32 = i32(blockIdx.z * gridDim.x + blockIdx.x);
  for (var v_6 : i32 = 0i; v_6 < 2i; v_6++) {
    if ((((v__1 * 128i) + (i32(threadIdx.x) * 2i)) + v_6) < podArgs.vocab_size) {
      v[((i32(threadIdx.x) * 2i) + v_6)] = value_buf[((((v__1 * 128i) + (i32(threadIdx.x) * 2i)) + (i32(blockIdx.y) * podArgs.vocab_size)) + v_6)];
      v_1[((i32(threadIdx.x) * 2i) + v_6)] = out_buf[((((v__1 * 128i) + (i32(threadIdx.x) * 2i)) + (i32(blockIdx.y) * podArgs.vocab_size)) + v_6)];
    }
  }
  workgroupBarrier();
  for (var v_7 : i32 = 0i; v_7 < min(128i, (podArgs.vocab_size - (v__1 * 128i))); v_7++) {
    if ((((i32(threadIdx.x) * 2i) + (v_7 & 1i)) < 127i) && ((((i32(threadIdx.x) * 2i) + (v_7 & 1i)) + 1i) < (podArgs.vocab_size - (v__1 * 128i)))) {
      v_2[0i] = v[((i32(threadIdx.x) * 2i) + (v_7 & 1i))];
      v_3[0i] = v[(((i32(threadIdx.x) * 2i) + (v_7 & 1i)) + 1i)];
      if (v_2[0i] < v_3[0i]) {
        v_4[0i] = v[((i32(threadIdx.x) * 2i) + (v_7 & 1i))];
        v[((i32(threadIdx.x) * 2i) + (v_7 & 1i))] = v[(((i32(threadIdx.x) * 2i) + (v_7 & 1i)) + 1i)];
        v[(((i32(threadIdx.x) * 2i) + (v_7 & 1i)) + 1i)] = v_4[0i];
        v_5[0i] = v_1[((i32(threadIdx.x) * 2i) + (v_7 & 1i))];
        v_1[((i32(threadIdx.x) * 2i) + (v_7 & 1i))] = v_1[(((i32(threadIdx.x) * 2i) + (v_7 & 1i)) + 1i)];
        v_1[(((i32(threadIdx.x) * 2i) + (v_7 & 1i)) + 1i)] = v_5[0i];
      }
    }
    workgroupBarrier();
  }
  for (var v_8 : i32 = 0i; v_8 < 2i; v_8++) {
    if ((((v__1 * 128i) + (i32(threadIdx.x) * 2i)) + v_8) < podArgs.vocab_size) {
      let rmod : i32 = (i32(blockIdx.y) % podArgs.batch_size);
      let rmod_1 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
      let rdiv : i32 = (i32(blockIdx.y) / podArgs.batch_size);
      value_buf[(((((v__1 * 128i) + (i32(threadIdx.x) * 2i)) + (select((rmod + podArgs.batch_size), rmod, (((podArgs.batch_size >= 0i) && (rmod >= 0i)) || ((podArgs.batch_size < 0i) && (rmod <= 0i)))) * podArgs.vocab_size)) + select((rdiv - 1i), rdiv, (((podArgs.batch_size >= 0i) && (rmod_1 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_1 <= 0i))))) + v_8)] = v[((i32(threadIdx.x) * 2i) + v_8)];
      let rmod_2 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
      let rmod_3 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
      let rdiv_1 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
      value_swap_buf[(((((v__1 * 128i) + (i32(threadIdx.x) * 2i)) + (select((rmod_2 + podArgs.batch_size), rmod_2, (((podArgs.batch_size >= 0i) && (rmod_2 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_2 <= 0i)))) * podArgs.vocab_size)) + select((rdiv_1 - 1i), rdiv_1, (((podArgs.batch_size >= 0i) && (rmod_3 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_3 <= 0i))))) + v_8)] = v[((i32(threadIdx.x) * 2i) + v_8)];
      let rmod_4 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
      let rmod_5 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
      let rdiv_2 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
      out_buf[(((((v__1 * 128i) + (i32(threadIdx.x) * 2i)) + (select((rmod_4 + podArgs.batch_size), rmod_4, (((podArgs.batch_size >= 0i) && (rmod_4 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_4 <= 0i)))) * podArgs.vocab_size)) + select((rdiv_2 - 1i), rdiv_2, (((podArgs.batch_size >= 0i) && (rmod_5 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_5 <= 0i))))) + v_8)] = v_1[((i32(threadIdx.x) * 2i) + v_8)];
      let rmod_6 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
      let rmod_7 : i32 = (i32(blockIdx.y) % podArgs.batch_size);
      let rdiv_3 : i32 = (i32(blockIdx.y) / podArgs.batch_size);
      out_swap_buf[(((((v__1 * 128i) + (i32(threadIdx.x) * 2i)) + (select((rmod_6 + podArgs.batch_size), rmod_6, (((podArgs.batch_size >= 0i) && (rmod_6 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_6 <= 0i)))) * podArgs.vocab_size)) + select((rdiv_3 - 1i), rdiv_3, (((podArgs.batch_size >= 0i) && (rmod_7 >= 0i)) || ((podArgs.batch_size < 0i) && (rmod_7 <= 0i))))) + v_8)] = v_1[((i32(threadIdx.x) * 2i) + v_8)];
    }
  }
}

