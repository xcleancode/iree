// See: https://github.com/nicolasvasilache/nicolas.vasilache.github.io/blob/master/.venv/mlirdev/bin/activate#L242
// RUN: iree-transform-opt ./tests/transform_dialect/tpp/conv2d.mlir -b llvm-cpu -c ./tests/transform_dialect/tpp/conv2d_codegen_spec.mlir -d ./tests/transform_dialect/tpp/conv2d_dispatch_spec.mlir

!in_tensor_t  = tensor<1x56x56x64xf32>
!weight_tensor_t = tensor<3x3x64x64xf32>
!tmp_tensor_t = tensor<1x58x58x64xf32>
!out_tensor_t = tensor<1x56x56x64xf32>

func.func @conv2d_padded(%input: !in_tensor_t, 
                  %weight: !weight_tensor_t,
                  %out: !out_tensor_t ) -> !out_tensor_t {
  %cst_0 = arith.constant 0.000000e+00 : f32
  %out_filled = linalg.fill ins(%cst_0 : f32) 
                           outs(%out : !out_tensor_t) -> !out_tensor_t
  %tmp = tensor.pad %input low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%arg2: index, %arg3: index, %arg4: index, %arg5: index):
      tensor.yield %cst_0 : f32
  } : !in_tensor_t to !tmp_tensor_t
  %res = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
     ins(%tmp, %weight : !tmp_tensor_t, !weight_tensor_t) 
    outs(%out_filled : !out_tensor_t) -> !out_tensor_t

  // TODO: what is a proper check here ?
  // check.expect_almost_eq_const(%res, dense<576.0> : tensor<1x56x56x64xf32>) : tensor<1x56x56x64xf32>
  return %res : !out_tensor_t
}
