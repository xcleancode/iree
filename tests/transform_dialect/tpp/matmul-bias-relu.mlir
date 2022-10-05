// iree-transform-opt-dispatch-only ./tests/transform_dialect/tpp/matmul-bias-relu.mlir -b llvm-cpu -c tests/transform_dialect/tpp/matmul-bias-relu_codegen_spec.mlir  -d tests/transform_dialect/tpp/matmul-bias-relu_dispatch_spec.mlir

// iree-transform-opt ./tests/transform_dialect/tpp/matmul-bias-relu.mlir -b llvm-cpu -c tests/transform_dialect/tpp/matmul-bias-relu_codegen_spec.mlir  -d tests/transform_dialect/tpp/matmul-bias-relu_dispatch_spec.mlir

// TODO: error: multiple public symbols with the name: xsmm_unary_invoke
// iree-transform-compile ./tests/transform_dialect/tpp/matmul-bias-relu.mlir -b llvm-cpu -c tests/transform_dialect/tpp/matmul-bias-relu_codegen_spec.mlir  -d tests/transform_dialect/tpp/matmul-bias-relu_dispatch_spec.mlir 

!A_tensor_t = tensor<256x512xf32>
!B_tensor_t = tensor<512x1024xf32>
!C_tensor_t = tensor<256x1024xf32>
!Bias_tensor_t = tensor<1024xf32>

#map0 = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @matmul_static(
    %A : !A_tensor_t, %B : !B_tensor_t, %C : !C_tensor_t, %Bias: !Bias_tensor_t) -> !C_tensor_t {
  // Expanding bias beforehand may be easier to fuse and completely fold away than post-hoc addBias to matmul.
  %expanded_bias = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} 
      ins(%Bias : !Bias_tensor_t) outs(%C : !C_tensor_t) {
        ^bb0(%arg9: f32, %arg10: f32):
      linalg.yield %arg9 : f32
    } -> !C_tensor_t

  %matmul = linalg.matmul ins(%A, %B : !A_tensor_t, !B_tensor_t)
                     outs(%expanded_bias : !C_tensor_t) -> !C_tensor_t

  // ReLU has no "ins" operands.
  %res = linalg.generic {indexing_maps = [#map1], iterator_types = ["parallel", "parallel"]} 
      outs(%matmul : !C_tensor_t) {
    ^bb0(%arg9: f32):
      %16 = mathx.relu %arg9 : f32
      linalg.yield %16 : f32
    } -> !C_tensor_t

  return %res : !C_tensor_t
}
