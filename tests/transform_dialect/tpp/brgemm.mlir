// See: https://github.com/nicolasvasilache/nicolas.vasilache.github.io/blob/master/.venv/mlirdev/bin/activate#L242
// RUN: iree-transform-opt ./tests/transform_dialect/tpp/brgemm.mlir -b llvm-cpu -c ./tests/transform_dialect/tpp/brgemm_codegen_spec.mlir -d ./tests/transform_dialect/tpp/brgemm_dispatch_spec.mlir

func.func @brgemmtpp(%A: tensor<2x4x8xf32>,
                    %B: tensor<2x8x4xf32>, %C: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %D = linalg.batch_reduce_matmul ins(%A, %B: tensor<2x4x8xf32>, tensor<2x8x4xf32>) outs(%C: tensor<4x4xf32>) -> tensor<4x4xf32>
  return %D: tensor<4x4xf32>
}
