// iree-transform-opt-dispatch-only ./tests/transform_dialect/tpp/matmul.mlir -b llvm-cpu -c tests/transform_dialect/tpp/matmul_codegen_spec.mlir  -d tests/transform_dialect/tpp/matmul_dispatch_spec.mlir

// iree-transform-opt ./tests/transform_dialect/tpp/matmul.mlir -b llvm-cpu -c tests/transform_dialect/tpp/matmul_codegen_spec.mlir  -d tests/transform_dialect/tpp/matmul_dispatch_spec.mlir

// iree-transform-compile ./tests/transform_dialect/tpp/matmul.mlir -b llvm-cpu -c tests/transform_dialect/tpp/matmul_codegen_spec.mlir  -d tests/transform_dialect/tpp/matmul_dispatch_spec.mlir

// iree-transform-compile ./tests/transform_dialect/tpp/matmul.mlir -b llvm-cpu -c tests/transform_dialect/tpp/matmul_codegen_spec.mlir -d tests/transform_dialect/tpp/matmul_dispatch_spec.mlir  -- --iree-llvm-target-triple=x86_64-pc-linux-gnu   --iree-llvm-target-cpu-features=host  --iree-hal-benchmark-dispatch-repeat-count=100  | iree-benchmark-module --device=local-task --task_topology_group_count=0 --batch_size=100  --entry_function=matmul_static --function_input="256x512xf32=1"  --function_input="512x1024xf32=2" --function_input="256x1024xf32=0"
!A_size = tensor<256x512xf32>
!B_size = tensor<512x1024xf32>
!C_size = tensor<256x1024xf32>

func.func @matmul_static(
    %A : !A_size, %B : !B_size, %C : !C_size) -> !C_size {
  %0 = linalg.matmul ins(%A, %B : !A_size, !B_size)
                     outs(%C : !C_size) -> !C_size
  return %0 : !C_size
}
