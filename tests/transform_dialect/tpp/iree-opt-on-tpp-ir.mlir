// RUN: iree-opt -iree-transform-dialect-interpreter -transform-dialect-drop-schedule %s | FileCheck %s

func.func @brgemmtpp(%A: tensor<1x4x8xf32>,
                    %B: tensor<1x8x4xf32>, %C: tensor<4x4xf32>) -> tensor<4x4xf32> attributes {llvm.emit_c_interface} {
  %D = linalg.batch_reduce_matmul ins(%A, %B: tensor<1x4x8xf32>, tensor<1x8x4xf32>) outs(%C: tensor<4x4xf32>) -> tensor<4x4xf32>
  return %D: tensor<4x4xf32>
}

transform.sequence failures(propagate) {
  ^bb0(%outermost: !pdl.operation):
    %outermost_2 = transform.iree.bufferize %outermost
    %func = transform.structured.match ops{["func.func"]} in %outermost_2
    %func_2 = transform.iree.apply_patterns %func { linalg_to_tpp }
    %func_3 = transform.iree.apply_patterns %func_2 { tpp_to_xsmm }
    transform.iree.apply_patterns %func_3 { xsmm_to_func }
}
