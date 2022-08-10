transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.structured.canonicalized_sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg1
    %foreach_op, %tiled_op = tile_to_foreach_thread_op %0 {num_threads = [42, 67]}
    %dispatch_op = transform.iree.foreach_thread_to_flow %foreach_op
  }
}