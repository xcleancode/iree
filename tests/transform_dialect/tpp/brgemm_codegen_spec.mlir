transform.sequence failures(propagate) {
  ^bb0(%variant_op: !pdl.operation):
    // TODO: bufferization with scf.foreach seems to have regressed, figure it out.
    // Tile to one level of outer-parallelism, just because we can. 
    // %brgemm = transform.structured.match ops{["linalg.batch_reduce_matmul"]} in %variant_op
    // transform.structured.tile_to_foreach_thread_op %brgemm tile_sizes [1]

    %variant_op_2 = transform.iree.bufferize %variant_op

    // %func_2 = transform.structured.match ops{["func.func"]} in %variant_op_2
    // // %func_2 = transform.iree.foreach_thread_to_workgroup %func
    // %func_3 = transform.iree.apply_patterns %func_2 
    //   { linalg_to_tpp, tpp_to_xsmm, xsmm_to_func }
    // %func_4 = transform.iree.apply_patterns %func_3 { simplify_memref_metadata }

    
}
