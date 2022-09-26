transform.structured.canonicalized_sequence failures(propagate) {
  ^bb0(%variant_op: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %variant_op
    %1 = transform.structured.generalize %0
    %2 = transform.structured.interchange %1 { iterator_interchange = [0, 1, 4, 5, 2, 3, 6] }
    %3 = transform.structured.map_conv_to_matmul %2 (filter_height_pos = 0, filter_width_pos = 1)

    %matmul = transform.structured.match ops{["linalg.matmul"]} in %variant_op
    %matmul_as_generic = transform.structured.generalize %matmul

    // TODO: bufferization with scf.foreach seems to have regressed, figure it out.
    // Tile to one level of outer-parallelism, just because we can. 
    // %brgemm = transform.structured.match ops{["linalg.batch_reduce_matmul"]} in %variant_op
    // transform.structured.tile_to_foreach_thread_op %brgemm tile_sizes [1]

    %variant_op_2 = transform.iree.bufferize %variant_op

    %func_2 = transform.structured.match ops{["func.func"]} in %variant_op_2
    // %func_2 = transform.iree.foreach_thread_to_workgroup %func
    %func_3 = transform.iree.apply_patterns %func_2 
      { linalg_to_tpp, tpp_to_xsmm, xsmm_to_func }
    %func_4 = transform.iree.apply_patterns %func_3 { simplify_memref_metadata }
}
