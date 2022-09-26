transform.structured.canonicalized_sequence failures(propagate) {
  ^bb0(%variant_op: !pdl.operation):
    %conv = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %variant_op
    %pad = transform.structured.match ops{["tensor.pad"]} in %variant_op
    %fill = transform.structured.match ops{["linalg.fill"]} in %variant_op
    %non_conv = transform.merge_handles %pad, %fill

    // Tile to foreach_thread which gives us both the ability to fuse nicely and
    // to exploit parallelism.
    // TODO: Atm IREE allows only 3-D so choose dimensions of tiling wisely here.
    %foreach_thread, %_ = transform.structured.tile_to_foreach_thread_op %conv
               // n  h  w   f
      tile_sizes [0, 1, 8, 32]
      // TODO: bug in thread_dim_mapping here.
      // {thread_dim_mapping = [2, 0, 1]}
    transform.structured.fuse_into_containing_op %non_conv into %foreach_thread

    %func = transform.structured.match ops{["func.func"]} in %variant_op
    // TODO: bufferization fails after the swapping_pattern creates an scf.if, even
    // when the tensor.generate is vectorized with extra_fancy_patterns.
    // As a consequence also use swap_padding_elide_corner_case as we statically 
    // know we don't need to guard against the empty case (3x3 strides and 1x1 padding).
    transform.iree.apply_patterns %func 
      { swapping_patterns, swap_padding_elide_conditional }

    %0 = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %variant_op
    %1 = transform.structured.generalize %0
    %2 = transform.structured.interchange %1 { iterator_interchange = [0, 1, 4, 5, 2, 3, 6] }
    %3 = transform.structured.map_conv_to_matmul %2 (filter_height_pos = 0, filter_width_pos = 1)
    %4 = transform.structured.generalize %3

    // TODO: pad does not get vectorized and instead is lowered to loops during
    // buferization. This requires a better control of vectorization of specific
    // ops (an outline-vectorize-inline model is probably the easiest).
    %variant_op_2 = transform.iree.bufferize %variant_op

    %func_2 = transform.structured.match ops{["func.func"]} in %variant_op_2
    %func_3 = transform.iree.foreach_thread_to_workgroup %func_2
    %func_4 = transform.iree.apply_patterns %func_3
      { linalg_to_tpp, tpp_to_xsmm, xsmm_to_func }
    %func_5 = transform.iree.apply_patterns %func_4 { simplify_memref_metadata }
}
