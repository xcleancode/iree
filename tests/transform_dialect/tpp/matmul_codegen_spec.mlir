transform.sequence failures(propagate) {
  ^bb0(%variant_op: !pdl.operation):
    // This has to be done a the Flow level to:
    //   1. allow linalgx.relayout (i.e. tensor.pack/unpack) swap patterns 
    //      across the whole program.
    //   2. generally avoid huge allocs within dispatch regions.
    // %matmul = transform.structured.match ops{["linalg.matmul"]} in %variant_op
    // transform.structured.blocking %matmul { blocking_factors = [32, 32] }

    // TODO: bufferization with scf.foreach has regressed, reenable parallelism
    // once D135342 has landed and is integrated in IREE.
    // Tile to one level of outer-parallelism, just because we can. 
    // %matmul = transform.structured.match ops{["linalg.matmul"]} in %variant_op
    // transform.structured.tile_to_foreach_thread_op %matmul tile_sizes [16]
    %variant_op_2 = transform.iree.bufferize %variant_op
    %generic = transform.structured.match ops{["linalg.generic"]} in %variant_op_2
    transform.structured.map_to_brgemm %generic

    %func_2 = transform.structured.match ops{["func.func"]} in %variant_op_2
    // TODO: reenable for parallelism once bufferization is fixed.
    // %func_2 = transform.iree.foreach_thread_to_workgroup %func
    // TODO: xsmm.ternary.dispatch currently does not hoist, investigate.
    %func_3 = transform.iree.apply_patterns %func_2 
      { linalg_to_tpp, tpp_to_xsmm, xsmm_to_func }
    transform.iree.apply_patterns %func_3 { simplify_memref_metadata }
}
