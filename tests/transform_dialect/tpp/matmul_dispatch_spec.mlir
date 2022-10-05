transform.structured.canonicalized_sequence failures(propagate){
^bb1(%func: !pdl.operation):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %func
  %generic = transform.structured.blocking %matmul { blocking_factors = [32, 32] }
  %relayouts = transform.structured.match ops{["linalgx.relayout"]} in %func
  
  // TODO: Cannot be too smart with fusion in IREE. This is a longer discussion 
  // on conflations between what the IREE runtime wants as atomic units of:
  //   1. memory allocation
  //   2. scheduling
  //   3. (synchronization-free) parallelization
  //   4. codegen
  // %region_op = transform.iree.wrap_in_dispatch_region %r4
  // %pre_relayouts = transform.merge_handles %r1, %r2, %r3, %generic
  // %region_op_2 = transform.iree.move_preceding_op_into_dispatch_region %pre_relayouts into %region_op
  // transform.iree.region_to_workgroups %region_op_2

  // For now we need to be explicit and split out everything relayout into its
  // own region to avoid gynormous allocas.
  %relayout_regions = transform.iree.wrap_in_dispatch_region %relayouts
  %generic_region = transform.iree.wrap_in_dispatch_region %generic
  transform.iree.region_to_workgroups %relayout_regions
  transform.iree.region_to_workgroups %generic_region
}
