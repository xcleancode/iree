transform.structured.canonicalized_sequence failures(propagate){
^bb1(%func: !pdl.operation):
  // This has to be done a the Flow level to:
  //   1. allow linalgx.relayout (i.e. tensor.pack/unpack) swap patterns 
  //      across the whole program.
  //   2. generally avoid huge allocs within dispatch regions.
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %func
  %generic = transform.structured.blocking %matmul { blocking_factors = [32, 32] }
 
  %func_2 = transform.iree.apply_patterns %func { swapping_relayout_patterns }
  %relayouts = transform.structured.match ops{["linalgx.relayout"]} in %func_2
  %generics = transform.structured.match ops{["linalg.generic"]} in %func_2

  // For now we need to be explicit and split out everything relayout into its
  // own region to avoid gynormous allocas.
  %relayout_regions = transform.iree.wrap_in_dispatch_region %relayouts
  %generic_regions = transform.iree.wrap_in_dispatch_region %generics
  transform.iree.region_to_workgroups %relayout_regions
  transform.iree.region_to_workgroups %generic_regions
}
