// RUN: iree-opt %s

// Dispatch brgemm.
transform.structured.canonicalized_sequence failures(propagate){
^bb1(%variant_op: !pdl.operation):
  %brgemm = transform.structured.match ops{["linalg.batch_reduce_matmul"]} in %variant_op

  // TODO: this could be replaced by a C++ only version.
  // Atm the IR produced is not the same so all pieces do not connect.
  %region_op = transform.iree.wrap_in_dispatch_region %brgemm
  transform.iree.region_to_workgroups %region_op
}
