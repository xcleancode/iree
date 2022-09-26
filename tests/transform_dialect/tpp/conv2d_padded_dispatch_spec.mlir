transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  %conv = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %variant_op
  %pad = transform.structured.match ops{["tensor.pad"]} in %variant_op
  %fill = transform.structured.match ops{["linalg.fill"]} in %variant_op
  %non_conv = transform.merge_handles %pad, %fill

  // TODO: this could be replaced by a C++ only version.
  // Atm the IR produced is not the same so all pieces do not connect.
  %region_op = transform.iree.wrap_in_dispatch_region %conv
  %region_op_2 = transform.iree.move_preceding_op_into_dispatch_region %non_conv into %region_op
  transform.iree.region_to_workgroups %region_op_2
}
