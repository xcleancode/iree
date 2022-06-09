func.func @large_aligned() {
  %lhs = util.unfoldable_constant dense<1.0> : tensor<2048x1024xf16>
  %rhs = util.unfoldable_constant dense<0.4> : tensor<1024x512xf16>
  %res = "mhlo.dot"(%lhs, %rhs) : (tensor<2048x1024xf16>, tensor<1024x512xf16>) -> tensor<2048x512xf16>
  check.expect_almost_eq_const(%res, dense<409.596> : tensor<2048x512xf16>) : tensor<2048x512xf16>
  return
}
