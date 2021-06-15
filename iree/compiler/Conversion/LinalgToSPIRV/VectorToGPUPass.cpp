// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===---- VectorToGPUPass.cpp - Pass for the final SPIR-V conversion ------===//
//
// This file implement a pass to convert vector dialect operations to GPU
// operations distributed across a subgroup.
//
//===----------------------------------------------------------------------===//
#include <memory>

#include "iree/compiler/Conversion/CodegenUtils/FunctionUtils.h"
#include "iree/compiler/Conversion/CodegenUtils/MarkerUtils.h"
#include "iree/compiler/Conversion/CodegenUtils/TransformUtils.h"
#include "iree/compiler/Conversion/PassDetail.h"
#include "iree/compiler/Conversion/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace {

struct LinalgToSPIRVConvertVectorToGPUPass
    : public LinalgToSPIRVConvertVectorToGPUBase<
          LinalgToSPIRVConvertVectorToGPUPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, gpu::GPUDialect, memref::MemRefDialect,
                    scf::SCFDialect, vector::VectorDialect>();
  }

  void runOnOperation() override;

 private:
  void tileAndVectorizeLinalgCopy(FuncOp funcOp, MLIRContext *context);
  void lowerVectorOps(FuncOp funcOp, MLIRContext *context);
};

class VectorToGPUConversionTarget : public ConversionTarget {
 public:
  using ConversionTarget::ConversionTarget;

 protected:
  // Standard operation are legal if they operate on scalars. We need to
  // legalize operations on vectors.
  bool isDynamicallyLegal(Operation *op) const override {
    auto isVectorType = [](Type t) { return t.isa<VectorType>(); };
    if (llvm::any_of(op->getResultTypes(), isVectorType) ||
        llvm::any_of(op->getOperandTypes(), isVectorType))
      return false;
    return true;
  }
};

void LinalgToSPIRVConvertVectorToGPUPass::tileAndVectorizeLinalgCopy(
    FuncOp funcOp, MLIRContext *context) {
  // 1. Tile linalg and distribute it on invocations.
  std::unique_ptr<ConversionTarget> target =
      std::make_unique<ConversionTarget>(*context);
  target->addDynamicallyLegalOp<linalg::CopyOp>([&](linalg::CopyOp copy) {
    return !(hasMarker(copy, getCopyToWorkgroupMemoryMarker()));
  });
  target->markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  OwningRewritePatternList tileAndDistributePattern(&getContext());
  populateTileAndDistributeLinalgCopyPatterns(context,
                                              tileAndDistributePattern);
  if (failed(applyPartialConversion(funcOp, *target,
                                    std::move(tileAndDistributePattern)))) {
    return signalPassFailure();
  }

  // 2. Canonicalize the IR generated by tiling.
  OwningRewritePatternList canonicalizePatterns =
      linalg::getLinalgTilingCanonicalizationPatterns(context);
  canonicalizePatterns.insert<AffineMinCanonicalizationPattern,
                              linalg::AffineMinSCFCanonicalizationPattern>(
      context);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(canonicalizePatterns));

  // 3. Vectorize the tiled linalg to be able to map it to load/store vector.
  OwningRewritePatternList vectorizationPatterns(&getContext());
  linalg::insertVectorizationPatterns<linalg::CopyOp>(
      vectorizationPatterns, linalg::LinalgVectorizationOptions(),
      linalg::LinalgTransformationFilter(
          Identifier::get(getVectorizeMarker(), context), {}));
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(vectorizationPatterns));
}

void LinalgToSPIRVConvertVectorToGPUPass::runOnOperation() {
  MLIRContext *context = &getContext();
  FuncOp funcOp = getOperation();
  tileAndVectorizeLinalgCopy(funcOp, context);
}
}  // namespace

//===----------------------------------------------------------------------===//
// Pass entry point and registration
//===----------------------------------------------------------------------===//
std::unique_ptr<OperationPass<FuncOp>> createLinalgToSPIRVConvertVectorToGPU() {
  return std::make_unique<LinalgToSPIRVConvertVectorToGPUPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
