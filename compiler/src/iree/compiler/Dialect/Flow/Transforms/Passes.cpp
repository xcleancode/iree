// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"

#include <memory>

#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"

// TODO(ravishankarm): Change to a pipeline option.
static llvm::cl::opt<bool> clExportBenchmarkFuncs(
    "iree-flow-export-benchmark-funcs",
    llvm::cl::desc(
        "Exports one function per original module entry point and "
        "unique flow.executable that dispatches with dummy arguments."),
    llvm::cl::init(false));

// TODO(ravishankarm): Change to a pipeline option.
static llvm::cl::opt<bool> clTraceDispatchTensors(
    "iree-flow-trace-dispatch-tensors",
    llvm::cl::desc(
        "Trace runtime input/output tensors for each dispatch function."),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clDemoteI64ToI32(
    "iree-flow-demote-i64-to-i32",
    llvm::cl::desc("Converts all i64 ops and values into i32 counterparts "
                   "unconditionally before main flow conversions."),
    llvm::cl::init(false));
static llvm::cl::opt<bool> clDemoteF32ToF16(
    "iree-flow-demote-f32-to-f16",
    llvm::cl::desc("Converts all f32 ops and values into f16 counterparts "
                   "unconditionally before main flow conversions."),
    llvm::cl::init(false));
static llvm::cl::opt<bool> clPromoteF16ToF32(
    "iree-flow-promote-f16-to-f32",
    llvm::cl::desc("Converts all f16 ops and values into f32 counterparts "
                   "unconditionally before main flow conversions."),
    llvm::cl::init(false));
static llvm::cl::opt<bool> clDemoteF64ToF32(
    "iree-flow-demote-f64-to-f32",
    llvm::cl::desc("Converts all f64 ops and values into f32 counterparts "
                   "unconditionally before main flow conversions."),
    llvm::cl::init(true));

static llvm::cl::opt<bool> clEnableConvToImg2Col(
    "iree-flow-enable-conv-img2col-transform",
    llvm::cl::desc("Enable converting convolution ops to img2col form."),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clEnableConvToWinograd(
    "iree-flow-enable-conv-winograd-transform",
    llvm::cl::desc("Enable converting convolution ops to winograd form."),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clEnablePaddingLinalgOps(
    "iree-flow-enable-padding-linalg-ops",
    llvm::cl::desc("Enable padding linalg ops to an integer multiple of "
                   "flow-padding-size"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clEnableFusePaddingIntoLinalgConsumerOps(
    "iree-flow-enable-fuse-padding-into-linalg-consumer-ops",
    llvm::cl::desc("Enable fusing tensor.pad ops into Linalg consumer ops"),
    llvm::cl::init(false));

static llvm::cl::opt<int> clLinalgOpsPaddingSize(
    "iree-flow-linalg-ops-padding-size",
    llvm::cl::desc("Enable padding linalg ops to an integer multiple of "
                   "flow-padding-size"),
    llvm::cl::init(4));

// TODO(#1159): enable by default or remove this option once it works on
//              a broader set of programs
static llvm::cl::opt<bool> clEnableLinalgDetensorize(
    "iree-flow-enable-linalg-detensorize",
    llvm::cl::desc("Enable detensorizing linalg ops to operate on primitives"),
    llvm::cl::init(true));

static llvm::cl::opt<bool> clEnableAggressiveFusion(
    "iree-flow-enable-aggressive-fusion",
    llvm::cl::desc(
        "Enable the aggressive fusion heuristic to fuse multiuse ops and ops "
        "with reduction loops"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clDispatchGenerateWorkloadRegion(
    "iree-flow-dispatch-generate-workload-region",
    llvm::cl::desc("Generate the workload region"), llvm::cl::init(true));

static llvm::cl::opt<bool> clCollapseDimensions(
    "iree-flow-form-dispatch-regions-collapse",
    llvm::cl::desc("Collapse dimensions"), llvm::cl::init(true));

static llvm::cl::opt<bool> clEnableDataTiling(
    "iree-flow-enable-data-tiling", llvm::cl::desc("Enable data tiling path"),
    llvm::cl::init(false));

static llvm::cl::opt<std::string> clMmt4dTargetOptions(
    "iree-flow-mmt4d-target-options",
    llvm::cl::desc("Convert linalg.matmul ops to MMT4D ops targetting the "
                   "given architecture"),
    llvm::cl::init(""));

static llvm::cl::opt<bool> clNormalizeInputIndexingMap(
    "iree-flow-normalize-input-indexing-map",
    llvm::cl::desc("Enable normalizing input indexing map to identity"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> clDumpDispatchGraph(
    "iree-flow-dump-dispatch-graph",
    llvm::cl::desc("Dump a dot graph for dispatches"), llvm::cl::init(false));

static llvm::cl::opt<std::string> clDumpDispatchGraphOutputFile(
    "iree-flow-dump-dispatch-graph-output-file",
    llvm::cl::desc("Output file name for a dispatch graph dump"),
    llvm::cl::init("dispatch.dot"));

static llvm::cl::opt<std::string> clDispatchTransformFileName(
    "iree-flow-dispatch-use-transform-dialect",
    llvm::cl::desc("mlir file containing a top-level module that specifies "
                   "the transformations to apply to form dispatch regions."),
    llvm::cl::init(""));

static llvm::cl::opt<bool> clZeroFillEmptyTensors(
    "iree-flow-zero-fill-empty-tensors",
    llvm::cl::desc(
        "Zero fill empty tensors instead of leaving them uninitialized"),
    llvm::cl::init(false));

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

using FunctionLikeNest = MultiOpNest<func::FuncOp, IREE::Util::InitializerOp>;

// Subset of the overall pass pipeline for optimizing globals and numerics.
// We may ultimately break this out separately so creating a syntactic
// distinction to keep that as an option.
void buildGlobalOptimizationPassPipeline(
    OpPassManager &mainPassManager, const TransformOptions &transformOptions) {
  OpPassManager pipeline(ModuleOp::getOperationName());

  FunctionLikeNest(pipeline)
      // Simplify util.global accesses early on; this can help with dispatch
      // region formation as redundant store-loads are removed.
      .addPass(IREE::Util::createSimplifyGlobalAccessesPass);

  // Module level cleanup and canonicalization of util.global (and other util
  // ops).
  pipeline.addPass(IREE::Util::createApplyPatternsPass());
  pipeline.addPass(IREE::Util::createFoldGlobalsPass());
  pipeline.addPass(IREE::Util::createIPOPass());

  if (transformOptions.constExprHoisting) {
    pipeline.addPass(IREE::Util::createHoistIntoGlobalsPass());
  }

  if (transformOptions.buildConstEvalPassPipeline) {
    transformOptions.buildConstEvalPassPipeline(pipeline);
  }

  if (transformOptions.numericPrecisionReduction) {
    pipeline.addPass(createInferNumericNarrowingPass());
    pipeline.addPass(createOptimizeNumericsPass());
    pipeline.addPass(createCleanupNumericNarrowingPass());
  }

  FunctionLikeNest(pipeline)
      .addPass(mlir::createCanonicalizerPass)
      .addPass(mlir::createCSEPass);

  // Add the whole fixed point iterator.
  mainPassManager.addPass(
      IREE::Util::createFixedPointIteratorPass(std::move(pipeline)));
}

}  // namespace

/// Optional pre-processing passes that transform the program for specialist
/// uses case.
static void buildOptionalPreprocessingPassPipeline(OpPassManager &passManager) {
  FunctionLikeNest(passManager)
      .addPredicatedPass(clEnableConvToWinograd,
                         IREE::LinalgExt::createConvertConv2DToWinogradPass)
      .addPredicatedPass(clEnableConvToImg2Col,
                         IREE::Flow::createConvertConv2DToImg2ColPass)
      .addPredicatedPass(
          !clMmt4dTargetOptions.empty(),
          []() {
            return IREE::Flow::createConvertLinalgMatmulToMmt4DPass(
                clMmt4dTargetOptions);
          })
      .addPredicatedPass(clEnablePaddingLinalgOps, []() {
        return IREE::Flow::createPadLinalgOpsToIntegerMultiplePass(
            clLinalgOpsPaddingSize);
      });
}

void buildFlowTransformPassPipeline(OpPassManager &passManager,
                                    const TransformOptions &transformOptions) {
  // ML frontends have very uneven support for user-controlled types _and_ users
  // tend to use types not well suited for the work they are doing. These
  // demotions/promotions allow users to change the types after lowering out of
  // the frontends. It'll always be better to do this higher up in the stack
  // as these kind of blanket conversions have corner cases and potential
  // accuracy/precision losses beyond what the user may expect.
  if (clDemoteF64ToF32) {
    passManager.addPass(IREE::Util::createDemoteF64ToF32Pass());
  }
  if (clDemoteF32ToF16) {
    passManager.addPass(IREE::Util::createDemoteF32ToF16Pass());
  }
  if (clPromoteF16ToF32) {
    passManager.addPass(IREE::Util::createPromoteF16ToF32Pass());
  }
  if (clDemoteI64ToI32) {
    passManager.addPass(IREE::Util::createDemoteI64ToI32Pass());
  }

  // Preprocessing passes to get the program into a canonical state.
  FunctionLikeNest(passManager)
      .addPass(IREE::Flow::createDetachElementwiseFromNamedOpsPass)
      .addPass(mlir::createLinalgNamedOpConversionPass)
      .addPass(IREE::Flow::createConvert1X1FilterConv2DToMatmulPass);

  // Optional pre-processing passes.
  buildOptionalPreprocessingPassPipeline(passManager);
  passManager.addPass(IREE::Flow::createVerifyInputLegalityPass());

  // Expand tensor shapes into SSA values and optimize the whole program.
  // The more we are able to equate shape dimensions at this level the better
  // our fusions will be.
  passManager.addPass(IREE::Flow::createExpandTensorShapesPass());
  buildGlobalOptimizationPassPipeline(passManager, transformOptions);

  FunctionLikeNest(passManager)
      // Preprocess the input to a form more amenable for fusion
      // - Convert all elementwise ops to Linalg
      // - Remove unit-extent dimensions.
      .addPass(mlir::createConvertElementwiseToLinalgPass)
      .addPass(mlir::createLinalgFoldUnitExtentDimsPass)
      .addPass(createInterchangeGenericOpsPass)
      .addPass(memref::createResolveShapedTypeResultDimsPass)
      .addPass(mlir::createCanonicalizerPass)
      .addPass(mlir::createCSEPass)
      // Elementwise fusion.
      .addPass([]() {
        return createFusionOfTensorOpsPass(clEnableAggressiveFusion);
      })
      .addPredicatedPass(clEnableLinalgDetensorize,
                         mlir::createLinalgDetensorizePass)
      .addPass(mlir::createCanonicalizerPass)
      .addPass(mlir::createCSEPass)
      .addPass(createCollapseDimsPass)
      // Split reduction operations into parallel and reduction.
      .addPass(createSplitReductionPass)
      // SplitReductionPass may create reduction dimension that are not the last
      // dimension.
      .addPass(createInterchangeGenericOpsPass)
      // Normalize the input indexing map to make the input indexing map
      // identity. This helps fusing named linalg op with a generic op with
      // transpose.
      .addPredicatedPass(clNormalizeInputIndexingMap,
                         createInterchangeTransposeGenericOpsPass)
      // Enable data tiling after all linalg level transformations.
      .addPredicatedPass(clEnableDataTiling, createSetEncodingPass)
      ////////////////////////////////////////////////////////////////////////
      // Dispatch region formation.
      .addPredicatedPass(!clDispatchTransformFileName.empty(),
                         [&]() {
                           return createDispatchWithTransformDialect(
                               clDispatchTransformFileName);
                         })
      // Only want use the transform dialect for some dispatch regions and let
      // the FormDispatchRegions handle the rest.
      .addPass([&]() {
        return createFormDispatchRegionsPass(clEnableAggressiveFusion,
                                             clDispatchGenerateWorkloadRegion,
                                             clCollapseDimensions);
      })
      // Form dispatch region into dispatch workgroups
      .addPass([&]() {
        return createFormDispatchWorkgroupsPass(
            clDispatchGenerateWorkloadRegion);
      })
      ////////////////////////////////////////////////////////////////////////
      .addPass(createCaptureDispatchDynamicDimsPass)
      .addPass(mlir::createCanonicalizerPass)
      .addPass(createCSEPass);

  // Initialize any empty tensors to zero.
  passManager.addPass(createInitializeEmptyTensorsPass(clZeroFillEmptyTensors));

  // Module pass to outline the dispatch regions into their own functions
  // wrapped in executables.
  passManager.addPass(IREE::Flow::createOutlineDispatchRegionsPass());

  // Strip assertions from executables. We could support them with a bunch of
  // work but our generated executables are designed to be safe in the face of
  // invalid values and it'd only be useful for debugging.
  passManager.addNestedPass<IREE::Flow::ExecutableOp>(
      IREE::Util::createStripDebugOpsPass());

  // Cleanup identity ops that clutter up the IR and canonicalize.
  FunctionLikeNest(passManager).addPass(mlir::createCanonicalizerPass);

  // Deduplicate executables created from dispatch regions.
  // Note: this only deduplicates equivalent executables. We could in addition
  // generalize executables to prune further (e.g. by promoting a dimension to
  // an argument if two executables differ only in that one dimension).
  passManager.addPass(IREE::Flow::createDeduplicateExecutablesPass());

  // Create one function per exported program entry point that can be used with
  // iree-benchmark-module to benchmark each function individually. Whether
  // a model supports execution like this (handles zero/null args, has state
  // resets, etc) is up to the author.
  if (clExportBenchmarkFuncs) {
    passManager.addPass(IREE::Flow::createExportBenchmarkFuncsPass());
  }

  FunctionLikeNest(passManager)
      // Inject tracing that logs both input and output tensors from all
      // dispatches. We do this after deduping so that the executable names
      // match later stages.
      .addPredicatedPass(clTraceDispatchTensors,
                         IREE::Flow::createInjectDispatchTracingPass)
      // Cleanup the IR after we are done.
      .addPass(IREE::Flow::createCleanupTensorShapesPass)
      .addPass(mlir::createCanonicalizerPass)
      .addPass(mlir::createCSEPass);

  passManager.addNestedPass<IREE::Flow::ExecutableOp>(
      mlir::createCanonicalizerPass());
  passManager.addNestedPass<IREE::Flow::ExecutableOp>(mlir::createCSEPass());

  // Symbol DCE any remaining variables/functions that are now no longer
  // required.
  passManager.addPass(mlir::createSymbolDCEPass());

  /// Print the dispatch graph in the Graphviz format.
  if (clDumpDispatchGraph) {
    std::string errorMessage;
    static auto dotFile =
        openOutputFile(clDumpDispatchGraphOutputFile, &errorMessage);
    if (!dotFile) {
      llvm::errs() << errorMessage << "\n";
    } else {
      passManager.addPass(
          IREE::Flow::createDumpDispatchGraphPass(dotFile->os()));
      dotFile->keep();
    }
  }
}

void registerFlowTransformPassPipeline() {
  PassPipelineRegistration<TransformOptions> transformPassPipeline(
      "iree-flow-transformation-pipeline",
      "Runs the full IREE flow dialect transformation pipeline",
      [](OpPassManager &passManager, const TransformOptions &transformOptions) {
        buildFlowTransformPassPipeline(passManager, transformOptions);
      });
}

namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"  // IWYU pragma: export
}  // namespace

void registerFlowPasses() {
  // Generated.
  registerPasses();

  // Pipelines.
  registerFlowTransformPassPipeline();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
