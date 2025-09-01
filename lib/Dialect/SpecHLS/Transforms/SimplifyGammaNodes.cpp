//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include <circt/Support/LLVM.h>
#include <mlir/Dialect/PDL/IR/PDL.h>
#include <mlir/Dialect/PDLInterp/IR/PDLInterp.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/TypeID.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "Dialect/SpecHLS/IR/SpecHLS.h"
#include "Dialect/SpecHLS/Transforms/Passes.h"

#include "TestPDLLPatterns.h.inc"

namespace {

struct SimplifyGammaNodesPass : public mlir::PassWrapper<SimplifyGammaNodesPass, mlir::OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SimplifyGammaNodesPass);

  mlir::FrozenRewritePatternSet patterns;

  mlir::StringRef getArgument() const final { return "simplify-gamma-nodes"; }
  mlir::StringRef getDescription() const final { return "Simplify gamma nodes"; }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::pdl::PDLDialect, mlir::pdl_interp::PDLInterpDialect, spechls::SpecHLSDialect>();
  }

  mlir::LogicalResult initialize(mlir::MLIRContext *ctx) override {
    // Building the pattern set inside of the `initialize` method pre-compiles the patterns into bytecode. If we don't
    // provide this function, patterns would be recompiled for each `runOnOperation` invocation.
    mlir::RewritePatternSet patternList{ctx};
    populateGeneratedPDLLPatterns(patternList);
    patterns = std::move(patternList);
    return mlir::success();
  }

  void runOnOperation() final { (void)mlir::applyPatternsGreedily(getOperation(), patterns); }
};

} // namespace

namespace spechls {
void registerSpecHLSPDLLPasses() { mlir::PassRegistration<SimplifyGammaNodesPass>(); }
} // namespace spechls
