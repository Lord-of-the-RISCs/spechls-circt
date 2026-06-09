//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Dialect/SpecHLS/IR/SpecHLS.h"
#include "Dialect/SpecHLS/IR/SpecHLSOps.h"
#include "Dialect/SpecHLS/Transforms/Passes.h"
#include "Dialect/SpecHLS/Transforms/TopologicalSort.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/LogicalResult.h"

#include <circt/Dialect/Comb/CombDialect.h>
#include <circt/Dialect/HW/HWDialect.h>
#include <circt/Dialect/HW/HWOps.h>
#include <mlir/Analysis/TopologicalSortUtils.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace mlir;

namespace spechls {
#define GEN_PASS_DEF_INLINEOPTIMIZEDFUNCBODYPASS
#define GEN_PASS_DEF_INLINEOPTIMIZEDFUNCOPTBODYPASS
#include "Dialect/SpecHLS/Transforms/Passes.h.inc"

} // namespace spechls

namespace {

struct InlinePattern : OpRewritePattern<spechls::OptimizedFuncOp> {

  bool inlineOptBody;

  InlinePattern(mlir::MLIRContext *context, bool inlineOptBody)
      : OpRewritePattern(context), inlineOptBody(inlineOptBody) {}

  LogicalResult matchAndRewrite(spechls::OptimizedFuncOp task, PatternRewriter &rewriter) const override {
    if (inlineOptBody) {
      return task.inlineOptBody(rewriter);
    }
    return task.inlineBody(rewriter);
  }
};

struct InlineOptimizedFuncPass : public spechls::impl::InlineOptimizedFuncBodyPassBase<InlineOptimizedFuncPass> {
  using InlineOptimizedFuncBodyPassBase::InlineOptimizedFuncBodyPassBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    auto kernel = getOperation();

    mlir::FrozenRewritePatternSet patterns;
    RewritePatternSet patternList{ctx};
    patternList.add<InlinePattern>(ctx, false);
    patterns = std::move(patternList);
    if (failed(applyPatternsGreedily(kernel, patterns)))
      return signalPassFailure();
  }
};

struct InlineOptimizedFuncOptBodyPass
    : public spechls::impl::InlineOptimizedFuncOptBodyPassBase<InlineOptimizedFuncOptBodyPass> {
  using InlineOptimizedFuncOptBodyPassBase::InlineOptimizedFuncOptBodyPassBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    auto kernel = getOperation();

    mlir::FrozenRewritePatternSet patterns;
    RewritePatternSet patternList{ctx};
    patternList.add<InlinePattern>(ctx, true);
    patterns = std::move(patternList);
    if (failed(applyPatternsGreedily(kernel, patterns)))
      return signalPassFailure();
  }
};

} // namespace