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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"

#include <circt/Dialect/Comb/CombDialect.h>
#include <circt/Dialect/HW/HWDialect.h>
#include <circt/Dialect/HW/HWOps.h>
#include <mlir/Analysis/TopologicalSortUtils.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace mlir;

namespace spechls {
#define GEN_PASS_DEF_SPLITWITHSYNCPASS
#include "Dialect/SpecHLS/Transforms/Passes.h.inc"

} // namespace spechls

namespace {

struct SplitOperationPattern : RewritePattern {
  double clockPeriod;

  SplitOperationPattern(double clockPeriod, PatternBenefit benefit, MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context), clockPeriod(clockPeriod) {}

  LogicalResult matchAndRewrite(mlir::Operation *operation, PatternRewriter &rewriter) const override {
    if (auto combDelayAttr = operation->getAttrOfType<mlir::FloatAttr>("spechls.combDelay")) {
      auto combDelay = combDelayAttr.getValueAsDouble();
      if (combDelay > clockPeriod) {
        mlir::Type type = operation->getResultTypes()[0];
        mlir::Value current = operation->getResult(0);
        mlir::Operation *firstSync = nullptr;
        operation->setAttr("spechls.combDelay", rewriter.getF64FloatAttr(clockPeriod));
        combDelay -= clockPeriod;
        auto ip = rewriter.saveInsertionPoint();
        rewriter.setInsertionPointAfter(operation);
        while (combDelay > 0) {
          auto sync = rewriter.create<spechls::SyncOp>(rewriter.getUnknownLoc(), type, current);
          if (firstSync == nullptr)
            firstSync = sync;
          double currentCombDelay = std::min(clockPeriod, combDelay);
          sync->setAttr("spechls.combDelay", rewriter.getF64FloatAttr(currentCombDelay));
          current = sync.getResult();
          combDelay -= currentCombDelay;
        }
        rewriter.replaceOpUsesWithIf(operation, llvm::ArrayRef<mlir::Value>{current},
                                     [&](mlir::OpOperand &operand) -> bool { return operand.getOwner() != firstSync; });
        rewriter.restoreInsertionPoint(ip);
        return llvm::success();
      }
    }
    return llvm::failure();
  }
};

struct SplitWithSyncPass : public spechls::impl::SplitWithSyncPassBase<SplitWithSyncPass> {
  using SplitWithSyncPassBase::SplitWithSyncPassBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    auto kernel = getOperation();

    mlir::FrozenRewritePatternSet patterns;
    RewritePatternSet patternList{ctx};
    patternList.add<SplitOperationPattern>(clockPeriod, 1, ctx);
    patterns = std::move(patternList);
    if (task == "") {
      if (failed(applyPatternsGreedily(kernel, patterns)))
        return signalPassFailure();
    } else {
      kernel.walk([&](spechls::TaskOp t) {
        if (t.getSymName().str() == task) {
          if (failed(applyPatternsGreedily(t, patterns)))
            return signalPassFailure();
        }
      });
    }
  }
};

} // namespace