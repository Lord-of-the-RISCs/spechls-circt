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
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"

#include <circt/Dialect/Comb/CombDialect.h>
#include <circt/Dialect/HW/HWDialect.h>
#include <circt/Dialect/HW/HWOps.h>
#include <mlir/Analysis/TopologicalSortUtils.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace mlir;

namespace spechls {
#define GEN_PASS_DEF_MERGETASKSPASS
#include "Dialect/SpecHLS/Transforms/Passes.h.inc"

} // namespace spechls

namespace {

struct MergeTaskPattern : OpRewritePattern<spechls::TaskOp> {
  using OpRewritePattern<spechls::TaskOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(spechls::TaskOp task, PatternRewriter &rewriter) const override {
    if (task->hasAttr("spechls.speculativeTask"))
      return llvm::failure();
    for (unsigned index = 0; index < task.getArgs().size(); ++index) {
      auto arg = task.getArgs()[index];
      if (auto predTask = llvm::dyn_cast_or_null<spechls::TaskOp>(arg.getDefiningOp())) {
        if (!predTask->hasAttr("spechls.speculativeTask")) {
          auto predCommit = llvm::dyn_cast<spechls::CommitOp>(predTask.getBodyBlock()->getTerminator());
          rewriter.setInsertionPointToStart(task.getBodyBlock());
          auto pack = rewriter.create<spechls::PackOp>(rewriter.getUnknownLoc(), predTask.getResult().getType(),
                                                       predCommit.getOperands());
          llvm::SmallVector<mlir::Operation *> toMove;
          for (auto &op : predTask.getBodyBlock()->getOperations()) {
            if (!llvm::isa<spechls::CommitOp>(op))
              toMove.push_back(&op);
          }
          for (auto *op : toMove) {
            rewriter.moveOpBefore(op, pack);
          }
          task.getBodyBlock()->getArgument(index).replaceAllUsesWith(pack.getResult());
          task.getArgsMutable().erase(index);
          task.getBodyBlock()->eraseArgument(index);
          task.getArgsMutable().append(predTask.getArgs());
          for (auto arg : predTask.getBodyBlock()->getArguments()) {
            auto newArg = task.getBodyBlock()->addArgument(arg.getType(), arg.getLoc());
            arg.replaceAllUsesWith(newArg);
          }
          rewriter.eraseOp(predTask);
          return llvm::success();
        }
      }
    }
    return llvm::failure();
  }
};

struct MergeTasksPass : public spechls::impl::MergeTasksPassBase<MergeTasksPass> {
  using MergeTasksPassBase::MergeTasksPassBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    auto kernel = getOperation();

    mlir::FrozenRewritePatternSet patterns;
    RewritePatternSet patternList{ctx};
    patternList.add<MergeTaskPattern>(ctx);
    patterns = std::move(patternList);
    if (failed(applyPatternsGreedily(kernel, patterns)))
      return signalPassFailure();
  }
};

} // namespace