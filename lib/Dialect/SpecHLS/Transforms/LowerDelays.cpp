//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Dialect/SpecHLS/IR/SpecHLSOps.h"
#include "Dialect/SpecHLS/Transforms/Passes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/LogicalResult.h"

#include <circt/Dialect/Comb/CombDialect.h>
#include <circt/Dialect/HW/HWDialect.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace mlir;

namespace spechls {
#define GEN_PASS_DEF_LOWERDELAYSPASS
#include "Dialect/SpecHLS/Transforms/Passes.h.inc"

} // namespace spechls

namespace {

struct RollbackableDelayLowering : OpRewritePattern<spechls::RollbackableDelayOp> {
  using OpRewritePattern<spechls::RollbackableDelayOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(spechls::RollbackableDelayOp rbDelay, PatternRewriter &rewriter) const override {
    if (auto cst = llvm::dyn_cast_or_null<circt::hw::ConstantOp>(rbDelay.getRollback().getDefiningOp())) {
      if (!cst.getValue().getBoolValue()) {
        rewriter.replaceOpWithNewOp<spechls::DelayOp>(rbDelay, rbDelay.getInput(), rbDelay.getDepth(),
                                                      rbDelay.getEnable(), rbDelay.getInit());
        return success();
      }
    }
    auto delay = rewriter.create<spechls::DelayOp>(rbDelay.getLoc(), rbDelay.getInput(), rbDelay.getDepth(),
                                                   rbDelay.getEnable(), rbDelay.getInit());
    rewriter.replaceOpWithNewOp<spechls::RollbackOp>(rbDelay, rbDelay.getRollbackDepths(), rbDelay.getOffset(), delay,
                                                     rbDelay.getRollback(), rbDelay.getRbWe());
    return success();
  }
};

struct CancellableDelayLowering : OpRewritePattern<spechls::CancellableDelayOp> {
  using OpRewritePattern<spechls::CancellableDelayOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(spechls::CancellableDelayOp rbDelay, PatternRewriter &rewriter) const override {
    if (auto cst = llvm::dyn_cast_or_null<circt::hw::ConstantOp>(rbDelay.getCancel().getDefiningOp())) {
      if (!cst.getValue().getBoolValue()) {
        rewriter.replaceOpWithNewOp<spechls::DelayOp>(rbDelay, rbDelay.getInput(), rbDelay.getDepth(),
                                                      rbDelay.getEnable(), rbDelay.getInit());
        return success();
      }
    }
    auto delay = rewriter.create<spechls::DelayOp>(rbDelay.getLoc(), rbDelay.getInput(), rbDelay.getDepth(),
                                                   rbDelay.getEnable(), rbDelay.getInit());
    rewriter.replaceOpWithNewOp<spechls::CancelOp>(rbDelay, rbDelay.getOffset(), delay, rbDelay.getCancel(),
                                                   rbDelay.getCancelWe());
    return failure();
  }
};

struct LowerDelaysPass : public spechls::impl::LowerDelaysPassBase<LowerDelaysPass> {
  using LowerDelaysPassBase::LowerDelaysPassBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::FrozenRewritePatternSet patterns;
    RewritePatternSet patternList{ctx};
    patternList.add<RollbackableDelayLowering>(ctx);
    patternList.add<CancellableDelayLowering>(ctx);
    patterns = std::move(patternList);
    if (failed(applyPatternsGreedily(getOperation(), patterns)))
      return signalPassFailure();
  }
};

} // namespace
