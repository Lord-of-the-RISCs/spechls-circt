//===- SplitDelays.cpp - CIRCT Global Pass Registration ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definition of the merge-delay pass
//
//===----------------------------------------------------------------------===//

#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include <mlir/Dialect/PDL/IR/PDL.h>
#include <mlir/Dialect/PDLInterp/IR/PDLInterp.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/TypeID.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <Dialect/SpecHLS/IR/SpecHLSOps.h>

using namespace mlir;

namespace wcet {
#define GEN_PASS_DEF_SPLITDELAYSPASS
#include "Dialect/Wcet/Transforms/Passes.h.inc"
} // namespace wcet

#include "SplitDelays.h.inc"

namespace {

struct SplitDelaysPass : public wcet::impl::SplitDelaysPassBase<SplitDelaysPass> {
  FrozenRewritePatternSet patterns;

  using SplitDelaysPassBase::SplitDelaysPassBase;

  LogicalResult initialize(MLIRContext *ctx) override {
    RewritePatternSet patternList{ctx};
    registerNativeRewrite(patternList);
    populateGeneratedPDLLPatterns(patternList);
    patterns = std::move(patternList);
    return success();
  }

  void registerNativeRewrite(RewritePatternSet &patterns) {
    patterns.getPDLPatterns().registerRewriteFunction("SplitDelays", splitDelaysImp);
  }

  void runOnOperation() override { (void)applyPatternsGreedily(getOperation(), patterns); }

private:
  static Operation *splitDelaysImp(PatternRewriter &rewriter, Operation *op) {
    spechls::DelayOp root = cast<spechls::DelayOp>(op);
    auto oneDelay = rewriter.create<spechls::DelayOp>(rewriter.getUnknownLoc(), root.getType(), root.getInput(), 1,
                                                      root.getEnable(), root.getInit());
    auto rootMinusOne =
        rewriter.create<spechls::DelayOp>(rewriter.getUnknownLoc(), root.getType(), oneDelay.getResult(),
                                          root.getDepth() - 1, root.getEnable(), root.getInit());
    return rootMinusOne;
  }
};
} // namespace
