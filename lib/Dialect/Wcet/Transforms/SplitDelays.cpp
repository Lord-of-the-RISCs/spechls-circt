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

#include <climits>
#include <cstdio>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <map>
#include <stack>
#include <system_error>

#include "Dialect/SpecHLS/IR/SpecHLS.h"
#include "Dialect/SpecHLS/IR/SpecHLSOps.h"
#include "Dialect/Wcet/IR/WcetOps.h"
#include "Dialect/Wcet/Transforms/Passes.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/Utils/InferIntRangeCommon.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace circt;
using namespace spechls;

namespace wcet {
#define GEN_PASS_DEF_SPLITDELAYSPASS
#include "Dialect/Wcet/Transforms/Passes.h.inc"
} // namespace wcet

namespace {} // namespace

namespace wcet {

struct SplitDelaysPattern : OpRewritePattern<spechls::DelayOp> {

  SplitDelaysPattern(MLIRContext *ctx) : OpRewritePattern<spechls::DelayOp>(ctx) {}

  LogicalResult matchAndRewrite(spechls::DelayOp top, PatternRewriter &rewriter) const override {
    if (top.getDepth() == 1)
      return failure();
    rewriter.setInsertionPointAfter(top);
    DelayOp newDelay = rewriter.create<spechls::DelayOp>(
        rewriter.getUnknownLoc(), top.getResult(), rewriter.getUI32IntegerAttr(1), top.getEnable(), top.getInit());
    top.setDepth(top.getDepth() - 1);
    rewriter.replaceAllUsesExcept(top, newDelay.getResult(), newDelay);

    return success();
  }
};

struct SplitDelaysPass : public impl::SplitDelaysPassBase<SplitDelaysPass> {

  using SplitDelaysPassBase::SplitDelaysPassBase;

public:
  void runOnOperation() override {
    auto *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<SplitDelaysPattern>(ctx);
    if (failed(applyPatternsGreedily(getOperation()->getParentOp(), std::move(patterns)))) {
      llvm::errs() << "failed\n";
      signalPassFailure();
    }
  }
};
} // namespace wcet
