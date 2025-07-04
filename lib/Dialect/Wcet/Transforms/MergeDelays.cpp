//===- MergeDelays.cpp - CIRCT Global Pass Registration ---------*- C++ -*-===//
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
#define GEN_PASS_DEF_MERGEDELAYSPASS
#include "Dialect/Wcet/Transforms/Passes.h.inc"
} // namespace wcet

namespace {} // namespace

namespace wcet {

struct MergeDelaysPattern : OpRewritePattern<spechls::KernelOp> {

  MergeDelaysPattern(MLIRContext *ctx) : OpRewritePattern<spechls::KernelOp>(ctx) {}

  LogicalResult matchAndRewrite(spechls::KernelOp top, PatternRewriter &rewriter) const override {
    llvm::errs() << "start merge delays patterns \n";
    llvm::SmallVector<DelayOp> delays = llvm::SmallVector<DelayOp>();

    top->walk([&](DelayOp delay) {
      auto *input = delay.getInput().getDefiningOp();
      if (auto delayOp = dyn_cast_or_null<spechls::DelayOp>(input)) {
        delays.push_back(delay);
      }
    });

    llvm::errs() << delays.size() << "\n";

    if (delays.size() == 0)
      return failure();

    for (DelayOp delay : delays) {
      DelayOp pred = llvm::cast<DelayOp>(delay.getInput().getDefiningOp());
      delay.setDepth(delay.getDepth() + pred.getDepth());
      delay->setOperand(0, pred.getInput());
      if (pred->getUses().empty())
        rewriter.eraseOp(pred);
    }

    return success();
  }
};

struct MergeDelaysPass : public impl::MergeDelaysPassBase<MergeDelaysPass> {

public:
  void runOnOperation() override {
    auto *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<MergeDelaysPattern>(ctx);
    if (failed(applyPatternsGreedily(getOperation()->getParentOp(), std::move(patterns)))) {
      llvm::errs() << "failed\n";
      signalPassFailure();
    }
  }
};
} // namespace wcet
