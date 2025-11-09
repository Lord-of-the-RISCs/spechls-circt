//===- InlineTasks.cpp - CIRCT Global Pass Registration ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definition of the UnrollInstr pass
//
//===----------------------------------------------------------------------===//

#include "Dialect/SpecHLS/IR/SpecHLS.h"
#include "Dialect/SpecHLS/IR/SpecHLSOps.h"
#include "Dialect/SpecHLS/IR/SpecHLSTypes.h"
#include "Dialect/Wcet/IR/WcetOps.h"
#include "Dialect/Wcet/Transforms/Passes.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <cstdio>
#include <memory>
#include <string>

using namespace mlir;
using namespace spechls;
using namespace circt;

namespace wcet {
#define GEN_PASS_DEF_INLINETASKSPASS
#include "Dialect/Wcet/Transforms/Passes.h.inc"
} // namespace wcet

namespace {} // namespace

namespace wcet {

struct InlineTasksPattern : OpRewritePattern<spechls::KernelOp> {
  using OpRewritePattern<spechls::KernelOp>::OpRewritePattern;

  // Constructor to save pass arguments
  InlineTasksPattern(MLIRContext *ctx) : OpRewritePattern<spechls::KernelOp>(ctx) {}

  LogicalResult matchAndRewrite(spechls::KernelOp kernel, PatternRewriter &rewriter) const override {
    if (kernel.getOps<spechls::TaskOp>().empty())
      return failure();
    kernel->walk([&](spechls::TaskOp top) {
      rewriter.setInsertionPointAfter(top);
      top->walk([&](Operation *p) {
        if (p->getName().getStringRef() != spechls::TaskOp::getOperationName() ||
            llvm::dyn_cast<spechls::TaskOp>(p).getSymName() != top.getSymName()) {
          auto *newOp = rewriter.clone(*p);
          rewriter.replaceAllOpUsesWith(p, newOp);
          if (p->getName().getStringRef() == spechls::CommitOp::getOperationName()) {
            rewriter.replaceAllUsesWith(top.getResult(), dyn_cast_or_null<spechls::CommitOp>(newOp).getValue());
            rewriter.eraseOp(newOp);
          }
        }
      });

      for (size_t i = 0; i < top->getOperands().size(); i++) {
        rewriter.replaceAllUsesWith(top.getBody().getBlocks().front().getArgument(i), top->getOperand(i));
      }
      rewriter.eraseOp(top);
    });
    return success();
  }
};

struct InlineTasksPass : public impl::InlineTasksPassBase<InlineTasksPass> {

  using InlineTasksPassBase::InlineTasksPassBase;

public:
  void runOnOperation() override {
    auto *ctx = &getContext();
    auto top = getOperation();
    // InlinerInterface interface(ctx);
    // top->walk([&](TaskOp t) {
    //   if (failed(mlir::inlineRegion(interface, &t.getBody(), t, t->getOperands(), t.getResults(), t->getLoc(),
    //   true))) {
    //     llvm::errs() << "Inlining fail\n";
    //   }
    // });

    RewritePatternSet patterns(ctx);
    patterns.add<InlineTasksPattern>(ctx);
    if (failed(applyPatternsGreedily(getOperation()->getParentOp(), std::move(patterns)))) {
      llvm::errs() << "failed\n";
      signalPassFailure();
    }
  }
};

} // namespace wcet
