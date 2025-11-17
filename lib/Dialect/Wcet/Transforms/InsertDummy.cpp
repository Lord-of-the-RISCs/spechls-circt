//===- InsertDummy.cpp - CIRCT Global Pass Registration ---------*- C++ -*-===//
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

#include "Dialect/SpecHLS/IR/SpecHLSOps.h"
#include "Dialect/Wcet/IR/WcetOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include <mlir/Dialect/PDL/IR/PDL.h>
#include <mlir/Dialect/PDLInterp/IR/PDLInterp.h>
#include <mlir/Parser/Parser.h>

using namespace mlir;

namespace wcet {
#define GEN_PASS_DEF_INSERTDUMMYPASS
#include "Dialect/Wcet/Transforms/Passes.h.inc"
} // namespace wcet

#include "InsertDummy.h.inc"

namespace {

struct InsertDummyPass : public wcet::impl::InsertDummyPassBase<InsertDummyPass> {
  FrozenRewritePatternSet patterns;

  using InsertDummyPassBase::InsertDummyPassBase;

  LogicalResult initialize(MLIRContext *ctx) override {
    RewritePatternSet patternList{ctx};
    registerNativeRewrite(patternList);
    populateGeneratedPDLLPatterns(patternList);
    patterns = std::move(patternList);
    return success();
  }

  void registerNativeRewrite(RewritePatternSet &patterns) {
    patterns.getPDLPatterns().registerRewriteFunction("InsertDummy", insertDummyImp);
  }

  void runOnOperation() override { (void)applyPatternsGreedily(getOperation(), patterns); }

private:
  static Operation *insertDummyImp(PatternRewriter &rewritter, Operation *op) {
    auto root = cast<spechls::UnpackOp>(op);
    auto pack = root.getInput().getDefiningOp<spechls::PackOp>();
    auto dummy = rewritter.create<wcet::DummyOp>(rewritter.getUnknownLoc(), root->getResultTypes(), pack.getInputs());
    return dummy;
  }
};

} // namespace
