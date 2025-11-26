//===- GammaBinarization.cpp - CIRCT Global Pass Registration ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definition of the gamma-binarization pass
//
//===----------------------------------------------------------------------===//

#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
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
#include <circt/Dialect/Comb/CombOps.h>
#include <circt/Dialect/HW/HWOps.h>

using namespace mlir;

namespace wcet {
#define GEN_PASS_DEF_GAMMABINARIZATIONPASS
#include "Dialect/Wcet/Transforms/Passes.h.inc"
} // namespace wcet

#include "GammaBinarization.h.inc"

namespace {

struct GammaBinarizationPass : public wcet::impl::GammaBinarizationPassBase<GammaBinarizationPass> {
  FrozenRewritePatternSet patterns;

  using GammaBinarizationPassBase::GammaBinarizationPassBase;

  LogicalResult initialize(MLIRContext *ctx) override {
    RewritePatternSet patternList{ctx};
    registerNativeRewrite(patternList);
    registerNativeConstraints(patternList);
    populateGeneratedPDLLPatterns(patternList);
    patterns = std::move(patternList);
    return success();
  }

  void registerNativeRewrite(RewritePatternSet &patterns) {
    patterns.getPDLPatterns().registerRewriteFunction("GamBinarization", binarizationImpl);
  }

  void registerNativeConstraints(RewritePatternSet &patterns) {
    patterns.getPDLPatterns().registerConstraintFunction("IsMuxTree", isMuxTreeImpl);
  }

  void runOnOperation() override { (void)applyPatternsGreedily(getOperation(), patterns); }

private:
  //====-----------------------------------------------------------------------
  // Constraint Implementation
  //====-----------------------------------------------------------------------
  static LogicalResult isMuxTreeImpl(PatternRewriter &rewriter, Operation *op) {
    SmallVector<Operation *> visited;
    return success(isMuxTreeRec(op, visited));
  }

  static bool isMuxTreeRec(Operation *op, SmallVector<Operation *> visited) {
    // Ensure that there is no cycle.
    if (llvm::any_of(visited, [op](Operation *p) { return p == op; })) {
      return false;
    }
    visited.push_back(op);

    // Base case
    auto constantOp = dyn_cast_or_null<circt::hw::ConstantOp>(op);
    if (constantOp) {
      return true;
    }

    // Recursive case
    auto muxOp = dyn_cast_or_null<circt::comb::MuxOp>(op);
    if (muxOp) {
      return isMuxTreeRec(muxOp.getTrueValue().getDefiningOp(), visited) &&
             isMuxTreeRec(muxOp.getFalseValue().getDefiningOp(), visited);
    }
    return false;
  }

  //====-----------------------------------------------------------------------
  // Rewritter Implementation
  //====-----------------------------------------------------------------------
  static Operation *binarizationImpl(PatternRewriter &rewriter, Operation *op) {
    auto gamma = cast<spechls::GammaOp>(op);
    DenseMap<Operation *, Value> visited;
    auto *result = binarizationRec(rewriter, gamma, gamma.getSelect().getDefiningOp(), visited).getDefiningOp();
    rewriter.replaceAllOpUsesWith(op, result);
    return result;
  }

  static Value binarizationRec(PatternRewriter &rewriter, spechls::GammaOp gamma, Operation *currentOp,
                               DenseMap<Operation *, Value> &visited) {
    // if (llvm::any_of(visited, [currentOp](std::pair<Operation *, Value> p) { return p.first == currentOp; })) {
    //   return visited.lookup(currentOp);
    // }

    auto result = visited.lookup_or(currentOp, nullptr);
    if (result) {
      return result;
    }

    // Base case
    auto constant = dyn_cast_or_null<circt::hw::ConstantOp>(currentOp);
    if (constant) {
      uint idx = constant.getValue().getZExtValue();
      assert(idx < gamma.getInputs().size());
      return gamma->getOperand(idx + 1);
    }

    // Recursive case
    auto mux = dyn_cast_or_null<circt::comb::MuxOp>(currentOp);
    assert(mux);
    auto gammaCtrl = mux.getCond();
    SmallVector<Value> gammaInputs;
    gammaInputs.push_back(binarizationRec(rewriter, gamma, mux.getFalseValue().getDefiningOp(), visited));
    gammaInputs.push_back(binarizationRec(rewriter, gamma, mux.getTrueValue().getDefiningOp(), visited));
    auto binaryGamma = rewriter.create<spechls::GammaOp>(rewriter.getUnknownLoc(), gamma.getType(),
                                                         rewriter.getStringAttr("gammaBin"), gammaCtrl, gammaInputs);
    visited.insert(std::pair(currentOp, binaryGamma.getResult()));
    return binaryGamma.getResult();
  }
};
} // namespace
