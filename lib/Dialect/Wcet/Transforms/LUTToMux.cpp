//===- LUTToMuxcpp - CIRCT Global Pass Registration ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definition of the lut-to-mux pass
//
//===----------------------------------------------------------------------===//

#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
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
#define GEN_PASS_DEF_LUTTOMUXPASS
#include "Dialect/Wcet/Transforms/Passes.h.inc"
} // namespace wcet

#include "LUTToMux.h.inc"

namespace {

struct LUTToMuxPass : public wcet::impl::LUTToMuxPassBase<LUTToMuxPass> {
  FrozenRewritePatternSet patterns;

  using LUTToMuxPassBase::LUTToMuxPassBase;

  LogicalResult initialize(MLIRContext *ctx) override {
    RewritePatternSet patternList{ctx};
    registerNativeRewrite(patternList);
    populateGeneratedPDLLPatterns(patternList);
    patterns = std::move(patternList);
    return success();
  }

  void registerNativeRewrite(RewritePatternSet &patterns) {
    patterns.getPDLPatterns().registerRewriteFunction("ReplaceLut", replaceLutImpl);
  }

  void runOnOperation() override { (void)applyPatternsGreedily(getOperation(), patterns); }

private:
  static Operation *replaceLutImpl(PatternRewriter &rewriter, Operation *op) {
    auto root = cast<spechls::LUTOp>(op);
    Operation *result = nullptr;
    if (root.getIndex().getType().getWidth() == 1) {
      result = oneBitLut(rewriter, root);
    } else {
      result = multiBitsLut(rewriter, root);
    }
    assert(result != nullptr);

    rewriter.replaceAllOpUsesWith(op, result);

    return result;
  }

  static Operation *multiBitsLut(PatternRewriter &rewriter, spechls::LUTOp lut) {
    auto muxCtrl =
        rewriter.create<circt::comb::ExtractOp>(rewriter.getUnknownLoc(), rewriter.getI1Type(), lut.getIndex(), 0);
    auto lutsCtrl = rewriter.create<circt::comb::ExtractOp>(
        rewriter.getUnknownLoc(), rewriter.getIntegerType(lut.getIndex().getType().getWidth() - 1), lut.getIndex(), 1);

    SmallVector<long> oddLutContents;
    SmallVector<long> evenLutContents;
    for (auto it : llvm::enumerate(lut.getContents())) {
      if (it.index() % 2 == 0)
        evenLutContents.push_back(it.value());
      else
        oddLutContents.push_back(it.value());
    }

    auto evenLut = rewriter.create<spechls::LUTOp>(rewriter.getUnknownLoc(), lut.getType(), lutsCtrl, evenLutContents);
    auto oddLut = rewriter.create<spechls::LUTOp>(rewriter.getUnknownLoc(), lut.getType(), lutsCtrl, oddLutContents);
    auto result =
        rewriter.create<circt::comb::MuxOp>(rewriter.getUnknownLoc(), lut.getType(), muxCtrl, oddLut, evenLut);
    return result;
  }

  static Operation *oneBitLut(PatternRewriter &rewriter, spechls::LUTOp lut) {
    assert(lut.getIndex().getType().getWidth() == 1);
    auto flsValue =
        rewriter.create<circt::hw::ConstantOp>(rewriter.getUnknownLoc(), lut.getType(), lut.getContents().front());
    auto truValue =
        rewriter.create<circt::hw::ConstantOp>(rewriter.getUnknownLoc(), lut.getType(), lut.getContents().back());
    auto result = rewriter.create<circt::comb::MuxOp>(rewriter.getUnknownLoc(), lut.getType(), lut.getIndex(), truValue,
                                                      flsValue);
    return result;
  }
};
} // namespace
