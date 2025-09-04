//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Dialect/SpecHLS/Transforms/Passes.h" // IWYU pragma: keep

#include <circt/Dialect/Comb/CombOps.h>
#include <circt/Dialect/HW/HWOps.h>
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

#include "Dialect/SpecHLS/IR/SpecHLSOps.h"
#include "Utils.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;

namespace spechls {
#define GEN_PASS_DEF_MERGELUTSPASS
#include "Dialect/SpecHLS/Transforms/Passes.h.inc"
} // namespace spechls

#include "MergeLUTs.h.inc"

namespace {

struct MergeLUTsPass : public spechls::impl::MergeLUTsPassBase<MergeLUTsPass> {
  FrozenRewritePatternSet patterns;

  using MergeLUTsPassBase::MergeLUTsPassBase;

  LogicalResult initialize(MLIRContext *ctx) override {
    // Building the pattern set inside of the `initialize` method pre-compiles the patterns into bytecode. If we don't
    // provide this function, patterns would be recompiled for each `runOnOperation` invocation.
    RewritePatternSet patternList{ctx};
    registerNativeRewrite(patternList);
    populateGeneratedPDLLPatterns(patternList);
    patterns = std::move(patternList);
    return success();
  }

  void registerNativeRewrite(RewritePatternSet &patterns) {
    patterns.getPDLPatterns().registerRewriteFunction("CollapseLUTs", collapseLUTsImpl);
    patterns.getPDLPatterns().registerRewriteFunction("GetLUTValue", getLUTValueImpl);
    patterns.getPDLPatterns().registerRewriteFunction("Concat", concatImpl);
  }

  void runOnOperation() override { (void)applyPatternsGreedily(getOperation(), patterns); }

private:
  static Operation *collapseLUTsImpl(PatternRewriter &rewriter, Operation *op) {
    auto root = cast<spechls::LUTOp>(op);
    auto arg = cast<spechls::LUTOp>(root.getIndex().getDefiningOp());

    ArrayRef<int64_t> rootContents = root.getContents();
    ArrayRef<int64_t> argContents = arg.getContents();

    SmallVector<int64_t> collapsedContents;
    for (size_t i = 0; i < argContents.size(); ++i) {
      collapsedContents.push_back(rootContents[argContents[i]]);
    }

    return rewriter.create<spechls::LUTOp>(root.getLoc(), root.getType(), arg.getIndex(), collapsedContents);
  }

  static Attribute getLUTValueImpl(PatternRewriter &rewriter, Attribute n, Operation *op) {
    uint64_t index = cast<IntegerAttr>(n).getValue().getZExtValue();
    auto root = cast<spechls::LUTOp>(op);
    return rewriter.getIntegerAttr(root.getType(), root.getContents()[index]);
  }

  static Operation *concatImpl(PatternRewriter &rewriter, Operation *op) {
    auto root = cast<circt::comb::ConcatOp>(op);

    size_t lutIndex = 0;
    spechls::LUTOp lut{};
    size_t beforeWidth = 0, afterWidth = 0;
    for (size_t i = 0; i < root.getNumOperands(); ++i) {
      Value arg = root.getOperand(i);
      if (!lut && (lut = dyn_cast<spechls::LUTOp>(arg.getDefiningOp()))) {
        lutIndex = i;
      } else if (!lut) {
        beforeWidth += arg.getType().getIntOrFloatBitWidth();
      } else {
        afterWidth += arg.getType().getIntOrFloatBitWidth();
      }
    }

    SmallVector<Value> concatOperands;
    concatOperands.reserve(root.getNumOperands());
    for (size_t i = 0; i < root.getNumOperands(); ++i) {
      if (i == lutIndex) {
        concatOperands.push_back(lut.getIndex());
      } else {
        concatOperands.push_back(root.getOperand(i));
      }
    }

    // TODO: Verify that LUT contents is a power of two, and the index is of the minimal bitwidth.
    auto concat = rewriter.create<circt::comb::ConcatOp>(root.getLoc(), concatOperands);
    size_t concatIndexWidth = concat.getType().getIntOrFloatBitWidth();

    ArrayRef<int64_t> oldLutContents = lut.getContents();
    SmallVector<int64_t> newLutContents(1ull << concatIndexWidth);
    for (size_t before = 0; before <= APInt::getMaxValue(beforeWidth).getZExtValue(); ++before) {
      for (size_t after = 0; after <= APInt::getMaxValue(afterWidth).getZExtValue(); ++after) {
        for (size_t middle = 0; middle < lut.getContents().size(); ++middle) {
          // Build the concatenated index.
          size_t k = before;
          k <<= lut.getIndex().getType().getWidth();
          k |= middle;
          k <<= afterWidth;
          k |= after;

          // Build the concatenated value.
          size_t value = before;
          value <<= lut.getType().getWidth();
          value |= oldLutContents[middle];
          value <<= afterWidth;
          value |= after;

          newLutContents[k] = value;
        }
      }
    }

    return rewriter.create<spechls::LUTOp>(root.getLoc(), root.getType(), concat, newLutContents);
  }
};

} // namespace
