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

using namespace mlir;

namespace spechls {
#define GEN_PASS_DEF_SIMPLIFYGAMMASPASS
#include "Dialect/SpecHLS/Transforms/Passes.h.inc"
} // namespace spechls

#include "SimplifyGammas.h.inc"

namespace {

struct SimplifyGammasPass : public spechls::impl::SimplifyGammasPassBase<SimplifyGammasPass> {
  FrozenRewritePatternSet patterns;

  using SimplifyGammasPassBase::SimplifyGammasPassBase;

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
    patterns.getPDLPatterns().registerRewriteFunction("GetSelectedInput", getSelectedInputImpl);
    patterns.getPDLPatterns().registerRewriteFunction("MergeGammaNodes", mergeGammaNodesImpl);
  }

  void runOnOperation() override { (void)applyPatternsGreedily(getOperation(), patterns); }

private:
  static Value getSelectedInputImpl(PatternRewriter &rewriter, Attribute attr, ValueRange args) {
    uint64_t n = cast<IntegerAttr>(attr).getValue().getZExtValue();
    return *std::next(args.begin(), n);
  }

  static mlir::Value zextTo(mlir::Location loc, mlir::PatternRewriter &rewriter, mlir::Value v, unsigned targetWidth) {
    auto srcTy = llvm::cast<mlir::IntegerType>(v.getType());
    unsigned srcWidth = srcTy.getWidth();
    if (srcWidth == targetWidth)
      return v;

    if (srcWidth > targetWidth) {
      return rewriter.create<circt::comb::ExtractOp>(loc, v, 0, targetWidth);
    }

    assert(srcWidth < targetWidth && "zextTo only supports widening");

    auto padWidth = targetWidth - srcWidth;
    auto padTy = rewriter.getIntegerType(padWidth);
    auto zeroPad = rewriter.create<circt::hw::ConstantOp>(loc, padTy, rewriter.getIntegerAttr(padTy, 0));

    // MSBs first for concat: zero padding on top, then original value.
    return rewriter.create<circt::comb::ConcatOp>(loc, zeroPad, v);
  }

  static Operation *mergeGammaNodesImpl(PatternRewriter &rewriter, Operation *op) {
    auto root = cast<spechls::GammaOp>(op);

    spechls::GammaOp g{};
    size_t idx = 0;
    // Look for the first input produced by a gamma node.
    for (auto &&arg : root.getInputs()) {
      if ((g = dyn_cast_if_present<spechls::GammaOp>(arg.getDefiningOp())))
        break;
      ++idx;
    }

    SmallVector<Value> inputs;
    inputs.append(root.getInputs().begin(), root.getInputs().begin() + idx);
    inputs.append(g.getInputs().begin(), g.getInputs().end());
    if (idx < root.getInputs().size()) {
      inputs.append(root.getInputs().begin() + idx + 1, root.getInputs().end());
    }

    size_t rootControlWidth = utils::getMinBitwidth(root.getInputs().size() - 1);
    size_t gControlWidth = utils::getMinBitwidth(g.getInputs().size() - 1);
    size_t lutIndexWidth = rootControlWidth + gControlWidth;

    SmallVector<int64_t> lutContents(1 << lutIndexWidth);
    int64_t maxValue = 0;
    for (size_t i = 0; i < root.getInputs().size(); ++i) {
      for (size_t j = 0; j < g.getInputs().size(); ++j) {
        APInt lutIndex(lutIndexWidth, i);
        lutIndex <<= gControlWidth;
        lutIndex |= j;

        size_t k = lutIndex.getZExtValue();
        int64_t value = i;
        if (i == idx) {
          value = idx + j;
        } else if (i > idx) {
          value = i + g.getInputs().size() - 1;
        }

        if (value > maxValue)
          maxValue = value;

        lutContents[k] = value;
      }
    }

    size_t lutOutputWidth = utils::getMinBitwidth(maxValue);

    Location loc = root.getLoc();
    unsigned outW = lutOutputWidth;
    assert(outW < 100 && "outw test");
    assert(outW > 0 && "outw test");
    auto outTy = rewriter.getIntegerType(outW);

    auto rootCntrl = zextTo(loc, rewriter, root.getSelect(), outW);
    auto gCntrl = zextTo(loc, rewriter, g.getSelect(), outW);

    auto cstI = rewriter.create<circt::hw::ConstantOp>(loc, outTy, rewriter.getIntegerAttr(outTy, idx));
    auto cstShift =
        rewriter.create<circt::hw::ConstantOp>(loc, outTy, rewriter.getIntegerAttr(outTy, g.getInputs().size() - 1));

    auto isLt = rewriter.create<circt::comb::ICmpOp>(loc, circt::comb::ICmpPredicate::ult, rootCntrl, cstI);
    auto isEq = rewriter.create<circt::comb::ICmpOp>(loc, circt::comb::ICmpPredicate::eq, rootCntrl, cstI);
    auto aPlusB = rewriter.create<circt::comb::AddOp>(loc, outTy, mlir::ValueRange{rootCntrl, gCntrl});
    auto aPlusShift = rewriter.create<circt::comb::AddOp>(loc, outTy, mlir::ValueRange{rootCntrl, cstShift});
    auto geCase = rewriter.create<circt::comb::MuxOp>(loc, isEq, aPlusB, aPlusShift);
    auto newSel = rewriter.create<circt::comb::MuxOp>(loc, isLt, rootCntrl, geCase);

    /* auto lutIndex = rewriter.create<circt::comb::ConcatOp>(
         loc, rewriter.create<circt::comb::ExtractOp>(loc, root.getSelect(), 0, rootControlWidth),
         rewriter.create<circt::comb::ExtractOp>(loc, g.getSelect(), 0, gControlWidth));
     auto lut = rewriter.create<spechls::LUTOp>(loc, rewriter.getIntegerType(lutOutputWidth), lutIndex,
                                                rewriter.getDenseI64ArrayAttr(lutContents));*/

    auto result = rewriter.create<spechls::GammaOp>(loc, root.getType(), root.getSymName(), newSel, inputs);
    return result;
  }
};

} // namespace
