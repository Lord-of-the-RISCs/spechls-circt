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
#define GEN_PASS_DEF_SIMPLIFYGAMMANODESPASS
#include "Dialect/SpecHLS/Transforms/Passes.h.inc"
} // namespace spechls

#include "SimplifyGammaNodes.h.inc"

namespace {

struct SimplifyGammaNodesPass : public spechls::impl::SimplifyGammaNodesPassBase<SimplifyGammaNodesPass> {
  FrozenRewritePatternSet patterns;

  using SimplifyGammaNodesPassBase::SimplifyGammaNodesPassBase;

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
    int64_t n = cast<IntegerAttr>(attr).getValue().getSExtValue();
    return *std::next(args.begin(), n);
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
    for (size_t i = 0; i < root.getInputs().size(); ++i) {
      for (size_t j = 0; j < g.getInputs().size(); ++j) {
        APInt lutIndex(lutIndexWidth, i);
        lutIndex <<= gControlWidth;
        lutIndex |= j;
        size_t k = lutIndex.getZExtValue();
        if (i < idx) {
          lutContents[k] = i;
        } else if (i == idx) {
          lutContents[k] = idx + j;
        } else {
          lutContents[k] = i + g.getInputs().size() - 1;
        }
      }
    }

    Location loc = root.getLoc();
    auto lutIndex = rewriter.create<circt::comb::ConcatOp>(
        loc, rewriter.create<circt::comb::ExtractOp>(loc, root.getSelect(), 0, rootControlWidth),
        rewriter.create<circt::comb::ExtractOp>(loc, g.getSelect(), 0, gControlWidth));
    auto lut =
        rewriter.create<spechls::LUTOp>(loc, lutIndex.getType(), lutIndex, rewriter.getDenseI64ArrayAttr(lutContents));

    auto result = rewriter.create<spechls::GammaOp>(loc, root.getType(), root.getSymName(), lut, inputs);
    return result;
  }
};

} // namespace
