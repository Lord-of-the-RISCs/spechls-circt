//===- ConstantProp.cpp - CIRCT Global Pass Registration ---------*- C++ -*-===//
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
#include <circt/Dialect/HW/HWOps.h>

using namespace mlir;

namespace wcet {
#define GEN_PASS_DEF_CONSTANTPROPPASS
#include "Dialect/Wcet/Transforms/Passes.h.inc"
} // namespace wcet

#include "ConstantProp.h.inc"

namespace {

struct ConstantPropPass : public wcet::impl::ConstantPropPassBase<ConstantPropPass> {
  FrozenRewritePatternSet patterns;

  using ConstantPropPassBase::ConstantPropPassBase;

  LogicalResult initialize(MLIRContext *ctx) override {
    RewritePatternSet patternList{ctx};
    registerNativeRewrite(patternList);
    populateGeneratedPDLLPatterns(patternList);
    patterns = std::move(patternList);
    return success();
  }

  void registerNativeRewrite(RewritePatternSet &patterns) {
    patterns.getPDLPatterns().registerRewriteFunction("GammaFold", gammaFoldImp);
    patterns.getPDLPatterns().registerRewriteFunction("LUTFold", lutFoldImp);
  }

  void runOnOperation() override { (void)applyPatternsGreedily(getOperation(), patterns); }

private:
  static Value gammaFoldImp(PatternRewriter &rewriter, Operation *op) {
    auto top = dyn_cast<spechls::GammaOp>(op);
    auto selectOp = dyn_cast<circt::hw::ConstantOp>(top.getSelect().getDefiningOp());
    auto idxAttr = selectOp.getValueAttr();
    auto idx = idxAttr.getInt();
    auto sidx = (size_t)(idx & ((1 << idxAttr.getType().getIntOrFloatBitWidth()) - 1));
    if (sidx >= top.getInputs().size()) {
      return top.getInputs().back();
    }
    return top.getInputs()[sidx];
  }

  static Value lutFoldImp(PatternRewriter &rewriter, Operation *op) {
    auto lut = cast<spechls::LUTOp>(op);
    auto selectOp = dyn_cast<circt::hw::ConstantOp>(lut.getIndex().getDefiningOp());

    auto idxAttr = selectOp.getValueAttr();
    size_t idx = (size_t)(idxAttr.getInt() & ((1 << selectOp.getType().getIntOrFloatBitWidth()) - 1));
    auto result =
        rewriter.create<circt::hw::ConstantOp>(rewriter.getUnknownLoc(), lut.getType(), lut.getContents()[idx]);
    return result;
  }
};
} // namespace
