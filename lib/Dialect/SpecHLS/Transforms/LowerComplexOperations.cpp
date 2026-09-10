//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Dialect/SpecHLS/IR/SpecHLSOps.h"
#include "Dialect/SpecHLS/Transforms/Passes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"

#include <circt/Dialect/Comb/CombDialect.h>
#include <circt/Dialect/HW/HWDialect.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

using namespace mlir;

namespace spechls {
#define GEN_PASS_DEF_LOWERCOMPLEXOPERATIONSPASS
#include "Dialect/SpecHLS/Transforms/Passes.h.inc"

} // namespace spechls

namespace {

auto lowerIntegerDivMod(mlir::Operation *op, PatternRewriter &rewriter) {
  struct {
    mlir::Value quotient, remainder;
  } result;
  mlir::Value lval = op->getOperand(0);
  mlir::Value rval = op->getOperand(1);
  mlir::Type type = rval.getType();
  unsigned bw = type.getIntOrFloatBitWidth();

  mlir::Value Q = rewriter.create<circt::hw::ConstantOp>(op->getLoc(), type, 0);
  mlir::Value R = rewriter.create<circt::hw::ConstantOp>(op->getLoc(), type, 0);

  mlir::Type bitType = rewriter.getIntegerType(1);
  mlir::Type bwM1Type = rewriter.getIntegerType(bw - 1);

  for (int i = bw - 1; i >= 0; --i) {
    mlir::Value lvalBit = rewriter.create<circt::comb::ExtractOp>(op->getLoc(), bitType, lval, i);
    mlir::Value lowRBits = rewriter.create<circt::comb::ExtractOp>(op->getLoc(), bwM1Type, R, 0);
    R = rewriter.create<circt::comb::ConcatOp>(op->getLoc(), lowRBits, lvalBit);
    mlir::Value comp = rewriter.create<circt::comb::ICmpOp>(op->getLoc(), circt::comb::ICmpPredicate::uge, R, rval);
    mlir::Value subR = rewriter.create<circt::comb::SubOp>(op->getLoc(), R, rval);
    mlir::Value orQCst = rewriter.create<circt::hw::ConstantOp>(op->getLoc(), type, (1 << i));
    mlir::Value orQ = rewriter.create<circt::comb::OrOp>(op->getLoc(), Q, orQCst);

    R = rewriter.create<circt::comb::MuxOp>(op->getLoc(), comp, subR, R);
    Q = rewriter.create<circt::comb::MuxOp>(op->getLoc(), comp, orQ, Q);
  }
  result.quotient = Q;
  result.remainder = R;
  return result;
}

struct DivULowering : OpRewritePattern<circt::comb::DivUOp> {
  using OpRewritePattern<circt::comb::DivUOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(circt::comb::DivUOp div, PatternRewriter &rewriter) const override {
    mlir::Value lop = div.getOperand(0);
    mlir::Value rop = div.getOperand(1);
    mlir::Type type = rop.getType();
    if (!llvm::isa<mlir::IntegerType>(type))
      return llvm::failure();

    if (auto constant = llvm::dyn_cast_or_null<circt::hw::ConstantOp>(rop.getDefiningOp())) {
      if (constant.getValue().popcount() == 1) {
        unsigned log2 = constant.getValue().logBase2();
        rewriter.replaceOpWithNewOp<circt::comb::ShrUOp>(
            div, lop, rewriter.create<circt::hw::ConstantOp>(div->getLoc(), type, log2));
        return llvm::success();
      }
    }
    rewriter.replaceOp(div, lowerIntegerDivMod(div, rewriter).quotient);
    return llvm::success();
  }
};

struct ModULowering : OpRewritePattern<circt::comb::ModUOp> {
  using OpRewritePattern<circt::comb::ModUOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(circt::comb::ModUOp mod, PatternRewriter &rewriter) const override {
    mlir::Value lop = mod.getOperand(0);
    mlir::Value rop = mod.getOperand(1);
    mlir::Type type = rop.getType();
    if (!llvm::isa<mlir::IntegerType>(type))
      return llvm::failure();
    if (auto constant = llvm::dyn_cast_or_null<circt::hw::ConstantOp>(rop.getDefiningOp())) {
      if (constant.getValue().popcount() == 1) {
        unsigned vM1 = constant.getValue().getZExtValue() - 1;
        rewriter.replaceOpWithNewOp<circt::comb::AndOp>(
            mod, lop, rewriter.create<circt::hw::ConstantOp>(mod->getLoc(), type, vM1));
        return llvm::success();
      }
    }
    rewriter.replaceOp(mod, lowerIntegerDivMod(mod, rewriter).remainder);
    return llvm::success();
  }
};

struct MulLowering : OpRewritePattern<circt::comb::MulOp> {
  using OpRewritePattern<circt::comb::MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(circt::comb::MulOp mul, PatternRewriter &rewriter) const override {
    mlir::Value lop = mul.getOperand(0);
    mlir::Value rop = mul.getOperand(1);
    mlir::Type type = rop.getType();
    if (!llvm::isa<mlir::IntegerType>(type))
      return llvm::failure();

    if (auto constant = llvm::dyn_cast_or_null<circt::hw::ConstantOp>(rop.getDefiningOp())) {
      if (constant.getValue().popcount() == 1) {
        unsigned log2 = constant.getValue().logBase2();
        rewriter.replaceOpWithNewOp<circt::comb::ShlOp>(
            mul, lop, rewriter.create<circt::hw::ConstantOp>(mul->getLoc(), type, log2));
        return llvm::success();
      }
    }

    unsigned bw = type.getIntOrFloatBitWidth();
    mlir::Type bitType = rewriter.getIntegerType(1);

    mlir::Value extract = rewriter.create<circt::comb::ExtractOp>(mul.getLoc(), bitType, rop, 0);
    mlir::Value zero = rewriter.create<circt::hw::ConstantOp>(mul->getLoc(), type, 0);
    mlir::Value mux = rewriter.create<circt::comb::MuxOp>(mul->getLoc(), extract, lop, zero);
    mlir::Value current = mux;

    for (unsigned i = 1; i < bw; ++i) {
      extract = rewriter.create<circt::comb::ExtractOp>(mul.getLoc(), bitType, rop, i);
      mlir::Value cst = rewriter.create<circt::hw::ConstantOp>(mul->getLoc(), type, i);
      mlir::Value shift = rewriter.create<circt::comb::ShlOp>(mul->getLoc(), lop, cst);
      mux = rewriter.create<circt::comb::MuxOp>(mul->getLoc(), extract, shift, zero);
      current = rewriter.create<circt::comb::AddOp>(mul->getLoc(), current, mux);
    }

    rewriter.replaceOp(mul, current);
    return llvm::success();
  }
};

struct LowerComplexOperationsPass : public spechls::impl::LowerComplexOperationsPassBase<LowerComplexOperationsPass> {
  using LowerComplexOperationsPassBase::LowerComplexOperationsPassBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::FrozenRewritePatternSet patterns;
    RewritePatternSet patternList{ctx};
    patternList.add<MulLowering>(ctx);
    patternList.add<DivULowering>(ctx);
    patternList.add<ModULowering>(ctx);
    patterns = std::move(patternList);
    if (failed(applyPatternsGreedily(getOperation(), patterns)))
      return signalPassFailure();
  }
};

} // namespace
