//
// This file is part of the SpecHLS project.
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#include "Dialect/SpecHLS/IR/SpecHLS.h"
#include "Dialect/SpecHLS/IR/SpecHLSOps.h"
#include "Dialect/SpecHLS/Transforms/Passes.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"

#include <circt/Dialect/Comb/CombDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>

using namespace mlir;

namespace spechls {
#define GEN_PASS_DEF_SIMPLIFYCOMBPASS
#include "Dialect/SpecHLS/Transforms/Passes.h.inc"

} // namespace spechls

namespace {

struct OrConcatFalsePattern : public OpRewritePattern<circt::comb::OrOp> {
  using OpRewritePattern<circt::comb::OrOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(circt::comb::OrOp op, PatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    auto resType = op.getResult().getType();
    unsigned resBw = resType.getIntOrFloatBitWidth();

    llvm::SmallVector<int> valuePositions(resBw, -1);

    for (unsigned operandIndex = 0; operandIndex < op.getInputs().size(); ++operandIndex) {
      if (auto concat = llvm::dyn_cast_or_null<circt::comb::ConcatOp>(op.getInputs()[operandIndex].getDefiningOp())) {
        for (unsigned index = 0; index < concat.getInputs().size(); ++index) {
          bool isValue = false;
          auto concatArg = concat.getInputs()[index];
          auto concatArgType = concatArg.getType();
          if (concatArgType.getIntOrFloatBitWidth() != 1)
            return llvm::failure();
          auto *concatArgOp = concatArg.getDefiningOp();
          if (concatArgOp) {
            if (auto constOp = llvm::dyn_cast<circt::hw::ConstantOp>(concatArgOp)) {
              if (constOp.getValue().getBoolValue()) {
                isValue = true;
              }
            } else {
              isValue = true;
            }
          } else {
            isValue = true;
          }

          if (isValue) {
            if (valuePositions[index] != -1)
              return llvm::failure();
            valuePositions[index] = operandIndex;
          }
        }
      } else {
        return llvm::failure();
      }
    }
    // There is no conflict between the concats, the operation can be simplified as an or.
    llvm::SmallVector<mlir::Value> inputs;
    for (unsigned i = 0; i < resBw; ++i) {
      if (valuePositions[i] == -1) {
        auto falseOp = rewriter.create<circt::hw::ConstantOp>(op->getLoc(), mlir::APInt(1, 0, false));
        inputs.push_back(falseOp.getResult());
      } else {
        inputs.push_back(
            llvm::dyn_cast<circt::comb::ConcatOp>(op.getInputs()[valuePositions[i]].getDefiningOp()).getInputs()[i]);
      }
    }
    rewriter.replaceOpWithNewOp<circt::comb::ConcatOp>(op, inputs);
    return llvm::success();

    return llvm::failure();
  }
};

class SimplifyCombPass : public spechls::impl::SimplifyCombPassBase<SimplifyCombPass> {
public:
  using SimplifyCombPassBase::SimplifyCombPassBase;

  void runOnOperation() override;
};

} // namespace

void SimplifyCombPass::runOnOperation() {

  // First split concat into concat of bit to make the pattern easier
  // Canonicalization is expected to reverse this part if needed.
  mlir::OpBuilder builder(&getContext());
  getOperation().walk([&](circt::comb::ConcatOp concat) {
    bool needRewrite = false;
    llvm::SmallVector<mlir::Value> inputs;
    builder.setInsertionPoint(concat);
    for (auto in : concat.getInputs()) {
      auto inType = in.getType();
      int bw = inType.getIntOrFloatBitWidth();
      if (bw == 1) {
        inputs.push_back(in);
      } else {
        needRewrite = true;
        for (int i = bw; i > 0; --i) {
          auto extract = builder.create<circt::comb::ExtractOp>(in.getLoc(), in, i - 1, 1);
          inputs.push_back(extract.getResult());
        }
      }
    }
    if (needRewrite) {
      concat->setOperands(inputs);
    }
  });

  RewritePatternSet patterns(&getContext());
  patterns.add<OrConcatFalsePattern>(patterns.getContext());
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}